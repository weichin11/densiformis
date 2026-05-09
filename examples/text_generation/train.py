from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

import densiformis


PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
DEFAULT_DATA_PATH = Path(__file__).with_name("data.txt")
DEFAULT_CHECKPOINT_ROOT = Path("text_generation_checkpoints")
CONFIG_NAME = "text_config.json"


def read_corpus(path: Path = DEFAULT_DATA_PATH) -> List[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class SimpleBPETokenizer:
    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size
        self.merges: List[tuple[str, str]] = []
        self.token_to_ordered_id: dict[str, int] = {}
        self.ordered_id_to_token: list[str] = []

    def fit(self, texts: Iterable[str]) -> None:
        words = []
        for text in texts:
            words.extend([tuple(word) for word in text.split()])

        special_count = 3
        base_vocab = {" "}
        for word in words:
            base_vocab.update(word)

        token_limit = max(self.vocab_size - special_count, len(base_vocab))
        vocab = set(base_vocab)
        encoded_words = list(words)

        while len(vocab) < token_limit:
            pair_counts = Counter()
            for word in encoded_words:
                pair_counts.update(zip(word, word[1:]))
            if not pair_counts:
                break

            best_pair = None
            merged_token = None
            for pair, _ in pair_counts.most_common():
                candidate = "".join(pair)
                if candidate not in vocab:
                    best_pair = pair
                    merged_token = candidate
                    break
            if best_pair is None or merged_token is None:
                break

            encoded_words = [self._merge_word(word, best_pair, merged_token) for word in encoded_words]
            vocab.add(merged_token)
            self.merges.append(best_pair)

        ordered_tokens = [PAD_TOKEN, EOS_TOKEN, UNK_TOKEN] + sorted(vocab - {PAD_TOKEN, EOS_TOKEN, UNK_TOKEN})
        self.ordered_id_to_token = ordered_tokens
        self.token_to_ordered_id = {token: index for index, token in enumerate(ordered_tokens)}

    def encode(self, text: str, seq_len: int) -> list[int]:
        token_ids = self.tokenize(text)
        token_ids = token_ids[:seq_len]
        token_ids.extend([self.token_to_ordered_id[PAD_TOKEN]] * (seq_len - len(token_ids)))
        return token_ids

    def tokenize(self, text: str) -> list[int]:
        token_ids = []
        for word_index, word in enumerate(text.split()):
            if word_index > 0:
                token_ids.append(self.token_to_ordered_id.get(" ", self.token_to_ordered_id[UNK_TOKEN]))
            tokens = tuple(word)
            for pair in self.merges:
                merged_token = "".join(pair)
                tokens = self._merge_word(tokens, pair, merged_token)
            token_ids.extend(self.token_to_ordered_id.get(token, self.token_to_ordered_id[UNK_TOKEN]) for token in tokens)

        token_ids.append(self.token_to_ordered_id[EOS_TOKEN])
        return token_ids

    def decode(self, token_ids: Iterable[int]) -> str:
        tokens = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self.ordered_id_to_token):
                token = UNK_TOKEN
            else:
                token = self.ordered_id_to_token[token_id]
            if token == EOS_TOKEN:
                break
            if token == PAD_TOKEN:
                continue
            if token == UNK_TOKEN:
                token = "?"
            tokens.append(token)
        return "".join(tokens).strip()

    @staticmethod
    def _merge_word(word: tuple[str, ...], pair: tuple[str, str], merged_token: str) -> tuple[str, ...]:
        merged = []
        index = 0
        while index < len(word):
            if index < len(word) - 1 and (word[index], word[index + 1]) == pair:
                merged.append(merged_token)
                index += 2
            else:
                merged.append(word[index])
                index += 1
        return tuple(merged)


def token_ids_to_one_hot(token_ids: Iterable[int], vocab_size: int) -> torch.Tensor:
    token_tensor = torch.tensor(list(token_ids), dtype=torch.long)
    return F.one_hot(token_tensor, num_classes=vocab_size).float().permute(1, 0)


def categorical_to_token_ids(token_tensor: torch.Tensor) -> list[int]:
    return token_tensor.detach().cpu().argmax(dim=0).long().tolist()


def categorical_to_text(token_tensor: torch.Tensor, tokenizer: SimpleBPETokenizer) -> str:
    return tokenizer.decode(categorical_to_token_ids(token_tensor))


class BPETextDataset(Dataset):
    def __init__(self, texts: Iterable[str], tokenizer: SimpleBPETokenizer, seq_len: int):
        self.texts = list(texts)
        vocab_size = len(tokenizer.ordered_id_to_token)
        self.samples = torch.stack(
            [
                token_ids_to_one_hot(tokenizer.encode(text, seq_len), vocab_size)
                for text in self.texts
            ]
        )

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        return [self.samples[index]]

    def __len__(self) -> int:
        return self.samples.size(0)


def split_corpus(texts: List[str], valid_fraction: float, seed: int) -> tuple[List[str], List[str]]:
    valid_count = max(1, int(round(len(texts) * valid_fraction)))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(texts), generator=generator).tolist()
    valid_indices = set(indices[:valid_count])
    train_texts = [text for index, text in enumerate(texts) if index not in valid_indices]
    valid_texts = [text for index, text in enumerate(texts) if index in valid_indices]
    return train_texts, valid_texts


def group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        groups = group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x + residual


class TextCategoricalDenoiser(nn.Module):
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        hidden_dim: int = 512,
        layers: int = 8,
        kernel_size: int = 17,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self.in_conv = nn.Conv1d(vocab_size + 1, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.blocks = nn.ModuleList(
            [ResidualConvBlock(hidden_dim, kernel_size=kernel_size) for _ in range(layers)]
        )
        self.out_norm = nn.GroupNorm(group_count(hidden_dim), hidden_dim)
        self.out_conv = nn.Conv1d(hidden_dim, vocab_size, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, inputs: List[torch.Tensor], t: List[torch.Tensor]) -> List[torch.Tensor]:
        (token_probs,) = inputs
        (text_t,) = t
        time_channel = text_t.view(-1, 1, 1).expand(-1, 1, self.seq_len)
        y = torch.cat([token_probs, time_channel], dim=1)
        y = self.in_conv(y)
        for block in self.blocks:
            y = block(y)
        y = self.out_norm(y)
        y = F.silu(y)
        y = self.out_conv(y)
        return [y]


def prepare_tokenizer(texts: List[str], vocab_size: int, valid_fraction: float, seed: int):
    train_texts, valid_texts = split_corpus(texts, valid_fraction, seed)
    tokenizer = SimpleBPETokenizer(vocab_size=vocab_size)
    tokenizer.fit(train_texts)
    seq_len = max(len(tokenizer.tokenize(text)) for text in texts)
    return tokenizer, seq_len, train_texts, valid_texts


def save_config(
    checkpoint_root: Path,
    *,
    corpus_size: int,
    seq_len: int,
    token_count: int,
    vocab_size: int,
    hidden_dim: int,
    layers: int,
    kernel_size: int,
    seed: int,
    valid_fraction: float,
) -> None:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    config = {
        "corpus_size": corpus_size,
        "seq_len": seq_len,
        "token_count": token_count,
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "layers": layers,
        "kernel_size": kernel_size,
        "seed": seed,
        "valid_fraction": valid_fraction,
    }
    (checkpoint_root / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main():
    parser = ArgumentParser(description="Train a categorical BPE diffusion model for tiny text generation.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=17)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--checkpoint-save-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--save-period", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    corpus = read_corpus(args.data)
    tokenizer, seq_len, train_texts, valid_texts = prepare_tokenizer(
        corpus,
        vocab_size=args.vocab_size,
        valid_fraction=args.valid_fraction,
        seed=args.seed,
    )
    train_dataset = BPETextDataset(train_texts, tokenizer, seq_len)
    valid_dataset = BPETextDataset(valid_texts, tokenizer, seq_len)

    token_count = len(tokenizer.ordered_id_to_token)
    print(f"BPE vocabulary size: {token_count} tokens")
    print(f"Encoding each BPE token id as a {token_count}-class categorical distribution")
    print(f"Fixed token sequence length: {seq_len}")
    print(f"Corpus size: {len(corpus)} sentences")
    print(f"Training split: {len(train_dataset)} train, {len(valid_dataset)} validation")

    model = TextCategoricalDenoiser(
        seq_len=seq_len,
        vocab_size=token_count,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        kernel_size=args.kernel_size,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    densiformis.diffuser.fit(
        model=model,
        distribution_types=["categorical"],
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        optimizer=optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_save_root=str(args.checkpoint_save_root),
        save_period=args.save_period,
        device=device,
    )

    args.checkpoint_save_root.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint_save_root / "model.pt")
    torch.save(optimizer.state_dict(), args.checkpoint_save_root / "optimizer.pt")
    save_config(
        args.checkpoint_save_root,
        corpus_size=len(corpus),
        seq_len=seq_len,
        token_count=token_count,
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        kernel_size=args.kernel_size,
        seed=args.seed,
        valid_fraction=args.valid_fraction,
    )
    print(f"Final model checkpoint saved: {args.checkpoint_save_root / 'model.pt'}")
    print(f"Checkpoint config saved: {args.checkpoint_save_root / CONFIG_NAME}")


if __name__ == "__main__":
    main()
