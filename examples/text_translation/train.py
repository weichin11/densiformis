from argparse import ArgumentParser
from collections import Counter
import csv
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
DEFAULT_DATA_PATH = Path(__file__).with_name("train.tsv")
DEFAULT_VALID_DATA_PATH = Path(__file__).with_name("valid.tsv")
DEFAULT_CHECKPOINT_ROOT = Path("text_translation_checkpoints")
CONFIG_NAME = "translation_config.json"


def read_parallel_corpus(path: Path = DEFAULT_DATA_PATH) -> List[tuple[str, str]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            english = row["english"].strip()
            japanese = row["japanese"].strip()
            if english and japanese:
                rows.append((english, japanese))
    return rows


class SimpleBPETokenizer:
    def __init__(self, vocab_size: int = 256):
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


class BPETranslationDataset(Dataset):
    def __init__(
        self,
        pairs: Iterable[tuple[str, str]],
        tokenizer: SimpleBPETokenizer,
        english_len: int,
        japanese_len: int,
    ):
        self.pairs = list(pairs)
        vocab_size = len(tokenizer.ordered_id_to_token)
        self.english = torch.stack(
            [
                token_ids_to_one_hot(tokenizer.encode(english, english_len), vocab_size)
                for english, _ in self.pairs
            ]
        )
        self.japanese = torch.stack(
            [
                token_ids_to_one_hot(tokenizer.encode(japanese, japanese_len), vocab_size)
                for _, japanese in self.pairs
            ]
        )

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        return [self.english[index], self.japanese[index]]

    def __len__(self) -> int:
        return self.english.size(0)


def unknown_tokens(
    pairs: Iterable[tuple[str, str]],
    tokenizer: SimpleBPETokenizer,
) -> list[tuple[str, str, str]]:
    unknown_id = tokenizer.token_to_ordered_id[UNK_TOKEN]
    unknown = []
    for english, japanese in pairs:
        if unknown_id in tokenizer.tokenize(english):
            unknown.append(("english", english, japanese))
        if unknown_id in tokenizer.tokenize(japanese):
            unknown.append(("japanese", english, japanese))
    return unknown


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


class TranslationCategoricalDenoiser(nn.Module):
    def __init__(
        self,
        english_len: int,
        japanese_len: int,
        vocab_size: int,
        hidden_dim: int = 512,
        layers: int = 8,
        kernel_size: int = 17,
    ):
        super().__init__()
        self.english_len = english_len
        self.japanese_len = japanese_len
        self.total_len = english_len + japanese_len
        self.vocab_size = vocab_size

        self.in_conv = nn.Conv1d(vocab_size + 2, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.blocks = nn.ModuleList(
            [ResidualConvBlock(hidden_dim, kernel_size=kernel_size) for _ in range(layers)]
        )
        self.out_norm = nn.GroupNorm(group_count(hidden_dim), hidden_dim)
        self.out_conv = nn.Conv1d(hidden_dim, vocab_size, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, inputs: List[torch.Tensor], t: List[torch.Tensor]) -> List[torch.Tensor]:
        english_probs, japanese_probs = inputs
        english_t, japanese_t = t
        token_probs = torch.cat([english_probs, japanese_probs], dim=2)
        english_time = english_t.view(-1, 1, 1).expand(-1, 1, self.english_len)
        japanese_time = japanese_t.view(-1, 1, 1).expand(-1, 1, self.japanese_len)
        time_channel = torch.cat([english_time, japanese_time], dim=2)
        language_channel = torch.cat(
            [
                torch.zeros_like(english_time),
                torch.ones_like(japanese_time),
            ],
            dim=2,
        )
        y = torch.cat([token_probs, time_channel, language_channel], dim=1)
        y = self.in_conv(y)
        for block in self.blocks:
            y = block(y)
        y = self.out_norm(y)
        y = F.silu(y)
        y = self.out_conv(y)
        english, japanese = torch.split(y, [self.english_len, self.japanese_len], dim=2)
        return [english, japanese]


def prepare_tokenizer(
    train_pairs: List[tuple[str, str]],
    valid_pairs: List[tuple[str, str]],
    vocab_size: int,
):
    tokenizer = SimpleBPETokenizer(vocab_size=vocab_size)
    tokenizer.fit([text for pair in train_pairs for text in pair])
    unknown = unknown_tokens(valid_pairs, tokenizer)
    if unknown:
        examples = "\n".join(
            f"- {language}: {english} / {japanese}"
            for language, english, japanese in unknown[:5]
        )
        raise ValueError(
            "valid data contains characters that are not present in train data, "
            f"which would become {UNK_TOKEN}:\n{examples}"
        )

    all_pairs = train_pairs + valid_pairs
    english_len = max(len(tokenizer.tokenize(english)) for english, _ in all_pairs)
    japanese_len = max(len(tokenizer.tokenize(japanese)) for _, japanese in all_pairs)
    return tokenizer, english_len, japanese_len, train_pairs, valid_pairs


def save_config(
    checkpoint_root: Path,
    *,
    train_data: Path,
    valid_data: Path,
    corpus_size: int,
    train_size: int,
    valid_size: int,
    english_len: int,
    japanese_len: int,
    token_count: int,
    vocab_size: int,
    hidden_dim: int,
    layers: int,
    kernel_size: int,
    seed: int,
) -> None:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    config = {
        "train_data": str(train_data),
        "valid_data": str(valid_data),
        "corpus_size": corpus_size,
        "train_size": train_size,
        "valid_size": valid_size,
        "english_len": english_len,
        "japanese_len": japanese_len,
        "token_count": token_count,
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "layers": layers,
        "kernel_size": kernel_size,
        "seed": seed,
    }
    (checkpoint_root / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main():
    parser = ArgumentParser(description="Train a conditional English/Japanese BPE diffusion translation model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--valid-data", type=Path, default=DEFAULT_VALID_DATA_PATH)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=17)
    parser.add_argument("--vocab-size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checkpoint-save-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--save-period", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    train_pairs = read_parallel_corpus(args.data)
    valid_pairs = read_parallel_corpus(args.valid_data)
    tokenizer, english_len, japanese_len, train_pairs, valid_pairs = prepare_tokenizer(
        train_pairs,
        valid_pairs,
        vocab_size=args.vocab_size,
    )
    train_dataset = BPETranslationDataset(train_pairs, tokenizer, english_len, japanese_len)
    valid_dataset = BPETranslationDataset(valid_pairs, tokenizer, english_len, japanese_len)

    token_count = len(tokenizer.ordered_id_to_token)
    print(f"Shared BPE vocabulary size: {token_count} tokens")
    print(f"English token sequence length: {english_len}")
    print(f"Japanese token sequence length: {japanese_len}")
    print(f"Parallel corpus size: {len(train_pairs) + len(valid_pairs)} pairs")
    print(f"Training split: {len(train_dataset)} train, {len(valid_dataset)} validation")

    model = TranslationCategoricalDenoiser(
        english_len=english_len,
        japanese_len=japanese_len,
        vocab_size=token_count,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        kernel_size=args.kernel_size,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    densiformis.diffuser.fit(
        model=model,
        distribution_types=["categorical", "categorical"],
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
        train_data=args.data,
        valid_data=args.valid_data,
        corpus_size=len(train_pairs) + len(valid_pairs),
        train_size=len(train_pairs),
        valid_size=len(valid_pairs),
        english_len=english_len,
        japanese_len=japanese_len,
        token_count=token_count,
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        kernel_size=args.kernel_size,
        seed=args.seed,
    )
    print(f"Final model checkpoint saved: {args.checkpoint_save_root / 'model.pt'}")
    print(f"Checkpoint config saved: {args.checkpoint_save_root / CONFIG_NAME}")


if __name__ == "__main__":
    main()
