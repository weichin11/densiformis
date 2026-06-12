from argparse import ArgumentParser
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable, List
import warnings

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

import densiformis


PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
DEFAULT_DATA_PATH = Path(__file__).with_name("translate.csv")
DEFAULT_CHECKPOINT_ROOT = Path("text_translation_checkpoints")
CONFIG_NAME = "translation_config.json"


def read_parallel_corpus(path: Path = DEFAULT_DATA_PATH) -> List[tuple[str, str]]:
    dataframe = pd.read_csv(path, usecols=["english", "japanese"]).fillna("")
    dataframe["english"] = dataframe["english"].astype(str).str.strip()
    dataframe["japanese"] = dataframe["japanese"].astype(str).str.strip()
    dataframe = dataframe[(dataframe["english"] != "") & (dataframe["japanese"] != "")]
    return list(dataframe[["english", "japanese"]].itertuples(index=False, name=None))


def split_parallel_corpus(
    pairs: List[tuple[str, str]],
    valid_fraction: float,
    seed: int,
) -> tuple[List[tuple[str, str]], List[tuple[str, str]]]:
    if len(pairs) < 2:
        raise ValueError("translation data must contain at least 2 sentence pairs")
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("--valid-fraction must be greater than 0 and less than 1")

    valid_count = max(1, int(round(len(pairs) * valid_fraction)))
    valid_count = min(valid_count, len(pairs) - 1)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(pairs), generator=generator).tolist()
    valid_indices = set(indices[:valid_count])
    train_pairs = [pair for index, pair in enumerate(pairs) if index not in valid_indices]
    valid_pairs = [pair for index, pair in enumerate(pairs) if index in valid_indices]
    return train_pairs, valid_pairs


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
    pairs: List[tuple[str, str]],
    vocab_size: int,
):
    if not pairs:
        raise ValueError("translation data is empty")

    tokenizer = SimpleBPETokenizer(vocab_size=vocab_size)
    tokenizer.fit([text for pair in pairs for text in pair])
    english_len = max(len(tokenizer.tokenize(english)) for english, _ in pairs)
    japanese_len = max(len(tokenizer.tokenize(japanese)) for _, japanese in pairs)
    return tokenizer, english_len, japanese_len


def save_config(
    checkpoint_root: Path,
    *,
    data: Path,
    corpus_size: int,
    train_size: int,
    valid_size: int,
    valid_fraction: float,
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
        "data": str(data),
        "corpus_size": corpus_size,
        "train_size": train_size,
        "valid_size": valid_size,
        "valid_fraction": valid_fraction,
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


def load_config(checkpoint_root: Path) -> dict:
    config_path = checkpoint_root / CONFIG_NAME
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"


def encode_batch(texts, tokenizer, seq_len, vocab_size, device):
    encoded = [
        token_ids_to_one_hot(tokenizer.encode(text, seq_len), vocab_size)
        for text in texts
    ]
    return torch.stack(encoded).to(device)


def target_steps_for_animation(steps: int, frame_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    frame_count = max(2, frame_count)
    return sorted(
        {
            int(round(index * steps / (frame_count - 1)))
            for index in range(frame_count)
        }
    )


def sample_direction(
    model,
    texts,
    direction,
    tokenizer,
    english_len,
    japanese_len,
    vocab_size,
    steps,
    device,
    frame_count=0,
):
    if direction == "en-ja":
        english = encode_batch(texts, tokenizer, english_len, vocab_size, device)
        japanese = densiformis.functions.rand_soft_label((len(texts), vocab_size, japanese_len), dim=1, device=device)
        inputs = [english, japanese]
        t_init = [0.0, 1.0]
        output_index = 1
    elif direction == "ja-en":
        english = densiformis.functions.rand_soft_label((len(texts), vocab_size, english_len), dim=1, device=device)
        japanese = encode_batch(texts, tokenizer, japanese_len, vocab_size, device)
        inputs = [english, japanese]
        t_init = [1.0, 0.0]
        output_index = 0
    else:
        raise ValueError("--direction must be either en-ja or ja-en")

    generated = inputs
    frames = []
    target_steps = target_steps_for_animation(steps, frame_count)
    if target_steps and target_steps[0] == 0:
        frames.append((
            0,
            generated[output_index].detach().cpu(),
        ))

    sampler = densiformis.diffuser.sample(
        model=model,
        distribution_types=["categorical", "categorical"],
        inputs=inputs,
        t_init=t_init,
        steps=steps,
        direction="backward",
        device=device,
    )
    target_index = 1 if target_steps and target_steps[0] == 0 else 0
    for step_index, generated in enumerate(sampler, start=1):
        while target_index < len(target_steps) and step_index >= target_steps[target_index]:
            frames.append((
                step_index,
                generated[output_index].detach().cpu(),
            ))
            target_index += 1

    if target_steps and (not frames or frames[-1][0] != steps):
        frames.append((
            steps,
            generated[output_index].detach().cpu(),
        ))

    translations = [
        categorical_to_text(sample, tokenizer)
        for sample in generated[output_index]
    ]
    return translations, frames


def select_default_pairs(pairs: list[tuple[str, str]], example_count: int) -> list[tuple[str, str]]:
    if example_count < 1:
        raise ValueError("--examples must be at least 1")
    return pairs[: min(example_count, len(pairs))]


def build_jobs(pairs: list[tuple[str, str]], example_count: int, split: str, direction: str):
    jobs = []
    for english, japanese in select_default_pairs(pairs, example_count):
        if direction in ("both", "en-ja"):
            jobs.append({"split": split, "direction": "en-ja", "source": english, "reference": japanese})
        if direction in ("both", "ja-en"):
            jobs.append({"split": split, "direction": "ja-en", "source": japanese, "reference": english})
    return jobs


def build_split_jobs(
    train_pairs: list[tuple[str, str]],
    valid_pairs: list[tuple[str, str]],
    example_count: int,
    split: str,
    direction: str,
):
    jobs = []
    if split in ("train", "both"):
        jobs.extend(build_jobs(train_pairs, example_count, "train", direction))
    if split in ("valid", "both"):
        jobs.extend(build_jobs(valid_pairs, example_count, "valid", direction))
    return jobs


def combine_direction_results(jobs, en_ja_results, ja_en_results):
    en_ja_index = 0
    ja_en_index = 0
    rows = []
    for job in jobs:
        if job["direction"] == "en-ja":
            generated = en_ja_results[en_ja_index]
            en_ja_index += 1
        else:
            generated = ja_en_results[ja_en_index]
            ja_en_index += 1
        rows.append({**job, "generated": generated})
    return rows


def combine_direction_frames(jobs, en_ja_frames, ja_en_frames):
    if not en_ja_frames and not ja_en_frames:
        return []

    frame_count = max(len(en_ja_frames), len(ja_en_frames))
    combined_frames = []
    for frame_index in range(frame_count):
        if en_ja_frames:
            en_ja_step, en_ja_probs = en_ja_frames[min(frame_index, len(en_ja_frames) - 1)]
        else:
            en_ja_step, en_ja_probs = 0, []
        if ja_en_frames:
            ja_en_step, ja_en_probs = ja_en_frames[min(frame_index, len(ja_en_frames) - 1)]
        else:
            ja_en_step, ja_en_probs = 0, []

        en_ja_index = 0
        ja_en_index = 0
        row_values = []
        for job in jobs:
            if job["direction"] == "en-ja":
                generated_probs = en_ja_probs[en_ja_index] if en_ja_index < len(en_ja_probs) else None
                en_ja_index += 1
            else:
                generated_probs = ja_en_probs[ja_en_index] if ja_en_index < len(ja_en_probs) else None
                ja_en_index += 1
            row_values.append({
                "direction": job["direction"],
                "source": job["source"],
                "reference": job["reference"],
                "generated_probs": generated_probs,
            })
        combined_frames.append((max(en_ja_step, ja_en_step), row_values))
    return combined_frames


def filter_frames_for_split(frames, jobs, split: str):
    split_indices = [index for index, job in enumerate(jobs) if job["split"] == split]
    return [
        (step, [rows[index] for index in split_indices])
        for step, rows in frames
    ]


def output_path_for_split(path: Path, split: str, selected_split: str) -> Path:
    if selected_split == "both":
        return path.with_name(f"{path.stem}_{split}{path.suffix}")
    if path.stem.endswith(f"_{split}"):
        return path
    return path.with_name(f"{path.stem}_{split}{path.suffix}")


def token_spans_for_probs(token_tensor: torch.Tensor | None, tokenizer) -> list[tuple[str, float]]:
    if token_tensor is None:
        return [("", 0.0)]

    probs, token_ids = token_tensor.detach().cpu().max(dim=0)
    spans = []
    for token_id, prob in zip(token_ids.long().tolist(), probs.float().tolist()):
        token = tokenizer.ordered_id_to_token[token_id]
        if token == EOS_TOKEN:
            break
        if token == PAD_TOKEN:
            continue
        if token == UNK_TOKEN:
            token = "?"
        spans.append((token, prob))
    if not spans:
        return [("<empty>", 0.0)]
    return spans


def token_probability_color(prob: float) -> tuple[float, float, float]:
    value = 1.0 - max(0.0, min(1.0, prob))
    return (value, value, value)


def generated_text_and_confidence(token_tensor: torch.Tensor | None, tokenizer) -> tuple[str, float]:
    spans = token_spans_for_probs(token_tensor, tokenizer)
    text = "".join(token for token, _ in spans).strip()
    if not text:
        text = "<empty>"
    confidence = sum(prob for _, prob in spans) / max(1, len(spans))
    return text, confidence


def text_capacity(text: str) -> int:
    return sum(2 if ord(char) > 127 else 1 for char in text)


def animation_writer(output_path: Path, animation_module, fps: int):
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        return animation_module.PillowWriter(fps=fps)
    if suffix == ".mp4":
        if not animation_module.writers.is_available("ffmpeg"):
            raise RuntimeError("MP4 output requires ffmpeg to be available to matplotlib.")
        return animation_module.FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    raise ValueError(f"Unsupported animation output extension: {output_path.suffix}. Use .gif or .mp4.")


def save_translation_animation(
    frames: list[tuple[int, list[dict[str, str]]]],
    output_path: Path,
    total_steps: int,
    fps: int,
    tokenizer,
) -> None:
    if not frames:
        return

    matplotlib_cache = Path(tempfile.gettempdir()) / "densiformis_matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = len(frames[0][1])
    display_frames = [(step, rows[:row_count]) for step, rows in frames]
    char_capacity = max(
        28,
        max(
            text_capacity(row["source"])
            + text_capacity(generated_text_and_confidence(row["generated_probs"], tokenizer)[0])
            for _, rows in display_frames
            for row in rows
        ),
    )
    fig_width = max(10.5, min(14.5, char_capacity * 0.16))
    fig_height = max(5.4, row_count * 1.08 + 1.95)
    dpi = 140
    if output_path.suffix.lower() == ".mp4":
        pixel_width = int(round(fig_width * dpi))
        pixel_height = int(round(fig_height * dpi))
        fig_width = (pixel_width + pixel_width % 2) / dpi
        fig_height = (pixel_height + pixel_height % 2) / dpi

    fig = plt.figure(figsize=(fig_width, fig_height), layout=None)
    fig.patch.set_facecolor("#f8fafc")
    probability_cmap = LinearSegmentedColormap.from_list(
        "token_probability_white_black",
        [token_probability_color(0.0), token_probability_color(1.0)],
    )
    preferred_fonts = [
        "Hiragino Sans",
        "Hiragino Maru Gothic Pro",
        "Arial Unicode MS",
        "AppleGothic",
        "Yu Gothic",
        "Noto Sans CJK JP",
        "Noto Sans JP",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    text_font = next((font for font in preferred_fonts if font in available_fonts), "DejaVu Sans")

    title_artist = fig.text(
        0.06,
        0.94,
        "Conditional translation denoising",
        ha="left",
        va="top",
        fontsize=20,
        weight="bold",
        family=text_font,
        color="#0f172a",
    )
    step_artist = fig.text(
        0.94,
        0.94,
        "0%",
        ha="right",
        va="top",
        fontsize=17,
        family=text_font,
        color="#334155",
    )

    progress_axis = fig.add_axes((0.06, 0.875, 0.88, 0.03))
    progress_axis.set_xlim(0.0, 1.0)
    progress_axis.set_ylim(0.0, 1.0)
    progress_axis.axis("off")
    progress_axis.add_patch(Rectangle((0.0, 0.32), 1.0, 0.36, color="#e2e8f0", linewidth=0))
    progress_bar = progress_axis.add_patch(Rectangle((0.0, 0.32), 0.0, 0.36, color="#2a9d8f", linewidth=0))

    row_top = 0.79
    row_bottom = 0.17
    row_gap = 0.026
    row_height = (row_top - row_bottom - row_gap * (row_count - 1)) / row_count
    row_artists = []
    first_rows = display_frames[0][1]
    body_font_size = max(13.0, min(17.0, fig_width * 72 * 0.84 / max(1, char_capacity)))
    generated_font_size = max(14.0, min(19.0, body_font_size + 1.8))
    for row_index, row in enumerate(first_rows):
        row_y = row_top - (row_index + 1) * row_height - row_index * row_gap
        row_axis = fig.add_axes((0.06, row_y, 0.88, row_height))
        row_axis.set_xlim(0, 1)
        row_axis.set_ylim(0, 1)
        row_axis.axis("off")
        row_axis.add_patch(Rectangle((0, 0), 1, 1, facecolor="#ffffff", edgecolor="#dbe4ee", linewidth=0.9))
        badge = "EN -> JA" if row["direction"] == "en-ja" else "JA -> EN"
        direction_artist = row_axis.text(
            0.020,
            0.70,
            badge,
            ha="left",
            va="center",
            fontsize=11.5,
            family="monospace",
            color="#2a9d8f",
            clip_on=True,
        )
        row_axis.text(0.14, 0.72, "source", ha="left", va="center", fontsize=10.5, family=text_font, color="#64748b")
        row_axis.text(0.14, 0.34, "translated", ha="left", va="center", fontsize=10.5, family=text_font, color="#64748b")
        source_artist = row_axis.text(
            0.235,
            0.72,
            row["source"],
            ha="left",
            va="center",
            fontsize=body_font_size,
            family=text_font,
            color="#0f172a",
            clip_on=True,
        )
        generated_text, generated_confidence = generated_text_and_confidence(row["generated_probs"], tokenizer)
        generated_artist = row_axis.text(
            0.235,
            0.34,
            generated_text,
            ha="left",
            va="center",
            fontsize=generated_font_size,
            family=text_font,
            color=token_probability_color(generated_confidence),
            clip_on=True,
        )
        row_artists.append((direction_artist, source_artist, generated_artist))

    colorbar_axis = fig.add_axes((0.06, 0.07, 0.88, 0.035))
    colorbar_axis.imshow(
        [[value / 100 for value in range(101)]],
        extent=(0, 1, 0, 1),
        cmap=probability_cmap,
        aspect="auto",
        interpolation="nearest",
    )
    colorbar_axis.set_yticks([])
    colorbar_axis.set_xticks([0, 1])
    colorbar_axis.set_xticklabels(["uncertain", "certain"], fontsize=12, color="#475569")
    colorbar_axis.set_xlabel("generated confidence", fontsize=12, color="#475569", labelpad=2)
    for spine in colorbar_axis.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.8)

    def update(frame: tuple[int, list[dict[str, str]]]):
        step, rows = frame
        progress = step / max(1, total_steps)
        step_artist.set_text(f"{round(100 * progress):3d}%")
        progress_bar.set_width(progress)
        artists = [title_artist, step_artist, progress_bar]
        for (direction_artist, source_artist, generated_artist), row in zip(row_artists, rows):
            direction_artist.set_text("EN -> JA" if row["direction"] == "en-ja" else "JA -> EN")
            source_artist.set_text(row["source"])
            generated_text, generated_confidence = generated_text_and_confidence(row["generated_probs"], tokenizer)
            generated_artist.set_text(generated_text)
            generated_artist.set_color(token_probability_color(generated_confidence))
            artists.extend([direction_artist, source_artist, generated_artist])
        return tuple(artists)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=display_frames,
        interval=int(1000 / max(1, fps)),
        blit=False,
    )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Glyph .* missing from font")
            ani.save(output_path, writer=animation_writer(output_path, animation, fps), dpi=dpi)
    finally:
        plt.close(fig)


def run_train(args) -> None:
    torch.manual_seed(args.seed)
    device = detect_device()
    print(f"Using device: {device}")

    corpus_pairs = read_parallel_corpus(args.data)
    tokenizer, english_len, japanese_len = prepare_tokenizer(
        corpus_pairs,
        vocab_size=args.vocab_size,
    )
    train_pairs, valid_pairs = split_parallel_corpus(corpus_pairs, args.valid_fraction, args.seed)
    train_dataset = BPETranslationDataset(train_pairs, tokenizer, english_len, japanese_len)
    valid_dataset = BPETranslationDataset(valid_pairs, tokenizer, english_len, japanese_len)

    token_count = len(tokenizer.ordered_id_to_token)
    print(f"Shared BPE vocabulary size: {token_count} tokens")
    print(f"English token sequence length: {english_len}")
    print(f"Japanese token sequence length: {japanese_len}")
    print(f"Parallel corpus size: {len(corpus_pairs)} pairs")
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
        data=args.data,
        corpus_size=len(corpus_pairs),
        train_size=len(train_pairs),
        valid_size=len(valid_pairs),
        valid_fraction=args.valid_fraction,
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


def run_generate(args) -> None:
    config = load_config(args.checkpoint_save_root)

    device = detect_device()
    print(f"Using device: {device}")

    corpus_pairs = read_parallel_corpus(args.data)
    tokenizer, english_len, japanese_len = prepare_tokenizer(
        corpus_pairs,
        vocab_size=args.vocab_size,
    )
    train_pairs, valid_pairs = split_parallel_corpus(corpus_pairs, args.valid_fraction, args.tokenizer_seed)
    english_len = config.get("english_len", english_len)
    japanese_len = config.get("japanese_len", japanese_len)

    token_count = len(tokenizer.ordered_id_to_token)
    print(f"Shared BPE vocabulary size: {token_count} tokens")
    print(f"English token sequence length: {english_len}")
    print(f"Japanese token sequence length: {japanese_len}")
    print(f"Tokenizer seed: {args.tokenizer_seed}")
    print(f"Generation seed: {args.seed}")
    print(f"Parallel corpus size: {len(corpus_pairs)} pairs")
    print(f"Training split: {len(train_pairs)} train, {len(valid_pairs)} validation")

    model = TranslationCategoricalDenoiser(
        english_len=english_len,
        japanese_len=japanese_len,
        vocab_size=token_count,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        kernel_size=args.kernel_size,
    ).to(device)

    ckpt_path = args.checkpoint_save_root / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    print(f"Loading trained weights from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    jobs = build_split_jobs(
        train_pairs=train_pairs,
        valid_pairs=valid_pairs,
        example_count=args.examples,
        split=args.split,
        direction=args.direction,
    )

    torch.manual_seed(args.seed)
    en_ja_sources = [job["source"] for job in jobs if job["direction"] == "en-ja"]
    ja_en_sources = [job["source"] for job in jobs if job["direction"] == "ja-en"]

    en_ja_results, en_ja_frames = [], []
    if en_ja_sources:
        en_ja_results, en_ja_frames = sample_direction(
            model=model,
            texts=en_ja_sources,
            direction="en-ja",
            tokenizer=tokenizer,
            english_len=english_len,
            japanese_len=japanese_len,
            vocab_size=token_count,
            steps=args.steps,
            device=device,
            frame_count=0 if args.no_animation else args.frames,
        )

    ja_en_results, ja_en_frames = [], []
    if ja_en_sources:
        ja_en_results, ja_en_frames = sample_direction(
            model=model,
            texts=ja_en_sources,
            direction="ja-en",
            tokenizer=tokenizer,
            english_len=english_len,
            japanese_len=japanese_len,
            vocab_size=token_count,
            steps=args.steps,
            device=device,
            frame_count=0 if args.no_animation else args.frames,
        )

    rows = combine_direction_results(jobs, en_ja_results, ja_en_results)
    split_order = [split for split in ("train", "valid") if any(row["split"] == split for row in rows)]
    combined_frames = combine_direction_frames(jobs, en_ja_frames, ja_en_frames)
    print("Translations:")
    for split in split_order:
        split_rows = [row for row in rows if row["split"] == split]
        split_output = output_path_for_split(args.output, split, args.split)

        output_lines = []
        print(f"{split_output}:")
        for index, row in enumerate(split_rows):
            reference = f" | ref: {row['reference']}" if row["reference"] else ""
            line = f"{index:02d} | {row['direction']} | {row['source']} => {row['generated']}{reference}"
            output_lines.append(line)
            print(f"- {line}")

        split_output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
        print(f"Artifacts saved: {split_output}")

        if not args.no_animation:
            split_animation_output = output_path_for_split(args.output_path, split, args.split)
            try:
                save_translation_animation(
                    frames=filter_frames_for_split(combined_frames, jobs, split),
                    output_path=split_animation_output,
                    total_steps=args.steps,
                    fps=args.fps,
                    tokenizer=tokenizer,
                )
                print(f"Translation animation saved: {split_animation_output}")
            except (ImportError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
                print(f"Skipping animation because output could not be written: {exc}")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train or generate with a conditional English/Japanese BPE diffusion model.")
    parser.add_argument("--phase", choices=["train", "generate"], default="train")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=17)
    parser.add_argument("--vocab-size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checkpoint-save-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--save-period", type=int, default=100)
    parser.add_argument("--direction", choices=["both", "en-ja", "ja-en"], default="both")
    parser.add_argument("--split", choices=["train", "valid", "both"], default="train")
    parser.add_argument("--examples", type=int, default=3, help="Number of sentence pairs to generate per split.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--tokenizer-seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=Path("generated_text_translation.txt"))
    parser.add_argument("--output-path", type=Path, default=Path("generated_text_translation.gif"))
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--no-animation", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.phase == "train":
        run_train(args)
    elif args.phase == "generate":
        run_generate(args)
    else:
        raise ValueError(f"Unknown phase: {args.phase}")


if __name__ == "__main__":
    main()
