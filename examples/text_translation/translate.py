from argparse import ArgumentParser
import json
import os
from pathlib import Path
import tempfile
import warnings

import torch

import densiformis
from train import (
    CONFIG_NAME,
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_DATA_PATH,
    DEFAULT_VALID_DATA_PATH,
    EOS_TOKEN,
    PAD_TOKEN,
    TranslationCategoricalDenoiser,
    UNK_TOKEN,
    categorical_to_text,
    prepare_tokenizer,
    read_parallel_corpus,
    token_ids_to_one_hot,
)


def load_config(checkpoint_root: Path) -> dict:
    config_path = checkpoint_root / CONFIG_NAME
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def choose(cli_value, config: dict, key: str, default):
    if cli_value is not None:
        return cli_value
    return config.get(key, default)


def encode_batch(texts, tokenizer, seq_len, vocab_size, device):
    encoded = [
        token_ids_to_one_hot(tokenizer.encode(text, seq_len), vocab_size)
        for text in texts
    ]
    return torch.stack(encoded).to(device)


def target_steps_for_gif(steps: int, frame_count: int) -> list[int]:
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
    target_steps = target_steps_for_gif(steps, frame_count)
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


def save_translation_gif(
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
    fig_width = max(10.0, min(13.5, char_capacity * 0.14))
    fig_height = max(5.1, row_count * 0.98 + 1.85)
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
        fontsize=15,
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
    body_font_size = max(11.0, min(14.0, fig_width * 72 * 0.80 / max(1, char_capacity)))
    generated_font_size = max(12.0, min(16.0, body_font_size + 1.5))
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
            fontsize=9.5,
            family="monospace",
            color="#2a9d8f",
            clip_on=True,
        )
        row_axis.text(0.14, 0.72, "source", ha="left", va="center", fontsize=8.8, family=text_font, color="#64748b")
        row_axis.text(0.14, 0.34, "translated", ha="left", va="center", fontsize=8.8, family=text_font, color="#64748b")
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
    colorbar_axis.set_xticklabels(["uncertain", "certain"], fontsize=10, color="#475569")
    colorbar_axis.set_xlabel("generated confidence", fontsize=10, color="#475569", labelpad=2)
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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Glyph .* missing from font")
        ani.save(output_path, writer=animation.PillowWriter(fps=fps), dpi=140)
    plt.close(fig)


def main():
    parser = ArgumentParser(description="Translate with a conditional English/Japanese BPE diffusion model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--valid-data", type=Path, default=DEFAULT_VALID_DATA_PATH)
    parser.add_argument("--direction", choices=["both", "en-ja", "ja-en"], default="both")
    parser.add_argument("--split", choices=["train", "valid", "both"], default="train")
    parser.add_argument("--examples", type=int, default=3, help="Number of sentence pairs to translate per split.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--kernel-size", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tokenizer-seed", type=int, default=None)
    parser.add_argument("--checkpoint-save-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=Path("generated_text_translation.txt"))
    parser.add_argument("--gif-output", type=Path, default=Path("generated_text_translation.gif"))
    parser.add_argument("--gif-frames", type=int, default=100)
    parser.add_argument("--gif-fps", type=int, default=20)
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    config = load_config(args.checkpoint_save_root)
    tokenizer_seed = choose(args.tokenizer_seed, config, "seed", 1234)
    generation_seed = args.seed if args.seed is not None else config.get("seed", 1234)
    vocab_size = choose(args.vocab_size, config, "vocab_size", 256)
    hidden_dim = choose(args.hidden_dim, config, "hidden_dim", 512)
    layers = choose(args.layers, config, "layers", 8)
    kernel_size = choose(args.kernel_size, config, "kernel_size", 17)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    train_pairs = read_parallel_corpus(args.data)
    valid_pairs = read_parallel_corpus(args.valid_data)
    tokenizer, english_len, japanese_len, train_pairs, valid_pairs = prepare_tokenizer(
        train_pairs,
        valid_pairs,
        vocab_size=vocab_size,
    )
    english_len = config.get("english_len", english_len)
    japanese_len = config.get("japanese_len", japanese_len)

    token_count = len(tokenizer.ordered_id_to_token)
    print(f"Shared BPE vocabulary size: {token_count} tokens")
    print(f"English token sequence length: {english_len}")
    print(f"Japanese token sequence length: {japanese_len}")
    print(f"Tokenizer seed: {tokenizer_seed}")
    print(f"Generation seed: {generation_seed}")
    print(f"Training split: {len(train_pairs)} train, {len(valid_pairs)} validation")

    model = TranslationCategoricalDenoiser(
        english_len=english_len,
        japanese_len=japanese_len,
        vocab_size=token_count,
        hidden_dim=hidden_dim,
        layers=layers,
        kernel_size=kernel_size,
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

    torch.manual_seed(generation_seed)
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
            frame_count=0 if args.no_gif else args.gif_frames,
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
            frame_count=0 if args.no_gif else args.gif_frames,
        )

    rows = combine_direction_results(jobs, en_ja_results, ja_en_results)
    split_order = [split for split in ("train", "valid") if any(row["split"] == split for row in rows)]
    combined_frames = combine_direction_frames(jobs, en_ja_frames, ja_en_frames)
    print("Translations:")
    for split in split_order:
        split_jobs = [job for job in jobs if job["split"] == split]
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

        if not args.no_gif:
            split_gif_output = output_path_for_split(args.gif_output, split, args.split)
            try:
                save_translation_gif(
                    frames=filter_frames_for_split(combined_frames, jobs, split),
                    output_path=split_gif_output,
                    total_steps=args.steps,
                    fps=args.gif_fps,
                    tokenizer=tokenizer,
                )
                print(f"Translation GIF saved: {split_gif_output}")
            except ImportError as exc:
                print(f"Skipping GIF because matplotlib/Pillow is unavailable: {exc}")


if __name__ == "__main__":
    main()
