from argparse import ArgumentParser
import json
import os
from pathlib import Path
import tempfile
from typing import List

import torch
import torch.nn as nn

import densiformis
from train import (
    CONFIG_NAME,
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_DATA_PATH,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    TextCategoricalDenoiser,
    categorical_to_text,
    prepare_tokenizer,
    read_corpus,
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


def sample_to_text(
    model: nn.Module,
    sample_count: int,
    seq_len: int,
    vocab_size: int,
    tokenizer,
    steps: int,
    device: str,
    frame_count: int = 0,
    frame_sample_index: int = 0,
    frame_sample_count: int = 1,
) -> tuple[List[str], list[tuple[int, torch.Tensor, list[str]]]]:
    noise = densiformis.functions.rand_soft_label((sample_count, vocab_size, seq_len), dim=1, device=device)
    frame_sample_index = min(max(frame_sample_index, 0), sample_count - 1)
    frame_sample_count = max(1, min(frame_sample_count, sample_count - frame_sample_index))
    frame_sample_slice = slice(frame_sample_index, frame_sample_index + frame_sample_count)
    target_steps = []
    frames = []
    if frame_count > 0:
        frame_count = max(2, frame_count)
        target_steps = sorted(
            {
                int(round(index * steps / (frame_count - 1)))
                for index in range(frame_count)
            }
        )
        initial_samples = noise[frame_sample_slice].detach().cpu()
        frames.append((
            0,
            initial_samples,
            [categorical_to_text(sample, tokenizer) for sample in initial_samples],
        ))

    sampler = densiformis.diffuser.sample(
        model=model,
        distribution_types=["categorical"],
        inputs=[noise],
        t_init=[1.0],
        steps=steps,
        direction="backward",
        device=device,
    )

    generated = None
    target_index = 1 if target_steps and target_steps[0] == 0 else 0
    for step_index, generated in enumerate(sampler, start=1):
        while target_index < len(target_steps) and step_index >= target_steps[target_index]:
            frame_samples = generated[0][frame_sample_slice].detach().cpu()
            frames.append((
                step_index,
                frame_samples,
                [categorical_to_text(sample, tokenizer) for sample in frame_samples],
            ))
            target_index += 1

    if generated is None:
        generated = [noise]

    if frame_count > 0 and (not frames or frames[-1][0] != steps):
        final_samples = generated[0][frame_sample_slice].detach().cpu()
        frames.append((
            steps,
            final_samples,
            [categorical_to_text(sample, tokenizer) for sample in final_samples],
        ))

    return [
        categorical_to_text(sample, tokenizer)
        for sample in generated[0]
    ], frames


def token_spans_for_probs(token_tensor: torch.Tensor, tokenizer) -> list[tuple[str, float]]:
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


def save_generation_gif(
    frames: list[tuple[int, torch.Tensor, list[str]]],
    output_path: Path,
    total_steps: int,
    fps: int,
    tokenizer,
    sample_indices: list[int],
) -> None:
    if not frames:
        return

    matplotlib_cache = Path(tempfile.gettempdir()) / "densiformis_matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_step, first_probs_batch, _ = frames[0]
    sample_count = first_probs_batch.size(0)
    seq_len = first_probs_batch.size(2)
    char_capacity = max(
        24,
        max(
            sum(max(1, len(token)) for token, _ in token_spans_for_probs(sample, tokenizer))
            for _, prob_batch, _ in frames
            for sample in prob_batch
        ),
    )
    fig_width = max(11.5, min(22.0, seq_len * 0.50))
    fig_height = max(4.7, sample_count * 0.82 + 1.95)
    fig = plt.figure(figsize=(fig_width, fig_height), layout=None)
    fig.patch.set_facecolor("#f8fafc")
    probability_cmap = LinearSegmentedColormap.from_list(
        "token_probability_white_black",
        [token_probability_color(0.0), token_probability_color(1.0)],
    )

    title_artist = fig.text(
        0.08,
        0.94,
        "Text denoising dashboard",
        ha="left",
        va="top",
        fontsize=18,
        weight="bold",
        color="#0f172a",
    )
    step_artist = fig.text(
        0.92,
        0.94,
        f"{round(100 * first_step / max(1, total_steps)):3d}%",
        ha="right",
        va="top",
        fontsize=13,
        family="monospace",
        color="#334155",
    )

    progress_axis = fig.add_axes((0.08, 0.865, 0.84, 0.035))
    progress_axis.set_xlim(0.0, 1.0)
    progress_axis.set_ylim(0.0, 1.0)
    progress_axis.axis("off")
    progress_axis.add_patch(Rectangle((0.0, 0.32), 1.0, 0.36, color="#e2e8f0", linewidth=0))
    progress_bar = progress_axis.add_patch(
        Rectangle((0.0, 0.32), first_step / max(1, total_steps), 0.36, color="#2a9d8f", linewidth=0)
    )

    row_top = 0.80
    row_bottom = 0.19
    row_gap = 0.03
    row_height = (row_top - row_bottom - row_gap * (sample_count - 1)) / sample_count
    text_left = 0.08
    text_width = 0.88

    token_artist_rows = []
    available_text_points = fig_width * 72 * text_width
    token_font_size = max(9.0, min(20.0, available_text_points / (max(1, char_capacity) * 0.62)))
    for row_index in range(sample_count):
        row_y = row_top - (row_index + 1) * row_height - row_index * row_gap

        text_axis = fig.add_axes((text_left, row_y, text_width, row_height))
        text_axis.set_xlim(0, char_capacity)
        text_axis.set_ylim(0, 1)
        text_axis.axis("off")

        token_spans = token_spans_for_probs(first_probs_batch[row_index], tokenizer)
        token_row_artists = []
        cursor = 0
        for token, prob in token_spans:
            token_artist = text_axis.text(
                cursor,
                0.5,
                token,
                ha="left",
                va="center",
                fontsize=token_font_size,
                family="monospace",
                color=token_probability_color(prob),
                clip_on=True,
            )
            token_row_artists.append(token_artist)
            cursor += max(1, len(token))
        for _ in range(len(token_row_artists), seq_len):
            token_artist = text_axis.text(
                cursor,
                0.5,
                "",
                ha="left",
                va="center",
                fontsize=token_font_size,
                family="monospace",
                color=token_probability_color(0.0),
                clip_on=True,
            )
            token_row_artists.append(token_artist)
        token_artist_rows.append(token_row_artists)

    colorbar_axis = fig.add_axes((0.08, 0.075, 0.84, 0.035))
    colorbar_axis.imshow(
        [[value / 100 for value in range(101)]],
        extent=(0, 1, 0, 1),
        cmap=probability_cmap,
        aspect="auto",
        interpolation="nearest",
    )
    colorbar_axis.set_yticks([])
    colorbar_axis.set_xticks([0, 1])
    colorbar_axis.set_xticklabels(["0", "1"], fontsize=9, color="#475569")
    colorbar_axis.set_xlabel("token probability", fontsize=9, color="#475569", labelpad=2)
    for spine in colorbar_axis.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.8)

    def update(frame: tuple[int, torch.Tensor, list[str]]):
        step, prob_values_batch, _ = frame
        progress = step / max(1, total_steps)
        step_artist.set_text(f"{round(100 * progress):3d}%")
        progress_bar.set_width(progress)
        updated_artists = [title_artist, step_artist, progress_bar]
        for row_index, prob_values in enumerate(prob_values_batch):
            cursor = 0
            token_spans = token_spans_for_probs(prob_values, tokenizer)
            for token_artist, (token, prob) in zip(token_artist_rows[row_index], token_spans):
                token_artist.set_position((cursor, 0.5))
                token_artist.set_text(token)
                token_artist.set_color(token_probability_color(prob))
                cursor += max(1, len(token))
                updated_artists.append(token_artist)
            for token_artist in token_artist_rows[row_index][len(token_spans):]:
                token_artist.set_text("")
                updated_artists.append(token_artist)
        return tuple(updated_artists)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=int(1000 / max(1, fps)),
        blit=False,
    )
    ani.save(output_path, writer=animation.PillowWriter(fps=fps), dpi=140)
    plt.close(fig)


def main():
    parser = ArgumentParser(description="Generate text from a trained categorical BPE diffusion model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--kernel-size", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation noise only.",
    )
    parser.add_argument(
        "--tokenizer-seed",
        type=int,
        default=None,
        help="Seed used to reconstruct the tokenizer train/validation split.",
    )
    parser.add_argument("--valid-fraction", type=float, default=None)
    parser.add_argument("--checkpoint-save-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=Path("generated_text_samples.txt"))
    parser.add_argument("--gif-output", type=Path, default=Path("generated_text_generation.gif"))
    parser.add_argument("--gif-frames", type=int, default=100)
    parser.add_argument("--gif-fps", type=int, default=20)
    parser.add_argument("--gif-sample-index", type=int, default=0, help="First sample index to show in the GIF.")
    parser.add_argument("--gif-samples", type=int, default=5, help="Number of samples to show in the GIF.")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    if args.gif_samples < 1:
        raise ValueError("--gif-samples must be at least 1")

    config = load_config(args.checkpoint_save_root)
    tokenizer_seed = choose(args.tokenizer_seed, config, "seed", 1234)
    generation_seed = (
        args.seed
        if args.seed is not None
        else config.get("generation_seed", config.get("seed", 1234))
    )
    valid_fraction = choose(args.valid_fraction, config, "valid_fraction", 0.1)
    vocab_size = choose(args.vocab_size, config, "vocab_size", 128)
    hidden_dim = choose(args.hidden_dim, config, "hidden_dim", 512)
    layers = choose(args.layers, config, "layers", 8)
    kernel_size = choose(args.kernel_size, config, "kernel_size", 17)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    corpus = read_corpus(args.data)
    tokenizer, seq_len, train_texts, valid_texts = prepare_tokenizer(
        corpus,
        vocab_size=vocab_size,
        valid_fraction=valid_fraction,
        seed=tokenizer_seed,
    )
    seq_len = config.get("seq_len", seq_len)

    token_count = len(tokenizer.ordered_id_to_token)
    print(f"BPE vocabulary size: {token_count} tokens")
    print(f"Encoding each BPE token id as a {token_count}-class categorical distribution")
    print(f"Fixed token sequence length: {seq_len}")
    print(f"Tokenizer seed: {tokenizer_seed}")
    print(f"Generation seed: {generation_seed}")
    print(f"Corpus size: {len(corpus)} sentences")
    print(f"Training split: {len(train_texts)} train, {len(valid_texts)} validation")
    gif_sample_start = min(max(args.gif_sample_index, 0), args.samples - 1)
    gif_sample_count = max(1, min(args.gif_samples, args.samples - gif_sample_start))
    gif_sample_indices = list(range(gif_sample_start, gif_sample_start + gif_sample_count))
    if not args.no_gif:
        print(f"GIF samples: {gif_sample_indices[0]:02d}-{gif_sample_indices[-1]:02d}")

    model = TextCategoricalDenoiser(
        seq_len=seq_len,
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

    torch.manual_seed(generation_seed)
    samples, generation_frames = sample_to_text(
        model=model,
        sample_count=args.samples,
        seq_len=seq_len,
        vocab_size=token_count,
        tokenizer=tokenizer,
        steps=args.steps,
        device=device,
        frame_count=0 if args.no_gif else args.gif_frames,
        frame_sample_index=args.gif_sample_index,
        frame_sample_count=args.gif_samples,
    )

    print("Generated text:")
    output_lines = []
    for index, text in enumerate(samples):
        line = f"{index:02d} | {text}"
        output_lines.append(line)
        print(f"- {line}")
    args.output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"Artifacts saved: {args.output}")

    if not args.no_gif:
        try:
            save_generation_gif(
                frames=generation_frames,
                output_path=args.gif_output,
                total_steps=args.steps,
                fps=args.gif_fps,
                tokenizer=tokenizer,
                sample_indices=gif_sample_indices,
            )
            print(f"Generation GIF saved: {args.gif_output}")
        except ImportError as exc:
            print(f"Skipping GIF because matplotlib/Pillow is unavailable: {exc}")


if __name__ == "__main__":
    main()
