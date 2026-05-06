from argparse import ArgumentParser
import json
import os
from pathlib import Path
import tempfile
import textwrap
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
    TextBitDenoiser,
    gray_bits_to_token_ids,
    gray_bits_to_text,
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
    bits: int,
    tokenizer,
    steps: int,
    device: str,
    frame_count: int = 0,
    frame_sample_index: int = 0,
    frame_sample_count: int = 1,
) -> tuple[List[str], list[tuple[int, torch.Tensor, list[str]]]]:
    noise = torch.rand((sample_count, seq_len, bits), device=device)
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
            [gray_bits_to_text(sample, tokenizer) for sample in initial_samples],
        ))

    sampler = densiformis.diffuser.sample(
        model=model,
        distribution_types=["binary"],
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
                [gray_bits_to_text(sample, tokenizer) for sample in frame_samples],
            ))
            target_index += 1

    if generated is None:
        generated = [noise]

    if frame_count > 0 and (not frames or frames[-1][0] != steps):
        final_samples = generated[0][frame_sample_slice].detach().cpu()
        frames.append((
            steps,
            final_samples,
            [gray_bits_to_text(sample, tokenizer) for sample in final_samples],
        ))

    return [
        gray_bits_to_text(sample, tokenizer)
        for sample in generated[0]
    ], frames


def token_labels_for_bits(bit_tensor: torch.Tensor, tokenizer) -> list[str]:
    token_ids = gray_bits_to_token_ids(bit_tensor, len(tokenizer.ordered_id_to_token))
    labels = []
    for token_id in token_ids:
        token = tokenizer.ordered_id_to_token[token_id]
        if token == PAD_TOKEN:
            labels.append("")
        elif token == EOS_TOKEN:
            labels.append("<eos>")
        elif token == " ":
            labels.append("space")
        else:
            labels.append(token if len(token) <= 8 else f"{token[:7]}...")
    return labels


def save_generation_gif_debug(
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

    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_step, first_bits_batch, first_texts = frames[0]
    sample_count = first_bits_batch.size(0)
    seq_len = first_bits_batch.size(1)
    bits = first_bits_batch.size(2)
    fig_width = max(9.0, seq_len * 0.28)
    fig_height = max(5.8, sample_count * 2.1 + 0.8)
    fig = plt.figure(figsize=(fig_width, fig_height), layout=None)
    title_artist = fig.text(
        0.5,
        0.975,
        f"Samples {sample_indices[0]:02d}-{sample_indices[-1]:02d} denoising step {first_step}/{total_steps}",
        ha="center",
        va="top",
        fontsize=15,
    )

    left = 0.12
    width = 0.78
    bottom = 0.035
    top = 0.93
    row_gap = 0.014
    row_height = (top - bottom - row_gap * (sample_count - 1)) / sample_count
    inner_gap = row_height * 0.035
    bit_height = row_height * 0.55
    token_height = row_height * 0.23
    text_height = row_height * 0.145
    color_axis = fig.add_axes((0.93, bottom + 0.03, 0.02, top - bottom - 0.06))

    bit_images = []
    token_artist_rows = []
    text_artists = []
    for row_index in range(sample_count):
        row_bottom = top - (row_index + 1) * row_height - row_index * row_gap
        text_bottom = row_bottom
        token_bottom = text_bottom + text_height + inner_gap
        bit_bottom = token_bottom + token_height + inner_gap
        sample_index = sample_indices[row_index]
        first_bits = first_bits_batch[row_index]

        bit_axis = fig.add_axes((left, bit_bottom, width, bit_height))
        bit_image = bit_axis.imshow(
            first_bits.T.numpy(),
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            aspect="auto",
            interpolation="nearest",
        )
        bit_axis.set_ylabel(f"Sample {sample_index:02d}\nGray-code bit", fontsize=8.5)
        bit_axis.set_xlim(-0.5, seq_len - 0.5)
        bit_axis.set_ylim(bits - 0.5, -0.5)
        bit_axis.set_xticks(range(seq_len))
        bit_axis.set_xticklabels([])
        bit_axis.tick_params(axis="y", labelsize=8)
        bit_images.append(bit_image)

        token_axis = fig.add_axes((left, token_bottom, width, token_height))
        token_axis.set_xlim(-0.5, seq_len - 0.5)
        token_axis.set_ylim(0.0, 1.0)
        token_axis.set_yticks([])
        token_axis.set_xticks(range(seq_len))
        token_axis.set_xticklabels([])
        for spine in token_axis.spines.values():
            spine.set_visible(False)
        token_axis.axhline(0.98, color="0.85", linewidth=0.8)
        token_axis.axhline(0.02, color="0.85", linewidth=0.8)
        for token_position in range(seq_len + 1):
            token_axis.axvline(token_position - 0.5, color="0.88", linewidth=0.45)
        token_artist_rows.append([
            token_axis.text(
                token_position,
                0.5,
                label,
                ha="center",
                va="center",
                rotation=90,
                fontsize=6.2,
                family="monospace",
                clip_on=True,
            )
            for token_position, label in enumerate(token_labels_for_bits(first_bits, tokenizer))
        ])

        text_axis = fig.add_axes((left, text_bottom, width, text_height))
        text_axis.axis("off")
        text_artists.append(text_axis.text(
            0.0,
            0.5,
            "\n".join(textwrap.wrap(f"decoded: {first_texts[row_index] or '<empty>'}", width=96)),
            ha="left",
            va="center",
            fontsize=9.5,
            family="monospace",
            transform=text_axis.transAxes,
        ))

    fig.colorbar(bit_images[0], cax=color_axis, label="bit value")

    def update(frame: tuple[int, torch.Tensor, list[str]]):
        step, bit_values_batch, decoded_texts = frame
        title_artist.set_text(
            f"Samples {sample_indices[0]:02d}-{sample_indices[-1]:02d} denoising step {step}/{total_steps}"
        )
        updated_artists = [title_artist]
        for row_index, bit_values in enumerate(bit_values_batch):
            bit_images[row_index].set_data(bit_values.T.numpy())
            updated_artists.append(bit_images[row_index])
            for artist, label in zip(token_artist_rows[row_index], token_labels_for_bits(bit_values, tokenizer)):
                artist.set_text(label)
                updated_artists.append(artist)
            text_artists[row_index].set_text(
                "\n".join(textwrap.wrap(f"decoded: {decoded_texts[row_index] or '<empty>'}", width=96))
            )
            updated_artists.append(text_artists[row_index])
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


def save_generation_gif_compact(
    frames: list[tuple[int, torch.Tensor, list[str]]],
    output_path: Path,
    total_steps: int,
    fps: int,
    sample_indices: list[int],
) -> None:
    if not frames:
        return

    matplotlib_cache = Path(tempfile.gettempdir()) / "densiformis_matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_step, first_bits_batch, first_texts = frames[0]
    sample_count = first_bits_batch.size(0)
    seq_len = first_bits_batch.size(1)
    bits = first_bits_batch.size(2)
    fig_width = max(9.5, seq_len * 0.18)
    fig_height = max(4.8, sample_count * 0.82 + 1.7)
    fig = plt.figure(figsize=(fig_width, fig_height), layout=None)
    fig.patch.set_facecolor("white")

    title_artist = fig.text(
        0.08,
        0.94,
        "Denoising text samples",
        ha="left",
        va="top",
        fontsize=18,
        weight="bold",
    )
    step_artist = fig.text(
        0.92,
        0.94,
        f"{round(100 * first_step / max(1, total_steps)):3d}%",
        ha="right",
        va="top",
        fontsize=13,
        family="monospace",
    )

    progress_axis = fig.add_axes((0.08, 0.865, 0.84, 0.035))
    progress_axis.set_xlim(0.0, 1.0)
    progress_axis.set_ylim(0.0, 1.0)
    progress_axis.axis("off")
    progress_axis.add_patch(Rectangle((0.0, 0.32), 1.0, 0.36, color="0.90", linewidth=0))
    progress_bar = progress_axis.add_patch(
        Rectangle((0.0, 0.32), first_step / max(1, total_steps), 0.36, color="#2a9d8f", linewidth=0)
    )

    row_top = 0.80
    row_bottom = 0.08
    row_gap = 0.02
    row_height = (row_top - row_bottom - row_gap * (sample_count - 1)) / sample_count
    heat_left = 0.08
    heat_width = 0.19
    text_left = 0.30
    text_width = 0.62

    heat_images = []
    text_artists = []
    for row_index in range(sample_count):
        row_y = row_top - (row_index + 1) * row_height - row_index * row_gap
        sample_index = sample_indices[row_index]

        label_axis = fig.add_axes((0.02, row_y, 0.05, row_height))
        label_axis.axis("off")
        label_axis.text(
            1.0,
            0.5,
            f"{sample_index:02d}",
            ha="right",
            va="center",
            fontsize=10,
            family="monospace",
            color="0.35",
            transform=label_axis.transAxes,
        )

        heat_axis = fig.add_axes((heat_left, row_y + row_height * 0.12, heat_width, row_height * 0.76))
        heat_image = heat_axis.imshow(
            first_bits_batch[row_index].T.numpy(),
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            aspect="auto",
            interpolation="nearest",
        )
        heat_axis.set_xlim(-0.5, seq_len - 0.5)
        heat_axis.set_ylim(bits - 0.5, -0.5)
        heat_axis.set_xticks([])
        heat_axis.set_yticks([])
        for spine in heat_axis.spines.values():
            spine.set_color("0.82")
            spine.set_linewidth(0.8)
        heat_images.append(heat_image)

        text_axis = fig.add_axes((text_left, row_y, text_width, row_height))
        text_axis.axis("off")
        text_artists.append(text_axis.text(
            0.0,
            0.5,
            first_texts[row_index] or "<empty>",
            ha="left",
            va="center",
            fontsize=13,
            family="monospace",
            transform=text_axis.transAxes,
        ))

    def update(frame: tuple[int, torch.Tensor, list[str]]):
        step, bit_values_batch, decoded_texts = frame
        progress = step / max(1, total_steps)
        step_artist.set_text(f"{round(100 * progress):3d}%")
        progress_bar.set_width(progress)
        updated_artists = [title_artist, step_artist, progress_bar]
        for row_index, bit_values in enumerate(bit_values_batch):
            heat_images[row_index].set_data(bit_values.T.numpy())
            text_artists[row_index].set_text(decoded_texts[row_index] or "<empty>")
            updated_artists.extend([heat_images[row_index], text_artists[row_index]])
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


def save_generation_gif(
    frames: list[tuple[int, torch.Tensor, list[str]]],
    output_path: Path,
    total_steps: int,
    fps: int,
    tokenizer,
    sample_indices: list[int],
    style: str,
) -> None:
    if style == "debug":
        save_generation_gif_debug(
            frames=frames,
            output_path=output_path,
            total_steps=total_steps,
            fps=fps,
            tokenizer=tokenizer,
            sample_indices=sample_indices,
        )
    else:
        save_generation_gif_compact(
            frames=frames,
            output_path=output_path,
            total_steps=total_steps,
            fps=fps,
            sample_indices=sample_indices,
        )


def main():
    parser = ArgumentParser(description="Generate text from a trained binary BPE/Gray-code diffusion model.")
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
    parser.add_argument(
        "--gif-style",
        choices=("compact", "debug"),
        default="compact",
        help="GIF layout style: compact for demos, debug for bit/token details.",
    )
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

    print(f"BPE vocabulary size: {len(tokenizer.ordered_id_to_token)} tokens")
    print(f"Encoding each BPE token id with {tokenizer.bits} Gray-code bits")
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
        print(f"GIF style: {args.gif_style}")

    model = TextBitDenoiser(
        seq_len=seq_len,
        bits=tokenizer.bits,
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
        bits=tokenizer.bits,
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
                style=args.gif_style,
            )
            print(f"Generation GIF saved: {args.gif_output}")
        except ImportError as exc:
            print(f"Skipping GIF because matplotlib/Pillow is unavailable: {exc}")


if __name__ == "__main__":
    main()
