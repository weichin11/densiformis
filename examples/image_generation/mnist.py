from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets

import densiformis


class MNISTDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "./data"):
        train = split == "train"
        mnist = datasets.MNIST(root=data_dir, train=train, download=True)
        images = mnist.data.float().unsqueeze(1) / 255.0
        images = 2 * (images - 0.5)
        self.images = images
        self.labels = mnist.targets.clone()

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, index: int):
        image = self.images[index]
        label = self.labels[index]
        return [image, F.one_hot(label, num_classes=10).float()]


class Unet(nn.Module):
    def __init__(self, filters: int = 64, blocks: int = 4, image_size_hw=(28, 28), label_dim: int = 10):
        super().__init__()
        self.image_size_hw = image_size_hw
        self.filters = filters

        self.in_image = nn.Conv2d(1, filters // 3, (3, 3), padding="same")
        self.in_label = nn.Linear(label_dim, (filters // 3) * image_size_hw[0] * image_size_hw[1])
        self.linear_t = nn.Linear(2, (filters // 3) * image_size_hw[0] * image_size_hw[1])

        self.in_conv = nn.Conv2d(3 * (filters // 3), filters, (3, 3), padding="same")
        self.in_norm = nn.InstanceNorm2d(3 * (filters // 3))

        self.out_conv = nn.Conv2d(filters, 2 * (filters // 2), (3, 3), padding="same")
        self.out_norm = nn.InstanceNorm2d(filters)

        self.out_image = nn.Conv2d(filters // 2, 1, (3, 3), padding="same")
        self.out_label = nn.Linear(filters // 2, label_dim)

        self.down_scale_blocks = nn.ModuleList(
            [
                ScaleBlock(filters, filters * 2, 0.5),
                ScaleBlock(filters * 2, filters * 4, 0.5),
            ]
        )
        self.down_res_blocks = nn.ModuleList(
            [
                ResidualBlock(filters, blocks),
                ResidualBlock(filters * 2, blocks),
                ResidualBlock(filters * 4, blocks),
            ]
        )
        self.up_scale_blocks = nn.ModuleList(
            [
                ScaleBlock(filters * 4, filters * 2, 2.0),
                ScaleBlock(filters * 2, filters, 2.0),
            ]
        )
        self.up_res_blocks = nn.ModuleList(
            [
                ResidualBlock(filters * 4, blocks),
                ResidualBlock(filters * 2, blocks),
                ResidualBlock(filters, blocks),
            ]
        )

    def forward(self, inputs, t):
        image, label = inputs
        batch_size = image.size(0)
        time_inputs = torch.stack([ti.view(-1) for ti in t], dim=1)

        label_feat = self.in_label(label).view(batch_size, self.filters // 3, *self.image_size_hw)
        time_feat = self.linear_t(time_inputs).view(batch_size, self.filters // 3, *self.image_size_hw)

        x = torch.cat(
            [
                self.in_image(image),
                label_feat,
                time_feat,
            ],
            dim=1,
        )
        x = F.silu(x)
        x = self.in_norm(x)
        x = self.in_conv(x)
        x = F.silu(x)

        f1 = self.down_res_blocks[0](x)
        x = self.down_scale_blocks[0](f1)
        f2 = self.down_res_blocks[1](x)
        x = self.down_scale_blocks[1](f2)
        x = self.down_res_blocks[2](x)

        x = self.up_res_blocks[0](x)
        x = self.up_scale_blocks[0](x)
        x = x + f2
        x = self.up_res_blocks[1](x)
        x = self.up_scale_blocks[1](x)
        x = x + f1
        x = self.up_res_blocks[2](x)

        x = self.out_norm(x)
        x = self.out_conv(x)
        x = F.silu(x)

        img_feat, label_feat = torch.split(x, [self.filters // 2] * 2, dim=1)
        label_feat = label_feat.mean(dim=(2, 3))
        return [self.out_image(img_feat), self.out_label(label_feat)]


class ScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.norm = nn.InstanceNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, (3, 3), padding="same")

    def forward(self, x):
        if self.scale_factor > 1.0:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        x = self.norm(x)
        x = self.conv(x)
        x = F.silu(x)
        if self.scale_factor < 1.0:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, blocks):
        super().__init__()
        self.blocks = blocks
        layer_dict = nn.ModuleDict()
        for i in range(blocks):
            layer_dict[f"b{i}_norm1"] = nn.InstanceNorm2d(in_channels)
            layer_dict[f"b{i}_conv1"] = nn.Conv2d(in_channels, in_channels, (3, 3), padding="same", groups=in_channels)
            layer_dict[f"b{i}_norm2"] = nn.InstanceNorm2d(in_channels)
            layer_dict[f"b{i}_conv2"] = nn.Conv2d(in_channels, in_channels, (1, 1), padding="same")
        self.layers = layer_dict

    def forward(self, x):
        for i in range(self.blocks):
            res = x
            x = self.layers[f"b{i}_norm1"](x)
            x = self.layers[f"b{i}_conv1"](x)
            x = F.silu(x)
            x = self.layers[f"b{i}_norm2"](x)
            x = self.layers[f"b{i}_conv2"](x)
            x = F.silu(x)
            x = x + res
        return x


def _label_colors(label_count: int) -> np.ndarray:
    return plt.get_cmap("tab10")(np.arange(label_count))[:, :3]


def _label_names(label_count: int) -> list[str]:
    return [str(index) for index in range(label_count)]


def _make_mnist_panel(
    images: torch.Tensor,
    labels: torch.Tensor,
    sample_rows: int = 4,
    sample_cols: int = 2,
) -> np.ndarray:
    """Return a compact image montage with a class-probability strip per digit."""
    images = images.detach().cpu().clamp(-1, 1)
    labels = labels.detach().cpu().clamp(0, 1)
    tile_count = min(sample_rows * sample_cols, images.size(0), labels.size(0))

    digit_h, digit_w = images.shape[-2:]
    label_h = 6
    pad = 2
    tile_h = digit_h + label_h
    canvas_h = sample_rows * tile_h + (sample_rows - 1) * pad
    canvas_w = sample_cols * digit_w + (sample_cols - 1) * pad
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.float32)
    class_colors = _label_colors(labels.size(1))

    for idx in range(tile_count):
        row, col = divmod(idx, sample_cols)
        y0 = row * (tile_h + pad)
        x0 = col * (digit_w + pad)

        image_np = ((images[idx, 0].numpy() + 1.0) / 2.0)[..., None]
        canvas[y0 : y0 + digit_h, x0 : x0 + digit_w] = np.repeat(image_np, 3, axis=2)

        probs = labels[idx].numpy()
        for cls_idx, prob in enumerate(probs):
            x_start = x0 + int(round(cls_idx * digit_w / labels.size(1)))
            x_end = x0 + int(round((cls_idx + 1) * digit_w / labels.size(1)))
            color = (1.0 - prob) * np.ones(3) + prob * class_colors[cls_idx]
            canvas[y0 + digit_h : y0 + tile_h, x_start:x_end] = color

    return canvas


def make_grid(
    model: nn.Module,
    distribution_types: List[str],
    images: torch.Tensor,
    labels: torch.Tensor,
    image_noise: torch.Tensor,
    label_noise: torch.Tensor,
    device: str,
    steps: int = 48,
    save_path: str = "mnist_grid.gif",
    fps: int = 24,
    sample_rows: int = 4,
    sample_cols: int = 2,
):
    """Create an animation illustrating MNIST denoising progress."""
    if sample_rows < 1 or sample_cols < 1:
        raise ValueError("sample_rows and sample_cols must be positive")

    sample_count = min(
        sample_rows * sample_cols,
        images.size(0),
        labels.size(0),
        image_noise.size(0),
        label_noise.size(0),
    )
    if sample_count < 1:
        raise ValueError("make_grid needs at least one sample")

    images = images[:sample_count]
    labels = labels[:sample_count]
    image_noise = image_noise[:sample_count]
    label_noise = label_noise[:sample_count]

    conditions = [
        {
            "name": "Generate both",
            "inputs": [image_noise, label_noise],
            "t_init": [1.0, 1.0],
        },
        {
            "name": "Generate images only",
            "inputs": [image_noise, labels],
            "t_init": [1.0, 0.0],
        },
        {
            "name": "Generate labels only",
            "inputs": [images, label_noise],
            "t_init": [0.0, 1.0],
        },
    ]

    save_path = Path(save_path)
    output_suffix = save_path.suffix.lower()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if output_suffix not in {".gif", ".mp4"}:
        raise ValueError("save_path must end with .gif or .mp4")

    import matplotlib.animation as animation
    from matplotlib.patches import Patch

    condition_frames = []
    for cond in conditions:
        frames = [[tensor.detach().cpu() for tensor in cond["inputs"]]]
        sampler = densiformis.diffuser.sample(
            model=model,
            distribution_types=distribution_types,
            inputs=cond["inputs"],
            t_init=cond["t_init"],
            steps=steps,
            direction="backward",
            device=device,
        )

        for xt in sampler:
            frames.append([tensor.detach().cpu() for tensor in xt])

        while len(frames) < steps + 1:
            frames.append([tensor.clone() for tensor in frames[-1]])

        condition_frames.append(frames[: steps + 1])

    frame_count = min(len(frames) for frames in condition_frames)
    fig, axes = plt.subplots(
        1,
        len(conditions),
        figsize=(8, 4.5),
    )
    axes = np.atleast_1d(axes)
    artists = []

    for col, cond in enumerate(conditions):
        image_frame, label_frame = condition_frames[col][0]
        artist = axes[col].imshow(
            _make_mnist_panel(image_frame, label_frame, sample_rows, sample_cols),
            animated=True,
        )
        axes[col].set_xticks([])
        axes[col].set_yticks([])
        axes[col].set_xlabel(cond["name"], fontsize=7, labelpad=4)
        artists.append(artist)

    title = fig.suptitle("0% of diffusion", fontsize=9)
    legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in zip(_label_names(labels.size(1)), _label_colors(labels.size(1)))
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=min(labels.size(1), 10),
        frameon=False,
        fontsize=6,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.94), w_pad=0.35)

    def update(frame_index: int):
        for col, artist in enumerate(artists):
            image_frame, label_frame = condition_frames[col][frame_index]
            artist.set_data(_make_mnist_panel(image_frame, label_frame, sample_rows, sample_cols))
        progress = int(round(frame_index / max(1, frame_count - 1) * 100))
        title.set_text(f"{progress}% of diffusion")
        return (*artists, title)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frame_count,
        interval=int(1000 / max(1, fps)),
        blit=False,
    )
    if output_suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps)
    ani.save(save_path, writer=writer, dpi=200)
    plt.close(fig)


def _sample_last(
    model: nn.Module,
    distribution_types: List[str],
    inputs: List[torch.Tensor],
    t_init: List[float],
    steps: int,
    direction: str,
    device: str,
) -> List[torch.Tensor]:
    last_xt = [tensor.detach().cpu() for tensor in inputs]
    sampler = densiformis.diffuser.sample(
        model=model,
        distribution_types=distribution_types,
        inputs=inputs,
        t_init=t_init,
        steps=steps,
        direction=direction,
        device=device,
    )

    for xt in sampler:
        last_xt = [tensor.detach().cpu() for tensor in xt]

    return last_xt


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model = Unet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    distribution_types = ["numerical", "categorical"]

    train_dataset = MNISTDataset(split="train")
    valid_dataset = MNISTDataset(split="valid")

    print("Training diffusion model on MNIST...")
    densiformis.diffuser.fit(
        model=model,
        distribution_types=distribution_types,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        optimizer=optimizer,
        epochs=200,
        batch_size=32,
        checkpoint_save_root="./mnist_checkpoints",
        save_period=10,
        device=device,
    )

    sample_rows = 4
    sample_cols = 4
    sample_count = sample_rows * sample_cols
    images = valid_dataset.images[:sample_count]
    labels = F.one_hot(valid_dataset.labels[:sample_count], num_classes=10).float()
    image_noise = torch.randn_like(images)
    label_noise = densiformis.functions.rand_soft_label(labels.size())

    make_grid(
        model=model,
        distribution_types=distribution_types,
        images=images,
        labels=labels,
        image_noise=image_noise,
        label_noise=label_noise,
        device=device,
        save_path="mnist_grid.mp4",
        sample_rows=sample_rows,
        sample_cols=sample_cols,
    )
    print("Artifacts saved: mnist_grid.mp4")


if __name__ == "__main__":
    main()
