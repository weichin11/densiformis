import os
from io import BytesIO
from pathlib import Path
from typing import List, Sequence
import zipfile
import numpy as np
import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.nn import Conv2d, InstanceNorm2d, Linear, Module, ModuleDict, ModuleList
from torch.nn.functional import interpolate, silu
from torch.utils.data import Dataset, Subset

import densiformis

class CelebAMaskHQ(Dataset):
    def __init__(self, zipfile_path="./data/CelebAMask-HQ.zip", image_size_hw = (128,128)) -> None:
        self.image_size_hw = image_size_hw
        self.zipfile = zipfile.ZipFile(zipfile_path, "r")
        self.image_folder = "CelebAMask-HQ/CelebA-HQ-img/"
        self.mask_folder = "CelebAMask-HQ/CelebAMask-HQ-mask-anno/"

        label_names = [
            "skin",
            "nose",
            "eye_g",
            "l_eye",
            "r_eye",
            "l_brow",
            "r_brow",
            "l_ear",
            "r_ear",
            "mouth",
            "u_lip",
            "l_lip",
            "hair",
            "hat",
            "ear_r",
            "neck_l",
            "neck",
            "cloth",
        ]
        self.label_index_map = {name: idx for idx, name in enumerate(label_names)}

        self.image_bytes_dict = {}
        self.mask_bytes_dict = {}

        for name in tqdm.tqdm(self.zipfile.namelist(), ncols=0, desc="Loading dataset"):
            if name.startswith(self.image_folder) and name.endswith(".jpg"):
                index = name.split("/")[-1].split(".")[0]
                index = int(index)
                self.image_bytes_dict[index] = self.zipfile.read(name)
            
            if name.startswith(self.mask_folder) and name.endswith(".png"):
                index, label = name.split("/")[-1].split(".")[0].split("_",1)
                index = int(index)
                if index not in self.mask_bytes_dict:
                    self.mask_bytes_dict[index] = {}
                self.mask_bytes_dict[index][label] = self.zipfile.read(name)
        
        assert len(self.image_bytes_dict) == len(self.mask_bytes_dict)

    def __len__(self):
        return len(self.image_bytes_dict)

    def __getitem__(self, index: int):
        image = Image.open(BytesIO(self.image_bytes_dict[index])).convert("RGB")
        image = image.resize((self.image_size_hw[1], self.image_size_hw[0]), Image.Resampling.LANCZOS)
        image = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 127.5 - 1.0

        mask = np.zeros((len(self.label_index_map), *self.image_size_hw), dtype=np.uint8)
        for label in self.mask_bytes_dict[index].keys():
            label_mask = Image.open(BytesIO(self.mask_bytes_dict[index][label])).convert("L")
            label_mask = label_mask.resize((self.image_size_hw[1], self.image_size_hw[0]), Image.Resampling.NEAREST)
            mask[self.label_index_map[label]] = np.array(label_mask)
        mask = np.stack(mask, axis=0)
        mask = torch.from_numpy(mask > 0).float()        
        return [image, mask]

class Unet(Module):
    def __init__(self, filters, blocks, image_size_hw=(128,128), image_channels=3, mask_channels=18) -> None:
        super().__init__()
        self.image_size_hw = image_size_hw
        self.filters = filters

        self.in_image = Conv2d(image_channels, filters // 3, (3, 3), padding="same")
        self.out_image = Conv2d(filters // 2, image_channels, (3, 3), padding="same")

        self.in_mask = Conv2d(mask_channels, filters // 3, (3, 3), padding="same")
        self.out_mask = Conv2d(filters // 2, mask_channels, (3, 3), padding="same")

        self.linear_t = Linear(2, (filters // 3) * image_size_hw[0] * image_size_hw[1])

        self.in_conv = Conv2d(3 * (filters // 3), filters, (3, 3), padding="same")
        self.in_norm = InstanceNorm2d(3 * (filters // 3))

        self.out_conv = Conv2d(filters, 2 * (filters // 2), (3, 3), padding="same")
        self.out_norm = InstanceNorm2d(filters)

        self.down_scale_blocks = ModuleList(
            [
                ScaleBlock(filters, filters * 2, 0.5),
                ScaleBlock(filters * 2, filters * 4, 0.5),
                ScaleBlock(filters * 4, filters * 8, 0.5),
                ScaleBlock(filters * 8, filters * 16, 0.5),
                ScaleBlock(filters * 16, filters * 32, 0.5),
            ]
        )
        self.down_res_blocks = ModuleList(
            [
                ResidualBlock(filters, blocks),
                ResidualBlock(filters * 2, blocks),
                ResidualBlock(filters * 4, blocks),
                ResidualBlock(filters * 8, blocks),
                ResidualBlock(filters * 16, blocks),
                ResidualBlock(filters * 32, blocks),
            ]
        )
        self.up_scale_blocks = ModuleList(
            [
                ScaleBlock(filters * 32, filters * 16, 2.0),
                ScaleBlock(filters * 16, filters * 8, 2.0),
                ScaleBlock(filters * 8, filters * 4, 2.0),
                ScaleBlock(filters * 4, filters * 2, 2.0),
                ScaleBlock(filters * 2, filters, 2.0),
            ]
        )
        self.up_res_blocks = ModuleList(
            [
                ResidualBlock(filters * 32, blocks),
                ResidualBlock(filters * 16, blocks),
                ResidualBlock(filters * 8, blocks),
                ResidualBlock(filters * 4, blocks),
                ResidualBlock(filters * 2, blocks),
                ResidualBlock(filters, blocks),
            ]
        )

    def forward(self, inputs: Sequence[torch.Tensor], t: Sequence[torch.Tensor]):
        image, mask = inputs
        time_inputs = torch.stack([ti.view(-1) for ti in t], dim=1)
        x = torch.concat(
            [
                self.in_image(image),
                self.in_mask(mask),
                self.linear_t(time_inputs).reshape((-1, self.filters // 3, *self.image_size_hw)),
            ],
            dim=1,
        )
        x = silu(x)

        x = self.in_norm(x)
        x = self.in_conv(x)
        x = silu(x)

        f1 = self.down_res_blocks[0](x)
        x = self.down_scale_blocks[0](f1)
        f2 = self.down_res_blocks[1](x)
        x = self.down_scale_blocks[1](f2)
        f3 = self.down_res_blocks[2](x)
        x = self.down_scale_blocks[2](f3)
        f4 = self.down_res_blocks[3](x)
        x = self.down_scale_blocks[3](f4)
        f5 = self.down_res_blocks[4](x)
        x = self.down_scale_blocks[4](f5)
        x = self.down_res_blocks[5](x)

        x = self.up_res_blocks[0](x)
        x = self.up_scale_blocks[0](x)
        x = x + f5
        x = self.up_res_blocks[1](x)
        x = self.up_scale_blocks[1](x)
        x = x + f4
        x = self.up_res_blocks[2](x)
        x = self.up_scale_blocks[2](x)
        x = x + f3
        x = self.up_res_blocks[3](x)
        x = self.up_scale_blocks[3](x)
        x = x + f2
        x = self.up_res_blocks[4](x)
        x = self.up_scale_blocks[4](x)
        x = x + f1
        x = self.up_res_blocks[5](x)

        x = self.out_norm(x)
        x = self.out_conv(x)
        x = silu(x)

        x, m = torch.split(x, split_size_or_sections=[self.filters // 2] * 2, dim=1)
        return [self.out_image(x), self.out_mask(m)]

class ScaleBlock(Module):
    def __init__(self, in_channels, out_channels, scale_factor) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.norm = InstanceNorm2d(in_channels)
        self.conv = Conv2d(in_channels, out_channels, (3, 3), padding="same")

    def forward(self, x):
        if self.scale_factor > 1.0:
            x = interpolate(x, scale_factor=self.scale_factor, mode="bilinear")
        x = self.norm(x)
        x = self.conv(x)
        x = silu(x)
        if self.scale_factor < 1.0:
            x = interpolate(x, scale_factor=self.scale_factor, mode="bilinear")
        return x

class ResidualBlock(Module):
    def __init__(self, in_channels, blocks) -> None:
        super().__init__()
        self.blocks = blocks
        layer_dict = {}
        for i in range(blocks):
            layer_dict[f"b{i}_norm1"] = InstanceNorm2d(in_channels)
            layer_dict[f"b{i}_conv1"] = Conv2d(in_channels, in_channels, (3, 3), padding="same", groups=in_channels)
            layer_dict[f"b{i}_norm2"] = InstanceNorm2d(in_channels)
            layer_dict[f"b{i}_conv2"] = Conv2d(in_channels, in_channels, (1, 1), padding="same")
        self.layers = ModuleDict(layer_dict)

    def forward(self, x):
        for i in range(self.blocks):
            res = x
            x = self.layers[f"b{i}_norm1"](x)
            x = self.layers[f"b{i}_conv1"](x)
            x = silu(x)
            x = self.layers[f"b{i}_norm2"](x)
            x = self.layers[f"b{i}_conv2"](x)
            x = silu(x)
            x = x + res
        return x

def _image_to_numpy(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().clamp(-1, 1)
    return ((image.permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0, 1)

def _mask_to_rgb(mask: torch.Tensor) -> np.ndarray:
    mask = mask.detach().cpu().clamp(0, 1)
    h, w = mask.shape[-2:]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    colors = plt.get_cmap("tab20")(np.arange(mask.size(0)))[:, :3]

    for channel, color in enumerate(colors):
        alpha = mask[channel].numpy()[..., None] * 0.8
        rgb = rgb * (1.0 - alpha) + color * alpha

    return rgb.clip(0, 1)

def _make_celeb_panel(
    images: torch.Tensor,
    masks: torch.Tensor,
    sample_rows: int = 4,
    sample_cols: int = 2,
) -> np.ndarray:
    images = images.detach().cpu()
    masks = masks.detach().cpu()
    tile_count = min(sample_rows * sample_cols, images.size(0), masks.size(0))

    image_h, image_w = images.shape[-2:]
    pad = 4
    tile_w = image_w * 2 + pad
    canvas_h = sample_rows * image_h + (sample_rows - 1) * pad
    canvas_w = sample_cols * tile_w + (sample_cols - 1) * pad
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.float32)

    for idx in range(tile_count):
        row, col = divmod(idx, sample_cols)
        y0 = row * (image_h + pad)
        x0 = col * (tile_w + pad)

        canvas[y0 : y0 + image_h, x0 : x0 + image_w] = _image_to_numpy(images[idx])
        mask_x0 = x0 + image_w + pad
        canvas[y0 : y0 + image_h, mask_x0 : mask_x0 + image_w] = _mask_to_rgb(masks[idx])

    return canvas

def make_grid(
    model: Module,
    distribution_types: List[str],
    images: torch.Tensor,
    masks: torch.Tensor,
    image_noise: torch.Tensor,
    mask_noise: torch.Tensor,
    device: str,
    steps: int = 48,
    save_path: str = "celeb_a_mask_hq_grid.gif",
    fps: int = 24,
    sample_rows: int = 4,
    sample_cols: int = 2,
):
    if sample_rows < 1 or sample_cols < 1:
        raise ValueError("sample_rows and sample_cols must be positive")

    sample_count = min(
        sample_rows * sample_cols,
        images.size(0),
        masks.size(0),
        image_noise.size(0),
        mask_noise.size(0),
    )
    if sample_count < 1:
        raise ValueError("make_grid needs at least one sample")

    images = images[:sample_count]
    masks = masks[:sample_count]
    image_noise = image_noise[:sample_count]
    mask_noise = mask_noise[:sample_count]

    conditions = [
        {
            "name": "Generate both",
            "inputs": [image_noise, mask_noise],
            "t_init": [1.0, 1.0],
        },
        {
            "name": "Generate images only",
            "inputs": [image_noise, masks],
            "t_init": [1.0, 0.0],
        },
        {
            "name": "Generate masks only",
            "inputs": [images, mask_noise],
            "t_init": [0.0, 1.0],
        },
    ]

    save_path = Path(save_path)
    output_suffix = save_path.suffix.lower()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if output_suffix not in {".gif", ".mp4"}:
        raise ValueError("save_path must end with .gif or .mp4")

    import matplotlib.animation as animation

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
        image_frame, mask_frame = condition_frames[col][0]
        artist = axes[col].imshow(
            _make_celeb_panel(image_frame, mask_frame, sample_rows, sample_cols),
            animated=True,
        )
        axes[col].set_xticks([])
        axes[col].set_yticks([])
        axes[col].set_xlabel(cond["name"], fontsize=7, labelpad=4)
        artists.append(artist)

    title = fig.suptitle("0% of diffusion", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94), w_pad=0.35)

    def update(frame_index: int):
        for col, artist in enumerate(artists):
            image_frame, mask_frame = condition_frames[col][frame_index]
            artist.set_data(_make_celeb_panel(image_frame, mask_frame, sample_rows, sample_cols))
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

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

    image_size = (64, 64)
    distribution_types=["numerical","binary"]
    validation_number = 3000

    model = Unet(filters=64, blocks=4, image_size_hw=image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    full_dataset = CelebAMaskHQ(image_size_hw=image_size)
    valid_split = max(1, min(validation_number, len(full_dataset) - 1))
    train_indices = list(range(len(full_dataset) - valid_split))
    valid_indices = list(range(len(full_dataset) - valid_split, len(full_dataset)))
    train_dataset = Subset(full_dataset, train_indices)
    valid_dataset = Subset(full_dataset, valid_indices)


    densiformis.diffuser.fit(
        model=model,
        distribution_types=distribution_types,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        optimizer=optimizer,
        epochs=500,
        batch_size=32,
        checkpoint_save_root="./celeb_a_mask_hq_checkpoints",
        save_period=1,
        device=device
    )

    sample_rows = 5
    sample_cols = 2
    samples = [valid_dataset[index] for index in range(min(sample_rows * sample_cols, len(valid_dataset)))]
    images = torch.stack([sample[0] for sample in samples])
    masks = torch.stack([sample[1] for sample in samples])
    image_noise = torch.randn_like(images)
    mask_noise = torch.rand_like(masks)

    make_grid(
        model=model,
        distribution_types=distribution_types,
        images=images,
        masks=masks,
        image_noise=image_noise,
        mask_noise=mask_noise,
        device=device,
        save_path="celeb_a_mask_hq_grid.mp4",
        sample_rows=sample_rows,
        sample_cols=sample_cols,
    )
    print("Artifact saved: celeb_a_mask_hq_grid.mp4")
