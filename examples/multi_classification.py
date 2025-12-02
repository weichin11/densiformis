from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import densiformis


class MultiClassDataset(Dataset):
    """Return [positions, one-hot labels] tensors for a given split."""

    def __init__(self, split: str, n_samples: int = 10000, random_state: int = 1234):
        pos, label = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            n_classes=3,
            random_state=random_state,
        )
        pos, label = pos.reshape((n_samples, 2)), label.reshape((n_samples,))

        # Normalize coordinates for a balanced regression target.
        pos = (pos - pos.mean()) / (pos.std() + 1e-8)

        pos_train, pos_test, label_train, label_test = train_test_split(
            pos, label, test_size=0.2, random_state=random_state
        )

        if split == "train":
            self.pos = torch.tensor(pos_train, dtype=torch.float32)
            self.label = torch.tensor(label_train, dtype=torch.int64)
        elif split == "valid":
            self.pos = torch.tensor(pos_test, dtype=torch.float32)
            self.label = torch.tensor(label_test, dtype=torch.int64)
        else:
            raise ValueError(f"split should be train or valid, but got {split}")

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        return [self.pos[index], F.one_hot(self.label[index], 3).float()]

    def __len__(self) -> int:
        return self.pos.shape[0]


class SimpleDNN(nn.Module):
    """Small MLP that predicts residuals for positions (2-dim) and labels (3-dim)."""

    def __init__(self, x_dim: int = 5, hidden_dim: int = 256):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(x_dim + 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, x_dim)

    def forward(self, inputs: List[torch.Tensor], t: List[torch.Tensor]) -> List[torch.Tensor]:
        y = torch.concat(inputs + [i.view((-1, 1)) for i in t], dim=-1)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        y = self.fc4(y)
        pos, label = torch.split(y, [2, 3], dim=1)
        return [pos, label]


def make_grid(
    model: nn.Module,
    distribution_types: List[str],
    pos: torch.Tensor,
    label: torch.Tensor,
    pos_noise: torch.Tensor,
    label_noise: torch.Tensor,
    device: str,
    time_slices: int = 5,
    steps: int = 100,
    save_path: str = "multi_grid.png",
):
    """Create a 3x5 (conditions x time) grid illustrating denoising progress."""
    snapshot_steps = np.linspace(0, steps, time_slices, dtype=int)
    step_labels = [f"{int(round(s / steps * 100))}%" for s in snapshot_steps]

    conditions = [
        {
            "name": "Denoise both",
            "inputs": [pos_noise, label_noise],
            "t_init": [1.0, 1.0],
        },
        {
            "name": "Denoise positions (labels fixed)",
            "inputs": [pos_noise, label],
            "t_init": [1.0, 0.0],
        },
        {
            "name": "Denoise labels (positions fixed)",
            "inputs": [pos, label_noise],
            "t_init": [0.0, 1.0],
        },
    ]

    fig, axes = plt.subplots(len(conditions), time_slices, figsize=(time_slices * 3.5, len(conditions) * 3.5))

    for row, cond in enumerate(conditions):
        snapshots = [[tensor.detach().cpu() for tensor in cond["inputs"]]]
        sampler = densiformis.diffuser.sample(
            model=model,
            distribution_types=distribution_types,
            inputs=cond["inputs"],
            t_init=cond["t_init"],
            steps=steps,
            direction="backward",
            device=device,
        )
        target_steps = list(snapshot_steps[1:])
        target_index = 0
        last_xt = None

        for step_idx, xt in enumerate(sampler, start=1):
            last_xt = xt
            if target_index < len(target_steps) and step_idx >= target_steps[target_index]:
                snapshots.append([tensor.detach().cpu() for tensor in xt])
                target_index += 1
            if target_index >= len(target_steps):
                break

        if len(snapshots) < len(target_steps) + 1 and last_xt is not None:
            snapshots.append([tensor.detach().cpu() for tensor in last_xt])

        for col, (pos_frame, lbl_frame) in enumerate(snapshots):
            ax = axes[row, col]
            pos_np, lbl_np = pos_frame.numpy(), lbl_frame.numpy()
            ax.scatter(pos_np[:, 0], pos_np[:, 1], c=np.clip(lbl_np,0.0,1.0), edgecolor="k")
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{step_labels[col]} of diffusion")
            if col == 0:
                ax.set_ylabel(cond["name"])

    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Data
    train_dataset = MultiClassDataset(split="train")
    valid_dataset = MultiClassDataset(split="valid")

    # 2) Model + optimizer
    model = SimpleDNN().to(device)
    optimizer = optim.Adam(model.parameters())
    distribution_types = ["numerical", "categorical"]

    # 3) Train and save checkpoints
    print("Training diffusion model on synthetic multi-class dataset...")
    densiformis.diffuser.fit(
        model=model,
        distribution_types=distribution_types,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        optimizer=optimizer,
        epochs=100,
        batch_size=32,
        checkpoint_save_root="./",
        save_period=10,
        device=device,
    )

    # 4) Load the final weights back for sampling
    ckpt_path = Path("./model.pt")
    if ckpt_path.exists():
        print(f"Loading trained weights from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    else:
        print(f"No checkpoint found at {ckpt_path}; using freshly trained weights.")

    # 5) Prepare clean inputs and noisy starting points
    pos = valid_dataset.pos
    label = F.one_hot(valid_dataset.label, 3).float()
    pos_noise = torch.randn_like(pos)
    label_noise = densiformis.functions.rand_soft_label(label.size())

    # 6) Build a compact grid that shows the denoising trajectory
    make_grid(model, distribution_types, pos, label, pos_noise, label_noise, device)
    print("Artifacts saved: multi_grid.png")


if __name__ == "__main__":
    main()
