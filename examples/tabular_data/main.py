from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

import densiformis


@dataclass
class ColumnSpec:
    name: str
    distribution: str
    dim: int
    mean: float | None = None
    std: float | None = None
    categories: list[Any] | None = None


def infer_distribution(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "binary"
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        return "numerical"
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_categorical_dtype(series) or series.dtype == object:
        return "categorical"
    raise TypeError(f"Unsupported dtype for column {series.name!r}: {series.dtype}")


def drop_index_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in list(df.columns):
        values = df[col]
        if str(col).startswith("Unnamed") and pd.api.types.is_integer_dtype(values):
            expected = np.arange(1, len(values) + 1)
            if np.array_equal(values.to_numpy(), expected):
                df = df.drop(columns=[col])
    return df


def build_specs(df: pd.DataFrame) -> list[ColumnSpec]:
    specs = []
    for name in df.columns:
        series = df[name]
        distribution = infer_distribution(series)
        if distribution == "categorical":
            categories = sorted(series.dropna().astype(str).unique().tolist())
            specs.append(ColumnSpec(name=name, distribution=distribution, dim=len(categories), categories=categories))
        elif distribution == "binary":
            specs.append(ColumnSpec(name=name, distribution=distribution, dim=1))
        else:
            values = series.astype("float32")
            specs.append(
                ColumnSpec(
                    name=name,
                    distribution=distribution,
                    dim=1,
                    mean=float(values.mean()),
                    std=float(values.std() + 1e-8),
                )
            )
    return specs


def encode_column(series: pd.Series, spec: ColumnSpec) -> torch.Tensor:
    if spec.distribution == "categorical":
        assert spec.categories is not None
        index = {value: i for i, value in enumerate(spec.categories)}
        encoded = torch.zeros((len(series), spec.dim), dtype=torch.float32)
        for row, value in enumerate(series.astype(str)):
            if value in index:
                encoded[row, index[value]] = 1.0
        return encoded
    if spec.distribution == "binary":
        return torch.tensor(series.astype("float32").to_numpy().reshape(-1, 1), dtype=torch.float32)
    assert spec.mean is not None and spec.std is not None
    values = (series.astype("float32").to_numpy().reshape(-1, 1) - spec.mean) / spec.std
    return torch.tensor(values, dtype=torch.float32)


def decode_samples(samples: torch.Tensor, spec: ColumnSpec) -> np.ndarray:
    values = samples.detach().cpu().numpy()
    if spec.distribution == "categorical":
        assert spec.categories is not None
        labels = np.argmax(values, axis=1)
        return np.array([spec.categories[i] for i in labels], dtype=object)
    if spec.distribution == "binary":
        return (values[:, 0] > 0.5).astype(int)
    assert spec.mean is not None and spec.std is not None
    return values[:, 0] * spec.std + spec.mean


def decode_ground_truth(tensor: torch.Tensor, spec: ColumnSpec) -> Any:
    return decode_samples(tensor.reshape(1, -1), spec)[0]


class TabularDataset(Dataset):
    def __init__(
        self,
        encoded_columns: list[torch.Tensor],
        clean_df: pd.DataFrame,
        specs: list[ColumnSpec],
        missing_rate: float,
        seed: int,
    ):
        self.clean_columns = [x.clone() for x in encoded_columns]
        self.columns = [x.clone() for x in encoded_columns]
        self.clean_df = clean_df.reset_index(drop=True)
        self.specs = specs

        rng = np.random.default_rng(seed)
        for tensor in self.columns:
            row_mask = rng.random(tensor.shape[0]) < missing_rate
            if tensor.shape[1] == 1:
                tensor[row_mask, 0] = torch.nan
            else:
                tensor[row_mask, :] = torch.nan

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        return [column[index] for column in self.columns]

    def __len__(self) -> int:
        return self.columns[0].shape[0]


class TabularDNN(nn.Module):
    def __init__(self, feature_dims: list[int], hidden_dim: int = 512, hidden_layers: int = 5):
        super().__init__()
        self.feature_dims = feature_dims
        total_dim = sum(feature_dims)
        self.input_layer = nn.Linear(total_dim + len(feature_dims), hidden_dim)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)
        )
        self.hidden_norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(hidden_layers)
        )
        self.output_layer = nn.Linear(hidden_dim, total_dim)

    def forward(self, inputs: List[torch.Tensor], t: List[torch.Tensor]) -> List[torch.Tensor]:
        x = torch.concat(inputs + [ti.reshape((-1, 1)) for ti in t], dim=-1)
        x = F.silu(self.input_layer(x))
        for layer, norm in zip(self.hidden_layers, self.hidden_norms):
            x = F.silu(norm(layer(x))) + x
        x = self.output_layer(x)
        return list(torch.split(x, self.feature_dims, dim=1))


def split_dataframe(df: pd.DataFrame, validation_rate: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))
    valid_size = max(1, int(round(len(df) * validation_rate)))
    valid_idx = indices[:valid_size]
    train_idx = indices[valid_size:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[valid_idx].reset_index(drop=True)


def load_datasets(
    csv_path: Path,
    missing_rate: float,
    validation_rate: float,
    seed: int,
    max_rows: int | None,
) -> tuple[TabularDataset, TabularDataset, list[ColumnSpec]]:
    df = pd.read_csv(csv_path)
    df = drop_index_like_columns(df)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)

    train_df, valid_df = split_dataframe(df, validation_rate, seed)
    specs = build_specs(train_df)
    train_encoded = [encode_column(train_df[spec.name], spec) for spec in specs]
    valid_encoded = [encode_column(valid_df[spec.name], spec) for spec in specs]
    train_dataset = TabularDataset(train_encoded, train_df, specs, missing_rate, seed)
    valid_dataset = TabularDataset(valid_encoded, valid_df, specs, missing_rate, seed + 1)
    return train_dataset, valid_dataset, specs


def noise_like(shape: torch.Size, distribution: str, device: str) -> torch.Tensor:
    behavior = densiformis.distributions.get_distribution_behavior(distribution)
    return behavior.generate_noise(shape, device=device)


def sample_missing_distribution(
    model: nn.Module,
    specs: list[ColumnSpec],
    row_tensors: list[torch.Tensor],
    target_col: int,
    sample_count: int,
    steps: int,
    device: str,
) -> torch.Tensor:
    inputs = []
    t_init = []
    for col_idx, (tensor, spec) in enumerate(zip(row_tensors, specs)):
        repeated = tensor.reshape(1, -1).repeat(sample_count, 1).to(device)
        if col_idx == target_col:
            repeated = noise_like(repeated.shape, spec.distribution, device)
            t_init.append(1.0)
        else:
            t_init.append(0.0)
        inputs.append(repeated)

    sampler = densiformis.diffuser.sample(
        model=model,
        distribution_types=[spec.distribution for spec in specs],
        inputs=inputs,
        t_init=t_init,
        steps=steps,
        direction="backward",
        device=device,
    )
    last = None
    for last in sampler:
        pass
    assert last is not None
    return last[target_col].detach().cpu()


def predict_value(decoded_samples: np.ndarray, spec: ColumnSpec) -> Any:
    if spec.distribution == "numerical":
        return float(np.mean(decoded_samples.astype(float)))
    values, counts = np.unique(decoded_samples.astype(str), return_counts=True)
    return values[int(np.argmax(counts))]


def numerical_error_summary(truth: np.ndarray, prediction: np.ndarray, zero_atol: float = 1e-12) -> dict[str, float | int]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    absolute_error = np.abs(prediction - truth)
    nonzero_mask = np.abs(truth) > zero_atol
    if np.any(nonzero_mask):
        mpae = float(np.mean(absolute_error[nonzero_mask] / np.abs(truth[nonzero_mask])) * 100.0)
    else:
        mpae = float("nan")
    return {
        "mae": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "mpae": mpae,
        "zero_truth_count": int(np.size(truth) - np.count_nonzero(nonzero_mask)),
        "count": int(np.size(truth)),
    }


def format_metric_percent(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    return f"{value:.2f}%"


def format_cell_value(value: Any, spec: ColumnSpec) -> str:
    if spec.distribution == "numerical":
        return f"{float(value):.4g}"
    return str(value)


def make_validation_plot(
    records: list[dict[str, Any]],
    dataset_name: str,
    save_path: Path,
):
    selected_records = []
    for distribution in ("numerical", "categorical", "binary"):
        match = next(
            (record for record in records if record["spec"].distribution == distribution),
            None,
        )
        if match is not None:
            selected_records.append(match)
        if len(selected_records) == 2:
            break
    for record in records:
        if len(selected_records) == 2:
            break
        if all(record is not selected for selected in selected_records):
            selected_records.append(record)

    record_count = len(selected_records)
    fig_height = 7.2 if record_count == 1 else 12.5
    fig = plt.figure(figsize=(10, fig_height), facecolor="#fbfaf7")
    grid = fig.add_gridspec(
        record_count * 2,
        1,
        height_ratios=[0.8, 3.0] * record_count,
        hspace=0.58,
    )
    fig.suptitle(
        "One missing cell. Many plausible answers.",
        x=0.07,
        y=0.975,
        ha="left",
        fontsize=24,
        fontweight="bold",
        color="#17324d",
    )
    fig.text(
        0.07,
        0.938,
        f"{dataset_name}  |  Condition on observed columns, then generate the missing one.",
        ha="left",
        fontsize=11,
        color="#52606d",
    )

    for idx, record in enumerate(selected_records):
        table_ax = fig.add_subplot(grid[idx * 2, 0])
        ax = fig.add_subplot(grid[idx * 2 + 1, 0])
        spec = record["spec"]
        samples = record["samples"]
        target_col = record["target_col"]

        table_ax.axis("off")
        context_columns = [target_col] + [
            col_idx for col_idx in range(len(record["specs"])) if col_idx != target_col
        ][:4]
        column_names = [record["specs"][col_idx].name for col_idx in context_columns]
        row_values = [
            "?\nMISSING"
            if col_idx == target_col
            else format_cell_value(record["row_values"][col_idx], record["specs"][col_idx])
            for col_idx in context_columns
        ]
        cell_colours = [
            ["#fff0ed"] + ["#ffffff"] * (len(column_names) - 1)
            for _ in range(2)
        ]
        table = table_ax.table(
            cellText=[column_names, row_values],
            cellColours=cell_colours,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.75)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#d8dee4")
            if col == 0:
                cell.get_text().set_color("#c9473a")
                cell.get_text().set_weight("bold")
            else:
                cell.get_text().set_color("#17324d")
            if row == 0:
                cell.get_text().set_weight("bold")

        plot_label = (
            "Generated numerical possibilities"
            if spec.distribution == "numerical"
            else "Generated category probabilities"
        )
        if spec.distribution == "numerical":
            ax.hist(
                samples.astype(float),
                bins=20,
                color="#3d7ea6",
                alpha=0.88,
                edgecolor="#fbfaf7",
            )
            ax.set_ylabel("generated samples")
        else:
            values, counts = np.unique(samples.astype(str), return_counts=True)
            order = np.argsort(-counts)
            values, counts = values[order], counts[order]
            probabilities = counts / counts.sum()
            ax.bar(values, probabilities, color="#3d7ea6", alpha=0.88)
            ax.set_ylabel("probability")
            ax.tick_params(axis="x", rotation=20)
        ax.set_title(
            f"What could {spec.name} be?",
            loc="left",
            fontsize=16,
            fontweight="bold",
            color="#17324d",
            pad=14,
        )
        ax.text(
            1.0,
            1.04,
            plot_label,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#52606d",
        )
        ax.set_facecolor("#fbfaf7")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#b8c1ca")
        ax.tick_params(colors="#52606d", labelsize=9)
        ax.grid(axis="y", color="#e4e8ec", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)

    fig.text(
        0.93,
        0.022,
        "Densiformis  ·  Tabular data imputation",
        ha="right",
        fontsize=9,
        color="#52606d",
    )
    top = 0.895 if record_count == 1 else 0.905
    fig.subplots_adjust(left=0.09, right=0.95, top=top, bottom=0.065)
    fig.savefig(save_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_numerical_metric(ax: plt.Axes, metric: dict[str, Any]) -> None:
    truth = np.array(metric["truth"], dtype=float)
    prediction = np.array(metric["prediction"], dtype=float)
    error = numerical_error_summary(truth, prediction)
    lower = float(min(truth.min(), prediction.min()))
    upper = float(max(truth.max(), prediction.max()))
    if np.isclose(lower, upper):
        lower -= 1.0
        upper += 1.0
    padding = (upper - lower) * 0.05
    lower -= padding
    upper += padding

    ax.scatter(truth, prediction, color="#4c78a8", alpha=0.75, s=22)
    ax.plot([lower, upper], [lower, upper], color="#e45756", linewidth=1.8, linestyle="--")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("ground truth")
    ax.set_ylabel("prediction")
    ax.set_title(
        f"{metric['spec'].name} - MAE={error['mae']:.4g}, MPAE(nonzero)={format_metric_percent(error['mpae'])}",
        loc="left",
        fontsize=10,
    )
    ax.grid(True, color="#eeeeee", linewidth=0.8)


def plot_confusion_metric(ax: plt.Axes, metric: dict[str, Any]) -> None:
    truth = np.array(metric["truth"], dtype=str)
    prediction = np.array(metric["prediction"], dtype=str)
    labels = sorted(np.unique(np.concatenate([truth, prediction])).tolist())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true_value, predicted_value in zip(truth, prediction):
        matrix[label_to_idx[true_value], label_to_idx[predicted_value]] += 1

    accuracy = float(np.mean(truth == prediction) * 100.0)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("prediction")
    ax.set_ylabel("ground truth")
    ax.set_title(f"{metric['spec'].name} - ACC={accuracy:.2f}%", loc="left", fontsize=10)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "#222222"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color, fontsize=8)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def make_metric_plot(metrics: list[dict[str, Any]], dataset_name: str, save_path: Path) -> None:
    col_count = min(3, max(1, len(metrics)))
    row_count = int(np.ceil(len(metrics) / col_count))
    fig, axes = plt.subplots(row_count, col_count, figsize=(5.2 * col_count, 4.6 * row_count), squeeze=False)

    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")

    for ax, metric in zip(axes.ravel(), metrics):
        if metric["spec"].distribution == "numerical":
            plot_numerical_metric(ax, metric)
        else:
            plot_confusion_metric(ax, metric)

    fig.suptitle(f"{dataset_name}: validation metrics by column", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def evaluate_column_metrics(
    model: nn.Module,
    valid_dataset: TabularDataset,
    specs: list[ColumnSpec],
    dataset_name: str,
    output_dir: Path,
    metric_rows: int,
    sample_count: int,
    steps: int,
    device: str,
) -> list[dict[str, Any]]:
    row_count = len(valid_dataset) if metric_rows == 0 else min(metric_rows, len(valid_dataset))
    metrics = []
    for target_col, spec in enumerate(specs):
        truth_values = []
        prediction_values = []
        for row in range(row_count):
            row_tensors = [column[row] for column in valid_dataset.clean_columns]
            raw_samples = sample_missing_distribution(model, specs, row_tensors, target_col, sample_count, steps, device)
            decoded_samples = decode_samples(raw_samples, spec)
            truth = decode_ground_truth(row_tensors[target_col], spec)
            prediction = predict_value(decoded_samples, spec)
            truth_values.append(truth)
            prediction_values.append(prediction)
        metrics.append(
            {
                "spec": spec,
                "truth": truth_values,
                "prediction": prediction_values,
            }
        )

    plot_path = output_dir / f"{dataset_name}_metrics.png"
    make_metric_plot(metrics, dataset_name, plot_path)
    return metrics


def validate_imputation(
    model: nn.Module,
    valid_dataset: TabularDataset,
    specs: list[ColumnSpec],
    dataset_name: str,
    output_dir: Path,
    display_rows: int | None,
    sample_count: int,
    steps: int,
    device: str,
) -> list[dict[str, Any]]:
    records = []
    row_count = len(specs) if display_rows is None else display_rows
    for row in range(min(row_count, len(valid_dataset), len(specs))):
        target_col = row
        row_tensors = [column[row] for column in valid_dataset.clean_columns]
        spec = specs[target_col]
        raw_samples = sample_missing_distribution(model, specs, row_tensors, target_col, sample_count, steps, device)
        decoded_samples = decode_samples(raw_samples, spec)
        truth = decode_ground_truth(row_tensors[target_col], spec)
        row_values = [decode_ground_truth(row_tensor, col_spec) for row_tensor, col_spec in zip(row_tensors, specs)]
        prediction = predict_value(decoded_samples, spec)
        condition_names = [s.name for i, s in enumerate(specs) if i != target_col]
        records.append(
            {
                "row": row,
                "spec": spec,
                "specs": specs,
                "target_col": target_col,
                "row_values": row_values,
                "samples": decoded_samples,
                "truth": truth,
                "prediction": prediction,
                "conditions": ", ".join(condition_names[:4]) + ("..." if len(condition_names) > 4 else ""),
            }
        )

    plot_path = output_dir / f"{dataset_name}_validation.png"
    if records:
        make_validation_plot(records, dataset_name, plot_path)
    return records


def build_model(specs: list[ColumnSpec], args: argparse.Namespace, device: str) -> TabularDNN:
    return TabularDNN(
        [spec.dim for spec in specs],
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
    ).to(device)


def checkpoint_path_for(output_dir: Path, dataset_name: str) -> Path:
    return output_dir / dataset_name / "model.pt"


def print_evaluation_metrics(metrics: list[dict[str, Any]]) -> None:
    for metric in metrics:
        spec = metric["spec"]
        if spec.distribution == "numerical":
            truth = np.array(metric["truth"], dtype=float)
            prediction = np.array(metric["prediction"], dtype=float)
            error = numerical_error_summary(truth, prediction)
            print(
                f"metric column={spec.name} MAE={error['mae']:.4g} RMSE={error['rmse']:.4g} "
                f"MPAE(nonzero)={format_metric_percent(error['mpae'])} "
                f"zero_truth={error['zero_truth_count']}/{error['count']}"
            )
        else:
            truth = np.array(metric["truth"], dtype=str)
            prediction = np.array(metric["prediction"], dtype=str)
            print(f"metric column={spec.name} ACC={float(np.mean(truth == prediction) * 100.0):.2f}%")


def evaluate_one(args: argparse.Namespace, csv_path: Path) -> None:
    dataset_name = csv_path.stem.replace("-", "_")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, valid_dataset, specs = load_datasets(
        csv_path=csv_path,
        missing_rate=args.missing_rate,
        validation_rate=args.validation_rate,
        seed=args.seed,
        max_rows=args.max_rows,
    )
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_model(specs, args, device)
    checkpoint_path = checkpoint_path_for(output_dir, dataset_name)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    print(f"\nDataset: {csv_path} train={len(train_dataset)} valid={len(valid_dataset)} device={device}")
    print("Distributions:", {spec.name: spec.distribution for spec in specs})
    print(f"Loaded checkpoint: {checkpoint_path}")

    records = validate_imputation(
        model=model,
        valid_dataset=valid_dataset,
        specs=specs,
        dataset_name=dataset_name,
        output_dir=output_dir,
        display_rows=args.display_rows,
        sample_count=args.sample_count,
        steps=args.sample_steps,
        device=device,
    )
    for record in records:
        print(
            f"validation row={record['row']} missing={record['spec'].name} "
            f"truth={record['truth']} prediction={record['prediction']}"
        )
    print(f"Saved plot: {output_dir / f'{dataset_name}_validation.png'}")

    metrics = evaluate_column_metrics(
        model=model,
        valid_dataset=valid_dataset,
        specs=specs,
        dataset_name=dataset_name,
        output_dir=output_dir,
        metric_rows=args.metric_rows,
        sample_count=args.metric_sample_count,
        steps=args.metric_sample_steps,
        device=device,
    )
    print_evaluation_metrics(metrics)
    print(f"Saved metrics plot: {output_dir / f'{dataset_name}_metrics.png'}")


def train_one(args: argparse.Namespace, csv_path: Path) -> None:
    dataset_name = csv_path.stem.replace("-", "_")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, valid_dataset, specs = load_datasets(
        csv_path=csv_path,
        missing_rate=args.missing_rate,
        validation_rate=args.validation_rate,
        seed=args.seed,
        max_rows=args.max_rows,
    )
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_model(specs, args, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"\nDataset: {csv_path} train={len(train_dataset)} valid={len(valid_dataset)} device={device}")
    print("Distributions:", {spec.name: spec.distribution for spec in specs})
    densiformis.diffuser.fit(
        model=model,
        distribution_types=[spec.distribution for spec in specs],
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        optimizer=optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_save_root=str(output_dir / dataset_name),
        save_period=max(1, args.epochs),
        device=device,
        verbose=not args.quiet,
    )
    print(f"Saved checkpoint: {checkpoint_path_for(output_dir, dataset_name)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate tabular DataFrame diffusion examples.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("csv", nargs="*", type=Path, default=[Path("winequality-red.csv"), Path("diamonds.csv")])
        command_parser.add_argument("--hidden-dim", type=int, default=2048)
        command_parser.add_argument("--hidden-layers", type=int, default=16)
        command_parser.add_argument("--missing-rate", type=float, default=0.0)
        command_parser.add_argument("--validation-rate", type=float, default=0.10)
        command_parser.add_argument("--max-rows", type=int, default=None)
        command_parser.add_argument("--seed", type=int, default=1234)
        command_parser.add_argument("--output-dir", type=str, default="./checkpoints/tabular")

    train_parser = subparsers.add_parser("train", help="Train models and save checkpoints.")
    add_common_arguments(train_parser)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--quiet", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate", help="Load checkpoints and generate validation plots/metrics.")
    add_common_arguments(evaluate_parser)
    evaluate_parser.add_argument("--display-rows", type=int, default=None)
    evaluate_parser.add_argument("--sample-count", type=int, default=128)
    evaluate_parser.add_argument("--sample-steps", type=int, default=100)
    evaluate_parser.add_argument("--metric-rows", type=int, default=64, help="Validation rows per column for metrics. Use 0 for all rows.")
    evaluate_parser.add_argument("--metric-sample-count", type=int, default=32)
    evaluate_parser.add_argument("--metric-sample-steps", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for csv_path in args.csv:
        if args.command == "train":
            train_one(args, csv_path)
        elif args.command == "evaluate":
            evaluate_one(args, csv_path)
        else:
            raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
