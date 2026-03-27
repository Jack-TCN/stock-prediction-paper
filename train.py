from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_WINDOW_SIZE = 60
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_DROPOUT = 0.2
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-3
DEFAULT_SEED = 42
DEFAULT_PATIENCE = 10
DEFAULT_VAL_RATIO = 0.1
DEFAULT_MIN_DELTA = 0.0
DEFAULT_OUTPUT_ROOT_DIR = "training_runs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TimeSeriesWindowDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, metadata: dict | None = None) -> None:
        self.X = X.float()
        self.y = y.float()
        self.metadata = metadata or {}

    @classmethod
    def from_pt(cls, pt_path: Path) -> "TimeSeriesWindowDataset":
        bundle = torch.load(pt_path)
        return cls(bundle["X"], bundle["y"], bundle)

    def slice(self, start: int, end: int) -> "TimeSeriesWindowDataset":
        sliced_metadata = dict(self.metadata)
        for key in ["sample_start_dates", "sample_end_dates", "label_dates", "label_row_indices"]:
            if key in sliced_metadata:
                sliced_metadata[key] = sliced_metadata[key][start:end]
        sliced_metadata["input_shape"] = [end - start, *self.X.shape[1:]]
        sliced_metadata["target_shape"] = [end - start]
        return TimeSeriesWindowDataset(self.X[start:end], self.y[start:end], sliced_metadata)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class RecurrentRegressor(nn.Module):
    def __init__(
        self,
        model_type: str,
        input_size: int,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()

        if model_type not in {"lstm", "gru"}:
            raise ValueError(f"Unsupported model_type: {model_type}")

        rnn_cls = nn.LSTM if model_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        # For a single recurrent layer, external dropout is the controllable regularization term.
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        last_hidden = output[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden).squeeze(-1)


@dataclass
class TrainConfig:
    ticker: str
    feature_set: str
    model_type: str
    window_size: int
    hidden_size: int
    dropout: float
    batch_size: int
    epochs: int
    learning_rate: float
    seed: int
    patience: int
    val_ratio: float
    min_delta: float
    device: str
    output_root_dir: str = DEFAULT_OUTPUT_ROOT_DIR


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_datasets(
    project_root: Path,
    ticker: str,
    feature_set: str,
    window_size: int,
) -> tuple[TimeSeriesWindowDataset, TimeSeriesWindowDataset]:
    base_dir = project_root / "data" / "tensors" / f"window_{window_size}" / ticker / feature_set
    train_path = base_dir / "train.pt"
    test_path = base_dir / "test.pt"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Tensor files not found for ticker={ticker}, feature_set={feature_set}, window_size={window_size}. "
            f"Expected: {train_path} and {test_path}"
        )

    return TimeSeriesWindowDataset.from_pt(train_path), TimeSeriesWindowDataset.from_pt(test_path)


def split_train_validation(
    train_dataset: TimeSeriesWindowDataset,
    val_ratio: float,
) -> tuple[TimeSeriesWindowDataset, TimeSeriesWindowDataset]:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    total = len(train_dataset)
    val_size = max(1, int(total * val_ratio))
    train_size = total - val_size
    if train_size <= 0:
        raise ValueError("Validation ratio is too large for the dataset size.")

    train_subset = train_dataset.slice(0, train_size)
    val_subset = train_dataset.slice(train_size, total)
    return train_subset, val_subset


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    y_true = y_true.float()
    y_pred = y_pred.float()
    mse = torch.mean((y_true - y_pred) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(y_true - y_pred))
    mape = torch.mean(torch.abs((y_true - y_pred) / torch.clamp(torch.abs(y_true), min=1e-8))) * 100

    return {
        "RMSE": float(rmse.item()),
        "MAE": float(mae.item()),
        "MAPE": float(mape.item()),
    }


def run_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    batch_count = 0
    all_preds = []
    all_targets = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if is_train:
                optimizer.zero_grad()

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            batch_count += 1
            all_preds.append(preds.detach().cpu())
            all_targets.append(y_batch.detach().cpu())

    epoch_loss = running_loss / max(batch_count, 1)
    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)
    metrics = regression_metrics(y_true, y_pred)
    return epoch_loss, metrics


def prepare_output_dir(project_root: Path, config: TrainConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{timestamp}_{config.ticker}_{config.feature_set}_{config.model_type}"
        f"_w{config.window_size}_d{str(config.dropout).replace('.', 'p')}"
    )
    output_dir = project_root / config.output_root_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def train_one_experiment(config: TrainConfig, project_root: Path | None = None) -> dict:
    set_seed(config.seed)
    project_root = project_root or resolve_project_root()
    device = torch.device(config.device)

    full_train_dataset, test_dataset = load_datasets(
        project_root=project_root,
        ticker=config.ticker,
        feature_set=config.feature_set,
        window_size=config.window_size,
    )
    train_dataset, val_dataset = split_train_validation(full_train_dataset, config.val_ratio)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config.batch_size,
    )

    input_size = train_dataset.X.shape[-1]
    model = RecurrentRegressor(
        model_type=config.model_type,
        input_size=input_size,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    output_dir = prepare_output_dir(project_root, config)

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_path = output_dir / "best_model.pt"
    best_record: dict | None = None

    print(f"Device: {device}")
    print(
        f"Train samples: {len(train_dataset)} | Validation samples: {len(val_dataset)} | "
        f"Test samples: {len(test_dataset)}"
    )
    print(f"Input shape per sample: {tuple(train_dataset.X.shape[1:])}")
    print(f"Feature names: {full_train_dataset.metadata.get('feature_names', [])}")

    for epoch in range(1, config.epochs + 1):
        train_loss, train_metrics = run_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = run_one_epoch(model, val_loader, criterion, None, device)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_RMSE": train_metrics["RMSE"],
            "train_MAE": train_metrics["MAE"],
            "train_MAPE": train_metrics["MAPE"],
            "val_RMSE": val_metrics["RMSE"],
            "val_MAE": val_metrics["MAE"],
            "val_MAPE": val_metrics["MAPE"],
        }
        history.append(record)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.6f} | Train RMSE {train_metrics['RMSE']:.4f} | "
            f"Train MAE {train_metrics['MAE']:.4f} | Train MAPE {train_metrics['MAPE']:.4f} | "
            f"Val Loss {val_loss:.6f} | Val RMSE {val_metrics['RMSE']:.4f} | "
            f"Val MAE {val_metrics['MAE']:.4f} | Val MAPE {val_metrics['MAPE']:.4f}"
        )

        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_record = record.copy()
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best epoch: {best_epoch}, best val_loss: {best_val_loss:.6f}"
            )
            break

    stopped_epoch = history[-1]["epoch"]

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_metrics = run_one_epoch(model, test_loader, criterion, None, device)
    torch.save(model.state_dict(), output_dir / "final_model_reloaded_best.pt")
    pd.DataFrame(history).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "ticker": config.ticker,
        "feature_set": config.feature_set,
        "model_type": config.model_type,
        "window_size": config.window_size,
        "hidden_size": config.hidden_size,
        "learning_rate": config.learning_rate,
        "dropout": config.dropout,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val_loss": best_val_loss,
        "best_val_RMSE": best_record["val_RMSE"] if best_record else None,
        "best_val_MAE": best_record["val_MAE"] if best_record else None,
        "best_val_MAPE": best_record["val_MAPE"] if best_record else None,
        "test_loss": test_loss,
        "test_RMSE": test_metrics["RMSE"],
        "test_MAE": test_metrics["MAE"],
        "test_MAPE": test_metrics["MAPE"],
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "input_shape": list(train_dataset.X.shape[1:]),
        "device": str(device),
        "best_model_path": best_model_path.relative_to(project_root).as_posix(),
        "output_dir": output_dir.relative_to(project_root).as_posix(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"Test | Loss {test_loss:.6f} | RMSE {test_metrics['RMSE']:.4f} | "
        f"MAE {test_metrics['MAE']:.4f} | MAPE {test_metrics['MAPE']:.4f}"
    )
    print(f"\nArtifacts saved to: {output_dir.relative_to(project_root).as_posix()}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single-layer LSTM or GRU for stock price prediction.")
    parser.add_argument("--ticker", type=str, default="AAPL", choices=["AAPL", "MSFT", "TSLA"])
    parser.add_argument("--feature-set", type=str, default="baseline", choices=["baseline", "proposed"])
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "gru"])
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-root-dir", type=str, default=DEFAULT_OUTPUT_ROOT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    config = TrainConfig(
        ticker=args.ticker,
        feature_set=args.feature_set,
        model_type=args.model,
        window_size=args.window_size,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        patience=args.patience,
        val_ratio=args.val_ratio,
        min_delta=args.min_delta,
        device=str(device),
        output_root_dir=args.output_root_dir,
    )
    train_one_experiment(config=config, project_root=resolve_project_root())


if __name__ == "__main__":
    main()
