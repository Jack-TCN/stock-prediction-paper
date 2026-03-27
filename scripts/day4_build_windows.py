from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


TICKERS = ["AAPL", "MSFT", "TSLA"]
TRAIN_RATIO = 0.8
DEFAULT_WINDOW_SIZES = [60]

BASELINE_FEATURES = ["Open", "High", "Low", "Close", "Volume"]
PROPOSED_FEATURES = BASELINE_FEATURES + [
    "RSI_14",
    "MACD",
    "MACD_SIGNAL",
    "MACD_DIFF",
    "BB_MAVG",
    "BB_HIGH",
    "BB_LOW",
]
TARGET_COLUMN = "Target_Close_Next"


def resolve_project_root() -> Path:
    cwd = Path.cwd()
    script_root = Path(__file__).resolve().parents[1]
    if (cwd / "scripts").exists() and (cwd / "data").exists():
        return cwd
    return script_root


def load_scaled_and_clean_data(project_root: Path, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    clean_path = project_root / "data" / "processed" / "featured" / f"{ticker}_featured_clean.csv"
    train_scaled_path = project_root / "data" / "processed" / "scaled" / f"{ticker}_train_scaled.csv"
    test_scaled_path = project_root / "data" / "processed" / "scaled" / f"{ticker}_test_scaled.csv"
    raw_path = project_root / "data" / "raw" / f"{ticker}.csv"

    clean_df = pd.read_csv(clean_path, parse_dates=["Date"])
    train_scaled_df = pd.read_csv(train_scaled_path, parse_dates=["Date"])
    test_scaled_df = pd.read_csv(test_scaled_path, parse_dates=["Date"])
    raw_df = pd.read_csv(raw_path, parse_dates=["Date"])

    full_scaled_df = pd.concat([train_scaled_df, test_scaled_df], ignore_index=True)
    raw_last_date = raw_df["Date"].iloc[-1].strftime("%Y-%m-%d")
    return clean_df, full_scaled_df, raw_last_date


def build_split_samples(
    clean_df: pd.DataFrame,
    scaled_df: pd.DataFrame,
    feature_columns: list[str],
    split_name: str,
    train_boundary: int,
    window_size: int,
    raw_last_date: str,
) -> dict:
    if split_name == "train":
        label_start = window_size - 1
        label_end = train_boundary - 2
    elif split_name == "test":
        label_start = max(window_size - 1, train_boundary - 1)
        label_end = len(clean_df) - 1
    else:
        raise ValueError(f"Unsupported split: {split_name}")

    if label_end < label_start:
        raise ValueError(
            f"Not enough rows to create {split_name} windows. "
            f"window_size={window_size}, label_start={label_start}, label_end={label_end}"
        )

    x_samples = []
    y_samples = []
    sample_start_dates = []
    sample_end_dates = []
    label_dates = []
    label_row_indices = []

    for label_row_idx in range(label_start, label_end + 1):
        start_idx = label_row_idx - window_size + 1
        end_idx = label_row_idx

        window_features = scaled_df.iloc[start_idx : end_idx + 1][feature_columns].to_numpy(dtype="float32")
        target_value = float(clean_df.iloc[label_row_idx][TARGET_COLUMN])

        x_samples.append(window_features)
        y_samples.append(target_value)
        sample_start_dates.append(clean_df.iloc[start_idx]["Date"].strftime("%Y-%m-%d"))
        sample_end_dates.append(clean_df.iloc[end_idx]["Date"].strftime("%Y-%m-%d"))
        if label_row_idx + 1 < len(clean_df):
            label_dates.append(clean_df.iloc[label_row_idx + 1]["Date"].strftime("%Y-%m-%d"))
        else:
            label_dates.append(raw_last_date)
        label_row_indices.append(label_row_idx)

    x_array = np.asarray(x_samples, dtype="float32")
    x_tensor = torch.from_numpy(x_array)
    y_tensor = torch.tensor(y_samples, dtype=torch.float32)

    return {
        "X": x_tensor,
        "y": y_tensor,
        "sample_start_dates": sample_start_dates,
        "sample_end_dates": sample_end_dates,
        "label_dates": label_dates,
        "label_row_indices": label_row_indices,
        "feature_names": feature_columns,
        "window_size": window_size,
        "split": split_name,
        "input_shape": list(x_tensor.shape),
        "target_shape": list(y_tensor.shape),
    }


def verify_alignment(dataset: dict, clean_df: pd.DataFrame) -> None:
    for idx, label_row_idx in enumerate(dataset["label_row_indices"][:3]):
        expected_target = float(clean_df.iloc[label_row_idx][TARGET_COLUMN])
        actual_target = float(dataset["y"][idx].item())
        if abs(expected_target - actual_target) > 1e-6:
            raise ValueError(
                f"Target misalignment detected at sample {idx}: "
                f"expected {expected_target}, got {actual_target}"
            )

    if dataset["X"].shape[1] != dataset["window_size"]:
        raise ValueError("Window tensor time dimension does not match configured window_size.")


def save_dataset_bundle(
    output_root: Path,
    ticker: str,
    feature_set_name: str,
    split_name: str,
    dataset: dict,
    project_root: Path,
) -> str:
    save_dir = output_root / ticker / feature_set_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{split_name}.pt"
    torch.save(dataset, save_path)
    return save_path.relative_to(project_root).as_posix()


def build_window_tensors(project_root: Path, window_size: int) -> list[dict]:
    output_root = project_root / "data" / "tensors" / f"window_{window_size}"
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = []

    for ticker in TICKERS:
        clean_df, scaled_df, raw_last_date = load_scaled_and_clean_data(project_root, ticker)
        train_boundary = int(len(clean_df) * TRAIN_RATIO)

        for feature_set_name, feature_columns in [
            ("baseline", BASELINE_FEATURES),
            ("proposed", PROPOSED_FEATURES),
        ]:
            for split_name in ["train", "test"]:
                dataset = build_split_samples(
                    clean_df=clean_df,
                    scaled_df=scaled_df,
                    feature_columns=feature_columns,
                    split_name=split_name,
                    train_boundary=train_boundary,
                    window_size=window_size,
                    raw_last_date=raw_last_date,
                )
                verify_alignment(dataset, clean_df)
                saved_path = save_dataset_bundle(
                    output_root=output_root,
                    ticker=ticker,
                    feature_set_name=feature_set_name,
                    split_name=split_name,
                    dataset=dataset,
                    project_root=project_root,
                )
                manifest.append(
                    {
                        "ticker": ticker,
                        "feature_set": feature_set_name,
                        "split": split_name,
                        "window_size": window_size,
                        "x_shape": list(dataset["X"].shape),
                        "y_shape": list(dataset["y"].shape),
                        "saved_path": saved_path,
                    }
                )

    manifest_path = output_root / "tensor_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PyTorch window tensors for one or more look-back sizes.")
    parser.add_argument("--window-sizes", type=int, nargs="+", default=DEFAULT_WINDOW_SIZES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = resolve_project_root()

    all_rows = []
    for window_size in args.window_sizes:
        manifest = build_window_tensors(project_root=project_root, window_size=window_size)
        all_rows.extend(manifest)

    summary_df = pd.DataFrame(all_rows)[["window_size", "ticker", "feature_set", "split", "x_shape", "y_shape"]]
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
