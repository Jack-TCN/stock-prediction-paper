from __future__ import annotations

import argparse
import csv
import itertools
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from scripts.day4_build_windows import build_window_tensors
from train import (
    DEFAULT_MIN_DELTA,
    DEFAULT_SEED,
    DEFAULT_VAL_RATIO,
    TrainConfig,
    resolve_device,
    resolve_project_root,
    train_one_experiment,
)


DEFAULT_TICKERS = ["AAPL", "MSFT", "TSLA"]
DEFAULT_FEATURE_SETS = ["baseline", "proposed"]
DEFAULT_MODELS = ["lstm", "gru"]
DEFAULT_WINDOW_SIZES = [20, 60]
DEFAULT_HIDDEN_SIZES = [32, 64, 128]
DEFAULT_LEARNING_RATES = [0.001, 0.0005]
DEFAULT_DROPOUTS = [0.0, 0.2, 0.4]

DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 500
DEFAULT_PATIENCE = 20
DEFAULT_OUTPUT_ROOT_DIR = "_local/runs/grid_search_runs"
DEFAULT_RESULTS_FILE = "grid_search_results.csv"

UNIQUE_KEY_FIELDS = [
    "ticker",
    "feature_set",
    "model_type",
    "window_size",
    "hidden_size",
    "learning_rate",
    "dropout",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a detailed grid search for stock price prediction experiments.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--feature-sets", nargs="+", default=DEFAULT_FEATURE_SETS, choices=DEFAULT_FEATURE_SETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    parser.add_argument("--window-sizes", nargs="+", type=int, default=DEFAULT_WINDOW_SIZES)
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=DEFAULT_HIDDEN_SIZES)
    parser.add_argument("--learning-rates", nargs="+", type=float, default=DEFAULT_LEARNING_RATES)
    parser.add_argument("--dropouts", nargs="+", type=float, default=DEFAULT_DROPOUTS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-root-dir", type=str, default=DEFAULT_OUTPUT_ROOT_DIR)
    parser.add_argument("--results-file", type=str, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--skip-build", action="store_true", help="Skip tensor generation and assume needed window tensors already exist.")
    return parser.parse_args()


def ensure_window_tensors(project_root: Path, window_sizes: list[int]) -> None:
    for window_size in window_sizes:
        tensor_dir = project_root / "data" / "tensors" / f"window_{window_size}"
        manifest_path = tensor_dir / "tensor_manifest.json"
        if manifest_path.exists():
            print(f"Window tensors already exist for window_size={window_size}: {manifest_path.relative_to(project_root).as_posix()}")
            continue
        print(f"Building tensors for window_size={window_size} ...")
        build_window_tensors(project_root=project_root, window_size=window_size)


def normalize_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def combo_key(combo: dict) -> tuple[str, ...]:
    return tuple(normalize_value(combo[field]) for field in UNIQUE_KEY_FIELDS)


def load_completed_keys(results_path: Path) -> set[tuple[str, ...]]:
    if not results_path.exists() or results_path.stat().st_size == 0:
        return set()

    df = pd.read_csv(results_path)
    if df.empty:
        return set()

    missing_fields = [field for field in UNIQUE_KEY_FIELDS if field not in df.columns]
    if missing_fields:
        raise ValueError(f"Existing results file is missing required fields: {missing_fields}")

    completed = set()
    for _, row in df.iterrows():
        completed.add(combo_key(row.to_dict()))
    return completed


def append_result(results_path: Path, row: dict) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_path.exists() or results_path.stat().st_size == 0

    with results_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_combinations(args: argparse.Namespace) -> list[dict]:
    combos = []
    for ticker, feature_set, model_type, window_size, hidden_size, learning_rate, dropout in itertools.product(
        args.tickers,
        args.feature_sets,
        args.models,
        args.window_sizes,
        args.hidden_sizes,
        args.learning_rates,
        args.dropouts,
    ):
        combos.append(
            {
                "ticker": ticker,
                "feature_set": feature_set,
                "model_type": model_type,
                "window_size": window_size,
                "hidden_size": hidden_size,
                "learning_rate": learning_rate,
                "dropout": dropout,
            }
        )
    return combos


def main() -> None:
    args = parse_args()
    project_root = resolve_project_root()
    device = resolve_device(args.device)
    results_path = project_root / args.results_file

    if not args.skip_build:
        ensure_window_tensors(project_root=project_root, window_sizes=args.window_sizes)

    all_combos = build_combinations(args)
    completed_keys = load_completed_keys(results_path)
    pending_combos = [combo for combo in all_combos if combo_key(combo) not in completed_keys]

    metadata = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "total_combinations": len(all_combos),
        "already_completed": len(completed_keys),
        "pending": len(pending_combos),
        "results_file": results_path.relative_to(project_root).as_posix(),
        "output_root_dir": args.output_root_dir,
        "search_space": {
            "tickers": args.tickers,
            "feature_sets": args.feature_sets,
            "models": args.models,
            "window_sizes": args.window_sizes,
            "hidden_sizes": args.hidden_sizes,
            "learning_rates": args.learning_rates,
            "dropouts": args.dropouts,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "val_ratio": args.val_ratio,
            "min_delta": args.min_delta,
            "seed": args.seed,
        },
    }
    metadata_path = project_root / "_local" / "logs" / "grid_search_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Total combinations: {len(all_combos)}")
    print(f"Already completed: {len(completed_keys)}")
    print(f"Pending combinations: {len(pending_combos)}")
    print(f"Results file: {results_path.relative_to(project_root).as_posix()}")

    progress = tqdm(pending_combos, total=len(pending_combos), dynamic_ncols=True, desc="Grid Search")

    for combo in progress:
        progress.set_postfix(
            ticker=combo["ticker"],
            feature=combo["feature_set"],
            model=combo["model_type"],
            window=combo["window_size"],
            hidden=combo["hidden_size"],
            lr=combo["learning_rate"],
            dropout=combo["dropout"],
        )

        config = TrainConfig(
            ticker=combo["ticker"],
            feature_set=combo["feature_set"],
            model_type=combo["model_type"],
            window_size=combo["window_size"],
            hidden_size=combo["hidden_size"],
            dropout=combo["dropout"],
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=combo["learning_rate"],
            seed=args.seed,
            patience=args.patience,
            val_ratio=args.val_ratio,
            min_delta=args.min_delta,
            device=str(device),
            output_root_dir=args.output_root_dir,
        )

        summary = train_one_experiment(config=config, project_root=project_root)
        row = {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "ticker": summary["ticker"],
            "feature_set": summary["feature_set"],
            "model_type": summary["model_type"],
            "window_size": summary["window_size"],
            "hidden_size": summary["hidden_size"],
            "learning_rate": summary["learning_rate"],
            "dropout": summary["dropout"],
            "max_epochs": args.epochs,
            "patience": args.patience,
            "seed": args.seed,
            "best_epoch": summary["best_epoch"],
            "stopped_epoch": summary["stopped_epoch"],
            "best_val_loss": summary["best_val_loss"],
            "best_val_RMSE": summary["best_val_RMSE"],
            "best_val_MAE": summary["best_val_MAE"],
            "best_val_MAPE": summary["best_val_MAPE"],
            "test_loss": summary["test_loss"],
            "test_RMSE": summary["test_RMSE"],
            "test_MAE": summary["test_MAE"],
            "test_MAPE": summary["test_MAPE"],
            "train_samples": summary["train_samples"],
            "validation_samples": summary["validation_samples"],
            "test_samples": summary["test_samples"],
            "input_shape": json.dumps(summary["input_shape"]),
            "device": summary["device"],
            "best_model_path": summary["best_model_path"],
            "output_dir": summary["output_dir"],
        }
        append_result(results_path, row)

    final_df = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    if not final_df.empty:
        final_df = final_df.sort_values(
            by=["ticker", "feature_set", "model_type", "window_size", "hidden_size", "learning_rate", "dropout"]
        ).reset_index(drop=True)
        final_df.to_csv(results_path, index=False, encoding="utf-8-sig")

    final_metadata = {
        **metadata,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "final_row_count": int(len(final_df)),
    }
    metadata_path.write_text(json.dumps(final_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGrid search completed. Detailed results saved to: {results_path.relative_to(project_root).as_posix()}")


if __name__ == "__main__":
    main()
