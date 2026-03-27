from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from train import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LR,
    DEFAULT_MIN_DELTA,
    DEFAULT_OUTPUT_ROOT_DIR,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_VAL_RATIO,
    DEFAULT_WINDOW_SIZE,
    TrainConfig,
    resolve_device,
    resolve_project_root,
    train_one_experiment,
)


TICKERS = ["AAPL", "MSFT", "TSLA"]
FEATURE_SETS = ["baseline", "proposed"]
MODELS = ["lstm", "gru"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all stock prediction experiments and aggregate results.")
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
    project_root = resolve_project_root()
    device = resolve_device(args.device)

    batch_started = datetime.now().isoformat(timespec="seconds")
    final_rows = []

    total_runs = len(TICKERS) * len(FEATURE_SETS) * len(MODELS)
    run_index = 0

    for ticker in TICKERS:
        for feature_set in FEATURE_SETS:
            for model_type in MODELS:
                run_index += 1
                print(
                    f"\n=== Running experiment {run_index}/{total_runs}: "
                    f"{ticker} | {feature_set} | {model_type} ==="
                )

                config = TrainConfig(
                    ticker=ticker,
                    feature_set=feature_set,
                    model_type=model_type,
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

                summary = train_one_experiment(config=config, project_root=project_root)
                final_rows.append(summary)

    final_df = pd.DataFrame(final_rows)
    final_df = final_df[
        [
            "ticker",
            "feature_set",
            "model_type",
            "best_epoch",
            "stopped_epoch",
            "best_val_loss",
            "window_size",
            "hidden_size",
            "dropout",
            "test_loss",
            "test_RMSE",
            "test_MAE",
            "test_MAPE",
            "train_samples",
            "validation_samples",
            "test_samples",
            "input_shape",
            "device",
            "best_model_path",
            "output_dir",
        ]
    ].sort_values(by=["ticker", "feature_set", "model_type"]).reset_index(drop=True)

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    final_csv_path = results_dir / "final_results.csv"
    final_json_path = results_dir / "final_results.json"

    final_df.to_csv(final_csv_path, index=False, encoding="utf-8-sig")

    batch_summary = {
        "batch_started": batch_started,
        "batch_finished": datetime.now().isoformat(timespec="seconds"),
        "total_runs": total_runs,
        "device": str(device),
        "results": final_rows,
    }
    final_json_path.write_text(json.dumps(batch_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nAll experiments completed. Aggregated CSV: {final_csv_path.relative_to(project_root).as_posix()}")


if __name__ == "__main__":
    main()
