from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MIN_DELTA,
    DEFAULT_SEED,
    DEFAULT_VAL_RATIO,
    TrainConfig,
    resolve_device,
    train_one_experiment,
)
from scripts.day4_build_windows import build_split_samples, load_scaled_and_clean_data, verify_alignment


GRID_RESULTS_PATH = PROJECT_ROOT / "grid_search_results.csv"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "ablation_results.csv"
DEFAULT_OUTPUT_ROOT_DIR = "_local/runs/ablation_runs"
DEFAULT_MAX_EPOCHS = 500
DEFAULT_PATIENCE = 20
TRAIN_RATIO = 0.8

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

# The ablation is defined at the indicator-family level rather than on a single
# scalar feature. This is important because MACD and Bollinger Bands are not
# individual variables in practice; they are structured indicator groups.
FEATURE_GROUPS: dict[str, list[str]] = {
    "proposed_all": PROPOSED_FEATURES,
    "no_rsi": [feature for feature in PROPOSED_FEATURES if feature != "RSI_14"],
    "no_macd": [
        feature
        for feature in PROPOSED_FEATURES
        if feature not in {"MACD", "MACD_SIGNAL", "MACD_DIFF"}
    ],
    "no_bollinger": [
        feature
        for feature in PROPOSED_FEATURES
        if feature not in {"BB_MAVG", "BB_HIGH", "BB_LOW"}
    ],
}


def select_validation_best_stock_configs(grid_results_path: Path) -> pd.DataFrame:
    df = pd.read_csv(grid_results_path)
    best = (
        df.sort_values(["ticker", "best_val_loss"])
        .groupby("ticker", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return best[
        [
            "ticker",
            "feature_set",
            "model_type",
            "window_size",
            "hidden_size",
            "learning_rate",
            "dropout",
            "best_val_loss",
            "test_RMSE",
        ]
    ]


def save_ablation_tensor_bundle(
    ticker: str,
    feature_group_name: str,
    split_name: str,
    dataset: dict,
    window_size: int,
) -> str:
    save_dir = PROJECT_ROOT / "data" / "tensors" / f"window_{window_size}" / ticker / feature_group_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{split_name}.pt"
    torch.save(dataset, save_path)
    return save_path.relative_to(PROJECT_ROOT).as_posix()


def build_ablation_tensors(ticker: str, window_size: int, feature_group_name: str) -> dict[str, str]:
    clean_df, scaled_df, raw_last_date = load_scaled_and_clean_data(PROJECT_ROOT, ticker)
    train_boundary = int(len(clean_df) * TRAIN_RATIO)
    feature_columns = FEATURE_GROUPS[feature_group_name]

    saved_paths: dict[str, str] = {}
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
        saved_paths[split_name] = save_ablation_tensor_bundle(
            ticker=ticker,
            feature_group_name=feature_group_name,
            split_name=split_name,
            dataset=dataset,
            window_size=window_size,
        )

    return saved_paths


def load_existing_results(results_path: Path) -> pd.DataFrame:
    if results_path.exists():
        return pd.read_csv(results_path)
    return pd.DataFrame()


def append_result(results_path: Path, result_row: dict) -> None:
    new_df = pd.DataFrame([result_row])
    if results_path.exists():
        existing_df = pd.read_csv(results_path)
        merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        merged_df = new_df
    merged_df.to_csv(results_path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a full three-stock ablation study by removing one technical-indicator family at a time."
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--results-path", type=str, default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--output-root-dir", type=str, default=DEFAULT_OUTPUT_ROOT_DIR)
    parser.add_argument("--force", action="store_true", help="Rerun combinations that already exist in ablation_results.csv.")
    parser.add_argument("--dry-run", action="store_true", help="Print the experiment plan without launching training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    results_path = Path(args.results_path)
    if not results_path.is_absolute():
        results_path = PROJECT_ROOT / results_path

    best_configs = select_validation_best_stock_configs(GRID_RESULTS_PATH)
    existing_results = load_existing_results(results_path)
    completed_keys: set[tuple[str, str]] = set()
    if not existing_results.empty:
        completed_keys = {
            (str(row["ticker"]), str(row["feature_group"]))
            for _, row in existing_results.iterrows()
        }

    print("Validation-best stock-level configurations fixed for the ablation study:")
    print(best_configs.to_string(index=False))
    print()

    planned_runs: list[tuple[pd.Series, str]] = []
    for _, config_row in best_configs.iterrows():
        for feature_group_name in FEATURE_GROUPS:
            run_key = (str(config_row["ticker"]), feature_group_name)
            if not args.force and run_key in completed_keys:
                continue
            planned_runs.append((config_row, feature_group_name))

    print(f"Planned runs: {len(planned_runs)}")
    for config_row, feature_group_name in planned_runs:
        print(
            f"- {config_row['ticker']} | {feature_group_name} | "
            f"{config_row['model_type']} | w={config_row['window_size']} | "
            f"h={config_row['hidden_size']} | lr={config_row['learning_rate']} | d={config_row['dropout']}"
        )
    print()

    if args.dry_run:
        return

    removed_component_map = {
        "proposed_all": "none",
        "no_rsi": "RSI",
        "no_macd": "MACD family",
        "no_bollinger": "Bollinger family",
    }

    for config_row, feature_group_name in planned_runs:
        ticker = str(config_row["ticker"])
        window_size = int(config_row["window_size"])

        # Rebuild tensors from the already processed/scaled data so that the
        # only changing factor is the active feature subset.
        saved_paths = build_ablation_tensors(
            ticker=ticker,
            window_size=window_size,
            feature_group_name=feature_group_name,
        )
        print(
            f"Prepared tensors for {ticker} / {feature_group_name}: "
            f"{saved_paths['train']} | {saved_paths['test']}"
        )

        # We intentionally freeze each stock's validation-best architecture and
        # hyperparameters from the prior grid search. This keeps the ablation
        # focused on feature contribution rather than re-optimizing the model.
        config = TrainConfig(
            ticker=ticker,
            feature_set=feature_group_name,
            model_type=str(config_row["model_type"]),
            window_size=window_size,
            hidden_size=int(config_row["hidden_size"]),
            dropout=float(config_row["dropout"]),
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=float(config_row["learning_rate"]),
            seed=args.seed,
            patience=args.patience,
            val_ratio=args.val_ratio,
            min_delta=args.min_delta,
            device=str(device),
            output_root_dir=args.output_root_dir,
        )
        summary = train_one_experiment(config=config, project_root=PROJECT_ROOT)

        result_row = {
            "ticker": ticker,
            "feature_group": feature_group_name,
            "removed_component": removed_component_map[feature_group_name],
            "feature_names": json.dumps(FEATURE_GROUPS[feature_group_name], ensure_ascii=False),
            "base_feature_set_from_grid_search": str(config_row["feature_set"]),
            "model_type": str(config_row["model_type"]),
            "window_size": window_size,
            "hidden_size": int(config_row["hidden_size"]),
            "learning_rate": float(config_row["learning_rate"]),
            "dropout": float(config_row["dropout"]),
            "grid_best_val_loss": float(config_row["best_val_loss"]),
            "grid_best_test_RMSE": float(config_row["test_RMSE"]),
            **summary,
        }
        append_result(results_path, result_row)
        print(f"Saved ablation result to: {results_path.relative_to(PROJECT_ROOT).as_posix()}")
        print("-" * 88)


if __name__ == "__main__":
    main()
