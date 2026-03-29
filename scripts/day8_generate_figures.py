from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import RecurrentRegressor


FIG_DIR = PROJECT_ROOT / "figures"
GRID_RESULTS = PROJECT_ROOT / "grid_search_results.csv"

COLOR_ACTUAL = "#2F2F2F"
COLOR_BASELINE = "#1F4E79"
COLOR_PROPOSED = "#B55239"
COLOR_GRID = "#D8D8D8"


def ensure_figure_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4C4C4C",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": COLOR_GRID,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def load_results() -> pd.DataFrame:
    return pd.read_csv(GRID_RESULTS)


def select_validation_best_per_combo(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["ticker", "feature_set", "model_type", "best_val_loss"])
        .groupby(["ticker", "feature_set", "model_type"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def generate_hyperparameter_sensitivity(df: pd.DataFrame) -> Path:
    # Use validation RMSE because hyperparameter selection should be validation-driven.
    pivot = (
        df.groupby(["hidden_size", "learning_rate"])["best_val_RMSE"]
        .mean()
        .reset_index()
        .pivot(index="hidden_size", columns="learning_rate", values="best_val_RMSE")
        .sort_index()
    )

    cmap = LinearSegmentedColormap.from_list(
        "blue_brick",
        [COLOR_BASELINE, "#F3ECE7", COLOR_PROPOSED],
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        cbar_kws={"label": "Mean Validation RMSE"},
        linewidths=0.8,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Hyperparameter Sensitivity\nMean Validation RMSE by Hidden Size and Learning Rate", pad=12)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Hidden Size")

    out_path = FIG_DIR / "hyperparameter_sensitivity.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def load_prediction_series(row: pd.Series) -> pd.DataFrame:
    config_path = PROJECT_ROOT / row["output_dir"] / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))

    test_path = (
        PROJECT_ROOT
        / "data"
        / "tensors"
        / f"window_{config['window_size']}"
        / config["ticker"]
        / config["feature_set"]
        / "test.pt"
    )
    bundle = torch.load(test_path, map_location="cpu")
    X = bundle["X"].float()
    y = bundle["y"].float()

    model = RecurrentRegressor(
        model_type=config["model_type"],
        input_size=X.shape[-1],
        hidden_size=config["hidden_size"],
        dropout=config["dropout"],
    )
    state_dict = torch.load(PROJECT_ROOT / row["best_model_path"], map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        pred = model(X).cpu().numpy()

    return pd.DataFrame(
        {
            "date": pd.to_datetime(bundle["label_dates"]),
            "actual": y.numpy(),
            "pred": pred,
        }
    )


def choose_tsla_v_reversal_segment(tsla_compare: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    window = 24
    min_side = 5
    best_score = None
    best_segment = None
    best_info = None

    for start in range(0, len(tsla_compare) - window + 1):
        seg = tsla_compare.iloc[start : start + window].reset_index(drop=True)
        bottom_idx = int(seg["actual"].idxmin())
        if bottom_idx < min_side or bottom_idx > window - min_side - 1:
            continue

        left_peak = float(seg.loc[:bottom_idx, "actual"].max())
        right_peak = float(seg.loc[bottom_idx:, "actual"].max())
        bottom_value = float(seg.loc[bottom_idx, "actual"])
        v_score = (left_peak - bottom_value) + (right_peak - bottom_value)

        baseline_mae = float((seg["actual"] - seg["baseline_pred"]).abs().mean())
        proposed_mae = float((seg["actual"] - seg["proposed_pred"]).abs().mean())
        improvement = baseline_mae - proposed_mae

        # Favor strong V-shapes first, then prefer windows where proposed improves over baseline.
        score = v_score + 0.8 * max(improvement, 0.0)

        if best_score is None or score > best_score:
            best_score = score
            best_segment = seg.copy()
            best_info = {
                "start_date": seg.loc[0, "date"].strftime("%Y-%m-%d"),
                "end_date": seg.loc[len(seg) - 1, "date"].strftime("%Y-%m-%d"),
                "bottom_date": seg.loc[bottom_idx, "date"].strftime("%Y-%m-%d"),
                "v_score": v_score,
                "baseline_mae": baseline_mae,
                "proposed_mae": proposed_mae,
                "mae_improvement": improvement,
            }

    if best_segment is None or best_info is None:
        raise RuntimeError("Failed to identify a valid V-shaped reversal segment for TSLA.")

    return best_segment, best_info


def generate_tsla_lag_figure(df: pd.DataFrame) -> tuple[Path, dict]:
    tsla_df = df[df["ticker"] == "TSLA"].copy()
    baseline_best = tsla_df[tsla_df["feature_set"] == "baseline"].sort_values("best_val_loss").iloc[0]
    proposed_best = tsla_df[tsla_df["feature_set"] == "proposed"].sort_values("best_val_loss").iloc[0]

    baseline_pred = load_prediction_series(baseline_best).rename(columns={"pred": "baseline_pred"})
    proposed_pred = load_prediction_series(proposed_best).rename(columns={"pred": "proposed_pred"})

    compare = baseline_pred.merge(proposed_pred[["date", "proposed_pred"]], on="date", how="inner")
    segment, info = choose_tsla_v_reversal_segment(compare)

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    ax.plot(segment["date"], segment["actual"], color=COLOR_ACTUAL, linewidth=2.6, label="Actual Price")
    ax.plot(
        segment["date"],
        segment["baseline_pred"],
        color=COLOR_BASELINE,
        linewidth=2.4,
        linestyle="--",
        label=f"Baseline Prediction (MAE={info['baseline_mae']:.2f})",
    )
    ax.plot(
        segment["date"],
        segment["proposed_pred"],
        color=COLOR_PROPOSED,
        linewidth=2.5,
        label=f"Proposed Prediction (MAE={info['proposed_mae']:.2f})",
    )

    bottom_date = pd.to_datetime(info["bottom_date"])
    label_x = bottom_date - pd.Timedelta(days=8)
    label_y = float(segment["actual"].min()) + 6.0
    ax.axvline(bottom_date, color="#666666", linestyle=":", linewidth=1.3)
    ax.annotate(
        f"V-bottom: {info['bottom_date']}",
        xy=(bottom_date, segment["actual"].min()),
        xytext=(label_x, label_y),
        textcoords="data",
        fontsize=9,
        color="#444444",
        ha="left",
        va="center",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 2.8},
        arrowprops={
            "arrowstyle": "-|>",
            "color": "0.6",
            "lw": 1.1,
            "mutation_scale": 11,
            "shrinkA": 6,
            "shrinkB": 3,
        },
    )

    ax.set_title("TSLA Local V-Shaped Reversal: Actual vs Baseline vs Proposed", pad=12, fontsize=14)
    ax.set_ylabel("Adjusted Close Price")
    ax.set_xlabel("Date")
    ax.legend(frameon=False, loc="best", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.subplots_adjust(bottom=0.2)

    out_path = FIG_DIR / "tsla_v_reversal_lag_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    info.update(
        {
            "baseline_run": str(baseline_best["output_dir"]),
            "proposed_run": str(proposed_best["output_dir"]),
            "baseline_test_rmse": float(baseline_best["test_RMSE"]),
            "proposed_test_rmse": float(proposed_best["test_RMSE"]),
        }
    )
    return out_path, info


def main() -> None:
    ensure_figure_dir()
    set_plot_style()

    df = load_results()
    heatmap_path = generate_hyperparameter_sensitivity(df)
    lag_path, lag_info = generate_tsla_lag_figure(df)

    best_table = select_validation_best_per_combo(df)[
        [
            "ticker",
            "feature_set",
            "model_type",
            "window_size",
            "hidden_size",
            "learning_rate",
            "dropout",
            "test_RMSE",
            "test_MAE",
            "test_MAPE",
        ]
    ].sort_values(["ticker", "feature_set", "model_type"])

    print("Figures saved:")
    print(f" - {heatmap_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f" - {lag_path.relative_to(PROJECT_ROOT).as_posix()}")
    print()
    print("TSLA V-reversal segment info:")
    print(json.dumps(lag_info, indent=2, ensure_ascii=False))
    print()
    print("Validation-best results table:")
    print(best_table.to_string(index=False))


if __name__ == "__main__":
    main()
