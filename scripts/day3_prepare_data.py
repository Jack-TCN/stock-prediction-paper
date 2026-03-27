from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


TICKERS = ["AAPL", "MSFT", "TSLA"]
TRAIN_RATIO = 0.8

RAW_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
INDICATOR_COLUMNS = [
    "RSI_14",
    "MACD",
    "MACD_SIGNAL",
    "MACD_DIFF",
    "BB_MAVG",
    "BB_HIGH",
    "BB_LOW",
]
FEATURE_COLUMNS = RAW_PRICE_COLUMNS + INDICATOR_COLUMNS
TARGET_COLUMN = "Target_Close_Next"


def resolve_project_root() -> Path:
    cwd = Path.cwd()
    script_root = Path(__file__).resolve().parents[1]
    if (cwd / "scripts").exists() and (cwd / "docs").exists():
        return cwd
    return script_root


def load_raw_data(raw_dir: Path, ticker: str) -> pd.DataFrame:
    csv_path = raw_dir / f"{ticker}.csv"
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()

    rsi_indicator = RSIIndicator(close=featured["Close"], window=14)
    macd_indicator = MACD(close=featured["Close"], window_slow=26, window_fast=12, window_sign=9)
    bb_indicator = BollingerBands(close=featured["Close"], window=20, window_dev=2)

    featured["RSI_14"] = rsi_indicator.rsi()
    featured["MACD"] = macd_indicator.macd()
    featured["MACD_SIGNAL"] = macd_indicator.macd_signal()
    featured["MACD_DIFF"] = macd_indicator.macd_diff()
    featured["BB_MAVG"] = bb_indicator.bollinger_mavg()
    featured["BB_HIGH"] = bb_indicator.bollinger_hband()
    featured["BB_LOW"] = bb_indicator.bollinger_lband()

    # The target is the next trading day's close price.
    featured[TARGET_COLUMN] = featured["Close"].shift(-1)

    return featured


def clean_featured_data(featured_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    nan_by_column = featured_df.isna().sum()
    nan_by_column = nan_by_column[nan_by_column > 0].sort_values(ascending=False)

    # Academic red line: drop rows with NaN directly. Do not fill financial time series.
    cleaned_df = featured_df.dropna().reset_index(drop=True)
    return cleaned_df, nan_by_column


def split_by_time(cleaned_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(cleaned_df) * TRAIN_RATIO)
    if split_index <= 0 or split_index >= len(cleaned_df):
        raise ValueError("Invalid split index. Check dataset length and TRAIN_RATIO.")

    train_df = cleaned_df.iloc[:split_index].copy()
    test_df = cleaned_df.iloc[split_index:].copy()
    return train_df, test_df


def scale_features_without_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    scaler = MinMaxScaler()

    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    # Academic red line: fit only on the training set, then transform the test set.
    train_scaled[FEATURE_COLUMNS] = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    test_scaled[FEATURE_COLUMNS] = scaler.transform(test_df[FEATURE_COLUMNS])

    return train_scaled, test_scaled, scaler


def save_outputs(
    project_root: Path,
    ticker: str,
    featured_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_scaled: pd.DataFrame,
    test_scaled: pd.DataFrame,
    scaler: MinMaxScaler,
    nan_by_column: pd.Series,
) -> dict:
    processed_dir = project_root / "data" / "processed"
    featured_dir = processed_dir / "featured"
    split_dir = processed_dir / "splits"
    scaled_dir = processed_dir / "scaled"
    scaler_dir = processed_dir / "scalers"
    metadata_dir = processed_dir / "metadata"

    for path in [featured_dir, split_dir, scaled_dir, scaler_dir, metadata_dir]:
        path.mkdir(parents=True, exist_ok=True)

    featured_with_nan_path = featured_dir / f"{ticker}_featured_with_nan.csv"
    cleaned_path = featured_dir / f"{ticker}_featured_clean.csv"
    train_path = split_dir / f"{ticker}_train_unscaled.csv"
    test_path = split_dir / f"{ticker}_test_unscaled.csv"
    train_scaled_path = scaled_dir / f"{ticker}_train_scaled.csv"
    test_scaled_path = scaled_dir / f"{ticker}_test_scaled.csv"
    scaler_path = scaler_dir / f"{ticker}_feature_scaler.joblib"
    metadata_path = metadata_dir / f"{ticker}_day3_metadata.json"

    featured_df.to_csv(featured_with_nan_path, index=False, encoding="utf-8-sig")
    cleaned_df.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")
    train_scaled.to_csv(train_scaled_path, index=False, encoding="utf-8-sig")
    test_scaled.to_csv(test_scaled_path, index=False, encoding="utf-8-sig")
    joblib.dump(scaler, scaler_path)

    metadata = {
        "ticker": ticker,
        "raw_rows": int(len(featured_df)),
        "clean_rows": int(len(cleaned_df)),
        "dropped_rows": int(len(featured_df) - len(cleaned_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "split_ratio": TRAIN_RATIO,
        "split_date": str(test_df["Date"].iloc[0].date()),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "nan_counts_before_dropna": {key: int(value) for key, value in nan_by_column.items()},
        "academic_red_lines": [
            "Use dropna() after technical indicator calculation. No zero fill, ffill, or bfill.",
            "Split by time order before scaling.",
            "Fit MinMaxScaler on the training set only, then transform the test set.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return metadata


def plot_price_and_volume(ticker: str, cleaned_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(cleaned_df["Date"], cleaned_df["Close"], color="#124E78", linewidth=1.3)
    axes[0].set_title(f"{ticker} Close Price")
    axes[0].set_ylabel("Close")
    axes[0].grid(alpha=0.25)

    axes[1].plot(cleaned_df["Date"], cleaned_df["Volume"], color="#F18F01", linewidth=1.0)
    axes[1].set_title(f"{ticker} Volume")
    axes[1].set_ylabel("Volume")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / f"{ticker}_01_price_volume.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_indicators(ticker: str, cleaned_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(cleaned_df["Date"], cleaned_df["Close"], label="Close", color="#1D3557", linewidth=1.2)
    axes[0].plot(cleaned_df["Date"], cleaned_df["BB_MAVG"], label="BB_MAVG", color="#2A9D8F", linewidth=1.0)
    axes[0].plot(cleaned_df["Date"], cleaned_df["BB_HIGH"], label="BB_HIGH", color="#E76F51", linewidth=0.9)
    axes[0].plot(cleaned_df["Date"], cleaned_df["BB_LOW"], label="BB_LOW", color="#E9C46A", linewidth=0.9)
    axes[0].set_title(f"{ticker} Close Price with Bollinger Bands")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)

    axes[1].plot(cleaned_df["Date"], cleaned_df["RSI_14"], color="#8E6C88", linewidth=1.1)
    axes[1].axhline(70, color="#C8553D", linestyle="--", linewidth=0.9)
    axes[1].axhline(30, color="#3C6E71", linestyle="--", linewidth=0.9)
    axes[1].set_title(f"{ticker} RSI(14)")
    axes[1].set_ylabel("RSI")
    axes[1].grid(alpha=0.25)

    axes[2].plot(cleaned_df["Date"], cleaned_df["MACD"], label="MACD", color="#264653", linewidth=1.1)
    axes[2].plot(cleaned_df["Date"], cleaned_df["MACD_SIGNAL"], label="MACD_SIGNAL", color="#F4A261", linewidth=1.0)
    axes[2].bar(cleaned_df["Date"], cleaned_df["MACD_DIFF"], label="MACD_DIFF", color="#A8DADC", width=3)
    axes[2].set_title(f"{ticker} MACD")
    axes[2].set_xlabel("Date")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / f"{ticker}_02_indicators.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cleaning_summary(
    ticker: str,
    featured_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    nan_by_column: pd.Series,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    row_summary = pd.DataFrame(
        {
            "Stage": ["Before dropna", "After dropna"],
            "Rows": [len(featured_df), len(cleaned_df)],
        }
    )
    sns.barplot(data=row_summary, x="Stage", y="Rows", hue="Stage", dodge=False, palette=["#457B9D", "#2A9D8F"], ax=axes[0])
    if axes[0].legend_ is not None:
        axes[0].legend_.remove()
    axes[0].set_title(f"{ticker} Rows Before and After dropna()")
    axes[0].grid(axis="y", alpha=0.25)

    if nan_by_column.empty:
        axes[1].text(0.5, 0.5, "No NaN detected", ha="center", va="center", fontsize=12)
        axes[1].set_axis_off()
    else:
        nan_plot_df = nan_by_column.reset_index()
        nan_plot_df.columns = ["Column", "NaN_Count"]
        sns.barplot(
            data=nan_plot_df,
            x="NaN_Count",
            y="Column",
            hue="Column",
            dodge=False,
            palette="crest",
            ax=axes[1],
        )
        if axes[1].legend_ is not None:
            axes[1].legend_.remove()
        axes[1].set_title(f"{ticker} NaN Counts Before dropna()")
        axes[1].grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / f"{ticker}_03_cleaning_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_split_and_distribution(
    ticker: str,
    cleaned_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    split_date = test_df["Date"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    axes[0].plot(train_df["Date"], train_df["Close"], label="Train", color="#2A9D8F", linewidth=1.2)
    axes[0].plot(test_df["Date"], test_df["Close"], label="Test", color="#E76F51", linewidth=1.2)
    axes[0].axvline(split_date, color="#1D3557", linestyle="--", linewidth=1.0)
    axes[0].set_title(f"{ticker} Time-based Split (80/20)")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Close")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)

    sns.histplot(train_df["Close"], color="#2A9D8F", label="Train Close", kde=True, stat="density", ax=axes[1], alpha=0.45)
    sns.histplot(test_df["Close"], color="#E76F51", label="Test Close", kde=True, stat="density", ax=axes[1], alpha=0.45)
    axes[1].set_title(f"{ticker} Train/Test Close Distribution")
    axes[1].set_xlabel("Close")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / f"{ticker}_04_split_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_feature_distributions(ticker: str, cleaned_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_columns = [
        ("Close", "#1D3557"),
        ("Volume", "#F4A261"),
        ("RSI_14", "#6D597A"),
        ("MACD", "#2A9D8F"),
    ]

    for ax, (column, color) in zip(axes.flatten(), plot_columns):
        sns.histplot(cleaned_df[column], kde=True, color=color, ax=ax)
        ax.set_title(f"{ticker} {column} Distribution")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / f"{ticker}_05_feature_distributions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_day3_report(project_root: Path, metadata_list: list[dict]) -> None:
    docs_dir = project_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Day3 Report",
        "",
        "本日完成内容：",
        "",
        "- 生成技术指标：RSI、MACD、Bollinger Bands",
        "- 使用 `dropna()` 删除技术指标和目标列产生的 NaN",
        "- 按时间顺序进行 80/20 硬切分",
        "- 仅用训练集 `fit` MinMaxScaler，再对测试集执行 `transform`",
        "- 输出处理后数据、缩放后数据、元数据和可视化图",
        "",
        "学术红线执行说明：",
        "",
        "- 未使用 `0`、`ffill` 或 `bfill` 补全技术指标生成的缺失值",
        "- 未在全量数据上执行 `fit_transform`",
        "- 训练集和测试集切分发生在归一化之前",
        "",
        "## Per-Ticker Summary",
        "",
        "| Ticker | Raw Rows | Clean Rows | Dropped Rows | Train Rows | Test Rows | Split Date |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for item in metadata_list:
        lines.append(
            f"| {item['ticker']} | {item['raw_rows']} | {item['clean_rows']} | {item['dropped_rows']} | "
            f"{item['train_rows']} | {item['test_rows']} | {item['split_date']} |"
        )

    report_path = docs_dir / "day3_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def update_progress_log(project_root: Path, metadata_list: list[dict]) -> None:
    log_path = project_root / "docs" / "progress_log.md"
    lines = [
        "# Project Progress Log",
        "",
        "## Fixed Topic",
        "",
        "`LSTM与GRU在股票价格预测中的对比研究：技术指标融合与Dropout正则化分析`",
        "",
        "## Fixed Research Questions",
        "",
        "1. 在相同数据和实验设置下，LSTM 与 GRU 哪个预测效果更好？",
        "2. 加入 RSI、MACD 和 Bollinger Bands 后，预测性能是否提升？",
        "3. 加入 Dropout 后，测试集表现和泛化能力是否改善？",
        "4. 模型、特征和正则化三者的哪种组合最稳定？",
        "",
        "## Completed Work",
        "",
        "### Day1",
        "",
        "- 固定题目、研究问题和实验变量",
        "- 建立项目目录和基础文档",
        "",
        "### Day2",
        "",
        "- 安装 `yfinance`、`scikit-learn`、`ta`、`torch`",
        "- 下载 `AAPL`、`MSFT`、`TSLA` 原始历史数据",
        "- 完成原始数据基础检查",
        "",
        "### Day3",
        "",
        "- 计算 `RSI_14`、`MACD`、`MACD_SIGNAL`、`MACD_DIFF`、`BB_MAVG`、`BB_HIGH`、`BB_LOW`",
        "- 通过 `dropna()` 删除技术指标和下一日目标值产生的 NaN",
        "- 按时间顺序执行 80/20 硬切分",
        "- 仅在训练集上 `fit` MinMaxScaler，避免信息泄露",
        "- 保存原尺度和归一化后的 train/test 数据",
        "- 生成 Day3 可视化图和元数据",
        "",
        "## Academic Red Lines",
        "",
        "- 缺失值处理：只能 `dropna()`，不做 `0` 填充，不做前向/后向填充",
        "- 信息泄露防范：先按时间切分，再做归一化",
        "- 归一化顺序：只能在训练集上 `fit`，测试集只能 `transform`",
        "",
        "## Day3 Output Snapshot",
        "",
        "| Ticker | Raw Rows | Clean Rows | Dropped Rows | Train Rows | Test Rows | Split Date |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for item in metadata_list:
        lines.append(
            f"| {item['ticker']} | {item['raw_rows']} | {item['clean_rows']} | {item['dropped_rows']} | "
            f"{item['train_rows']} | {item['test_rows']} | {item['split_date']} |"
        )

    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "- Day4: 构建滑动窗口样本，形成 LSTM / GRU 可直接训练的输入张量",
        ]
    )

    log_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    sns.set_theme(style="whitegrid")

    project_root = resolve_project_root()
    raw_dir = project_root / "data" / "raw"
    visual_dir = project_root / "results" / "day3_visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)

    metadata_list = []

    for ticker in TICKERS:
        raw_df = load_raw_data(raw_dir, ticker)
        featured_df = add_technical_indicators(raw_df)
        cleaned_df, nan_by_column = clean_featured_data(featured_df)
        train_df, test_df = split_by_time(cleaned_df)
        train_scaled, test_scaled, scaler = scale_features_without_leakage(train_df, test_df)

        metadata = save_outputs(
            project_root=project_root,
            ticker=ticker,
            featured_df=featured_df,
            cleaned_df=cleaned_df,
            train_df=train_df,
            test_df=test_df,
            train_scaled=train_scaled,
            test_scaled=test_scaled,
            scaler=scaler,
            nan_by_column=nan_by_column,
        )
        metadata_list.append(metadata)

        ticker_visual_dir = visual_dir / ticker
        ticker_visual_dir.mkdir(parents=True, exist_ok=True)
        plot_price_and_volume(ticker, cleaned_df, ticker_visual_dir)
        plot_indicators(ticker, cleaned_df, ticker_visual_dir)
        plot_cleaning_summary(ticker, featured_df, cleaned_df, nan_by_column, ticker_visual_dir)
        plot_split_and_distribution(ticker, cleaned_df, train_df, test_df, ticker_visual_dir)
        plot_feature_distributions(ticker, cleaned_df, ticker_visual_dir)

    write_day3_report(project_root, metadata_list)
    update_progress_log(project_root, metadata_list)

    summary_df = pd.DataFrame(metadata_list)[
        ["ticker", "raw_rows", "clean_rows", "dropped_rows", "train_rows", "test_rows", "split_date"]
    ]
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
