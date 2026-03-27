from pathlib import Path

import pandas as pd
import yfinance as yf


TICKERS = ["AAPL", "MSFT", "TSLA"]
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"


def download_one(ticker: str, output_dir: Path, project_root: Path) -> dict:
    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"{ticker} download failed: empty dataframe")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    expected_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{ticker} missing columns: {missing_cols}")

    output_path = output_dir / f"{ticker}.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return {
        "ticker": ticker,
        "rows": len(df),
        "start": str(df["Date"].min().date()),
        "end": str(df["Date"].max().date()),
        "missing_values": int(df.isna().sum().sum()),
        "output": output_path.relative_to(project_root).as_posix(),
    }


def write_data_overview(summary_df: pd.DataFrame, docs_dir: Path) -> None:
    lines = [
        "# Data Overview",
        "",
        "数据来源：Yahoo Finance",
        "",
        f"时间范围：{START_DATE} 到 {END_DATE}",
        "",
        "下载股票：`AAPL`、`MSFT`、`TSLA`",
        "",
        "## 数据摘要",
        "",
        "| Ticker | Rows | Start | End | Missing Values | File |",
        "| --- | ---: | --- | --- | ---: | --- |",
    ]

    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['ticker']} | {row['rows']} | {row['start']} | {row['end']} | "
            f"{row['missing_values']} | {row['output']} |"
        )

    overview_path = docs_dir / "data_overview.md"
    overview_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cwd = Path.cwd()
    script_root = Path(__file__).resolve().parents[1]
    project_root = cwd if (cwd / "scripts").exists() and (cwd / "docs").exists() else script_root
    raw_dir = project_root / "data" / "raw"
    docs_dir = project_root / "docs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for ticker in TICKERS:
        summaries.append(download_one(ticker, raw_dir, project_root))

    summary_df = pd.DataFrame(summaries)
    summary_path = project_root / "data" / "raw" / "download_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_data_overview(summary_df, docs_dir)

    print("Download completed.")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to: {summary_path.relative_to(project_root).as_posix()}")


if __name__ == "__main__":
    main()
