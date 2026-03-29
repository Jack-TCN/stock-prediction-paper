# Stock Prediction Paper

This repository contains a student research project on stock price forecasting with `LSTM` and `GRU`, with a particular focus on technical-indicator fusion, lag mitigation under extreme market conditions, and leakage-free experimental design.

## Main Manuscripts

The project root keeps the three current master manuscripts:

- [Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators_zh.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators_zh.md)
- [Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators.md)
- [Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators.tex](C:/Users/27476/Desktop/论文/stock_prediction_paper/Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators.tex)

These are the primary files for final review, PDF generation, and submission preparation.

## New Chat Handoff

If a new chat needs to take over the project, read these three files first:

- [context_anchor.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/context_anchor.md)
- [paper_memory.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/paper_memory.md)
- [path_map.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/path_map.md)

## Repository Structure

- [configs](C:/Users/27476/Desktop/论文/stock_prediction_paper/configs): project-level configuration notes
- [data](C:/Users/27476/Desktop/论文/stock_prediction_paper/data): raw and processed datasets
- [docs](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs): active chapter drafts and core project memory
- [figures](C:/Users/27476/Desktop/论文/stock_prediction_paper/figures): publication figures used in the paper
- [scripts](C:/Users/27476/Desktop/论文/stock_prediction_paper/scripts): data processing, figure generation, and ablation scripts
- [train.py](C:/Users/27476/Desktop/论文/stock_prediction_paper/train.py): single-experiment training engine
- [run_grid_search.py](C:/Users/27476/Desktop/论文/stock_prediction_paper/run_grid_search.py): grid-search entry point
- [run_ultimate_experiments.py](C:/Users/27476/Desktop/论文/stock_prediction_paper/run_ultimate_experiments.py): window-sensitivity experiment runner
- [run_all_experiments.py](C:/Users/27476/Desktop/论文/stock_prediction_paper/run_all_experiments.py): earlier batch baseline runner

## Key Result Files

- [grid_search_results.csv](C:/Users/27476/Desktop/论文/stock_prediction_paper/grid_search_results.csv)
- [ultimate_results.csv](C:/Users/27476/Desktop/论文/stock_prediction_paper/ultimate_results.csv)
- [ablation_results.csv](C:/Users/27476/Desktop/论文/stock_prediction_paper/ablation_results.csv)

## Local-Only Workspace

The folder below is intentionally excluded from GitHub:

- [_local](C:/Users/27476/Desktop/论文/stock_prediction_paper/_local)

It stores:

- detailed run folders
- logs
- archived drafts
- temporary review materials

## Environment

Typical local environment setup:

```bash
conda activate paper_env
pip install -r requirements.txt
```

Note:

- On the server, commands are expected to run under `paper_env`.
- In some `zsh` shells, `conda` may require manual shell initialization before activation.
