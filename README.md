# Stock Prediction Paper

[中文说明](README.zh-CN.md)

This repository contains the **code-side public workspace** of a student research project on stock price forecasting with `LSTM` and `GRU`, with a particular focus on technical-indicator fusion, lag mitigation under extreme market conditions, and leakage-free experimental design.

## Public Scope of This Repository

This GitHub repository is intentionally limited to:

- experiment scripts
- data-processing pipelines
- figure-generation code
- result summary tables
- lightweight project documentation

The **full manuscript drafts** (`.md` / `.tex`) are intentionally kept out of the public repository before submission in order to reduce pre-submission disclosure risk.

## Repository Structure

- `configs/`: project-level configuration notes
- `data/`: raw and processed datasets
- `docs/`: lightweight active project notes for repo orientation
- `figures/`: publication figures used in the study
- `scripts/`: data processing, figure generation, and ablation scripts
- `train.py`: single-experiment training engine
- `run_grid_search.py`: grid-search entry point
- `run_ultimate_experiments.py`: window-sensitivity experiment runner
- `run_all_experiments.py`: earlier batch baseline runner

## Key Public Result Files

- `grid_search_results.csv`
- `ultimate_results.csv`
- `ablation_results.csv`

## Private Local Workspace

The following folder is intentionally excluded from GitHub:

- `_local/`

It stores:

- detailed run folders
- logs
- archived drafts
- temporary review materials
- private manuscript support files

## Environment

Typical local environment setup:

```bash
conda activate paper_env
pip install -r requirements.txt
```

Notes:

- on the server, commands are expected to run under `paper_env`
- in some `zsh` shells, `conda` may require manual shell initialization before activation
