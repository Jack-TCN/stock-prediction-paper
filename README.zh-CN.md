# Stock Prediction Paper

[English](README.md)

这个仓库目前是这篇论文项目的**公开代码版工作区**，核心方向是：

- `LSTM / GRU` 股票价格预测
- 技术指标融合
- 极端行情下的滞后效应缓解
- 严格防止数据泄露的实验设计

## 这个公开仓库包含什么

为了避免论文在正式投稿前过早公开，当前 GitHub 仓库**只保留代码与结果侧内容**，主要包括：

- 实验脚本
- 数据处理流程
- 图表生成脚本
- 结果总表
- 轻量级项目说明文档

## 这个公开仓库不包含什么

以下内容目前**不会上传到公开 GitHub**：

- 完整论文总稿（`.md` / `.tex`）
- 分章节论文草稿
- 本地私有日志与详细 run 目录
- 其他投稿准备材料

这些内容会继续保留在本地私有工作区中，等正式投稿后再决定是否公开。

## 仓库结构

- `configs/`：项目配置说明
- `data/`：原始数据与处理后数据
- `docs/`：公开仓库保留的核心说明文档
- `figures/`：论文中使用的正式图表
- `scripts/`：数据处理、画图、消融实验等脚本
- `train.py`：单组实验训练引擎
- `run_grid_search.py`：网格搜索入口
- `run_ultimate_experiments.py`：窗口敏感性实验入口
- `run_all_experiments.py`：早期基础批量实验入口

## 当前公开结果文件

- `grid_search_results.csv`
- `ultimate_results.csv`
- `ablation_results.csv`

## 本地私有工作区

以下目录已经被 `.gitignore` 排除，不会上传 GitHub：

- `_local/`

里面主要放：

- 详细训练结果目录
- 运行日志
- 归档草稿
- 本地检查 PDF 的临时文件
- 私有论文主稿及其配套材料

## 环境

常用环境启动方式：

```bash
conda activate paper_env
pip install -r requirements.txt
```

补充说明：

- 服务器默认在 `paper_env` 中运行
- 某些 `zsh` 环境里，`conda` 可能需要先做 shell 初始化
