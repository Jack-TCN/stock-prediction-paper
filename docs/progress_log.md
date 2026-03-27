# Project Progress Log

## Fixed Topic

`LSTM与GRU在股票价格预测中的对比研究：技术指标融合与Dropout正则化分析`

## Fixed Research Questions

1. 在相同数据和实验设置下，LSTM 与 GRU 哪个预测效果更好？
2. 加入 RSI、MACD 和 Bollinger Bands 后，预测性能是否提升？
3. 加入 Dropout 后，测试集表现和泛化能力是否改善？
4. 模型、特征和正则化三者的哪种组合最稳定？

## Completed Work

### Day1

- 固定题目、研究问题和实验变量
- 建立项目目录和基础文档

### Day2

- 安装 `yfinance`、`scikit-learn`、`ta`、`torch`
- 下载 `AAPL`、`MSFT`、`TSLA` 原始历史数据
- 完成原始数据基础检查

### Day3

- 计算 RSI、MACD、Bollinger Bands
- 用 `dropna()` 删除技术指标和目标列产生的 NaN
- 按时间顺序执行 80/20 硬切分
- 仅在训练集上 `fit` MinMaxScaler，避免信息泄露
- 生成可视化图、缩放器和元数据

### Day4

- 将窗口大小设为全局可配置参数，当前默认值为 `60`
- 构建标准 PyTorch 三维张量 `(samples, time_steps, features)`
- 生成 Baseline: `OHLCV`
- 生成 Proposed: `OHLCV + RSI + MACD + Bollinger Bands`
- 使用 `torch.save` 将 train/test 张量序列化为 `.pt` 文件

### Day5

- 编写统一训练脚本 `train.py`
- 实现 `Dataset` 和 `DataLoader`
- 实现单层 `LSTM` 与单层 `GRU`
- 隐藏层神经元设置为 `64`
- 将 `Dropout` 作为可控参数传入，默认值为 `0.2`
- 统一使用 `Adam` 和 `MSELoss`
- 每个 Epoch 输出 `RMSE`、`MAE` 和 `MAPE`
- 实现 `cuda/cpu` 自动切换逻辑
- 在本地 CPU 上完成 `LSTM` 和 `GRU` 的 1 epoch 冒烟测试

### Day6

- 将训练脚本升级为验证集驱动的 `Early Stopping`
- 默认最大 `100 epochs`
- 默认 `patience = 10`
- 保存 `best_model.pt` 和每次运行的 `summary.json`
- 编写批量脚本 `run_all_experiments.py`
- 自动遍历 3 只股票和 4 组核心对照实验
- 自动汇总生成 `results/final_results.csv`
- 在本地 CPU 上完成 12 组轻量批量测试，验证调度与汇总逻辑

### Day7

- 编写 `run_ultimate_experiments.py`
- 自动检查并生成 `window_size = 20, 60, 100` 的窗口张量
- 将正式实验规模扩展到 `36` 组
- 目标训练上限提升到 `500 epochs`
- `patience` 提升到 `20`
- 新增独立汇总文件 `ultimate_results.csv`
- 支持 `nohup` 或 `tmux` 场景下的通宵后台运行

### Day8

- 编写 `run_grid_search.py`
- 支持 `432` 组超参数网格搜索
- 支持 `tqdm` 进度条
- 支持断点续跑
- 支持实时追加写入 `grid_search_results.csv`
- 支持自动生成所需窗口张量
- 本地完成缩小版冒烟测试，验证实时写入和断点续跑逻辑

### Documentation Update

- 重写 `docs/novice_experiment_flow.md`，补充 `epoch`、`patience`、`window_size`、`张量`、`config` 等术语的白话解释
- 更新 `docs/project_structure_guide.md`，补充 `configs/` 的角色说明
- 重写 `docs/file_cleanup_plan.md`，明确正式结果、测试残留和归档方案
- 新增 `configs/README.md`，解释项目总配置和单次实验配置的区别
- 新增 `docs/path_map.md`，固定“以后新结果放哪里”的路径规则
- 实际创建 `archive/`，并将 smoke 测试与旧结果从项目根目录移入归档区
- 新增 `docs/context_anchor.md`，作为后续长对话的最短上下文入口
- 新增 `.gitignore`，为后续 GitHub 协作提前排除缓存、大结果目录和模型权重
- 更新根目录 `README.md`，明确新会话优先阅读的文档和依赖安装方式

## Academic Red Lines

- 缺失值处理：只能 `dropna()`，不做 `0` 填充，不做前向/后向填充
- 信息泄露防范：先按时间切分，再做归一化
- 归一化顺序：只能在训练集上 `fit`，测试集只能 `transform`
- 窗口标签对齐：`y` 必须是窗口期之后第一天的收盘价
- 早停依据：使用验证集，不使用测试集做模型选择

## Key Output Files

- `data/tensors/window_60/`
- `train.py`
- `run_all_experiments.py`
- `run_ultimate_experiments.py`
- `run_grid_search.py`
- `results/final_results.csv`
- `results/final_results.json`
- `ultimate_results.csv`
- `grid_search_results.csv`
- `docs/paper_memory.md`
- `docs/project_structure_guide.md`
- `docs/novice_experiment_flow.md`

## Server Environment Note

- 后续在 Ubuntu 服务器上执行任何论文相关训练命令前，先运行：`conda activate paper_env`

## Next Step

- 优先整理目录结构，归档 smoke/旧结果
- 然后基于 `ultimate_results.csv` 和后续网格搜索结果，开始写论文图表与 Results 初稿
