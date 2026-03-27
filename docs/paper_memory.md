# Paper Memory

## Fixed Topic

`LSTM与GRU在股票价格预测中的对比研究：技术指标融合与Dropout正则化分析`

## Fixed Research Questions

1. 在相同数据和实验设置下，LSTM 与 GRU 哪个预测效果更好？
2. 加入 RSI、MACD 和 Bollinger Bands 后，预测性能是否提升？
3. 加入 Dropout 后，测试集表现和泛化能力是否改善？
4. 模型、特征和正则化三者的哪种组合最稳定？

## Must Keep In The Paper

- 必须保留“时间窗口敏感性分析”这一节
- 该节的核心是比较 `window_size = 20, 60, 100`
- 该节属于论文的重要增值点，不是可删项
- 论文写作时要明确说明滑动窗口长度会影响模型对时间依赖的捕捉能力
- 论文结果部分需要单独报告不同窗口下的 `RMSE`、`MAE`、`MAPE`
- 论文讨论部分需要解释为什么某些股票更适合短窗口或长窗口

## Fixed Experimental Rules

- 缺失值处理：技术指标产生的 NaN 只能 `dropna()`
- 禁止用 `0`、`ffill`、`bfill` 补缺失值
- 必须先按时间顺序切分，再做归一化
- `MinMaxScaler` 只能在训练集上 `fit`
- 测试集只能 `transform`
- 标签 `y` 必须是窗口结束后下一交易日的收盘价

## Current Core Design

- 股票：`AAPL`、`MSFT`、`TSLA`
- Baseline 特征：`OHLCV`
- Proposed 特征：`OHLCV + RSI + MACD + Bollinger Bands`
- 模型：单层 `LSTM`、单层 `GRU`
- Hidden Units：`64`
- Dropout：默认 `0.2`
- 优化器：`Adam`
- 损失函数：`MSE`
- 评价指标：`RMSE`、`MAE`、`MAPE`

## Server Rule

- 以后在 Ubuntu 服务器上执行任何论文相关命令前，先运行：`conda activate paper_env`

## Current Important Result

- 正式服务器结果中，3 只股票当前最佳组合均为 `Proposed + LSTM`
- 这条结论后续要写进 Results 和 Discussion

## Grid Search Plan

- 已编写 `run_grid_search.py`
- 默认搜索空间总数为 `432`
- 该搜索的主要目的：
  - 补强论文的超参数严谨性
  - 比较不同 `hidden_size / learning_rate / dropout` 的影响
- 正式写论文时，超参数最优组合必须基于验证集表现选择

## Beginner Note

- 如果后面看文件觉得混乱，先看：`docs/project_structure_guide.md`
- 如果后面担心上下文太长、想先快速进入状态，先看：`docs/context_anchor.md`
- 如果想按“我问了什么 / 做了什么 / 为什么这样做”来回顾，先看：`docs/novice_experiment_flow.md`
- 如果想搞清楚哪些文件该保留、哪些只是测试残留，先看：`docs/file_cleanup_plan.md`
- 如果想搞清楚每类文件应该放在哪条路径，先看：`docs/path_map.md`
- 如果别人问你“有没有 config”，先看：`configs/README.md`
