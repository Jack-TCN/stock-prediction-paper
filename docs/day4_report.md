# Day4 Report

当前滑动窗口大小：`60`

候选窗口大小：`[20, 60, 100]`

本日完成内容：

- 构建可配置的滑动窗口样本生成脚本
- 输出标准 PyTorch 三维张量 `(samples, time_steps, features)`
- 分开生成 Baseline 和 Proposed 两套数据
- 将 train/test 样本分别保存为 `.pt` 文件
- 记录每个样本的起始日期、结束日期和标签日期，便于后续审计

学术严谨性说明：

- 训练集样本的标签只允许落在训练区间内
- 测试集样本的标签从测试期第一天开始，允许窗口引用此前历史数据
- 标签 `y` 严格对齐为窗口结束后第一天的收盘价

## Tensor Summary

| Ticker | Feature Set | Split | X Shape | y Shape | Saved Path |
| --- | --- | --- | --- | --- | --- |
| AAPL | baseline | train | [2124, 60, 5] | [2124] | data/tensors/window_60/AAPL/baseline/train.pt |
| AAPL | baseline | test | [548, 60, 5] | [548] | data/tensors/window_60/AAPL/baseline/test.pt |
| AAPL | proposed | train | [2124, 60, 12] | [2124] | data/tensors/window_60/AAPL/proposed/train.pt |
| AAPL | proposed | test | [548, 60, 12] | [548] | data/tensors/window_60/AAPL/proposed/test.pt |
| MSFT | baseline | train | [2124, 60, 5] | [2124] | data/tensors/window_60/MSFT/baseline/train.pt |
| MSFT | baseline | test | [548, 60, 5] | [548] | data/tensors/window_60/MSFT/baseline/test.pt |
| MSFT | proposed | train | [2124, 60, 12] | [2124] | data/tensors/window_60/MSFT/proposed/train.pt |
| MSFT | proposed | test | [548, 60, 12] | [548] | data/tensors/window_60/MSFT/proposed/test.pt |
| TSLA | baseline | train | [2124, 60, 5] | [2124] | data/tensors/window_60/TSLA/baseline/train.pt |
| TSLA | baseline | test | [548, 60, 5] | [548] | data/tensors/window_60/TSLA/baseline/test.pt |
| TSLA | proposed | train | [2124, 60, 12] | [2124] | data/tensors/window_60/TSLA/proposed/train.pt |
| TSLA | proposed | test | [548, 60, 12] | [548] | data/tensors/window_60/TSLA/proposed/test.pt |