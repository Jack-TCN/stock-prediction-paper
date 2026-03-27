# Day3 Report

本日完成内容：

- 生成技术指标：RSI、MACD、Bollinger Bands
- 使用 `dropna()` 删除技术指标和目标列产生的 NaN
- 按时间顺序进行 80/20 硬切分
- 仅用训练集 `fit` MinMaxScaler，再对测试集执行 `transform`
- 输出处理后数据、缩放后数据、元数据和可视化图

学术红线执行说明：

- 未使用 `0`、`ffill` 或 `bfill` 补全技术指标生成的缺失值
- 未在全量数据上执行 `fit_transform`
- 训练集和测试集切分发生在归一化之前

## Per-Ticker Summary

| Ticker | Raw Rows | Clean Rows | Dropped Rows | Train Rows | Test Rows | Split Date |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| AAPL | 2765 | 2731 | 34 | 2184 | 547 | 2023-10-24 |
| MSFT | 2765 | 2731 | 34 | 2184 | 547 | 2023-10-24 |
| TSLA | 2765 | 2731 | 34 | 2184 | 547 | 2023-10-24 |