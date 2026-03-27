# Stock Prediction Paper

论文题目：
`LSTM与GRU在股票价格预测中的对比研究：技术指标融合与Dropout正则化分析`

这个项目用于完成一篇面向英文期刊投稿的实证型论文。

## 如果是新会话，先看哪里

如果后面对话很多，先看这两个文件再继续：

- [context_anchor.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/context_anchor.md)
- [paper_memory.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/paper_memory.md)

如果你是论文小白，再看：

- [novice_experiment_flow.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/novice_experiment_flow.md)

## 项目当前主路径

- [ultimate_training_runs](C:/Users/27476/Desktop/论文/stock_prediction_paper/ultimate_training_runs)：当前正式主实验每组详细结果
- [ultimate_results.csv](C:/Users/27476/Desktop/论文/stock_prediction_paper/ultimate_results.csv)：当前正式主实验总表
- [docs](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs)：项目知识库和说明文档
- [configs](C:/Users/27476/Desktop/论文/stock_prediction_paper/configs)：项目默认配置说明

## 项目结构

- `data/`：数据区
- `scripts/`：数据处理脚本
- `train.py`：单组训练引擎
- `run_ultimate_experiments.py`：正式窗口实验批量脚本
- `run_grid_search.py`：正式网格搜索脚本
- `archive/`：旧结果和冒烟测试归档

## 当前默认实验设定

- 股票：`AAPL`、`MSFT`、`TSLA`
- 数据源：`Yahoo Finance`
- 预测目标：下一交易日收盘价
- 基础特征：`Open`、`High`、`Low`、`Close`、`Volume`
- 技术指标：`RSI`、`MACD`、`Bollinger Bands`
- 比较模型：`LSTM`、`GRU`

## `requirements.txt` 是什么

它是 Python 依赖包清单。

如果以后在新环境安装依赖，可以执行：

```bash
conda activate paper_env
pip install -r requirements.txt
```
