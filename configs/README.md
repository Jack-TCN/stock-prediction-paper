# Configs Guide

## 这个文件夹是干什么的

这个文件夹放的是“项目级别的默认配置”。

它不是每次训练都会唯一读取的总入口，但它很适合：

- 你自己快速回顾项目默认设定
- 别人快速理解你的实验默认框架
- 写论文时查项目全局参数

## 当前最重要的文件

- [project_defaults.json](C:/Users/27476/Desktop/论文/stock_prediction_paper/configs/project_defaults.json)

## 这个文件和每个实验文件夹里的 `config.json` 有什么区别

### `configs/project_defaults.json`

作用：

- 记录整个项目默认怎么设计

比如：

- 默认题目
- 默认股票
- 默认模型
- 默认窗口
- 默认指标

### `ultimate_training_runs/.../config.json`

作用：

- 记录某一组具体实验到底用了什么参数

比如：

- 这次是 `AAPL` 还是 `MSFT`
- 用的是 `LSTM` 还是 `GRU`
- `window_size` 是多少
- `dropout` 是多少

## 最白话的理解

你可以这样记：

- `project_defaults.json`：项目总说明书
- 每组里的 `config.json`：单次实验说明书
