# Context Anchor

## 这份文件的作用

这是给后续长对话准备的“最短上下文入口”。

以后如果聊天很多、怕上下文太长，可以先看这份文件，再看 `paper_memory.md`。

## 当前论文主题

`LSTM与GRU在股票价格预测中的对比研究：技术指标融合与Dropout正则化分析`

## 当前固定研究问题

1. `LSTM` 和 `GRU` 哪个更好
2. 技术指标有没有帮助
3. `Dropout` 有没有帮助
4. 哪种模型-特征-正则化组合最稳定

## 当前最重要的正式结果路径

- [ultimate_training_runs](C:/Users/27476/Desktop/论文/stock_prediction_paper/ultimate_training_runs)
- [ultimate_results.csv](C:/Users/27476/Desktop/论文/stock_prediction_paper/ultimate_results.csv)

## 当前最重要的说明文件

- [paper_memory.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/paper_memory.md)
- [novice_experiment_flow.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/novice_experiment_flow.md)
- [path_map.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/path_map.md)

## 当前代码角色

- [train.py](C:/Users/27476/Desktop/论文/stock_prediction_paper/train.py)：单组训练引擎
- [run_ultimate_experiments.py](C:/Users/27476/Desktop/论文/stock_prediction_paper/run_ultimate_experiments.py)：正式窗口实验批量脚本
- [run_grid_search.py](C:/Users/27476/Desktop/论文/stock_prediction_paper/run_grid_search.py)：正式网格搜索批量脚本

## 当前目录规则

- 根目录只保留当前正式主线
- 冒烟测试放 `archive/smoke_tests/`
- 旧结果放 `archive/old_runs/`

## 服务器规则

任何论文相关命令前先执行：

`conda activate paper_env`

## 如果以后要快速进入状态

最省 token 的顺序：

1. 先看这份 `context_anchor.md`
2. 再看 `paper_memory.md`
3. 需要回顾过程时再看 `novice_experiment_flow.md`
