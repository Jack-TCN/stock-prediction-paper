# Context Anchor

## 这份文件的作用

这是给后续长对话准备的“最短上下文入口”。

以后如果聊天很多、怕上下文太长，可以先看这份文件，再看 `paper_memory.md`。

## 新聊天的固定接管方式

以后如果你新开一个聊天页面，第一句直接发：

`先读取 docs/context_anchor.md、docs/paper_memory.md、docs/path_map.md，再继续。`

这 3 个文件现在就是本项目的唯一可信入口。

- `context_anchor.md`：最短总览
- `paper_memory.md`：项目固定规则、当前结论、写作状态
- `path_map.md`：文件到底放在哪里、哪些是主线、哪些是归档

除了这 3 个文件之外，其他 md 都默认是补充材料，不需要在新聊天里全部重新读一遍。

## 当前论文主题

`LSTM与GRU在股票价格预测中的对比研究：技术指标融合与Dropout正则化分析`

## 当前固定研究问题

1. `LSTM` 和 `GRU` 哪个更好
2. 技术指标有没有帮助
3. `Dropout` 有没有帮助
4. 哪种模型-特征-正则化组合最稳定

## 当前最重要的正式结果路径

- [ultimate_training_runs](C:/Users/27476/Desktop/论文/stock_prediction_paper/_local/runs/ultimate_training_runs)
- [ultimate_results.csv](C:/Users/27476/Desktop/论文/stock_prediction_paper/ultimate_results.csv)
- [grid_search_runs](C:/Users/27476/Desktop/论文/stock_prediction_paper/_local/runs/grid_search_runs)
- [grid_search_results.csv](C:/Users/27476/Desktop/论文/stock_prediction_paper/grid_search_results.csv)
- [ablation_results.csv](C:/Users/27476/Desktop/论文/stock_prediction_paper/ablation_results.csv)

## 当前最重要的说明文件

- [paper_memory.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/paper_memory.md)
- [path_map.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/path_map.md)

如果只是为了让新聊天快速接管项目，只看这两个加上当前文件就够了。

低频说明文档已经归到：

- [_local/archive/docs_reference](C:/Users/27476/Desktop/论文/stock_prediction_paper/_local/archive/docs_reference)

这些文件默认不是新聊天的首轮必读材料。

## 当前代码角色

- [train.py](C:/Users/27476/Desktop/论文/stock_prediction_paper/train.py)：单组训练引擎
- [run_ultimate_experiments.py](C:/Users/27476/Desktop/论文/stock_prediction_paper/run_ultimate_experiments.py)：正式窗口实验批量脚本
- [run_grid_search.py](C:/Users/27476/Desktop/论文/stock_prediction_paper/run_grid_search.py)：正式网格搜索批量脚本
- [run_ablation_study.py](C:/Users/27476/Desktop/论文/stock_prediction_paper/scripts/run_ablation_study.py)：审稿阶段的技术指标消融实验脚本

## 当前目录规则

- 根目录只保留当前正式主线
- 冒烟测试放 `_local/archive/smoke_tests/`
- 旧结果放 `_local/archive/old_runs/`

## 服务器规则

任何论文相关命令前先执行：

`conda activate paper_env`

但要记住：

- 某些 `zsh` 终端里 `conda` 可能未初始化
- 出现 `command not found: conda` 时，要先处理 shell 的 conda 初始化

## 如果以后要快速进入状态

最省 token 的顺序：

1. 先看这份 `context_anchor.md`
2. 再看 `paper_memory.md`
3. 需要找文件时再看 `path_map.md`
4. 只有在需要追溯历史过程时，才去看 `_local/archive/docs_reference/` 里的归档说明文档

