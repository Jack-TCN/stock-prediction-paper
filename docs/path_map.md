# Path Map

## 你以后只要按这张地图放文件就行

这个项目以后按 5 个区来理解：

1. 数据区
2. 代码区
3. 正式结果区
4. 文档区
5. 归档区

## 1. 数据区

### [data](C:/Users/27476/Desktop/论文/stock_prediction_paper/data)

这里放所有数据。

#### [data/raw](C:/Users/27476/Desktop/论文/stock_prediction_paper/data/raw)

作用：

- 原始股票数据

不要轻易改。

#### [data/processed](C:/Users/27476/Desktop/论文/stock_prediction_paper/data/processed)

作用：

- 清洗后数据
- 技术指标数据
- metadata

#### [data/tensors](C:/Users/27476/Desktop/论文/stock_prediction_paper/data/tensors)

作用：

- 模型训练直接使用的 `.pt` 张量

## 2. 代码区

### [scripts](C:/Users/27476/Desktop/论文/stock_prediction_paper/scripts)

作用：

- 数据处理脚本
- 滑动窗口生成脚本

### [train.py](C:/Users/27476/Desktop/论文/stock_prediction_paper/train.py)

作用：

- 单组实验训练引擎

### [run_ultimate_experiments.py](C:/Users/27476/Desktop/论文/stock_prediction_paper/run_ultimate_experiments.py)

作用：

- 正式窗口实验批量运行

### [run_grid_search.py](C:/Users/27476/Desktop/论文/stock_prediction_paper/run_grid_search.py)

作用：

- 正式网格搜索批量运行

## 3. 正式结果区

### [ultimate_training_runs](C:/Users/27476/Desktop/论文/stock_prediction_paper/ultimate_training_runs)

作用：

- 当前论文主实验的每组详细结果

里面每个子文件夹通常有：

- `best_model.pt`
- `metrics.csv`
- `config.json`
- `summary.json`

### [ultimate_results.csv](C:/Users/27476/Desktop/论文/stock_prediction_paper/ultimate_results.csv)

作用：

- 当前论文主实验的总表

## 4. 文档区

### [docs](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs)

作用：

- 论文知识库
- 过程说明
- 结果解读

你最常看的几个：

- [paper_memory.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/paper_memory.md)
- [novice_experiment_flow.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/novice_experiment_flow.md)
- [project_structure_guide.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/project_structure_guide.md)
- [ultimate_results_analysis.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/ultimate_results_analysis.md)

### [configs](C:/Users/27476/Desktop/论文/stock_prediction_paper/configs)

作用：

- 项目默认配置说明

最重要的是：

- [project_defaults.json](C:/Users/27476/Desktop/论文/stock_prediction_paper/configs/project_defaults.json)
- [README.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/configs/README.md)

## 5. 归档区

### [archive](C:/Users/27476/Desktop/论文/stock_prediction_paper/archive)

作用：

- 存旧结果和测试残留

#### [archive/smoke_tests](C:/Users/27476/Desktop/论文/stock_prediction_paper/archive/smoke_tests)

作用：

- 冒烟测试
- 临时验证代码是否跑通

#### [archive/old_runs](C:/Users/27476/Desktop/论文/stock_prediction_paper/archive/old_runs)

作用：

- 旧实验结果
- 早期版本结果

## 以后新数据应该放哪里

### 以后如果你跑“正式主实验”

放在：

- [ultimate_training_runs](C:/Users/27476/Desktop/论文/stock_prediction_paper/ultimate_training_runs)
- [ultimate_results.csv](C:/Users/27476/Desktop/论文/stock_prediction_paper/ultimate_results.csv)

这两个路径代表：

- 你当前准备写进论文正文的正式版本

### 以后如果你跑“正式网格搜索”

建议放在：

- `grid_search_runs/`
- `grid_search_results.csv`

这两个会和 `ultimate_*` 并列，表示：

- 这是正式超参数搜索结果

### 以后如果你只是测试代码通不通

不要放在根目录长期留着。

可以临时命名成：

- `*_smoke`

跑通以后就移到：

- [archive/smoke_tests](C:/Users/27476/Desktop/论文/stock_prediction_paper/archive/smoke_tests)

### 以后如果你重新跑了一版正式结果，旧版本怎么办

不要直接删。

做法：

1. 把旧版本移到 [archive/old_runs](C:/Users/27476/Desktop/论文/stock_prediction_paper/archive/old_runs)
2. 再把新版本放到根目录正式位置

这样你永远只有一版“当前正式结果”留在主路径里。
