# Path Map

## 你以后只要按这张地图放文件就行

## 新聊天只要认这 3 个文件

以后如果你换到新聊天页面，不要让它从几十个 md 里盲找。

固定让它先读：

1. `docs/context_anchor.md`
2. `docs/paper_memory.md`
3. `docs/path_map.md`

含义是：

- `context_anchor.md`：最快进入状态
- `paper_memory.md`：固定规则和当前论文状态
- `path_map.md`：哪里是主线、哪里是归档

这 3 个文件之外，其它 md 都默认是补充材料。

新聊天还要额外记住两件事：

- 用户是学生，第一次系统写论文，需要更像导师而不是像审稿系统的引导
- 回答要尽量直接、少黑话、少冗余，先说明“现在做什么”和“为什么这样做”

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

### [scripts/run_ablation_study.py](C:/Users/27476/Desktop/论文/stock_prediction_paper/scripts/run_ablation_study.py)

作用：

- 审稿阶段的技术指标消融实验
- 固定每只股票的验证集最优超参数
- 比较 `proposed_all / no_rsi / no_macd / no_bollinger`

## 3. 正式结果区

### [ultimate_training_runs](C:/Users/27476/Desktop/论文/stock_prediction_paper/_local/runs/ultimate_training_runs)

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

- [context_anchor.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/context_anchor.md)
- [paper_memory.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/paper_memory.md)
- [path_map.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/path_map.md)

如果只是为了管理项目和让新聊天快速接管，到这里就够了。

根目录下还有 3 个最终总稿文件：

- [Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators_zh.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators_zh.md)
- [Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators.md)
- [Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators.tex](C:/Users/27476/Desktop/论文/stock_prediction_paper/Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators.tex)

真正投稿、总审稿和总修改时，应优先以这 3 个总稿文件为准，而不是分章节稿。

注意：

- 这 3 个总稿文件当前属于本地私有论文材料
- 公开 GitHub 仓库不会上传它们
- GitHub 上的 README 现在已明确区分“公开代码版仓库”和“私有论文主稿”
- 结果解释时还要额外记住：
  - `Table 1` 是“每个 stock × combo 各自 validation-best”的主结果表
  - `Table 2` 是“每只股票先固定一套 stock-level validation-best reference configuration，再做特征消融”的机制表
  - 两张表不能按“同名行应逐项数值相等”的方式理解

这里还有一个命名硬规则：

- `docs/paper_*.md` 只放正式论文章节
- 不再用同一个 `paper_*.md` 同时承载“正式章节”和“工作备忘”
- 如果某个章节需要保留旧版分析或写作过程，统一放到：
  - [_local/archive/docs_history](C:/Users/27476/Desktop/论文/stock_prediction_paper/_local/archive/docs_history)
  - 命名成 `*_working_notes.md`

### [configs](C:/Users/27476/Desktop/论文/stock_prediction_paper/configs)

作用：

- 项目默认配置说明

最重要的是：

- [project_defaults.json](C:/Users/27476/Desktop/论文/stock_prediction_paper/configs/project_defaults.json)
- [README.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/configs/README.md)

## 5. 归档区

### [archive](C:/Users/27476/Desktop/论文/stock_prediction_paper/_local/archive)

作用：

- 存旧结果和测试残留

#### [_local/archive/smoke_tests](C:/Users/27476/Desktop/论文/stock_prediction_paper/_local/archive/smoke_tests)

作用：

- 冒烟测试
- 临时验证代码是否跑通

#### [_local/archive/old_runs](C:/Users/27476/Desktop/论文/stock_prediction_paper/_local/archive/old_runs)

作用：

- 旧实验结果
- 早期版本结果

#### [_local/archive/docs_history](C:/Users/27476/Desktop/论文/stock_prediction_paper/_local/archive/docs_history)

作用：

- 已归档的历史文档
- 早期日报
- 被后续主线文档替代的旧版分析
- 例如：
  - [paper_results_working_notes.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/_local/archive/docs_history/paper_results_working_notes.md)
    - 这是 `paper_results.md` 被正式章节覆盖前的旧版结果备忘恢复件

#### [_local/archive/docs_reference](C:/Users/27476/Desktop/论文/stock_prediction_paper/_local/archive/docs_reference)

作用：

- 低频说明文档
- 只在需要回顾历史过程、清理规则、训练环境、GPU 优化建议时才看

当前已归档进去的包括：

- `file_cleanup_plan.md`
- `gpu_utilization_checklist.md`
- `novice_experiment_flow.md`
- `progress_log.md`
- `project_structure_guide.md`
- `training_environment.md`

## 以后新数据应该放哪里

### 以后如果你跑“正式主实验”

放在：

- [ultimate_training_runs](C:/Users/27476/Desktop/论文/stock_prediction_paper/_local/runs/ultimate_training_runs)
- [ultimate_results.csv](C:/Users/27476/Desktop/论文/stock_prediction_paper/ultimate_results.csv)

这两个路径代表：

- 你当前准备写进论文正文的正式版本

### 以后如果你跑“正式网格搜索”

建议放在：

- `_local/runs/grid_search_runs/`
- `grid_search_results.csv`

这两个会和 `ultimate_*` 并列，表示：

- 这是正式超参数搜索结果

### 以后如果你跑“正式消融实验”

建议放在：

- `_local/runs/ablation_runs/`
- `ablation_results.csv`

这两个代表：

- 审稿阶段用于解释技术指标作用机制的补充实验

### 以后如果你只是测试代码通不通

不要放在根目录长期留着。

可以临时命名成：

- `*_smoke`

跑通以后就移到：

- [_local/archive/smoke_tests](C:/Users/27476/Desktop/论文/stock_prediction_paper/_local/archive/smoke_tests)

### 以后如果你重新跑了一版正式结果，旧版本怎么办

不要直接删。

做法：

1. 把旧版本移到 [_local/archive/old_runs](C:/Users/27476/Desktop/论文/stock_prediction_paper/_local/archive/old_runs)
2. 再把新版本放到根目录正式位置

这样你永远只有一版“当前正式结果”留在主路径里。

## 当前正式论文章节放哪里

- 标题与摘要：
  - [paper_abstract.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/paper_abstract.md)
- 英文摘要：
  - [paper_abstract_en.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/paper_abstract_en.md)
- 引言：
  - [paper_introduction.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/paper_introduction.md)
- 英文引言：
  - [paper_introduction_en.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/paper_introduction_en.md)
- 文献综述：
  - [paper_related_work.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/paper_related_work.md)
- 英文文献综述：
  - [paper_related_work_en.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/paper_related_work_en.md)
- 方法：
  - [paper_methodology.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/paper_methodology.md)
- 英文方法：
  - [paper_methodology_en.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/paper_methodology_en.md)
- 结果与讨论：
  - [paper_results.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/paper_results.md)
- 英文结果与讨论：
  - [paper_results_en.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/paper_results_en.md)
- 结论：
  - [paper_conclusion.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/paper_conclusion.md)
- 英文结论：
  - [paper_conclusion_en.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/paper_conclusion_en.md)
- 参考文献：
  - [paper_references.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/paper_references.md)
- 英文参考文献：
  - [paper_references_en.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/paper_references_en.md)
- 根目录总稿：
  - [Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators_zh.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators_zh.md)
- 根目录英文总稿：
  - [Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators.md)

当前英文审稿修订后，结果与结论相关主文件已经同步到：

- [paper_results_en.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/paper_results_en.md)
- [paper_conclusion_en.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/docs/paper_conclusion_en.md)
- [Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators.md](C:/Users/27476/Desktop/论文/stock_prediction_paper/Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators.md)
- [Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators.tex](C:/Users/27476/Desktop/论文/stock_prediction_paper/Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators.tex)
