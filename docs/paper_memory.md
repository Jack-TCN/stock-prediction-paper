# Paper Memory

## Fixed Topic

`LSTM与GRU在股票价格预测中的对比研究：技术指标融合与Dropout正则化分析`

## User Profile And Working Style

- 用户是学生，这是第一次较完整地推进一篇英文论文
- 当前目标不是冲强刊，而是先做出一篇结构完整、实验可信、可以向老师请教和尝试投稿的稿子
- 用户不喜欢过度学术黑话、过多空泛框架和文档泛滥
- 解释时应优先给出“为什么这么做、现在做到哪一步、下一步该干什么”
- 任何影响主线理解的修改，都必须同步更新：
  - `docs/context_anchor.md`
  - `docs/paper_memory.md`
  - `docs/path_map.md`

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
- 最终整合完整 Markdown 终稿时，要在 `Results` 章节用 Markdown 图片语法插入：
  - `./figures/hyperparameter_sensitivity.png`
  - `./figures/tsla_v_reversal_lag_comparison.png`
- 同时要把“Results 核心结果总表”放在结果章节的合适位置，而不是只散落在说明文档里

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
- 当前服务器硬件信息单独记录在：`_local/archive/docs_reference/training_environment.md`
- 论文写作时需要在实验环境或实现细节部分简洁交代训练设备
- 注意：服务器某些 `zsh` 终端里 `conda` 可能未初始化，出现 `command not found: conda` 时要先处理 shell 初始化

## Current Directory Rule

- 为避开 Windows 下中文路径导致的 `pandoc/python` 编码问题，已建立一个英文入口路径：
  - `C:\Users\27476\Desktop\article\stock_prediction_paper`
- 该路径是指向当前项目根目录的 junction，后续做 `docx/pdf` 导出或其他依赖外部工具的任务时，优先走这个英文路径
- 根目录只保留适合上传 GitHub 的主线文件
- 本地私有训练资产、日志、旧结果统一放在 `_local/`
- `docs/paper_*.md` 只保留“正式论文章节草稿”，不能再混放备忘录或工作笔记
- 分析备忘、阶段性解释、被正式章节替代的旧内容，一律放到：
  - `_local/archive/docs_history/`
- 低频说明文档统一放到：
  - `_local/archive/docs_reference/`
- 命名规则固定为：
  - 正式章节：`paper_introduction.md`、`paper_methodology.md`、`paper_related_work.md`、`paper_results.md`、`paper_conclusion.md`
  - 工作备忘：`*_working_notes.md`
- 已发生过一次同名覆盖错误：旧版 `paper_results.md` 工作备忘被正式章节覆盖
- 该错误已修复，旧内容恢复到：
  - `_local/archive/docs_history/paper_results_working_notes.md`
- 当前最重要的详细运行目录是：
  - `_local/runs/grid_search_runs/`
  - `_local/runs/ultimate_training_runs/`
- 当前最重要的正式总表仍保留在根目录：
  - `grid_search_results.csv`
  - `ultimate_results.csv`
- 当前根目录下的正式总稿产物包括：
  - `Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators_zh.md`
  - `Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators.md`
  - `Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators.tex`
- 但当前公开 GitHub 仓库不会上传上述完整论文主稿；公开仓库仅保留代码、图、结果总表和必要说明文档
- 审稿修订阶段新增：
  - `scripts/run_ablation_study.py`
  - `ablation_results.csv`
- 英文总稿中的图表引用已统一为：
  - Markdown：`Fig. 1`、`Fig. 2`、`Table 1`
  - LaTeX：`Fig.~\ref{...}`、`Table~\ref{...}`
- 进一步修订后，英文总稿已新增：
  - `4.4 Ablation Study on Technical Indicators`
  - `Table 2: Ablation Study Results`
  - Discussion 末尾的“轻量模型 + 金融先验 + 严格防泄露验证”的防守性段落
  - Conclusion / Future Work 中关于中国 A 股、欧洲市场与更多资产类别验证的扩展方向

## Current Important Result

- 现在论文最重要的结果以 `grid_search_results.csv` 为准，不再以前一版 `ultimate_results.csv` 的结论为准
- 当前按“验证集最优”选配置时，每只股票的最佳组合分别为：
  - `AAPL`：`baseline + LSTM + window=60 + hidden=128 + lr=0.001 + dropout=0.2`
  - `MSFT`：`proposed + GRU + window=20 + hidden=128 + lr=0.001 + dropout=0.0`
  - `TSLA`：`proposed + GRU + window=60 + hidden=128 + lr=0.001 + dropout=0.2`
- 论文里必须坚持：验证集选配置，测试集只报告最终结果

## Ablation Study Snapshot

- `ablation_results.csv` 已跑完
- 范围：`AAPL`、`MSFT`、`TSLA`
- 特征组：
  - `proposed_all`
  - `no_rsi`
  - `no_macd`
  - `no_bollinger`
- 当前最重要的机制性结论：
  - `RSI` 与 `Bollinger Bands` 对缓解滞后效应的贡献最强
  - 在 `MSFT` 与 `TSLA` 上，移除 `RSI` 或 `Bollinger Bands` 会导致测试误差显著恶化
  - `MACD` 的贡献更偏资产依赖，不适合写成“所有股票上都不可替代”
- 写作时要坚持更严谨的表述：
  - `RSI` 和 `Bollinger Bands` 是主要驱动
  - `MACD` 是互补性、条件性的贡献
- 结果解释红线：
  - `Table 1` 是“每个 stock × combo 各自 validation-best”的主结果表
  - `Table 2` 是“每只股票先固定一套 stock-level validation-best reference configuration，再做特征消融”的机制表
  - 因此，`Table 2` 中的 `Proposed (All)` 不应该被解释为 `Table 1` 中对应 `Proposed` 行的数值复刻
  - 论文正文里还需要更直白地补一句：`Table 2` 是在单独的 ablation retraining protocol 下生成的，因此与 `Table 1` 出现小幅数值差异反映的是实验上下文不同，而不是结果矛盾

## Grid Search Result Snapshot

- `run_grid_search.py` 已完整跑完，`432 / 432` 组均已完成
- 运行设备：`cuda:0`
- 总耗时约 `7 小时 8 分`
- 大部分组合都在 `500 epochs` 之前触发早停：
  - `395` 组提前停止
  - `37` 组跑满 `500`

按“验证集最优”选择时，每只股票当前最佳配置为：

- `AAPL`：`baseline + LSTM + window=60 + hidden=128 + lr=0.001 + dropout=0.2`
- `MSFT`：`proposed + GRU + window=20 + hidden=128 + lr=0.001 + dropout=0.0`
- `TSLA`：`proposed + GRU + window=60 + hidden=128 + lr=0.001 + dropout=0.2`

相对 `ultimate_results.csv` 中的最佳配置，这次网格搜索对应测试集表现提升为：

- `AAPL`：`RMSE` 下降约 `7.71`
- `MSFT`：`RMSE` 下降约 `113.57`
- `TSLA`：`RMSE` 下降约 `1.07`

重要写作提醒：

- 论文里正式选“最优超参数”时，必须按验证集选择
- 不能按测试集挑最优，否则会有测试集泄露风险
- 当前 `MSFT` 和 `TSLA` 上，“验证集最优”和“测试集最优”不是同一组配置，这一点后面写论文时必须说明清楚

## Grid Search Plan

- 已编写 `run_grid_search.py`
- 默认搜索空间总数为 `432`
- 该搜索的主要目的：
  - 补强论文的超参数严谨性
  - 比较不同 `hidden_size / learning_rate / dropout` 的影响
- 正式写论文时，超参数最优组合必须基于验证集表现选择

## Minimal Reading Rule

- 以后优先只看：
  - `docs/context_anchor.md`
  - `docs/paper_memory.md`
  - `docs/path_map.md`
- 低频说明、历史过程、清理规则、GPU 优化备注等内容，统一放在：
  - `_local/archive/docs_reference/`
- 正式投稿与改稿时，应优先以根目录总稿为准，而不是分章节稿

## New Chat Rule

- 以后如果你换一个新聊天页面，不要试图让新页面读完所有 md
- 直接让它先读这 3 个文件：
  - `docs/context_anchor.md`
  - `docs/paper_memory.md`
  - `docs/path_map.md`
- 这 3 个文件是当前项目的唯一可信入口
- 其他 md 一律视为补充材料，不作为新聊天的首轮必读内容
- 如果后面还觉得 docs 太散，下一轮可以继续把低频说明文档并入这 3 个主文件

## Current Manuscript Files

- 正式摘要与标题：
  - `docs/paper_abstract.md`
- 英文摘要：
  - `docs/paper_abstract_en.md`
- 正式引言：
  - `docs/paper_introduction.md`
- 英文引言：
  - `docs/paper_introduction_en.md`
- 正式文献综述：
  - `docs/paper_related_work.md`
- 英文文献综述：
  - `docs/paper_related_work_en.md`
- 正式方法：
  - `docs/paper_methodology.md`
- 英文方法：
  - `docs/paper_methodology_en.md`
- 正式结果与讨论：
  - `docs/paper_results.md`
- 英文结果与讨论：
  - `docs/paper_results_en.md`
- 正式结论：
  - `docs/paper_conclusion.md`
- 英文结论：
  - `docs/paper_conclusion_en.md`
- 正式参考文献表：
  - `docs/paper_references.md`
- 英文参考文献表：
  - `docs/paper_references_en.md`
- 根目录总稿：
  - `Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators_zh.md`
- 根目录英文总稿：
  - `Stock_Price_Forecasting_LSTM_GRU_Technical_Indicators.md`
