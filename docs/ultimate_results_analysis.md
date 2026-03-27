# Ultimate Results Analysis

## 1. 实验完成情况

- 总实验数：`36`
- 组合方式：`3 stocks × 4 model/feature groups × 3 window sizes`
- 早停触发情况：`36 / 36` 组全部触发 Early Stopping
- 说明：这次实验已经比之前 `100 epoch` 的版本更可靠，收敛性问题得到了实质性改善

## 2. 每只股票的最优结果

- `AAPL`
  - 最优组合：`window_size = 60 + Baseline + LSTM`
  - `RMSE = 38.1750`
  - `MAE = 30.0276`
  - `MAPE = 12.6929`
- `MSFT`
  - 最优组合：`window_size = 60 + Baseline + GRU`
  - `RMSE = 104.7072`
  - `MAE = 94.0526`
  - `MAPE = 20.7163`
- `TSLA`
  - 最优组合：`window_size = 60 + Baseline + LSTM`
  - `RMSE = 27.8749`
  - `MAE = 17.3175`
  - `MAPE = 4.9778`

## 3. 时间窗口敏感性分析的核心结论

不同窗口长度的平均结果如下：

- `window_size = 20`
  - 平均 `RMSE = 86.2517`
  - 平均 `MAE = 68.3988`
  - 平均 `MAPE = 18.0698`
- `window_size = 60`
  - 平均 `RMSE = 68.2214`
  - 平均 `MAE = 55.7031`
  - 平均 `MAPE = 15.9423`
- `window_size = 100`
  - 平均 `RMSE = 74.2113`
  - 平均 `MAE = 60.0519`
  - 平均 `MAPE = 16.4254`

结论：

- `60` 天窗口是整体表现最好的窗口长度
- `20` 天窗口普遍过短，效果最差
- `100` 天窗口虽然优于 `20`，但整体仍不如 `60`
- 因此，“时间窗口敏感性分析”这一节必须保留，而且可以成为论文的一个亮点章节

## 4. 技术指标是否真的有用

按每只股票、每个窗口分别比较 `Baseline` 和 `Proposed` 的最优模型：

- `AAPL`
  - `20`：`Baseline` 更好
  - `60`：`Baseline` 更好
  - `100`：`Baseline` 更好
- `MSFT`
  - `20`：`Baseline` 更好
  - `60`：`Baseline` 更好
  - `100`：`Proposed` 更好
- `TSLA`
  - `20`：`Baseline` 更好
  - `60`：`Baseline` 更好
  - `100`：`Proposed` 更好

结论：

- `Baseline` 在 `9` 个股票-窗口组合中赢了 `7` 次
- `Proposed` 只在 `MSFT-100` 和 `TSLA-100` 上占优
- 所以不能写成“技术指标普遍提升了性能”
- 更准确的写法是：
  - 技术指标的增益具有条件性
  - 在较长时间窗口下，技术指标对部分股票有帮助
  - 但在多数情况下，基础 `OHLCV` 特征已经足够强

## 5. LSTM 与 GRU 的对比

- 在 `Baseline` 设定下，`LSTM` 赢 `6` 次，`GRU` 赢 `3` 次
- 在 `Proposed` 设定下，`LSTM` 也赢 `6` 次，`GRU` 赢 `3` 次
- 说明整体上 `LSTM` 更稳，但并不是在所有子任务上都绝对占优

## 6. Early Stopping 的结果可以怎么写

- `window_size = 20`
  - 平均 `best_epoch = 243.67`
  - 平均 `stopped_epoch = 263.67`
- `window_size = 60`
  - 平均 `best_epoch = 250.17`
  - 平均 `stopped_epoch = 270.17`
- `window_size = 100`
  - 平均 `best_epoch = 229.08`
  - 平均 `stopped_epoch = 249.08`

结论：

- 这次实验已经不是“100 epoch 没收敛”的情况
- 模型通常在 `200~300 epoch` 区间内达到较优状态
- 你可以在论文里明确说明：扩大训练上限并引入验证集驱动的早停，有助于获得更可信的实验结果

## 7. 接下来最应该做什么

优先级 1：开始写论文 Results

- 这批数据已经够支撑结果章节
- 现在不建议再无休止加实验

优先级 2：做论文图表

- 表 1：36 组完整结果总表
- 表 2：每只股票的最佳模型表
- 表 3：窗口敏感性分析汇总表
- 图 1：不同 `window_size` 的平均 `RMSE/MAE/MAPE` 柱状图
- 图 2：每只股票在 `20/60/100` 窗口下的最佳 `RMSE` 折线图
- 图 3：`best_epoch / stopped_epoch` 统计图

优先级 3：写 Results + Discussion

- 不要写成“技术指标一定有效”
- 不要写成“GRU 一定更强”或 “LSTM 一定更强”
- 正确写法应该强调：
  - `window_size = 60` 整体最优
  - `Baseline` 在大多数情况下更稳
  - `Proposed` 在个别长窗口场景下有优势
  - `LSTM` 总体占优，但优势具有数据依赖性
