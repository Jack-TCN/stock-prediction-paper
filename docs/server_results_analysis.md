# Server Results Analysis

数据来源：

- `final_results.csv`
- `training_runs/` 中 2026-03-27 的 12 组正式实验结果

## 1. 每只股票的最佳模型

- `AAPL` 最优：`Proposed + LSTM`
  - `RMSE = 63.6862`
  - `MAE = 56.7181`
  - `MAPE = 24.8334`
- `MSFT` 最优：`Proposed + LSTM`
  - `RMSE = 198.2162`
  - `MAE = 191.1941`
  - `MAPE = 43.1280`
- `TSLA` 最优：`Proposed + LSTM`
  - `RMSE = 88.6950`
  - `MAE = 58.9516`
  - `MAPE = 16.0238`

## 2. 总体规律

- 从 3 只股票的最优结果看，`Proposed + LSTM` 在全部数据集上表现最好
- 平均指标上，`Proposed + LSTM` 也是四种组合里最优
- 说明在当前实验设置下，引入技术指标并结合 LSTM，整体上优于只使用基础 OHLCV 特征

## 3. Proposed 相比 Baseline 的改进

- `AAPL`
  - 最优 Baseline：`GRU`
  - 最优 Proposed：`LSTM`
  - `RMSE` 改进：`0.2616`
  - `MAE` 改进：`0.4207`
  - `MAPE` 改进：`0.2139`
- `MSFT`
  - 最优 Baseline：`LSTM`
  - 最优 Proposed：`LSTM`
  - `RMSE` 改进：`3.5670`
  - `MAE` 改进：`3.3303`
  - `MAPE` 改进：`0.7431`
- `TSLA`
  - 最优 Baseline：`LSTM`
  - 最优 Proposed：`LSTM`
  - `RMSE` 改进：`3.0300`
  - `MAE` 改进：`2.9460`
  - `MAPE` 改进：`0.8707`

## 4. LSTM 与 GRU 的比较

- 在 `Proposed` 特征集下，`LSTM` 在 3 只股票上都优于 `GRU`
- 在 `Baseline` 特征集下，`AAPL` 上 `GRU` 略优于 `LSTM`
- 但整体看，`LSTM` 的平均表现仍然优于 `GRU`

## 5. Early Stopping 观察

- 12 组实验的 `best_epoch = 100`
- 12 组实验的 `stopped_epoch = 100`
- 说明当前设置下，`Early Stopping` 没有提前触发，模型全部跑满了最大轮数
- 这意味着：
  - 要么验证集损失仍在持续下降
  - 要么 100 个 epoch 对当前任务仍然不够长

## 6. 可直接写进论文的简短结论

可以写成：

“Across the three stock datasets, the proposed feature configuration combined with the LSTM model consistently achieved the best predictive performance. Compared with the strongest baseline setting, the proposed framework reduced RMSE by 0.2616 on AAPL, 3.5670 on MSFT, and 3.0300 on TSLA, indicating that the integration of technical indicators improved forecasting accuracy under the current experimental setup.”
