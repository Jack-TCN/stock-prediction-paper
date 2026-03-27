# Day5 Report

本日完成内容：

- 编写统一训练脚本 `train.py`
- 实现 PyTorch `Dataset` 和 `DataLoader`
- 实现单层 `LSTM` 与单层 `GRU`
- 隐藏层维度固定为 `64`
- 将 `Dropout` 作为可控参数传入，默认值为 `0.2`
- 统一使用 `Adam` 优化器和 `MSELoss`
- 在每个 Epoch 输出 `RMSE`、`MAE` 和 `MAPE`
- 加入 `device = cuda / cpu` 自动切换逻辑
- 在本地 CPU 上完成 `1 epoch` 冒烟测试

模型与训练约束：

- 模型深度：单层
- 隐藏层神经元：`64`
- Dropout：默认 `0.2`
- Optimizer：`Adam`
- Loss：`MSE`
- 评价指标：`RMSE`、`MAE`、`MAPE`

快速运行命令：

```powershell
python train.py --ticker AAPL --feature-set baseline --model lstm --epochs 1 --batch-size 32 --device cpu
```

本地冒烟测试结果：

- `AAPL + baseline + LSTM + 1 epoch + CPU`：正常运行
- `AAPL + baseline + GRU + 1 epoch + CPU`：正常运行

结果文件位置：

- `results/training_runs/20260326_232423_AAPL_baseline_lstm_w60_d0p2`
- `results/training_runs/20260326_232431_AAPL_baseline_gru_w60_d0p2`
