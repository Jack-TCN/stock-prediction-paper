# Day6 Report

本日完成内容：

- 编写批量实验脚本 `run_all_experiments.py`
- 自动遍历 `AAPL`、`MSFT`、`TSLA`
- 自动运行四组核心对照：
  - `baseline + lstm`
  - `baseline + gru`
  - `proposed + lstm`
  - `proposed + gru`
- 总计支持 `12` 组完整实验
- 在训练脚本中加入基于验证集的 `Early Stopping`
- 默认最大 `100 epochs`
- 默认 `patience = 10`
- 自动保存最佳模型权重 `best_model.pt`
- 所有实验结束后自动汇总生成 `results/final_results.csv`

学术规范说明：

- 提前停止监控的是验证集 `val_loss`，不是测试集
- 测试集只在训练完成后做最终评估
- 这样可以避免用测试集做模型选择，符合更规范的论文实验设计

正式运行命令：

```powershell
python run_all_experiments.py
```

轻量测试命令：

```powershell
python run_all_experiments.py --epochs 1 --patience 1 --batch-size 32 --device cpu
```

关键结果文件：

- `results/final_results.csv`
- `results/final_results.json`
- `results/training_runs/`
