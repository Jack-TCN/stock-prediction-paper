# Day7 Report

本日完成内容：

- 编写 `run_ultimate_experiments.py`
- 将窗口消融实验扩展到 `window_size = [20, 60, 100]`
- 总实验规模扩展为 `3 × 4 × 3 = 36` 组
- 将最大训练轮数提升到 `500`
- 将 `patience` 提升到 `20`
- 脚本自动检查并生成缺失的窗口张量
- 所有结果汇总输出到 `ultimate_results.csv`
- 每组模型仍保留自己独立的 `best_model.pt`

说明：

- 该脚本默认输出目录为 `ultimate_training_runs/`
- 默认结果汇总文件为 `ultimate_results.csv`
- 可通过 `--results-prefix` 改成别的结果文件名前缀

服务器运行前提：

- 先执行：`conda activate paper_env`
- 然后进入项目目录再启动脚本

建议：

- 论文里可以保留“100 epoch 未触发早停”的分析，但不要直接放聊天截图
- 更规范的做法是画成正式图，例如 `val_loss-epoch` 曲线，或做一张 `best_epoch / stopped_epoch` 统计表
