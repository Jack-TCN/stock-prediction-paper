# Day8 Report

本日完成内容：

- 编写 `run_grid_search.py`
- 搜索空间覆盖：
  - `ticker = [AAPL, MSFT, TSLA]`
  - `feature_set = [baseline, proposed]`
  - `model = [lstm, gru]`
  - `window_size = [20, 60]`
  - `hidden_size = [32, 64, 128]`
  - `learning_rate = [0.001, 0.0005]`
  - `dropout = [0.0, 0.2, 0.4]`
- 默认规模：`432` 组组合
- 保留 `Early Stopping`
  - `max_epochs = 500`
  - `patience = 20`
- 支持 `tqdm` 进度条
- 支持断点续跑
- 支持实时追加写入 `grid_search_results.csv`
- 支持自动生成缺失的窗口张量

学术边界：

- 最优超参数选择应基于验证集指标，而不是测试集指标
- `test_RMSE / test_MAE / test_MAPE` 应用于最终报告，不应用于调参

建议正式运行命令：

```bash
conda activate paper_env
cd /path/to/stock_prediction_paper
python run_grid_search.py --device cuda:0
```
