# File Cleanup Plan

## 你现在为什么会觉得乱

因为当前项目根目录里混着 5 种东西：

1. 正式代码
2. 正式论文结果
3. 早期旧结果
4. 冒烟测试结果
5. 缓存文件

开发阶段这样很正常，但写论文阶段继续这样放，就会越来越难看懂。

## 先给你一个最简单的判断标准

以后你看到一个文件或文件夹，先问它属于哪一类：

- `正式要写进论文的`
- `为了开发测试临时跑的`
- `历史记录，暂时留档`
- `缓存，可以忽略`

只要这样分，项目马上就不乱了。

## 这些内容属于“正式主线”，建议保留在根目录

- `data/`
- `docs/`
- `scripts/`
- `configs/`
- `train.py`
- `run_ultimate_experiments.py`
- `run_grid_search.py`
- `ultimate_training_runs/`
- `ultimate_results.csv`
- `requirements.txt`
- `README.md`

这些才是你论文现在最核心的资产。

## 这些内容不是正式主结果，而是测试或开发残留

- `ultimate_training_runs_smoke/`
- `ultimate_training_runs_smoke2/`
- `grid_search_runs_smoke/`
- `grid_search_smoke_results.csv`
- `__pycache__/`

这些文件看着多，但并不代表论文主结果很多。

它们主要是为了：

- 测代码有没有 bug
- 测断点续跑有没有生效
- 测脚本能不能跑通

所以它们不是“论文主战场”，只是“开发痕迹”。

## 这些内容可以归档，但先别直接删

- `training_runs/`
- `results/training_runs/`
- `final_results.csv`
- `grid_search_metadata.json`

为什么建议先归档，不建议直接删：

- 它们仍然是历史记录
- 后面如果你要回头解释某个阶段做过什么，会有用

## 你现在看到的命名，到底是什么意思

### `ultimate_`

表示：

- 这一批是正式窗口实验主结果

比如：

- `ultimate_training_runs/`
- `ultimate_results.csv`

### `grid_search_`

表示：

- 这一批是超参数搜索相关结果

### `_smoke`

表示：

- 冒烟测试
- 只是为了验证代码能不能跑通

这类文件一般不放进论文主叙事里。

### `training_runs`

表示：

- 比较早期的单组或基础批量运行结果

它不是错，只是现在已经不是项目主线了。

## 以后最适合你的命名理解方式

你不用硬记所有英文名。

只要记住这三条：

- 带 `ultimate` 的，大概率是正式主结果
- 带 `grid_search` 的，大概率是超参数搜索
- 带 `smoke` 的，大概率只是测试残留

## 推荐整理方案

### 方案 A：最稳

什么都不移动，只在脑子里分层看：

- 主结果只看 `ultimate_training_runs/` 和 `ultimate_results.csv`
- 其他都先当“辅助材料”

优点：

- 最安全
- 不会影响脚本路径

缺点：

- 你看目录时还是会觉得乱

### 方案 B：最适合你现在

新建两个目录：

- `archive/`
- `archive/smoke_tests/`

然后把这些移进去：

- `ultimate_training_runs_smoke/`
- `ultimate_training_runs_smoke2/`
- `grid_search_runs_smoke/`
- `grid_search_smoke_results.csv`

再把这些移到 `archive/old_runs/`：

- `training_runs/`
- `results/training_runs/`
- `final_results.csv`
- `grid_search_metadata.json`

这样根目录就会明显清爽。

### 方案 C：写论文阶段最终形态

根目录最后只保留：

- `data/`
- `docs/`
- `scripts/`
- `configs/`
- `train.py`
- `run_ultimate_experiments.py`
- `run_grid_search.py`
- `ultimate_training_runs/`
- `ultimate_results.csv`
- `requirements.txt`
- `README.md`

这会最像一个真正的论文项目。

## 关于 `config`，你可以怎么解释

你现在不是没有 `config`，而是有两层：

### 第一层：项目总配置

- `configs/project_defaults.json`

作用：

- 一眼看清这个项目默认怎么设

### 第二层：每组实验自己的配置

比如：

- `ultimate_training_runs/.../config.json`

作用：

- 记录这一次具体实验到底用了哪些参数

所以你朋友说“没 config”，更准确的情况是：

- `config` 不是集中写在一个脚本顶部，而是分成“项目默认配置 + 每次运行自动生成配置”

## 我给你的实际建议

现在先不要删正式结果。

最合理的顺序是：

1. 保留 `ultimate_training_runs/`
2. 保留 `ultimate_results.csv`
3. 把所有 `smoke` 文件夹挪到归档区
4. 把旧的 `training_runs` 也归档
5. 根目录只保留论文主线文件

## 如果你要我下一步直接帮你动手整理

我建议执行的版本是：

- 保留：`ultimate_training_runs/`
- 保留：`ultimate_results.csv`
- 保留：`docs/`、`data/`、`scripts/`、`configs/`
- 归档：所有带 `smoke` 的文件和文件夹
- 归档：`training_runs/`、`final_results.csv`、`grid_search_metadata.json`

这样你既不会丢结果，也不会继续被文件名淹没。
