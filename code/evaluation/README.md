# M5 预测实验

本目录围绕未来 28 天销量预测，比较从季节性朴素方法到信息共享混合模型的一系列改动。评价覆盖全部 30,490 条商品×门店序列，主要指标为 12 层级 WRMSSE。

## 运行入口与依赖

先按[数据准备说明](../data_prep/README.md)生成 `data/processed/sales_matrix.npy` 和 `sales_metadata.csv`。脚本还读取原始目录中的日历、价格及销量文件，环境安装见[项目说明](../../README.md#环境与运行)。

以下命令在项目根目录的 Windows PowerShell 中执行：

```powershell
.\.venv\Scripts\python.exe code/evaluation/01_snaive_wrmsse.py
.\.venv\Scripts\python.exe code/evaluation/02_select_x_wrmsse.py
.\.venv\Scripts\python.exe code/evaluation/03_visualize_x_selection.py
.\.venv\Scripts\python.exe code/evaluation/04_select_rho_wrmsse.py
.\.venv\Scripts\python.exe code/evaluation/05_grouped_trend_wrmsse.py
.\.venv\Scripts\python.exe code/evaluation/06_calendar_effect_wrmsse.py
.\.venv\Scripts\python.exe code/evaluation/07_sparse_zero_diagnostic.py
.\.venv\Scripts\python.exe code/evaluation/08_borrow_information_wrmsse.py
.\.venv\Scripts\python.exe code/evaluation/09_build_kaggle_submission.py
```

主要实验关系如下：

```text
01 sNaive 与层级评分
02 选择窗口 x ──→ 03 窗口选择图形
       ↓
04 选择分组 rho ──→ 05 分组趋势比较
       ↓
06 节日与 SNAP ──→ 07 稀疏销量诊断
       ↓
08 信息共享混合
       ↓
09 提交文件
```

03 读取 02 的结果，05 使用 04 的参数，07 使用 04 和 06 的参数。最终 BLX 主线按 01、02、04、06、08、09 运行；图形和诊断分支可按分析需要补充。

## 数据切分与评分

| 范围 | 用途 |
|---|---|
| `d_1` 至 `d_1913` | 历史销量及滚动实验 |
| `d_1914` 至 `d_1941` | 本地留出期评价 |
| `d_1942` 至 `d_1969` | 提交文件 evaluation 部分的预测目标 |

滚动实验共有 64 个起点，按 `d_121 + 28k` 排列，最后一个为 `d_1885`。每个起点使用截至当天的历史，预测随后 28 天。

WRMSSE 将底层预测聚合为 12 个层级、42,840 个节点。每个节点的 RMSSE 分母为首次非零销量之后的历史一阶差分平方均值，权重来自当前起点最近 28 天的销售额；最终对 12 个层级得分取等权平均。

早期滚动起点可能出现缩放分母为零的节点。滚动选择评分会排除这些节点，并在各层级内重新归一化剩余销售额权重，排除占比记录在 `origin_metric_coverage.csv`。留出期使用 01 中的完整层级评分实现。

## 01：季节性朴素基线

`01_snaive_wrmsse.py` 将最后七天 `d_1907` 至 `d_1913` 的销量重复四次，得到下一段 28 天预测。

它也提供后续实验复用的层级聚合、缩放分母、销售额权重和留出期评分函数。

输出目录：`outputs/evaluation/snaive/`。

- `snaive_predictions.csv`：底层预测，保留 `_validation` ID。
- `snaive_wrmsse_summary.json`、`snaive_wrmsse_by_level.csv`：总分与分层得分。
- 每日总销量、示例序列和分层误差的表格与图形。

## 02—03：选择与展示平均窗口

`02_select_x_wrmsse.py` 比较同星期平均窗口：

```text
x = {1, 2, 3, 4, 5, 6, 8, 12, 16}
```

例如预测星期一时，使用过去 x 个星期一的销量平均。开售日期根据首次价格记录确定；开售后的零销量参与平均。

实验同时比较所有品类使用同一 x，以及 FOODS、HOBBIES、HOUSEHOLD 分别使用不同 x 的 729 种组合。

64 个起点按时间拆分为：

| 阶段 | 起点数 | 作用 |
|---|---:|---|
| 开发期 | 32 | 搜索最优品类组合与统一窗口 |
| 确认期 | 16 | 比较开发期选定的两个候选和 x=1 |
| 审计期 | 16 | 评价选定方案及固定对照 |

确认期均分最低的候选被采用；并列时依次优先 x=1、统一窗口、品类组合。已有结果采用统一 `x=8`。

`03_visualize_x_selection.py` 绘制统一窗口曲线、品类组合、阶段比较、x=5 与 x=8 的配对差异及商品示例。首次运行会补算 x=5 的确认期得分并缓存，用于后续比较。

输出目录：`outputs/evaluation/x_selection/`。重点文件为 `selection_summary.json`、`development_vector_means.csv`、`rolling_origin_scores.csv`、`selected_vector_split_scores.csv` 及 `figures/`。

## 04：选择近期衰减与分组

`04_select_rho_wrmsse.py` 固定 x=8，对同星期历史使用归一化权重：

```text
w_k ∝ rho^k，k=0 表示最近一次观测
```

rho=0 对应最近一次有效同星期观测，rho=1 对应八周等权平均。候选网格为 `{0, .25, .5, .65, .75, .8, .85, .9, .95, 1}`。

分组从全局、州、州×品类扩展到门店×品类，以坐标下降搜索各组 rho。四折回测分别在前 32、40、48、56 个起点拟合，再在随后八个起点验证。按与最优模型的配对损失差及移动块自助抽样标准误，选择符合单标准误规则的最简单分组。

已有结果选择州×品类，共九个 rho，最终使用全部 64 个起点重拟合。

输出目录：`outputs/evaluation/rho_selection/`。包括各折和最终 rho、分组选择摘要、验证得分、预测及留出期分层得分。

已完成四折计算时，可从已有结果继续最终选择与拟合：

```powershell
.\.venv\Scripts\python.exe code/evaluation/04_select_rho_wrmsse.py --resume-selection
```

## 05：比较额外趋势

`05_grouped_trend_wrmsse.py` 在九个州×品类组内汇总最近八个完整周，使用 rho 加权最小二乘估计趋势，比较水平不变、线性外推和阻尼外推。分组水平的预测比率用于缩放已有底层星期曲线，负的外推水平截为零。

开发期选择阻尼系数 phi，确认期比较三个候选，审计期记录结果。已有结果选择水平不变模型。

输出目录：`outputs/evaluation/grouped_trend/`，包含滚动得分、阶段摘要、分组趋势分量、星期效应、预测和图形。

附加选项：

- `--reuse-rolling-scores`：读取已有滚动得分，重建最终起点输出。
- `--skip-final-holdout`：运行至预测生成，跳过留出期评分。

## 06：节日与 SNAP 修正

`06_calendar_effect_wrmsse.py` 在分组 rho 基线上比较 `baseline`、`holiday_only`、`snap_only` 和 `both` 四个候选。

日历乘数采用：

```text
exp(beta_g × SNAP + Σ gamma_e × event_e)
```

beta 为九个州×品类组的 SNAP 系数，gamma 为四个节日的共享系数：Pesach End、OrthodoxEaster、Cinco De Mayo、Mother's day。节日模型从参考均值中排除历史事件日；SNAP 模型先去除历史 SNAP 影响，再施加目标日乘数。

参数通过分组总销量的 Poisson 负对数似然拟合，对数乘数限制在 `[-0.7, 0.7]`，SNAP 岭惩罚为 2；各节日惩罚按其拟合日期上的基线预测总量设定。另输出节日惩罚敏感性结果。

四折划分与 rho 实验相同，每折使用对应折的 rho。候选按 32 个验证起点的平均 WRMSSE 选择，并列时依次优先 baseline、holiday_only、snap_only、both。已有结果选择 both。

输出目录：`outputs/evaluation/calendar_effect/`。包括滚动得分、各折与最终系数、四套预测、留出期得分和图形。

可用 `--smoke` 执行单个验证起点的小规模检查，输出使用同一目录。

## 07：零销量与稀疏序列诊断

`07_sparse_zero_diagnostic.py` 重建各验证折的 both 与 rho 基线预测，按当前起点的开售历史分组：

- 尚无开售历史。
- 开售不足 28 天。
- 成熟序列最近 56 天的零销量占比低于 50%、50% 至 80%、至少 80%。

统计零销量日预测超过 0、0.5、1 的比例、零销量日平方误差贡献，以及有销量日的低估、漏报和均值。对高稀疏序列，另按预测值区间分析校准，并比较高预测十分位的实际高销量频率。

输出目录：`outputs/evaluation/sparse_zero_diagnostic/`，包括逐起点和汇总统计、预测区间校准表及两张诊断图。

## 08：同店与跨门店信息共享

`08_borrow_information_wrmsse.py` 比较三个预测成分：

| 成分 | 信息来源 |
|---|---|
| B | 06 的日历与 SNAP 同星期基线 |
| L_perp | 同一商品门店过去 56 天中，目标星期之外的开售后非事件日 |
| X | 同一商品在其他门店过去 56 天的销量 |

L_perp 先去除 SNAP 和星期效应，再恢复目标星期与目标日日历效应。门店×品类星期乘数使用过去 364 天估计，并用 28 个伪观测日收缩。X 使用去日历化的历史销量和门店份额先验转换跨店销量率，收缩参数 `kappa={5,20,80}`，份额优势比上限为 20。所需历史不足时，相应成分回退到 B。

混合按过去八个目标星期中的正销量次数分为 0、1—2、至少 3 三组，每组的三个权重之和为一。实验比较 B、BL、BX、BLX，以训练起点的层级 WRMSSE 优化权重，再以四折验证均分选择成分和 kappa；最终在全部 64 个起点重拟合。

已有结果选择 `BLX`、`kappa=80`。输出目录为 `outputs/evaluation/borrow_information/`，重点查看：

- `validation_candidate_summary.csv`：候选验证均分。
- `final_mixture_weights.csv`：最终九个权重。
- `forecast_borrow_information.csv`：28 天底层预测。
- `experiment_summary.json`：选择结果、参数和留出期得分。

## 09：生成提交文件

`09_build_kaggle_submission.py` 读取 04、06、08 保存的参数及 BLX 预测：

1. validation 部分使用已有 `d_1914` 至 `d_1941` 预测。
2. 将新观测的 28 天销量追加到历史，起点推进到 `d_1941`。
3. 沿用 rho、日历系数、kappa 和混合权重，预测 `d_1942` 至 `d_1969`。
4. 按 `sample_submission.csv` 的 ID 顺序合并两部分并检查格式。

输出为 `outputs/submission/m5_accuracy_submission.csv`，共 60,980 行、29 列，列名为 `id,F1,...,F28`。此文件和完整底层预测在本地运行生成；上传版本保留评分摘要、参数表和精选图形，见[结果索引](../../outputs/README.md)。

## 已保存结果

以下为同一留出期 `d_1914` 至 `d_1941` 的 WRMSSE：

| 方法 | WRMSSE |
|---|---:|
| sNaive | 0.86970 |
| 州×品类 rho | 0.73699 |
| rho＋节日 | 0.71941 |
| rho＋SNAP | 0.68630 |
| rho＋节日＋SNAP | 0.66136 |
| BLX | 0.65357 |

数值来自各输出目录的 JSON 摘要。滚动验证用于候选选择，留出期得分用于观察选定配置在最后 28 天的表现。
