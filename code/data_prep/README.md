# M5 数据准备与探索

本目录把原始 CSV 转换为便于计算的中间数据，并生成探索性统计表和图形。流程为：

```text
01 检查原始数据 → 02 转换格式 → 03 计算统计 → 04 绘图
05 日历可视化可独立运行
```

原始文件放在 `data/m5-forecasting-accuracy/`。输入包括销量训练表、销量扩展表、日历、每周价格和提交模板，具体文件清单见[项目说明](../../README.md)。数据审计检查五个文件是否齐全，实际审计销量训练表、日历和价格的内容。

## 运行方式

在项目根目录的 Windows PowerShell 中执行：

```powershell
.\.venv\Scripts\python.exe code/data_prep/01_audit_raw.py
.\.venv\Scripts\python.exe code/data_prep/02_prepare_clean.py
.\.venv\Scripts\python.exe code/data_prep/03_summarize_eda.py
.\.venv\Scripts\python.exe code/data_prep/04_plot_eda.py
.\.venv\Scripts\python.exe code/data_prep/05_plot_calendar.py
```

环境安装见[项目说明](../../README.md#环境与运行)。脚本按项目内的位置定位输入、输出目录。

## 各脚本的作用

| 脚本 | 处理内容 | 主要输出 |
|---|---|---|
| `01_audit_raw.py` | 检查销量列结构、重复 ID、异常数值、分类归属、日期轴及价格键 | `reports/raw_audit.json` |
| `02_prepare_clean.py` | 根据审计结果转换销量类型，拆分身份信息，规范日历并压缩价格表 | `data/processed/`；`reports/cleaning_summary.json` |
| `03_summarize_eda.py` | 汇总每日、星期、品类、州及单条序列的销量统计，选择展示案例 | `tables/*.csv`；`reports/eda_summary.json` |
| `04_plot_eda.py` | 从统计表生成七组探索图 | `figures/*.png`、`figures/*.svg` |
| `05_plot_calendar.py` | 从原始日历展示事件和 SNAP 日期分布 | `calendar/figures/` |
| `plot_style.py` | 提供共用配色、字体、坐标轴和图形导出函数 | 由绘图脚本调用 |

表中的 `reports/`、`tables/`、`figures/` 均位于 `outputs/data_prep/` 下。

## 处理后数据

`02_prepare_clean.py` 读取审计报告，生成：

| 文件 | 含义 |
|---|---|
| `sales_matrix.npy` | `30490 × 1913` 的非负整数销量矩阵，现有数据采用 `uint16` |
| `sales_metadata.csv` | 矩阵每行对应的 ID、商品、部门、品类、门店和州 |
| `calendar_clean.csv` | 保留完整日历，增加数字编号 `d_num` 和事件标记 `has_event` |
| `sell_prices_clean.csv.gz` | 商品×门店×周的价格记录，使用 gzip 压缩 |
| `manifest.json` | 文件位置、矩阵形状、类型及行列顺序 |

矩阵第 `i` 行对应身份表第 `i` 行，第 `j` 列对应 `d_{j+1}`。销量范围为 `d_1` 至 `d_1913`；日历保留至 `d_1969`。

处理保留零销量和空事件字段，价格保持每周粒度，未出现的商品×门店×周记录不补值。销量分块读写为磁盘映射数组，原始 CSV 保留在输入目录。

## 探索性统计

`03_summarize_eda.py` 生成以下表：

| 表 | 内容 |
|---|---|
| `daily_summary.csv` | 每日总销量、零销量序列比例、28 天滚动均值 |
| `weekday_summary.csv` | 各星期的平均和中位总销量、观测天数 |
| `category_summary.csv`、`state_summary.csv` | 分组序列数、总销量、日均销量、每序列日均销量和零值比例 |
| `category_daily.csv`、`state_daily.csv` | 各组每日总销量与每序列销量 |
| `series_summary.csv` | 各序列全历史和最近 365 天的销量水平、零销量比例等 |
| `example_series_selection.csv`、`example_series_daily.csv` | 三个品类的展示序列及最近 365 天真实销量 |
| `example_price_series.csv` | 一条商品门店序列的周价格轨迹 |

每个品类的展示序列先筛选最近 365 天覆盖的各自然月均有正销量的候选，再按日均销量和零值比例与候选组中位数的标准化距离选择。并列时按商品、门店和 ID 排序。

价格示例从 240 条等距候选序列中，优先选取不同价格数量最多、记录周数更多的序列，用于展示价格变化。

## 图形输出

`04_plot_eda.py` 生成七组 PNG/SVG：

1. 每日总销量及 28 天滚动走势。
2. 星期销量模式。
3. 三个品类的每序列销量走势。
4. 各州的每序列销量走势。
5. 各品类序列的零销量比例分布。
6. 商品门店的价格轨迹。
7. 每个品类一条商品门店序列的最近一年销量。

只调整图形时，重跑 04 即可。

`05_plot_calendar.py` 直接读取原始 `calendar.csv`，生成三组 PNG/SVG：

- `01_training_calendar_overview`：训练期月度事件分布和各州 SNAP 标记比例。
- `02_training_28_day_example`：2015-04-25 至 2015-05-22 的日历示例。
- `03_forecast_28_days`：`d_1914` 至 `d_1941`，即 2016-04-25 至 2016-05-22 的预测期日历。

数据探索使用完整训练历史；后续[滚动预测实验](../evaluation/README.md)按各起点截取历史并计算模型输入。
