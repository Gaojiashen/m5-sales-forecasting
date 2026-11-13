# M5 数据下载与放置

从 [Kaggle M5 Forecasting — Accuracy 数据页](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)下载数据，按页面要求登录并获取数据包。解压后放置为：

```text
data/
├─ README.md
└─ m5-forecasting-accuracy/
   ├─ sales_train_validation.csv
   ├─ sales_train_evaluation.csv
   ├─ calendar.csv
   ├─ sell_prices.csv
   └─ sample_submission.csv
```

原始 CSV、压缩包和处理后矩阵保留在本地，仓库提供下载说明和处理代码。

在项目根目录安装依赖后，执行：

```powershell
.\.venv\Scripts\python.exe code/data_prep/01_audit_raw.py
.\.venv\Scripts\python.exe code/data_prep/02_prepare_clean.py
```

第二步生成 `data/processed/`，包括销量矩阵、元数据、日历、压缩价格表和数据清单。后续步骤见[数据准备说明](../code/data_prep/README.md)。
