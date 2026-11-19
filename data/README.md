# Downloading and Placing M5 Data

Download the dataset from the [Kaggle M5 Forecasting - Accuracy data page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data), following its sign-in and access steps. Extract the files into:

~~~text
data/
├─ README.md
└─ m5-forecasting-accuracy/
   ├─ sales_train_validation.csv
   ├─ sales_train_evaluation.csv
   ├─ calendar.csv
   ├─ sell_prices.csv
   └─ sample_submission.csv
~~~

Raw CSVs, archives, and processed matrices stay local. The repository supplies download instructions and processing code.

After installing dependencies, run from the project root:

~~~powershell
.\.venv\Scripts\python.exe code/data_prep/01_audit_raw.py
.\.venv\Scripts\python.exe code/data_prep/02_prepare_clean.py
~~~

Step 02 generates `data/processed/` with the sales matrix, metadata, calendar, compressed prices, and manifest. Continue with [data preparation](../code/data_prep/README.md).
