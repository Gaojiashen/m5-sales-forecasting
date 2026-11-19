# M5 Data Preparation and Exploration

This directory converts raw CSVs into analysis-ready intermediate files and produces exploratory tables and figures.

~~~text
01 Audit raw data -> 02 Convert formats -> 03 Compute summaries -> 04 Plot
05 Calendar visualization can run independently
~~~

Place the raw files in `data/m5-forecasting-accuracy/`. Inputs include training sales, extended sales, calendar, weekly prices, and the submission template; see the [project README](../../README.md). The audit checks that all five files exist and inspects training sales, calendar, and prices.

## Running the scripts

From the project root in Windows PowerShell:

~~~powershell
.\.venv\Scripts\python.exe code/data_prep/01_audit_raw.py
.\.venv\Scripts\python.exe code/data_prep/02_prepare_clean.py
.\.venv\Scripts\python.exe code/data_prep/03_summarize_eda.py
.\.venv\Scripts\python.exe code/data_prep/04_plot_eda.py
.\.venv\Scripts\python.exe code/data_prep/05_plot_calendar.py
~~~

See [environment setup](../../README.md#environment-and-data). Scripts locate inputs and outputs relative to the project root.

## Script roles

| Script | Work | Main outputs |
|---|---|---|
| `01_audit_raw.py` | Check sales layout, duplicate IDs, invalid values, hierarchy assignments, date continuity, and price keys | `reports/raw_audit.json` |
| `02_prepare_clean.py` | Convert sales dtypes, separate metadata, normalize dates, and compress prices | `data/processed/`; `reports/cleaning_summary.json` |
| `03_summarize_eda.py` | Summarize days, weekdays, categories, states, and individual series; select examples | `tables/*.csv`; `reports/eda_summary.json` |
| `04_plot_eda.py` | Draw seven exploratory figure pairs from summary tables | `figures/*.png` and `figures/*.svg` |
| `05_plot_calendar.py` | Display event and SNAP flags from the raw calendar | `calendar/figures/` |
| `plot_style.py` | Shared colors, fonts, axes, and export functions | Imported by plotting scripts |

The `reports/`, `tables/`, and `figures/` paths in the table are under `outputs/data_prep/`.

## Prepared data

`02_prepare_clean.py` reads the audit report and produces:

| File | Meaning |
|---|---|
| `sales_matrix.npy` | A `30490 x 1913` nonnegative integer matrix; the existing data use `uint16` |
| `sales_metadata.csv` | Row-aligned IDs, items, departments, categories, stores, and states |
| `calendar_clean.csv` | Full calendar with numeric day ID `d_num` and event flag `has_event` |
| `sell_prices_clean.csv.gz` | Gzip-compressed item-store-week prices |
| `manifest.json` | File locations, matrix shape, dtype, and row/column order |

Matrix row `i` corresponds to metadata row `i`; column `j` corresponds to `d_{j+1}`. Sales span `d_1` through `d_1913`, while the calendar extends through `d_1969`.

Preparation preserves observed zeros and empty event fields. Prices remain weekly; absent item-store-week records are not imputed. Sales are written in chunks to a memory-mapped array, and raw CSVs remain in the input directory.

## Exploratory statistics

`03_summarize_eda.py` produces:

| Table | Contents |
|---|---|
| `daily_summary.csv` | Daily total units, zero-series share, and 28-day rolling mean |
| `weekday_summary.csv` | Mean and median total units and observation counts by weekday |
| `category_summary.csv`, `state_summary.csv` | Group sizes, total and average daily sales, per-series means, and zero shares |
| `category_daily.csv`, `state_daily.csv` | Daily totals and per-series units for each group |
| `series_summary.csv` | Full-history and last-365-day sales levels and sparsity |
| `example_series_selection.csv`, `example_series_daily.csv` | Three category examples and their last 365 observed days |
| `example_price_series.csv` | A weekly item-store price trajectory |

A representative series must have positive sales in every calendar month covered by the last 365 days. Within each category, selection minimizes standardized absolute distances from the eligible pool's median sales level and zero share. Ties are resolved by item, store, and ID.

The price example is chosen from 240 evenly spaced candidate series, preferring the most distinct prices and then the most observed weeks.

## Figures

`04_plot_eda.py` produces seven PNG/SVG pairs:

1. Daily totals and a 28-day rolling trend.
2. Weekday sales patterns.
3. Per-series category sales profiles.
4. Per-series state sales profiles.
5. Within-category zero-sales share distributions.
6. An item-store price trajectory.
7. One recent-year item-store history per category.

Rerun 04 when changing figure styling.

`05_plot_calendar.py` reads raw `calendar.csv` and produces:

- `01_training_calendar_overview`: monthly event and SNAP-flag frequencies.
- `02_training_28_day_example`: the calendar from 2015-04-25 to 2015-05-22.
- `03_forecast_28_days`: `d_1914` through `d_1941`, or 2016-04-25 to 2016-05-22.

Exploration uses the full training history. The subsequent [rolling experiments](../evaluation/README.md) build model inputs from history truncated at each origin.
