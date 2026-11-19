# Selected Experiment Results

The uploaded version retains statistical summaries, scores, parameter tables, and representative PNGs. Full leaf predictions, submission CSVs, per-series exploratory tables, and other figures can be regenerated with the corresponding scripts. JSON and CSV files describe saved results; see [forecasting experiments](../code/evaluation/README.md) for reproduction.

## Data exploration

![Daily total sales and rolling trend](data_prep/figures/01_sales_overview.png)

![Zero-sales distributions by category](data_prep/figures/05_zero_sales_by_category.png)

Data-quality reports are under `data_prep/reports/`; other exploratory plots are under `data_prep/figures/`.

## Averaging windows and recency weights

![Shared-window comparison](evaluation/x_selection/figures/01_shared_x_curve.png)

![Recency-factor grouping comparison](evaluation/rho_selection/figures/02_partition_comparison.png)

## Calendar effects and sparse sales

![Calendar-model validation scores](evaluation/calendar_effect/figures/01_model_validation_scores.png)

![Sparse-series forecast calibration](evaluation/sparse_zero_diagnostic/sparse_forecast_calibration.png)

## Information-borrowing mixture

![Information-borrowing candidates](evaluation/borrow_information/figures/01_validation_candidates.png)

![Final mixture weights](evaluation/borrow_information/figures/02_final_weights.png)

The final selection is BLX with `kappa=80`. Local holdout WRMSSE scores are:

| Method | WRMSSE |
|---|---:|
| sNaive | 0.86970 |
| State-category recency weights | 0.73699 |
| Holidays and SNAP | 0.66136 |
| BLX | 0.65357 |

Results are recorded in JSON summaries under `evaluation/snaive/`, `evaluation/rho_selection/`, `evaluation/calendar_effect/`, and `evaluation/borrow_information/`.
