# M5 Forecasting Experiments

This directory compares a sequence of modifications from seasonal naive to an information-borrowing mixture for 28-day sales forecasting. Evaluation covers all 30,490 item-store series, using 12-level WRMSSE.

## Setup and execution

Follow [data preparation](../data_prep/README.md) to create `data/processed/sales_matrix.npy` and `sales_metadata.csv`. Experiments also read the raw calendar, prices, and sales. See [environment setup](../../README.md#environment-and-data).

Run from the project root in Windows PowerShell:

~~~powershell
.\.venv\Scripts\python.exe code/evaluation/01_snaive_wrmsse.py
.\.venv\Scripts\python.exe code/evaluation/02_select_x_wrmsse.py
.\.venv\Scripts\python.exe code/evaluation/03_visualize_x_selection.py
.\.venv\Scripts\python.exe code/evaluation/04_select_rho_wrmsse.py
.\.venv\Scripts\python.exe code/evaluation/05_grouped_trend_wrmsse.py
.\.venv\Scripts\python.exe code/evaluation/06_calendar_effect_wrmsse.py
.\.venv\Scripts\python.exe code/evaluation/07_sparse_zero_diagnostic.py
.\.venv\Scripts\python.exe code/evaluation/08_borrow_information_wrmsse.py
.\.venv\Scripts\python.exe code/evaluation/09_build_kaggle_submission.py
~~~

~~~text
01 sNaive and hierarchy scoring
02 Select x  -> 03 Window-selection figures
       |
04 Select grouped rho -> 05 Trend comparison
       |
06 Holiday and SNAP adjustments -> 07 Sparse-series diagnostics
       |
08 Information-borrowing mixture
       |
09 Submission export
~~~

03 reads results from 02; 05 uses parameters from 04; 07 uses saved parameters from 04 and 06. The final BLX path runs 01, 02, 04, 06, 08, and 09. Visualization and diagnostic branches can be run separately.

## Splits and scoring

| Period | Role |
|---|---|
| `d_1` through `d_1913` | Sales history and rolling experiments |
| `d_1914` through `d_1941` | Local holdout |
| `d_1942` through `d_1969` | Submission evaluation forecasts |

The 64 rolling origins follow `d_121 + 28k` through `d_1885`. Each origin predicts the following 28 days.

WRMSSE aggregates leaf predictions into 12 levels with 42,840 nodes. A node's RMSSE denominator is the mean squared first difference after its first nonzero historical sale. Weights use revenue from the 28 days preceding the origin; the final score averages all 12 levels equally.

At early origins, rolling selection excludes zero-scale nodes and renormalizes the remaining revenue weights within each level. Excluded shares are recorded in `origin_metric_coverage.csv`. Holdout scoring uses the full hierarchy implementation in 01.

## 01: Seasonal naive

`01_snaive_wrmsse.py` repeats `d_1907` through `d_1913` four times to forecast 28 days. It also provides reusable hierarchy aggregation, scales, revenue weights, and holdout scoring.

Output directory: `outputs/evaluation/snaive/`.

- `snaive_predictions.csv`: leaf predictions with `_validation` IDs.
- `snaive_wrmsse_summary.json` and `snaive_wrmsse_by_level.csv`: total and per-level scores.
- Daily-total, example-series, and hierarchy-error tables and figures.

## 02-03: Averaging-window selection and visualization

`02_select_x_wrmsse.py` compares same-weekday windows:

~~~text
x = {1, 2, 3, 4, 5, 6, 8, 12, 16}
~~~

For example, a Monday forecast averages up to x historical Mondays. The first price record determines release; post-release zeros count in the mean.

The experiment compares shared x values and all 729 vectors of separate FOODS, HOBBIES, and HOUSEHOLD windows.

| Stage | Origins | Purpose |
|---|---:|---|
| Development | 32 | Search the best category vector and shared window |
| Confirmation | 16 | Compare the two selected candidates and x=1 |
| Audit | 16 | Evaluate the adopted choice and fixed controls |

The lowest confirmation mean wins; ties prefer x=1, then shared x, then the category vector. Existing results adopt shared `x=8`.

`03_visualize_x_selection.py` plots shared-window curves, category vectors, stage comparisons, paired x=5 versus x=8 differences, and an item example. Its first run computes and caches x=5 confirmation scores.

Outputs: `outputs/evaluation/x_selection/`, particularly `selection_summary.json`, `development_vector_means.csv`, `rolling_origin_scores.csv`, `selected_vector_split_scores.csv`, and `figures/`.

## 04: Recency factors and grouping

`04_select_rho_wrmsse.py` fixes x=8 and uses normalized weights proportional to `rho^k`, with k=0 for the newest observation.

rho=0 uses the latest eligible matching weekday; rho=1 averages eight weeks equally. The search grid is `{0, .25, .5, .65, .75, .8, .85, .9, .95, 1}`.

Coordinate descent searches global, state, state-category, and store-category groupings. Four expanding folds fit on the first 32, 40, 48, or 56 origins and validate on the following eight. Paired loss differences and a moving-block bootstrap standard error define the one-standard-error simplicity rule.

Existing results select nine state-category rho values, refitted on all 64 origins.

Outputs: `outputs/evaluation/rho_selection/`, including fold/final parameters, selection summaries, validation scores, forecasts, and holdout hierarchy scores.

Resume selection and final refitting after completing the four folds:

~~~powershell
.\.venv\Scripts\python.exe code/evaluation/04_select_rho_wrmsse.py --resume-selection
~~~

## 05: Additional trend comparison

`05_grouped_trend_wrmsse.py` aggregates eight complete weeks in each of nine state-category groups and estimates rho-weighted least-squares slopes. It compares constant, linear, and damped levels. Group-level ratios scale the existing leaf weekday profiles; negative extrapolated levels are clipped to zero.

Development selects phi, confirmation compares three candidates, and audit records their scores. Existing results select the constant model.

Outputs: `outputs/evaluation/grouped_trend/`, including rolling scores, split summaries, group trend components, weekday effects, forecasts, and figures.

Options:

- `--reuse-rolling-scores`: rebuild final-origin outputs from saved rolling scores.
- `--skip-final-holdout`: stop after forecast generation, skipping holdout scoring.

## 06: Holiday and SNAP adjustments

`06_calendar_effect_wrmsse.py` compares `baseline`, `holiday_only`, `snap_only`, and `both` on the grouped-rho profile.

The calendar multiplier is `exp(beta_g * SNAP + sum(gamma_e * event_e))`. There are nine state-category SNAP coefficients and four shared event coefficients: Pesach End, OrthodoxEaster, Cinco De Mayo, and Mother's day.

Holiday models omit historical event dates from reference means. SNAP models remove historical SNAP effects before averaging and restore the target-day multiplier.

Parameters minimize group-total Poisson negative log likelihood. Log multipliers are bounded to `[-0.7, 0.7]`; the SNAP ridge penalty is 2, and each event penalty scales with its baseline fitting-date exposure. Additional event-penalty sensitivity coefficients are saved.

The four folds match rho selection and use fold-specific rho values. Mean WRMSSE over 32 validation origins selects the candidate; ties prefer baseline, holiday_only, snap_only, then both. Existing results select both.

Outputs: `outputs/evaluation/calendar_effect/`, including rolling scores, fold/final coefficients, four forecast tables, holdout scores, and figures.

Use `--smoke` for a one-origin real-data check. It writes to the same output directory.

## 07: Zero-day and sparse-series diagnostics

`07_sparse_zero_diagnostic.py` reconstructs fold-specific both and rho-baseline forecasts. Cohorts use release-aware history:

- No release history.
- Fewer than 28 released days.
- Mature series with below 50%, 50-80%, or at least 80% zeros over the preceding 56 days.

Outputs include zero-day predictions exceeding 0, 0.5, and 1; zero-day squared-loss contributions; and nonzero-day underprediction, missed sales, and means. For high-sparsity series, forecast bands describe calibration, and the top forecast decile is compared with the overall frequency of high-sales days.

Outputs: `outputs/evaluation/sparse_zero_diagnostic/`, including per-origin and pooled tables, calibration bands, and two figures.

## 08: Other-weekday and cross-store information

`08_borrow_information_wrmsse.py` compares:

| Component | Information |
|---|---|
| B | The holiday-plus-SNAP same-weekday model from 06 |
| L_perp | Released non-event days over 56 days for the same item-store series, excluding the target weekday |
| X | The same item's sales in other stores over 56 days |

L_perp removes SNAP and weekday effects before restoring the target weekday and calendar factors. Store-category weekday multipliers use 364 days and 28 pseudo-days of shrinkage.

X converts de-calendarized cross-store rates using historical volume and a store-share prior, with `kappa={5,20,80}` and an odds cap of 20. Components without sufficient history fall back to B.

Mixture bins distinguish 0, 1-2, or at least 3 positive observations among eight matching weekdays. Each bin's three weights sum to one. B, BL, BX, and BLX are compared by fitting weights to training-origin hierarchy WRMSSE and selecting components and kappa by four-fold validation. The selected mixture is refitted on all 64 origins.

Existing results select `BLX` and `kappa=80`.

Outputs: `outputs/evaluation/borrow_information/`. Key files:

- `validation_candidate_summary.csv`: validation means.
- `final_mixture_weights.csv`: final nine weights.
- `forecast_borrow_information.csv`: 28-day leaf forecasts.
- `experiment_summary.json`: selected configuration and holdout score.

## 09: Submission export

`09_build_kaggle_submission.py` reads parameters and predictions from 04, 06, and 08:

1. Reuse `d_1914`-`d_1941` validation forecasts.
2. Append newly observed sales and advance the origin to `d_1941`.
3. Keep selected rho, calendar coefficients, kappa, and mixture weights to predict `d_1942`-`d_1969`.
4. Align both parts with the template IDs and validate the layout.

Output: `outputs/submission/m5_accuracy_submission.csv`, with 60,980 rows and columns `id,F1,...,F28`. Full predictions and the submission are generated locally; selected scores, parameters, and figures are indexed in [results](../../outputs/README.md).

## Saved results

These are 12-level WRMSSE scores on the same local holdout, `d_1914` through `d_1941`.

| Method | WRMSSE |
|---|---:|
| sNaive | 0.86970 |
| State-category rho | 0.73699 |
| Rho + holidays | 0.71941 |
| Rho + SNAP | 0.68630 |
| Rho + holidays + SNAP | 0.66136 |
| BLX | 0.65357 |

Values come from saved JSON summaries. Rolling validation selects candidates; the holdout describes their performance on the final 28-day period.
