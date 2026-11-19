'Step 3: compute exploratory statistics for M5.\nInputs in data/processed/: the sales matrix, metadata, calendar, weekly prices, and manifest.\nOutputs in outputs/data_prep/tables/: daily and weekday summaries; category and state summaries and daily profiles; per-series statistics; representative series and their last 365 observed days; and a weekly-price example.\nSummary: outputs/data_prep/reports/eda_summary.json.'

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "data_prep" / "tables"
REPORTS = ROOT / "outputs" / "data_prep" / "reports"


def load_inputs() -> tuple[np.memmap, pd.DataFrame, pd.DataFrame]:
    'Check matrix dimensions and alignment with metadata and calendar.'
    paths = [PROCESSED / name for name in (
        "manifest.json", "sales_matrix.npy", "sales_metadata.csv", "calendar_clean.csv"
    )]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Run 02_prepare_clean.py first; missing: {missing}")
    matrix = np.load(paths[1], mmap_mode="r")
    metadata = pd.read_csv(paths[2])
    calendar = pd.read_csv(paths[3], parse_dates=["date"])
    calendar = calendar.loc[calendar["d_num"].between(1, 1913)].sort_values("d_num")
    return matrix, metadata, calendar.reset_index(drop=True)


def summarize_daily(matrix: np.memmap, calendar: pd.DataFrame) -> pd.DataFrame:
    'Summarize daily totals, zero-sales shares, and rolling trends.'
    daily = calendar[["d_num", "date", "wm_yr_wk", "has_event"]].copy()
    daily["total_units"] = matrix.sum(axis=0, dtype=np.int64)
    daily["zero_series_share"] = (matrix == 0).mean(axis=0)
    daily["rolling_28d_mean"] = daily["total_units"].rolling(28, min_periods=28).mean()
    daily.to_csv(TABLES / "daily_summary.csv", index=False, date_format="%Y-%m-%d")
    return daily


def summarize_weekday(daily: pd.DataFrame) -> pd.DataFrame:
    'Derive weekdays from dates using Monday=0 through Sunday=6.'
    work = daily.assign(weekday_num=daily["date"].dt.weekday)
    weekday = work.groupby("weekday_num", as_index=False).agg(
        mean_units=("total_units", "mean"),
        median_units=("total_units", "median"),
        observed_days=("total_units", "size"),
    )
    weekday["weekday"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday.to_csv(TABLES / "weekday_summary.csv", index=False)
    return weekday


def summarize_group(matrix: np.memmap, metadata: pd.DataFrame,
                    calendar: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    'Summarize differences across categories or states.'
    rows = []
    all_days = []
    for label in sorted(metadata[column].dropna().unique()):
        selected = matrix[metadata[column].to_numpy() == label]
        n_series = selected.shape[0]
        day_units = selected.sum(axis=0, dtype=np.int64)
        rows.append({
            column: label,
            "series_count": n_series,
            "total_units": int(day_units.sum()),
            "mean_daily_units": float(day_units.mean()),
            "mean_daily_units_per_series": float(day_units.mean() / n_series),
            "zero_cell_share": float((selected == 0).mean()),
        })
        all_days.append(pd.DataFrame({
            "date": calendar["date"].to_numpy(),
            column: label,
            "daily_units": day_units,
            "units_per_series": day_units / n_series,
        }))
    summary = pd.DataFrame(rows).sort_values("total_units", ascending=False)
    summary.to_csv(TABLES / f"{prefix}_summary.csv", index=False)
    pd.concat(all_days, ignore_index=True).to_csv(
        TABLES / f"{prefix}_daily.csv", index=False, date_format="%Y-%m-%d"
    )
    return summary


def summarize_series(matrix: np.memmap, metadata: pd.DataFrame,
                     calendar: pd.DataFrame) -> pd.DataFrame:
    'Summarize full-history and recent-year statistics and select reproducible examples.'
    series = metadata[["id", "item_id", "cat_id", "store_id"]].copy()
    series.insert(0, "row_index", np.arange(len(metadata), dtype=np.int32))
    series["mean_daily_units"] = matrix.mean(axis=1)
    series["zero_day_share"] = (matrix == 0).mean(axis=1)
    # Use the last 365 observed days; compare the first and last 90-day totals.
    recent_start = matrix.shape[1] - 365
    recent = matrix[:, recent_start:]
    series["mean_daily_units_last365"] = recent.mean(axis=1)
    series["zero_day_share_last365"] = (recent == 0).mean(axis=1)
    series["first90_units_last365"] = recent[:, :90].sum(axis=1)
    series["last90_units_last365"] = recent[:, -90:].sum(axis=1)
    series.to_csv(TABLES / "series_summary.csv", index=False)

    # Require positive sales in every covered calendar month.
    # Then select levels and zero shares close to category medians.
    recent_dates = calendar["date"].iloc[recent_start:].reset_index(drop=True)
    month_codes = recent_dates.dt.to_period("M").astype(str).to_numpy()
    if len(month_codes) != 365:
        raise ValueError('Recent-year calendar and sales lengths differ')
    candidate_rows = []
    for cat_id in sorted(series["cat_id"].dropna().unique()):
        group = series.loc[series["cat_id"] == cat_id]
        eligible = []
        for row in group.itertuples(index=False):
            values = recent[int(row.row_index)]
            if all(np.any(values[month_codes == month] > 0) for month in np.unique(month_codes)):
                eligible.append(int(row.row_index))
        if not eligible:
            raise ValueError(f"Category {cat_id} has no series with sales in every calendar month covered by the last 365 days")
        pool = group.loc[group["row_index"].isin(eligible)].copy()
        med_mean = float(pool["mean_daily_units_last365"].median())
        med_zero = float(pool["zero_day_share_last365"].median())
        # Standardize distances and use stable sort keys to break ties.
        mean_scale = max(float(pool["mean_daily_units_last365"].std(ddof=0)), 1e-12)
        zero_scale = max(float(pool["zero_day_share_last365"].std(ddof=0)), 1e-12)
        pool["_distance"] = (
            (pool["mean_daily_units_last365"] - med_mean).abs() / mean_scale
            + (pool["zero_day_share_last365"] - med_zero).abs() / zero_scale
        )
        chosen = pool.sort_values(
            ["_distance", "item_id", "store_id", "id"], kind="mergesort"
        ).iloc[0]
        candidate_rows.append(chosen.drop(labels="_distance"))
    selected = pd.DataFrame(candidate_rows).reset_index(drop=True)
    selected.to_csv(TABLES / "example_series_selection.csv", index=False)
    daily_parts = []
    for row in selected.itertuples(index=False):
        values = np.asarray(matrix[int(row.row_index), recent_start:], dtype=np.int64)
        daily_parts.append(pd.DataFrame({
            "date": recent_dates, "category": row.cat_id, "id": row.id,
            "actual_units": values,
        }))
    pd.concat(daily_parts, ignore_index=True).to_csv(
        TABLES / "example_series_daily.csv", index=False, date_format="%Y-%m-%d"
    )
    return series


def choose_price_example(metadata: pd.DataFrame, calendar: pd.DataFrame) -> dict:
    'Choose a price-varying series from a fixed candidate sample.'
    price_path = PROCESSED / "sell_prices_clean.csv.gz"
    if not price_path.is_file():
        raise FileNotFoundError(f"Missing processed price file: {price_path}")
    positions = np.unique(np.linspace(0, len(metadata) - 1, 240, dtype=int))
    candidates = metadata.iloc[positions][["store_id", "item_id"]].drop_duplicates()
    candidate_items = set(candidates["item_id"])
    candidate_stores = set(candidates["store_id"])
    pieces = []
    for chunk in pd.read_csv(price_path, chunksize=200_000):
        subset = chunk.loc[
            chunk["item_id"].isin(candidate_items) & chunk["store_id"].isin(candidate_stores)
        ]
        if not subset.empty:
            matched = subset.merge(candidates, on=["store_id", "item_id"], how="inner")
            if not matched.empty:
                pieces.append(matched)
    if not pieces:
        raise ValueError('No price records found for the candidate series')
    prices = pd.concat(pieces, ignore_index=True)
    ranking = prices.groupby(["store_id", "item_id"]).agg(
        distinct_prices=("sell_price", "nunique"), weeks=("sell_price", "size")
    ).sort_values(["distinct_prices", "weeks"], ascending=False)
    store_id, item_id = ranking.index[0]
    example = prices.loc[(prices["store_id"] == store_id) & (prices["item_id"] == item_id)].copy()
    week_dates = calendar.groupby("wm_yr_wk", as_index=False)["date"].min()
    example = example.merge(week_dates, on="wm_yr_wk", how="inner").sort_values("date")
    example.to_csv(TABLES / "example_price_series.csv", index=False, date_format="%Y-%m-%d")
    return {
        "store_id": store_id, "item_id": item_id,
        "observed_weeks": len(example),
        "distinct_prices": int(example["sell_price"].nunique()),
        "selection": 'Most distinct prices among 240 evenly spaced candidate series; illustrative example',
    }


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    matrix, metadata, calendar = load_inputs()
    print('1/5 Summarizing daily sales...')
    daily = summarize_daily(matrix, calendar)
    print('2/5 Summarizing weekday patterns...')
    summarize_weekday(daily)
    print('3/5 Summarizing categories and states...')
    categories = summarize_group(matrix, metadata, calendar, "cat_id", "category")
    states = summarize_group(matrix, metadata, calendar, "state_id", "state")
    print('4/5 Summarizing series sparsity...')
    series = summarize_series(matrix, metadata, calendar)
    print('5/5 Selecting a price example...')
    price_example = choose_price_example(metadata, calendar)
    summary = {
        "source_days": [1, 1913],
        "series": len(metadata),
        "total_units": int(daily["total_units"].sum()),
        "overall_zero_cell_share": float(series["zero_day_share"].mean()),
        "categories": categories.to_dict(orient="records"),
        "states": states.to_dict(orient="records"),
        "price_example": price_example,
        "series_examples": {
            "count": 3,
            "selection": 'Require positive sales in every calendar month covered by the last 365 days. Among eligible series in each category, minimize the sum of standardized absolute distances from the median daily sales and zero-sales share; break ties by item_id, store_id, and id.',
            "daily_table": 'example_series_daily.csv (actual daily sales for the last 365 days)',
        },
    }
    (REPORTS / "eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Summary tables: {TABLES}")


if __name__ == "__main__":
    main()
