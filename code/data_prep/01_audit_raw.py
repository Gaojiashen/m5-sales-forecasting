'Audit raw M5 data and write a data-quality report.\nRun this first; inspect outputs/data_prep/reports/raw_audit.json for issues.\nSales auditing uses sales_train_validation.csv.'

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "m5-forecasting-accuracy"
REPORT_DIR = ROOT / "outputs" / "data_prep" / "reports"
REPORT_PATH = REPORT_DIR / "raw_audit.json"

ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
DAY_COLUMNS = [f"d_{day}" for day in range(1, 1914)]
PRICE_KEY = ["store_id", "item_id", "wm_yr_wk"]


def require_raw_files() -> dict[str, Path]:
    'Check that the required raw files exist.'
    names = [
        "sales_train_validation.csv",
        "sales_train_evaluation.csv",
        "calendar.csv",
        "sell_prices.csv",
        "sample_submission.csv",
    ]
    paths = {name: RAW / name for name in names}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing raw files: {missing}; expected directory: {RAW}")
    return paths


def audit_sales(path: Path, issues: list[str]) -> dict:
    'Audit sales columns, duplicate IDs, invalid values, and item/department/category or store/state assignment conflicts.'
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    # Read only the header and convert column names to a list.
    expected = ID_COLUMNS + DAY_COLUMNS
    if columns != expected:
        issues.append('Sales columns or their order differ from the expected layout')
        return {"columns_found": len(columns), "expected_columns": len(expected)}

    dtype = {name: "float32" for name in DAY_COLUMNS}
    # Read all day columns as float32.
    row_count = missing_cells = infinite_cells = negative_cells = fractional_cells = 0
    # Accumulate row, missing, infinite, negative, and fractional-value counts.
    zero_cells = 0
    # Count zero-sales cells.
    minimum, maximum = float("inf"), float("-inf")
    # Initialize global extrema.
    seen_ids: set[str] = set()
    duplicate_ids = 0
    # Count duplicate IDs.
    stores, categories, states = Counter(), Counter(), Counter()
    # Create counters updated from each chunk.
    metadata_parts: list[pd.DataFrame] = []
    # Retain metadata to check hierarchy assignments.
    # Check item-to-department, department-to-category, and store-to-state assignments.

    for chunk in pd.read_csv(path, dtype=dtype, chunksize=256):
        # Read 256 rows per chunk with explicit sales dtypes.
        row_count += len(chunk)
        ids = chunk["id"].astype(str).tolist()
        # Convert the ID Series to a list of strings.
        duplicate_ids += len(ids) - len(set(ids)) # Count duplicates within this chunk.
        duplicate_ids += len(set(ids) & seen_ids) # Count duplicates shared with earlier chunks.
        seen_ids.update(ids)
        stores.update(chunk["store_id"].astype(str))
        categories.update(chunk["cat_id"].astype(str))
        states.update(chunk["state_id"].astype(str))
        metadata_parts.append(chunk[ID_COLUMNS].copy())
        # Copy metadata independently of the current chunk.

        values = chunk[DAY_COLUMNS].to_numpy(dtype=np.float32, copy=False)
        # Extract a NumPy array without DataFrame labels.
        # Reuse the existing array storage where possible.
        missing_cells += int(np.isnan(values).sum())
        # Create elementwise NaN and infinity masks.
        infinite_cells += int(np.isinf(values).sum())
        finite = values[np.isfinite(values)]
        # Select finite values using a Boolean mask.
        if finite.size:
            negative_cells += int((finite < 0).sum())
            fractional_cells += int((finite != np.floor(finite)).sum())
            # Floor values to detect fractional sales.
            zero_cells += int((finite == 0).sum())
            minimum = min(minimum, float(finite.min()))
            maximum = max(maximum, float(finite.max()))

    metadata = pd.concat(metadata_parts, ignore_index=True)
    # Concatenate metadata with a fresh sequential index.
    item_to_dept_violations = int(
        (metadata.groupby("item_id")["dept_id"].nunique() > 1).sum()
    ) # Count distinct department assignments per item.
    dept_to_cat_violations = int(
        (metadata.groupby("dept_id")["cat_id"].nunique() > 1).sum()
    )
    store_to_state_violations = int(
        (metadata.groupby("store_id")["state_id"].nunique() > 1).sum()
    )

    if row_count != 30_490: # Expected full-panel dataset size.
        issues.append(f"Sales row count is {row_count}; expected 30,490")
    if duplicate_ids:
        issues.append(f"Found {duplicate_ids} duplicate sales-series IDs")
    if missing_cells or infinite_cells or negative_cells or fractional_cells:
        issues.append('Sales contain missing, infinite, negative, or fractional values; inspect the report')
    if item_to_dept_violations or dept_to_cat_violations or store_to_state_violations:
        issues.append('An item, department, or store has conflicting parent assignments')

    total_cells = row_count * len(DAY_COLUMNS)
    return {
        "rows": row_count,
        "columns": len(columns),
        "days": len(DAY_COLUMNS),
        "unique_ids": len(seen_ids),
        "duplicate_ids": duplicate_ids,
        "missing_sales_cells": missing_cells,
        "infinite_sales_cells": infinite_cells,
        "negative_sales_cells": negative_cells,
        "fractional_sales_cells": fractional_cells,
        "zero_sales_ratio": round(zero_cells / total_cells, 6),
        "minimum_sale": None if minimum == float("inf") else minimum,
        "maximum_sale": None if maximum == float("-inf") else maximum,
        "stores": dict(sorted(stores.items())),
        "categories": dict(sorted(categories.items())),
        "states": dict(sorted(states.items())),
        "item_to_dept_violations": item_to_dept_violations,
        "dept_to_cat_violations": dept_to_cat_violations,
        "store_to_state_violations": store_to_state_violations,
    }


def audit_calendar(path: Path, issues: list[str]) -> dict:
    'Check that dates and day identifiers are complete, unique, and consecutive.'
    calendar = pd.read_csv(path)
    expected = {"date", "d", "wm_yr_wk", "wday", "event_name_1", "snap_CA", "snap_TX", "snap_WI"}
    missing_columns = sorted(expected - set(calendar.columns))
    # Sort missing column names for stable output.
    if missing_columns:
        issues.append(f"Missing calendar columns: {missing_columns}")
        return {"rows": len(calendar), "missing_columns": missing_columns}

    dates = pd.to_datetime(calendar["date"], errors="coerce")
    # Parse date strings; invalid or missing values become NaT.
    day_numbers = pd.to_numeric(calendar["d"].str.extract(r"^d_(\d+)$")[0], errors="coerce")
    # Extract the numeric part of d_1913; unmatched values become NaN.
    # Select the first extraction column as a Series.
    ordered = pd.DataFrame({"day": day_numbers, "date": dates}).sort_values("day")
    # Align day identifiers with dates and sort by day number.
    missing_training_days = sorted(set(range(1, 1914)) - set(day_numbers.dropna().astype(int)))
    duplicate_days = int(calendar["d"].duplicated().sum())
    duplicate_dates = int(dates.duplicated().sum())
    discontinuities = int((ordered["date"].diff().dt.days.dropna() != 1).sum())
    # Convert consecutive date differences to day counts.
    # The first difference is NaN because no previous date exists.
    if dates.isna().any() or day_numbers.isna().any() or missing_training_days:
    # any() returns true if at least one value is true.
        issues.append('Calendar dates or day identifiers are missing or invalid')
    if duplicate_days or duplicate_dates or discontinuities:
        issues.append('Calendar dates are duplicated or not consecutive')

    return {
        "rows": len(calendar),
        "first_date": None if dates.isna().all() else dates.min().date().isoformat(),
        "last_date": None if dates.isna().all() else dates.max().date().isoformat(),
        "duplicate_days": duplicate_days,
        "duplicate_dates": duplicate_dates,
        "discontinuities": discontinuities,
        "missing_training_days_count": len(missing_training_days),
        "event_name_1_nonempty": int(calendar["event_name_1"].notna().sum()),
    }


def audit_prices(path: Path, issues: list[str]) -> dict:
    'Inspect weekly item-store prices; absent records may have a business meaning.'
    prices = pd.read_csv(
        path,
        dtype={"store_id": "category", "item_id": "category", "wm_yr_wk": "int32", "sell_price": "float32"},
        # Use categorical dtypes for repeated item and store labels.
        # Categories are stored as labels plus integer codes.
    )
    missing = {name: int(prices[name].isna().sum()) for name in PRICE_KEY + ["sell_price"]}
    duplicate_keys = int(prices.duplicated(PRICE_KEY).sum())
    nonpositive_prices = int((prices["sell_price"] <= 0).sum())
    if any(missing.values()) or duplicate_keys or nonpositive_prices:
        issues.append('Prices contain missing fields, duplicate keys, or nonpositive values')
    return {
        "rows": len(prices),
        "unique_price_keys": len(prices) - duplicate_keys,
        "duplicate_price_keys": duplicate_keys,
        "missing": missing,
        "nonpositive_prices": nonpositive_prices,
        "unique_stores": int(prices["store_id"].nunique()),
        "unique_items": int(prices["item_id"].nunique()),
        "minimum_price": float(prices["sell_price"].min()),
        "maximum_price": float(prices["sell_price"].max()),
    }


def main() -> None:
    paths = require_raw_files()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    issues: list[str] = []
    print('1/3 Auditing sales...')
    sales = audit_sales(paths["sales_train_validation.csv"], issues)
    print('2/3 Auditing calendar...')
    calendar = audit_calendar(paths["calendar.csv"], issues)
    print('3/3 Auditing prices...')
    prices = audit_prices(paths["sell_prices.csv"], issues)
    report = {"source_directory": str(RAW), "sales": sales, "calendar": calendar,
              "prices": prices, "issues": issues}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report written: {REPORT_PATH}")
    if issues:
        raise ValueError('Raw-data checks failed; inspect issues in raw_audit.json')
    print('Raw-data checks passed. Continue with 02_prepare_clean.py.')


if __name__ == "__main__":
    main()
