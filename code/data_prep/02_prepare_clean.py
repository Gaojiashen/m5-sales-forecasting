'Convert audited M5 CSV files into analysis-ready intermediate data.\nRun after 01_audit_raw.py passes. Matrix rows match sales_metadata.csv; columns are d_1 through d_1913.\nInputs: raw_audit.json and the raw training sales, calendar, and price CSVs.\nOutputs in data/processed/: sales_matrix.npy, sales_metadata.csv, calendar_clean.csv, sell_prices_clean.csv.gz, and manifest.json.\nProcessing summary: outputs/data_prep/reports/cleaning_summary.json.'

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "m5-forecasting-accuracy"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "data_prep" / "reports"
AUDIT_PATH = REPORTS / "raw_audit.json"
SUMMARY_PATH = REPORTS / "cleaning_summary.json"
DAY_COLUMNS = [f"d_{day}" for day in range(1, 1914)]
ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def check_audit() -> dict:
    'Check the previous audit and its recorded issues.'
    if not AUDIT_PATH.is_file():
        raise FileNotFoundError(f"Run 01_audit_raw.py first: {AUDIT_PATH}")
    report = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if report.get("issues"):
        raise ValueError(f"Raw-data audit failed: {report['issues']}")
    if report["sales"]["rows"] != 30_490 or report["sales"]["days"] != 1913:
        raise ValueError('Audited sales dimensions differ from the expected dataset')
    return report


def prepare_sales(audit: dict) -> dict:
    'Store sales as a NumPy array and series identities as CSV.'
    max_sale = audit["sales"]["maximum_sale"]
    if max_sale is None or max_sale > np.iinfo(np.uint32).max:
        raise ValueError('Maximum sales exceed the supported integer range')
    # Choose the smallest unsigned integer dtype that fits nonnegative sales.
    storage_dtype = np.uint16 if max_sale <= np.iinfo(np.uint16).max else np.uint32
    # uint16 stores values from 0 to 65535.
    n_rows, n_days = 30_490, len(DAY_COLUMNS)
    matrix_path = PROCESSED / "sales_matrix.npy"
    matrix = np.lib.format.open_memmap(
        matrix_path, mode="w+", dtype=storage_dtype, shape=(n_rows, n_days)
    )
    # Create the memory-mapped NPY file.
    metadata_parts: list[pd.DataFrame] = []
    row_start = 0
    read_dtype = {name: "float32" for name in DAY_COLUMNS}
    for chunk in pd.read_csv(
        RAW / "sales_train_validation.csv", dtype=read_dtype, chunksize=256
    ):
        values = chunk[DAY_COLUMNS].to_numpy(dtype=np.float32, copy=False)
        row_end = row_start + len(chunk)
        matrix[row_start:row_end] = values.astype(storage_dtype)
        metadata_parts.append(chunk[ID_COLUMNS].copy())
        row_start = row_end
    matrix.flush()
    # Flush pending array writes to disk.
    del matrix
    # Release the matrix reference.

    if row_start != n_rows:
        raise ValueError(f"Sales row count changed: read {row_start}; expected {n_rows}")
    metadata = pd.concat(metadata_parts, ignore_index=True)
    if metadata["id"].duplicated().any():
        raise ValueError('Duplicate sales-series IDs; audit the raw data again')
    metadata.to_csv(PROCESSED / "sales_metadata.csv", index=False)
    return {"rows": row_start, "days": n_days, "matrix_dtype": np.dtype(storage_dtype).name}


def prepare_calendar() -> dict:
    'Parse dates, extract numeric day IDs, convert SNAP flags, and add an event indicator.'
    calendar = pd.read_csv(RAW / "calendar.csv")
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise")
    calendar["d_num"] = calendar["d"].str.extract(r"^d_(\d+)$")[0].astype("int32")
    calendar = calendar.sort_values("d_num").reset_index(drop=True)
    # Convert existing SNAP flags to small integers.
    for state in ("CA", "TX", "WI"):
        column = f"snap_{state}"
        calendar[column] = calendar[column].astype("int8")
    calendar["has_event"] = calendar[["event_name_1", "event_name_2"]].notna().any(axis=1).astype("int8")
    calendar.to_csv(PROCESSED / "calendar_clean.csv", index=False, date_format="%Y-%m-%d")
    return {
        "rows": len(calendar),
        "training_start": calendar.loc[calendar["d_num"] == 1, "date"].iloc[0].date().isoformat(),
        "training_end": calendar.loc[calendar["d_num"] == 1913, "date"].iloc[0].date().isoformat(),
        "days_with_event_in_training": int(calendar.loc[calendar["d_num"] <= 1913, "has_event"].sum()),
    }


def prepare_prices() -> dict:
    'Preserve prices at their original item-store-week granularity.'
    prices = pd.read_csv(
        RAW / "sell_prices.csv",
        dtype={"store_id": "category", "item_id": "category", "wm_yr_wk": "int32", "sell_price": "float32"},
    )
    prices.to_csv(PROCESSED / "sell_prices_clean.csv.gz", index=False, compression="gzip")
    # Write compressed CSV; pandas can read gzip directly.
    return {"rows": len(prices), "minimum_price": float(prices["sell_price"].min()),
            "maximum_price": float(prices["sell_price"].max())}


def main() -> None:
    audit = check_audit()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    # Remove the old manifest before rebuilding intermediate outputs.
    manifest_path = PROCESSED / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    print('1/3 Preparing sales matrix and metadata...')
    sales = prepare_sales(audit)
    print('2/3 Preparing dates and events...')
    calendar = prepare_calendar()
    print('3/3 Preparing weekly prices...')
    prices = prepare_prices()
    summary = {
        "sales": sales, "calendar": calendar, "prices": prices,
        "decisions": [
            'Preserve zero sales and all zero-sales days',
            'Preserve empty calendar event names as no recorded event'
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "source": "data/m5-forecasting-accuracy/sales_train_validation.csv",
        "sales_matrix": "data/processed/sales_matrix.npy",
        "sales_metadata": "data/processed/sales_metadata.csv",
        "calendar": "data/processed/calendar_clean.csv",
        "sell_prices": "data/processed/sell_prices_clean.csv.gz",
        "matrix_shape": [sales["rows"], sales["days"]],
        "matrix_dtype": sales["matrix_dtype"],
        "row_order": 'Row order in sales_metadata.csv',
        "column_order": "d_1, d_2, ..., d_1913",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Processed data written: {PROCESSED}")
    print(f"Processing summary written: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
