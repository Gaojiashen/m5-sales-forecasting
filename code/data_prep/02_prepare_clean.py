"""
把通过审计的 M5 原始 CSV 转成便于分析的中间数据。
运行顺序：第 2 个。先运行 01_audit_raw.py，且 raw_audit.json 的 issues 为空。
销售矩阵的行与sales_metadata.csv 严格同序，列依次为 d_1 ... d_1913。
输入：
- outputs/data_prep/reports/raw_audit.json：审计结果。
- data/m5-forecasting-accuracy/：销量训练表、日历表和价格表。

输出至 data/processed/：
- sales_matrix.npy：销量矩阵，行对应序列，列对应 d_1～d_1913。
- sales_metadata.csv：与矩阵行顺序一致的商品、门店等身份信息。
- calendar_clean.csv：规范日期，新增 d_num 和 has_event。
- sell_prices_clean.csv.gz：gzip 压缩的每周价格表。
- manifest.json：文件清单及矩阵结构说明。

处理摘要：outputs/data_prep/reports/cleaning_summary.json。
"""

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
    """检查数据是否经过审计、审计是否存在问题。"""
    if not AUDIT_PATH.is_file():
        raise FileNotFoundError(f"请先运行 01_audit_raw.py：{AUDIT_PATH}")
    report = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if report.get("issues"):
        raise ValueError(f"原始数据审计未通过：{report['issues']}")
    if report["sales"]["rows"] != 30_490 or report["sales"]["days"] != 1913:
        raise ValueError("审计报告中的销量规模与本项目预期不符")
    return report


def prepare_sales(audit: dict) -> dict:
    """读取销售表，存储为Numpy数组；将身份表存储为csv"""
    max_sale = audit["sales"]["maximum_sale"]
    if max_sale is None or max_sale > np.iinfo(np.uint32).max:
        raise ValueError("销量最大值超出当前整数存储范围")
    # M5 销量是非负整数。选能容纳最大值的类型以节约磁盘和内存。
    storage_dtype = np.uint16 if max_sale <= np.iinfo(np.uint16).max else np.uint32
    # np.uint16 16位无符号整数，0-65535
    n_rows, n_days = 30_490, len(DAY_COLUMNS)
    matrix_path = PROCESSED / "sales_matrix.npy"
    matrix = np.lib.format.open_memmap(
        matrix_path, mode="w+", dtype=storage_dtype, shape=(n_rows, n_days)
    )
    # 创建一个.npy文件
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
    # 将内存中数组写回磁盘
    del matrix
    # 删除matrix变量的引用

    if row_start != n_rows:
        raise ValueError(f"销量行数发生变化：读到 {row_start}，预期 {n_rows}")
    metadata = pd.concat(metadata_parts, ignore_index=True)
    if metadata["id"].duplicated().any():
        raise ValueError("销量序列 ID 重复；请重新审计原始数据")
    metadata.to_csv(PROCESSED / "sales_metadata.csv", index=False)
    return {"rows": row_start, "days": n_days, "matrix_dtype": np.dtype(storage_dtype).name}


def prepare_calendar() -> dict:
    """把日期文本解析为日期，提取整数作为d_num方便排序与筛选，转换SNAP标记类型，新增当天是否有事件标记，最终保存为csv"""
    calendar = pd.read_csv(RAW / "calendar.csv")
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise")
    calendar["d_num"] = calendar["d"].str.extract(r"^d_(\d+)$")[0].astype("int32")
    calendar = calendar.sort_values("d_num").reset_index(drop=True)
    # SNAP 标记在原始文件中已有；转换为小整数，不改变原意。
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
    """价格按商品、门店、周保留原始粒度。"""
    prices = pd.read_csv(
        RAW / "sell_prices.csv",
        dtype={"store_id": "category", "item_id": "category", "wm_yr_wk": "int32", "sell_price": "float32"},
    )
    prices.to_csv(PROCESSED / "sell_prices_clean.csv.gz", index=False, compression="gzip")
    # 压缩保存，后续pandas可以直接读取，节省空间，代价是读写时需要做压缩、解压计算
    return {"rows": len(prices), "minimum_price": float(prices["sell_price"].min()),
            "maximum_price": float(prices["sell_price"].max())}


def main() -> None:
    audit = check_audit()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    # 重新运行若中途失败，旧 manifest 不应让下一步误认半成品已完成。
    manifest_path = PROCESSED / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    print("1/3 整理销量矩阵与序列元数据…")
    sales = prepare_sales(audit)
    print("2/3 整理日期与事件信息…")
    calendar = prepare_calendar()
    print("3/3 整理每周售价…")
    prices = prepare_prices()
    summary = {
        "sales": sales, "calendar": calendar, "prices": prices,
        "decisions": [
            "销量 0 保持为 0；不删除零销量日",
            "日历事件空值表示无已记录事件；保留为空"
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
        "row_order": "sales_metadata.csv 的行顺序",
        "column_order": "d_1, d_2, ..., d_1913",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已写入处理后数据：{PROCESSED}")
    print(f"已写入清洗说明：{SUMMARY_PATH}")


if __name__ == "__main__":
    main()
