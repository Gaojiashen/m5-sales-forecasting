"""
读取 M5 日历，展示训练期和预测期的事件与 SNAP 标记。

以下路径均相对于项目根目录。
输入：data/m5-forecasting-accuracy/calendar.csv。
绘图样式由同目录的 plot_style.py 提供。

输出至 outputs/data_prep/calendar/figures/，每张图保存 PNG 和 SVG 两种格式：
- 01_training_calendar_overview：训练期每月事件分布及各州 SNAP 标记比例。
- 02_training_28_day_example：上一年相近季节的 28 天日历示例。
- 03_forecast_28_days：预测期 d_1914～d_1941 的事件与 SNAP 标记。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from plot_style import (
    BLUE, GRID, INK, MUTED, ORANGE, TEAL,
    add_context, apply_style, export, style_axis,
)


ROOT = Path(__file__).resolve().parents[2]
CALENDAR_FILE = ROOT / "data" / "m5-forecasting-accuracy" / "calendar.csv"
OUT = ROOT / "outputs" / "data_prep" / "calendar" / "figures"
TRAIN_END = 1913
FORECAST_FIRST = 1914
FORECAST_LAST = 1941
EXAMPLE_FIRST = pd.Timestamp("2015-04-25")
EXAMPLE_LAST = pd.Timestamp("2015-05-22")
EVENT_NAMES = ("event_name_1", "event_name_2")
EVENT_TYPES = ("event_type_1", "event_type_2")
SNAP_COLUMNS = ("snap_CA", "snap_TX", "snap_WI")
MONTH_LABELS = ("1月", "2月", "3月", "4月", "5月", "6月",
                "7月", "8月", "9月", "10月", "11月", "12月")


def read_calendar() -> pd.DataFrame:
    """读取日历，检查绘图所需的日期、事件和 SNAP 标记列。"""
    if not CALENDAR_FILE.exists():
        raise FileNotFoundError(f"找不到 M5 日历文件：{CALENDAR_FILE}")
    calendar = pd.read_csv(CALENDAR_FILE, parse_dates=["date"])
    calendar["d_num"] = calendar["d"].str.removeprefix("d_").astype(int)
    calendar = calendar.sort_values("d_num").reset_index(drop=True)
    for column in (*EVENT_NAMES, *EVENT_TYPES):
        calendar[column] = calendar[column].fillna("").astype(str).str.strip()
    for column in SNAP_COLUMNS:
        calendar[column] = pd.to_numeric(calendar[column], errors="raise")
        if not calendar[column].isin((0, 1)).all():
            raise ValueError(f"{column} 只能包含 0 或 1")
    return calendar


def validate_28_days(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """检查是否恰好包含连续的 28 天，避免绘制缺日或日期错序的区间。"""
    frame = frame.sort_values("date").reset_index(drop=True)
    if len(frame) != 28:
        raise ValueError(f"{label}：预期有 28 个日历日，实际有 {len(frame)} 个")
    if not np.all(np.diff(frame.d_num.to_numpy()) == 1):
        raise ValueError(f"{label}：M5 天数编号不连续")
    if not (frame.date.diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError(f"{label}：日历日期不连续")
    return frame


def event_records(frame: pd.DataFrame) -> list[tuple[int, str, str]]:
    """读取两组事件字段；同一天记录两个事件时均予以保留。"""
    records = []
    for day_index, row in frame.iterrows():
        for name_col, type_col in zip(EVENT_NAMES, EVENT_TYPES):
            name = row[name_col]
            if name:
                records.append((day_index, name, row[type_col]))
    return records


def plot_training_overview(train: pd.DataFrame) -> None:
    """展示整个训练期内事件类型、事件日比例和 SNAP 标记的月度分布。"""
    has_event = train[list(EVENT_NAMES)].ne("").any(axis=1)
    month = train.date.dt.month
    monthly_event_rate = has_event.groupby(month).mean().reindex(range(1, 13))
    snap_by_month = train.groupby(month)[list(SNAP_COLUMNS)].mean().reindex(range(1, 13))

    types = pd.concat([train[column] for column in EVENT_TYPES], ignore_index=True)
    type_counts = types[types.ne("")].value_counts().sort_values()

    fig = plt.figure(figsize=(13.8, 8.3))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.1),
                           left=0.08, right=0.95, top=0.81, bottom=0.10,
                           wspace=0.24, hspace=0.48)
    ax_month = fig.add_subplot(grid[0, 0])
    ax_types = fig.add_subplot(grid[0, 1])
    ax_snap = fig.add_subplot(grid[1, :])

    x = np.arange(12)
    ax_month.bar(x, monthly_event_rate.to_numpy(), color=BLUE, width=0.66)
    ax_month.set_xticks(x, MONTH_LABELS)
    ax_month.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_month.set_ylabel("至少有一个事件的日期比例")
    ax_month.set_title("各月事件日比例", loc="left", pad=12,
                       fontsize=12, weight="bold")
    ax_month.set_ylim(0, max(0.16, float(monthly_event_rate.max()) * 1.18))
    style_axis(ax_month)

    ax_types.barh(type_counts.index, type_counts.to_numpy(), color=ORANGE, height=0.62)
    ax_types.set_title("已记录的事件类型", loc="left", pad=12,
                       fontsize=12, weight="bold")
    ax_types.set_xlabel("事件记录数")
    ax_types.set_xlim(0, max(1.0, float(type_counts.max()) * 1.18))
    for y, count in enumerate(type_counts.to_numpy()):
        ax_types.text(count + max(1, type_counts.max() * 0.02), y, str(count),
                      va="center", color=INK, fontsize=9)
    style_axis(ax_types, horizontal_grid=False)
    ax_types.grid(axis="x", color=GRID, linewidth=0.7)
    ax_types.set_axisbelow(True)

    # 热力图统计的是日历日期比例，不是销量占比。
    matrix = snap_by_month.to_numpy(dtype=float).T
    heat = ax_snap.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=0.5)
    ax_snap.set_xticks(x, MONTH_LABELS)
    ax_snap.set_yticks(range(3), ("加利福尼亚州", "得克萨斯州", "威斯康星州"))
    ax_snap.set_title("各州各月 SNAP 标记日期比例",
                      loc="left", pad=12, fontsize=12, weight="bold")
    for row in range(3):
        for col in range(12):
            value = matrix[row, col]
            ax_snap.text(col, row, f"{value:.0%}", ha="center", va="center",
                         fontsize=8.5, color="white" if value >= 0.34 else INK)
    ax_snap.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax_snap.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax_snap.grid(which="minor", color="white", linewidth=1.5)
    ax_snap.tick_params(which="both", length=0)
    for spine in ax_snap.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(heat, ax=ax_snap, pad=0.015, fraction=0.022)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    colorbar.set_label("标记日期比例", color=MUTED)

    start = train.date.min().date().isoformat()
    end = train.date.max().date().isoformat()
    add_context(
        fig,
        "训练期日历｜预测前已知的日期标记",
        f"d_1–d_{TRAIN_END} · {start} 至 {end} · 展示标记频率，不代表销量影响",
        "来源：M5 calendar.csv。各月比例以该月已有日期为分母；事件类型按记录条数计数。",
    )
    export(fig, OUT / "01_training_calendar_overview")


def choose_annotation_lanes(records: list[tuple[int, str, str]]) -> list[int]:
    """将相邻事件的标签分配到不同高度，减少文字重叠。"""
    last_at_lane = [-100] * 4
    lanes = []
    for day_index, _, _ in records:
        eligible = [lane for lane, last in enumerate(last_at_lane)
                    if day_index - last >= 7]
        lane = eligible[0] if eligible else int(np.argmin(last_at_lane))
        last_at_lane[lane] = day_index
        lanes.append(lane)
    return lanes


def plot_28_day_strip(frame: pd.DataFrame, *, title: str,
                      subtitle: str, filename: str) -> None:
    """采用相同的坐标布局绘制训练期示例和预测期的 28 天日历。"""
    records = event_records(frame)
    lanes = choose_annotation_lanes(records)
    fig = plt.figure(figsize=(16.5, 6.8))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.45, 1),
                           left=0.075, right=0.97, top=0.77,
                           bottom=0.16, hspace=0.13)
    ax_events = fig.add_subplot(grid[0])
    ax_snap = fig.add_subplot(grid[1], sharex=ax_events)
    ax_events.axhline(0.10, color=GRID, linewidth=2)
    ax_events.set_xlim(-0.8, 27.8)
    ax_events.set_ylim(-0.12, 2.4)
    ax_events.set_yticks([])
    ax_events.tick_params(axis="x", bottom=False, labelbottom=False)
    for spine in ax_events.spines.values():
        spine.set_visible(False)
    ax_events.text(-0.7, 2.20, "已记录事件", color=MUTED,
                   fontsize=9, weight="bold")
    if not records:
        ax_events.text(13.5, 1.25, "这 28 天内没有已命名事件",
                       ha="center", va="center", color=MUTED)
    for (day_index, name, event_type), lane in zip(records, lanes):
        label = name if not event_type else f"{name}\n{event_type}"
        label_x = day_index + (0.30 if day_index < 23 else -0.30)
        ax_events.scatter(day_index, 0.10, s=68, color=ORANGE,
                          edgecolor="white", linewidth=1.3, zorder=4)
        ax_events.annotate(
            label, xy=(day_index, 0.12), xytext=(label_x, 0.70 + lane * 0.43),
            ha="left" if day_index < 23 else "right", va="bottom",
            color=INK, fontsize=8.6, linespacing=1.15,
            bbox={"boxstyle": "round,pad=0.23", "facecolor": "white",
                  "edgecolor": "#D7E0E8", "linewidth": 0.7},
            arrowprops={"arrowstyle": "-", "color": ORANGE,
                        "linewidth": 0.9, "shrinkA": 2, "shrinkB": 4},
        )

    data = frame[list(SNAP_COLUMNS)].to_numpy(dtype=int).T
    cmap = ListedColormap(["#E9F0F5", TEAL])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    ax_snap.imshow(data, aspect="auto", cmap=cmap, norm=norm)
    ax_snap.set_yticks(range(3), ("加利福尼亚州", "得克萨斯州", "威斯康星州"))
    ax_snap.set_ylabel("SNAP 标记", labelpad=16)
    ax_snap.set_xticks(np.arange(28))
    ax_snap.set_xticklabels(
        [f"{d.month}/{d.day}\n周{'一二三四五六日'[d.weekday()]}" for d in frame.date],
        fontsize=7.2, rotation=45, ha="right", rotation_mode="anchor",
    )
    ax_snap.set_xticks(np.arange(-0.5, 28, 1), minor=True)
    ax_snap.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax_snap.grid(which="minor", color="white", linewidth=1.2)
    ax_snap.tick_params(which="both", length=0)
    for spine in ax_snap.spines.values():
        spine.set_visible(False)
    for boundary in (6.5, 13.5, 20.5):
        ax_events.axvline(boundary, color=GRID, linewidth=1, linestyle="--", zorder=0)
        ax_snap.axvline(boundary, color="#9AAFC2", linewidth=1.5)
    fig.legend(handles=[Patch(facecolor=TEAL, label="SNAP = 1（有标记）"),
                        Patch(facecolor="#E9F0F5", label="SNAP = 0（无标记）")],
               loc="lower right", bbox_to_anchor=(0.97, 0.10),
               ncol=2, frameon=False, fontsize=9)
    add_context(
        fig, title, subtitle,
        "来源：M5 calendar.csv。事件与 SNAP 标记是预先已知的日历信息；颜色不表示销量或估计影响。",
    )
    export(fig, OUT / filename)


def main() -> None:
    """读取并校验日历，依次生成训练期概览和两张 28 天日历图。"""
    apply_style()
    # 在本脚本中优先使用中文字体，避免中文标题和标签显示为方框。
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    calendar = read_calendar()
    train = calendar.loc[calendar.d_num <= TRAIN_END].copy()
    if len(train) != TRAIN_END or not np.array_equal(
        train.d_num.to_numpy(), np.arange(1, TRAIN_END + 1)
    ):
        raise ValueError("训练期日历必须完整包含 d_1 至 d_1913")
    example = validate_28_days(
        train.loc[train.date.between(EXAMPLE_FIRST, EXAMPLE_LAST)].copy(),
        "训练期示例",
    )
    future = validate_28_days(
        calendar.loc[calendar.d_num.between(FORECAST_FIRST, FORECAST_LAST)].copy(),
        "预测区间",
    )
    plot_training_overview(train)
    plot_28_day_strip(
        example,
        title="训练期示例｜上一年春季的 28 天",
        subtitle=f"{EXAMPLE_FIRST.date()} 至 {EXAMPLE_LAST.date()} · 与预测期季节相近，均为 d_1913 之前的已知日期",
        filename="02_training_28_day_example",
    )
    plot_28_day_strip(
        future,
        title="预测期日历｜需要预测的 28 天",
        subtitle=f"d_{FORECAST_FIRST}–d_{FORECAST_LAST} · {future.date.min().date()} 至 {future.date.max().date()} · 预测前已知的日历信息",
        filename="03_forecast_28_days",
    )
    print(f"已生成三组日历图（PNG 和 SVG）：{OUT}")


if __name__ == "__main__":
    main()
