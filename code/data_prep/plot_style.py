'Shared visual styling for M5 exploratory figures: colors, fonts, spacing, and exports.\nUse color for grouping or emphasis, label curves directly, retain sources and dates, and keep axes simple. Imported by the plotting scripts.'

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Use a noninteractive backend for headless figure export.
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


INK = "#1E293B"
MUTED = "#65758B"
GRID = "#DDE4EC"
BLUE = "#2463A6"
ORANGE = "#D97932"
TEAL = "#21867A"
PURPLE = "#7357A4"
PALE = "#B9C7D5"
COLORS = (BLUE, ORANGE, TEAL, PURPLE)


def apply_style() -> None:
    'Apply the shared Matplotlib style.'
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.unicode_minus": False,
        "svg.fonttype": "none",  # Keep SVG text editable.
    })


def style_axis(ax, *, horizontal_grid: bool = True) -> None:
    'Simplify spines and grids to emphasize data.'
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, pad=7)
    if horizontal_grid:
        ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)


def add_context(fig, title: str, subtitle: str, source: str) -> None:
    'Add a title, measurement context, and data-source footer.'
    fig.text(0.07, 0.97, title, ha="left", va="top", fontsize=16, weight="bold", color=INK)
    fig.text(0.07, 0.91, subtitle, ha="left", va="top", fontsize=10, color=MUTED)
    fig.text(0.07, 0.035, source, ha="left", va="bottom", fontsize=8.5, color=MUTED)


def compact_number(value: float, _position: float | None = None) -> str:
    'Format sales-axis values compactly, for example 10k and 20k.'
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}m"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def set_compact_y(ax) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(compact_number))


def export(fig, path_without_extension: Path) -> None:
    'Export matching PNG and editable SVG figures.'
    path_without_extension.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_without_extension.with_suffix(".png"), dpi=180)
    fig.savefig(path_without_extension.with_suffix(".svg"))
    plt.close(fig)
