"""Render the combined coarse/refined MPC weight data as a boxed 2x3 grid."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_ROOT / "输出结果" / "MPC权重局部细化" / "综合分析" / "MPC权重粗细扫描_统一相对指标.csv"
OUTPUT_ROOT = PROJECT_ROOT / "输出结果" / "MPC权重局部细化" / "综合分析"

PANEL_LAYOUT = (
    (("temp_change_pct", "cv"), ("temp_change_pct", "compressor"), ("temp_change_pct", "pump")),
    (("energy_change_pct", "cv"), ("energy_change_pct", "compressor"), ("energy_change_pct", "pump")),
)

SCAN_TITLES = {"cv": "CV软约束权重", "compressor": "压缩机能耗权重", "pump": "水泵能耗权重"}
X_LABELS = {"cv": "CV软约束权重", "compressor": "压缩机能耗权重系数", "pump": "水泵能耗权重系数"}
METRIC_LABELS = {"temp_change_pct": "温度偏差相对变化 (%)", "energy_change_pct": "能耗相对变化 (%)"}
COLORS = {"peak": "#0072B2", "freq": "#D55E00"}


def load_relative_data():
    return pd.read_csv(INPUT_PATH, encoding="utf-8-sig")


def _close_box(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.85)


def _plot_panel(ax, data, metric, scan_key):
    for source, linestyle, filled, alpha, width in (
        ("粗扫描", "--", False, 0.68, 1.35),
        ("局部细化", "-", True, 1.0, 2.2),
    ):
        for scene in ("peak", "freq"):
            group = data[
                (data["scan_key"] == scan_key)
                & (data["source"] == source)
                & (data["scene"] == scene)
            ].sort_values("factor")
            if group.empty:
                continue
            color = COLORS[scene]
            ax.plot(
                group["factor"],
                group[metric],
                color=color,
                linestyle=linestyle,
                linewidth=width,
                marker="o",
                markersize=4.8,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color,
                markeredgewidth=1.0,
                alpha=alpha,
                zorder=3 if filled else 2,
            )
    ax.axhline(0.0, color="#777777", linewidth=0.8, zorder=1)
    ax.grid(True, alpha=0.18, linewidth=0.6)
    ax.set_xlabel(X_LABELS[scan_key])
    ax.set_ylabel(METRIC_LABELS[metric])
    if scan_key == "cv":
        ax.set_xscale("log")
    _close_box(ax)


def plot_two_by_three(data, output_root):
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS"],
            "axes.unicode_minus": False,
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "figure.dpi": 180,
            "savefig.dpi": 350,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2))
    for row, panel_row in enumerate(PANEL_LAYOUT):
        for col, (metric, scan_key) in enumerate(panel_row):
            ax = axes[row, col]
            _plot_panel(ax, data, metric, scan_key)
            if row == 0:
                ax.set_title(SCAN_TITLES[scan_key], pad=7)

    legend = [
        Line2D([0], [0], color=COLORS["peak"], linewidth=2.4, label="调峰"),
        Line2D([0], [0], color=COLORS["freq"], linewidth=2.4, label="调频"),
        Line2D(
            [0], [0], color="#555555", linestyle="--", linewidth=1.35,
            marker="o", markerfacecolor="white", markeredgecolor="#555555",
            markersize=5.0, label="粗扫描",
        ),
        Line2D(
            [0], [0], color="#555555", linestyle="-", linewidth=2.2,
            marker="o", markerfacecolor="#555555", markeredgecolor="#555555",
            markersize=5.0, label="局部细化",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.992),
        frameon=False,
        fontsize=10.5,
        columnspacing=1.8,
        handlelength=2.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.25, w_pad=1.2)
    output_root.mkdir(parents=True, exist_ok=True)
    png_path = output_root / "MPC权重粗细扫描_2x3封口.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main():
    data = load_relative_data()
    png_path = plot_two_by_three(data, OUTPUT_ROOT)
    print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()
