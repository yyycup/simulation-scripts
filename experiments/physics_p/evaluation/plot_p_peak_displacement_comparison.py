from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BASELINE_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "p_mpc_operational_v1"
    / "heat_flow_corrected_v3_final_full_after_safe_fallback_20260801"
    / "peak"
    / "mpc"
    / "peak_physics_p_operational_steps1280_horizon8.csv"
)
DEFAULT_SMALLER_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "p_mpc_operational_v1"
    / "displacement_scale_scan_v1"
    / "peak_scale_0p90_full1280"
    / "mpc"
    / "peak_physics_p_operational_steps1280_horizon8.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SMALLER_CSV.parents[1] / "figures"

BASELINE_COLOR = "#6F6F6F"
SMALLER_COLOR = "#0072B2"
TARGET_COLOR = "#D55E00"


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 9.5,
            "axes.titlesize": 11.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.0,
            "legend.fontsize": 9.0,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.65,
            "savefig.dpi": 350,
            "savefig.bbox": "tight",
        }
    )


def _load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "Time",
        "Average temperature",
        "Compressor command",
        "Cumulative energy consumption",
        "MPC solve time",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def _starts(frame: pd.DataFrame, threshold_rpm: float = 500.0) -> int:
    on = frame["Compressor command"].to_numpy(dtype=float) > threshold_rpm
    return int(np.sum((~on[:-1]) & on[1:]))


def _plot_pair(ax, baseline, smaller, column, *, raw_alpha=1.0) -> None:
    ax.plot(
        baseline["Time"] / 60.0,
        baseline[column],
        color=BASELINE_COLOR,
        linestyle="--",
        linewidth=1.35,
        alpha=raw_alpha,
        label="原排量 5.525 cm³/rev",
        zorder=2,
    )
    ax.plot(
        smaller["Time"] / 60.0,
        smaller[column],
        color=SMALLER_COLOR,
        linewidth=1.75,
        alpha=raw_alpha,
        label="0.90排量 4.973 cm³/rev",
        zorder=3,
    )


def make_figure(
    baseline_csv: Path = DEFAULT_BASELINE_CSV,
    smaller_csv: Path = DEFAULT_SMALLER_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    baseline = _load(Path(baseline_csv))
    smaller = _load(Path(smaller_csv))
    if len(baseline) != len(smaller):
        raise ValueError(
            f"row count mismatch: baseline={len(baseline)}, smaller={len(smaller)}"
        )

    _configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.0), constrained_layout=True)

    ax = axes[0, 0]
    _plot_pair(ax, baseline, smaller, "Average temperature")
    ax.axhline(25.0, color=TARGET_COLOR, linestyle=":", linewidth=1.25, label="目标25 ℃")
    ax.set_title("(a) 电池平均温度")
    ax.set_xlabel("时间 (min)")
    ax.set_ylabel("温度 (℃)")
    ax.set_ylim(24.99, max(baseline["Average temperature"].max(), smaller["Average temperature"].max()) + 0.025)
    ax.legend(loc="upper right", ncol=1)

    baseline_starts = _starts(baseline)
    smaller_starts = _starts(smaller)
    ax = axes[0, 1]
    _plot_pair(ax, baseline, smaller, "Compressor command")
    ax.axhline(300.0, color="#9A9A9A", linestyle=":", linewidth=0.9)
    ax.set_title(f"(b) 压缩机指令：启停循环 {baseline_starts} → {smaller_starts} 次")
    ax.set_xlabel("时间 (min)")
    ax.set_ylabel("转速 (rpm)")
    ax.set_ylim(0.0, 6300.0)
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    _plot_pair(ax, baseline, smaller, "Cumulative energy consumption")
    ax.set_title("(c) 累计能耗")
    ax.set_xlabel("时间 (min)")
    ax.set_ylabel("能耗 (kWh)")
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    window = 25
    for frame, color, linestyle, label in (
        (baseline, BASELINE_COLOR, "--", "原排量（25步滑动均值）"),
        (smaller, SMALLER_COLOR, "-", "0.90排量（25步滑动均值）"),
    ):
        time_min = frame["Time"] / 60.0
        raw = frame["MPC solve time"]
        smooth = raw.rolling(window, center=True, min_periods=1).mean()
        ax.plot(time_min, raw, color=color, linewidth=0.45, alpha=0.16)
        ax.plot(time_min, smooth, color=color, linestyle=linestyle, linewidth=1.8, label=label)
    ax.axhline(5.0, color=TARGET_COLOR, linestyle=":", linewidth=1.0, label="5秒控制周期")
    ax.set_title("(d) MPC单步求解时间")
    ax.set_xlabel("时间 (min)")
    ax.set_ylabel("时间 (s)")
    ax.set_ylim(0.0, 5.25)
    ax.legend(loc="upper left")

    fig.suptitle(
        "调峰完整工况：P模型压缩机排量缩小前后对比",
        fontsize=14,
        fontweight="bold",
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "fig_peak_p_displacement_0p90_full_comparison.png"
    pdf_path = output_dir / "fig_peak_p_displacement_0p90_full_comparison.pdf"
    fig.savefig(png_path, dpi=350)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def make_single_figure(
    smaller_csv: Path = DEFAULT_SMALLER_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    frame = _load(Path(smaller_csv))
    extra_columns = {"T_cell_min_C", "T_cell_max_C", "Pump command"}
    missing = extra_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{Path(smaller_csv).name} missing columns: {sorted(missing)}")
    for column in extra_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    _configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.35), constrained_layout=True)
    time_min = frame["Time"].to_numpy(dtype=float) / 60.0
    mean_temp = frame["Average temperature"].to_numpy(dtype=float)
    min_temp = frame["T_cell_min_C"].to_numpy(dtype=float)
    max_temp = frame["T_cell_max_C"].to_numpy(dtype=float)

    ax = axes[0]
    ax.fill_between(
        time_min,
        min_temp,
        max_temp,
        color="#56B4E9",
        alpha=0.20,
        linewidth=0.0,
        label="电芯最低—最高温度",
    )
    ax.plot(time_min, mean_temp, color=SMALLER_COLOR, linewidth=1.9, label="电池平均温度")
    ax.axhline(25.0, color=TARGET_COLOR, linestyle=":", linewidth=1.2, label="目标25 ℃")
    ax.set_title("(a) 电池温度跟踪")
    ax.set_xlabel("时间 (min)")
    ax.set_ylabel("温度 (℃)")
    ax.legend(loc="upper left")
    ax.text(
        0.98,
        0.04,
        f"MAE {np.mean(np.abs(mean_temp - 25.0)):.3f} ℃\n"
        f"最高 {mean_temp.max():.3f} ℃\n末端 {mean_temp[-1]:.3f} ℃",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.82, "edgecolor": "#BBBBBB"},
    )

    ax = axes[1]
    ax.plot(
        time_min,
        frame["Compressor command"],
        color="#D55E00",
        linewidth=1.65,
        label="压缩机指令",
    )
    ax.plot(
        time_min,
        frame["Pump command"],
        color="#56B4E9",
        linewidth=1.45,
        label="泵指令",
    )
    ax.axhline(300.0, color="#999999", linestyle=":", linewidth=0.85)
    ax.set_title(f"(b) 执行器指令：完整工况启停循环 {_starts(frame)} 次")
    ax.set_xlabel("时间 (min)")
    ax.set_ylabel("转速 (rpm)")
    ax.set_ylim(0.0, 6300.0)
    ax.legend(loc="upper right")

    ax = axes[2]
    solve_time = frame["MPC solve time"].to_numpy(dtype=float)
    ax.plot(time_min, solve_time, color="#009E73", linewidth=0.8, alpha=0.72, label="MPC单步求解时间")
    ax.axhline(5.0, color="#666666", linestyle="--", linewidth=1.0, label="5秒控制周期")
    ax.set_title("(c) 计算负荷与累计能耗")
    ax.set_xlabel("时间 (min)")
    ax.set_ylabel("单步求解时间 (s)")
    ax.set_ylim(0.0, 5.25)
    energy_ax = ax.twinx()
    energy_ax.plot(
        time_min,
        frame["Cumulative energy consumption"],
        color="#E69F00",
        linewidth=1.65,
        label="累计能耗",
    )
    energy_ax.set_ylabel("累计能耗 (kWh)", color="#A96900")
    energy_ax.tick_params(axis="y", colors="#A96900")
    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = energy_ax.get_legend_handles_labels()
    ax.legend(handles_left + handles_right, labels_left + labels_right, loc="upper left")
    ax.text(
        0.98,
        0.04,
        f"平均求解 {solve_time.mean():.3f} s\n"
        f"成功率 {pd.to_numeric(frame['MPC_Solved'], errors='coerce').mean() * 100:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.82, "edgecolor": "#BBBBBB"},
    )

    fig.suptitle(
        "调峰完整工况：P模型（压缩机排量4.973 cm³/rev）",
        fontsize=14,
        fontweight="bold",
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "fig_peak_p_displacement_0p90_full_single.png"
    pdf_path = output_dir / "fig_peak_p_displacement_0p90_full_single.pdf"
    fig.savefig(png_path, dpi=350)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--smaller-csv", type=Path, default=DEFAULT_SMALLER_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=("comparison", "single", "both"), default="comparison")
    args = parser.parse_args()
    if args.mode in {"comparison", "both"}:
        png_path, pdf_path = make_figure(
            baseline_csv=args.baseline_csv,
            smaller_csv=args.smaller_csv,
            output_dir=args.output_dir,
        )
        print(f"COMPARISON_PNG {png_path.resolve()}")
        print(f"COMPARISON_PDF {pdf_path.resolve()}")
    if args.mode in {"single", "both"}:
        png_path, pdf_path = make_single_figure(
            smaller_csv=args.smaller_csv,
            output_dir=args.output_dir,
        )
        print(f"SINGLE_PNG {png_path.resolve()}")
        print(f"SINGLE_PDF {pdf_path.resolve()}")


if __name__ == "__main__":
    main()
