from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parent
RESULT_ROOT = (
    WORKSPACE
    / "输出结果"
    / "最终控制器完整对比"
    / "服务器_12组_终端代价_candidateB_20260712_134913"
)
SUMMARY_PATH = RESULT_ROOT / "parallel_controller_summary.csv"
OUTPUT_DIR = RESULT_ROOT / "MPC内部汇报图"

SCENES = ("peak", "freq")
CONTROLS = ("on-off", "pid", "mpc")
CASES = (("peak", "single"), ("peak", "double"), ("freq", "single"), ("freq", "double"))
SCENE_LABELS = {"peak": "调峰", "freq": "调频"}
FLOW_LABELS = {"single": "单向", "double": "双向"}
CONTROL_LABELS = {"on-off": "On-Off", "pid": "PID", "mpc": "MPC"}
SCENE_COLORS = {"peak": "#2878B5", "freq": "#F28E2B"}
CONTROL_COLORS = {"on-off": "#7F7F7F", "pid": "#F28E2B", "mpc": "#2878B5"}
FLOW_STYLES = {"single": ("-", 1.8), "double": ("--", 2.1)}


def configure_style():
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
        }
    )


def load_summary():
    if not SUMMARY_PATH.is_file():
        raise FileNotFoundError(f"找不到汇总文件: {SUMMARY_PATH}")
    frame = pd.read_csv(SUMMARY_PATH)
    required = {
        "scene_key", "control", "flow_key", "temperature_MAE_C",
        "max_delta_T_C", "energy_kWh", "temperature_min_C", "mean_delta_T_C", "out_csv",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"汇总文件缺少字段: {sorted(missing)}")
    return frame


def _summary_row(scene, control, flow):
    summary = load_summary()
    rows = summary[
        (summary["scene_key"] == scene)
        & (summary["control"] == control)
        & (summary["flow_key"] == flow)
    ]
    if len(rows) != 1:
        raise ValueError(f"案例匹配数量异常: {scene}/{control}/{flow}, count={len(rows)}")
    return rows.iloc[0]


def case_csv_path(scene, control, flow):
    row = _summary_row(scene, control, flow)
    filename = Path(str(row["out_csv"])).name
    path = RESULT_ROOT / control / filename
    if not path.is_file():
        raise FileNotFoundError(f"找不到案例 CSV: {path}")
    return path


def load_case(scene, control, flow):
    path = case_csv_path(scene, control, flow)
    frame = pd.read_csv(path, low_memory=False)
    required = {"Time", "T_cell_mean_C", "Delta_T_cell_C"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} 缺少字段: {sorted(missing)}")
    return frame


def _as_bool(series):
    values = series.dropna().astype(str).str.strip().str.lower()
    return values.isin({"true", "1", "yes"})


def validate_mpc_terminal_cost(frame, scene):
    required = {"Terminal_Cost_Enabled", "W_Terminal_Temp"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"MPC CSV 缺少终端代价字段: {sorted(missing)}")
    flags = _as_bool(frame["Terminal_Cost_Enabled"])
    if flags.empty or not flags.all():
        raise ValueError(f"{SCENE_LABELS[scene]} MPC 终端代价未全程启用")
    expected = 1_000_000.0 if scene == "peak" else 500_000.0
    weights = pd.to_numeric(frame["W_Terminal_Temp"], errors="coerce").dropna()
    if weights.empty or not np.allclose(weights, expected):
        raise ValueError(f"{SCENE_LABELS[scene]} MPC 终端权重不是 {expected:g}")


def _minutes(frame):
    return pd.to_numeric(frame["Time"], errors="coerce") / 60.0


def _values(frame, column):
    return pd.to_numeric(frame[column], errors="coerce")


def plot_controller_temperature_timeseries():
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), sharey=True)
    for ax, scene in zip(axes, SCENES):
        for control in CONTROLS:
            frame = load_case(scene, control, "single")
            ax.plot(
                _minutes(frame), _values(frame, "T_cell_mean_C"),
                color=CONTROL_COLORS[control],
                linewidth=2.35 if control == "mpc" else 1.45,
                label=CONTROL_LABELS[control],
            )
        ax.set_xlabel("时间 (min)")
        ax.set_ylabel("电池平均温度 (℃)")
        ax.text(0.02, 0.95, SCENE_LABELS[scene], transform=ax.transAxes, va="top", fontsize=11)
        ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


def plot_controller_temperature_metrics(summary=None):
    configure_style()
    summary = load_summary() if summary is None else summary
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
    x = np.arange(len(CONTROLS))
    metrics = (("temperature_MAE_C", "温度 MAE (℃)"), ("energy_kWh", "累计能耗 (kWh)"))
    width = 0.34
    for ax, (column, ylabel) in zip(axes, metrics):
        for offset, scene in zip((-width / 2, width / 2), SCENES):
            values = [
                float(summary[(summary.scene_key == scene) & (summary.control == c) & (summary.flow_key == "single")][column].iloc[0])
                for c in CONTROLS
            ]
            bars = ax.bar(x + offset, values, width, color=SCENE_COLORS[scene], label=SCENE_LABELS[scene])
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
        ax.set_xticks(x, [CONTROL_LABELS[c] for c in CONTROLS])
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        ax.margins(y=0.16)
    fig.tight_layout()
    return fig


def plot_controller_overview_2x2(summary=None):
    configure_style()
    summary = load_summary() if summary is None else summary
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 5.8))

    for ax, scene in zip(axes[0], SCENES):
        for control in CONTROLS:
            frame = load_case(scene, control, "single")
            ax.plot(
                _minutes(frame),
                _values(frame, "T_cell_mean_C"),
                color=CONTROL_COLORS[control],
                linewidth=2.0 if control == "mpc" else 1.25,
                label=CONTROL_LABELS[control],
            )
        ax.set_xlabel("时间 (min)", fontsize=9)
        ax.set_ylabel("电池平均温度 (℃)", fontsize=9)
        ax.text(0.02, 0.94, SCENE_LABELS[scene], transform=ax.transAxes, va="top", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.legend(frameon=False, ncol=2, fontsize=8, loc="best")

    x = np.arange(len(CONTROLS))
    width = 0.34
    metrics = (("temperature_MAE_C", "温度 MAE (℃)"), ("energy_kWh", "累计能耗 (kWh)"))
    for ax, (column, ylabel) in zip(axes[1], metrics):
        for offset, scene in zip((-width / 2, width / 2), SCENES):
            values = [
                float(summary[(summary.scene_key == scene) & (summary.control == control) & (summary.flow_key == "single")][column].iloc[0])
                for control in CONTROLS
            ]
            bars = ax.bar(x + offset, values, width, color=SCENE_COLORS[scene], label=SCENE_LABELS[scene])
            ax.bar_label(bars, fmt="%.3f", padding=1.5, fontsize=7)
        ax.set_xticks(x, [CONTROL_LABELS[control] for control in CONTROLS])
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(frameon=False, fontsize=8)
        ax.margins(y=0.17)

    fig.tight_layout(pad=0.8, h_pad=1.0, w_pad=1.1)
    return fig


def plot_mpc_flow_temperature():
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2))
    for row, scene in enumerate(SCENES):
        for flow in ("single", "double"):
            frame = load_case(scene, "mpc", flow)
            linestyle, linewidth = FLOW_STYLES[flow]
            color = "#6BAED6" if flow == "single" else "#08519C"
            axes[row, 0].plot(_minutes(frame), _values(frame, "T_cell_mean_C"), linestyle, color=color, linewidth=linewidth, label=FLOW_LABELS[flow])
            axes[row, 1].plot(_minutes(frame), _values(frame, "Delta_T_cell_C"), linestyle, color=color, linewidth=linewidth, label=FLOW_LABELS[flow])
        axes[row, 0].set_ylabel(f"{SCENE_LABELS[scene]}平均温度 (℃)")
        axes[row, 1].set_ylabel(f"{SCENE_LABELS[scene]}最大温差 (℃)")
        axes[row, 0].legend(frameon=False, fontsize=9)
        axes[row, 1].legend(frameon=False, fontsize=9)
    for ax in axes.flat:
        ax.set_xlabel("时间 (min)")
    fig.tight_layout()
    return fig


def plot_mpc_flow_overview_2x2(summary=None):
    configure_style()
    summary = load_summary() if summary is None else summary
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 5.8))

    legend_handles = None
    legend_labels = None
    for ax, scene in zip(axes[0], SCENES):
        twin = ax.twinx()
        for flow, color in (("single", "#6BAED6"), ("double", "#08519C")):
            frame = load_case(scene, "mpc", flow)
            linestyle, _ = FLOW_STYLES[flow]
            ax.plot(
                _minutes(frame),
                _values(frame, "T_cell_mean_C"),
                linestyle,
                color=color,
                linewidth=1.15,
                alpha=0.58,
                label=f"{FLOW_LABELS[flow]}平均温度",
            )
            twin.plot(
                _minutes(frame),
                _values(frame, "Delta_T_cell_C"),
                linestyle,
                color=color,
                linewidth=2.05,
                label=f"{FLOW_LABELS[flow]}最大温差",
            )
        ax.set_xlabel("时间 (min)", fontsize=9)
        ax.set_ylabel("电池平均温度 (℃)", fontsize=9)
        twin.set_ylabel("最大温差 (℃)", fontsize=9)
        ax.text(0.02, 0.94, SCENE_LABELS[scene], transform=ax.transAxes, va="top", fontsize=10)
        ax.tick_params(labelsize=8)
        twin.tick_params(labelsize=8)
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = twin.get_legend_handles_labels()
        if legend_handles is None:
            legend_handles = handles1 + handles2
            legend_labels = labels1 + labels2

    labels = [f"{SCENE_LABELS[scene]}\n{FLOW_LABELS[flow]}" for scene, flow in CASES]
    colors = ["#9ECAE1", "#3182BD", "#FDD0A2", "#E6550D"]
    metrics = (
        ("max_delta_T_C", "最大温差 (℃)"),
        ("energy_kWh", "累计能耗 (kWh)"),
    )
    for ax, (column, ylabel) in zip(axes[1], metrics):
        values = [
            float(
                summary[
                    (summary.scene_key == scene)
                    & (summary.control == "mpc")
                    & (summary.flow_key == flow)
                ][column].iloc[0]
            )
            for scene, flow in CASES
        ]
        bars = ax.bar(np.arange(len(CASES)), values, color=colors, width=0.68)
        ax.set_xticks(np.arange(len(CASES)), labels)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
        ax.margins(y=0.18)

    fig.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        ncol=4,
        fontsize=7.5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955), pad=0.8, h_pad=1.0, w_pad=1.1)
    return fig


def plot_mpc_other_metrics(summary=None):
    configure_style()
    summary = load_summary() if summary is None else summary
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))
    labels = [f"{SCENE_LABELS[s]}\n{FLOW_LABELS[f]}" for s, f in CASES]
    colors = ["#9ECAE1", "#3182BD", "#FDD0A2", "#E6550D"]
    metrics = (
        ("energy_kWh", "累计能耗 (kWh)", "%.3f"),
        ("temperature_min_C", "最低单体温度 (℃)", "%.2f"),
        ("mean_delta_T_C", "平均温差 (℃)", "%.3f"),
    )
    for ax, (column, ylabel, fmt) in zip(axes, metrics):
        values = [
            float(summary[(summary.scene_key == s) & (summary.control == "mpc") & (summary.flow_key == f)][column].iloc[0])
            for s, f in CASES
        ]
        bars = ax.bar(np.arange(len(CASES)), values, color=colors, width=0.68)
        ax.set_xticks(np.arange(len(CASES)), labels)
        ax.set_ylabel(ylabel)
        ax.bar_label(bars, fmt=fmt, padding=2, fontsize=8)
        low, high = min(values), max(values)
        if column == "temperature_min_C":
            pad = max((high - low) * 0.8, 0.25)
            ax.set_ylim(low - pad, high + pad)
        else:
            ax.margins(y=0.18)
    fig.tight_layout()
    return fig


def generate_all_figures(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    summary = load_summary()
    for scene, flow in CASES:
        validate_mpc_terminal_cost(load_case(scene, "mpc", flow), scene)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = (
        ("01_单向控制器实时温度对比", plot_controller_temperature_timeseries()),
        ("02_单向控制器温控指标", plot_controller_temperature_metrics(summary)),
        ("02A_PPT单向控制器综合对比_2x2", plot_controller_overview_2x2(summary)),
        ("03_MPC单双向温度对比", plot_mpc_flow_temperature()),
        ("03A_PPT流向反转效果_2x2", plot_mpc_flow_overview_2x2(summary)),
        ("04_MPC其他指标柱状图", plot_mpc_other_metrics(summary)),
    )
    paths = []
    for stem, fig in figures:
        path = output_dir / f"{stem}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        paths.append(path)
        plt.close(fig)
    return paths


def main():
    for path in generate_all_figures():
        print(path.resolve())


if __name__ == "__main__":
    main()
