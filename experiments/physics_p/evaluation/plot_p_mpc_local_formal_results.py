"""Generate publication-quality plots for the local formal Physics-P run."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from p_mpc_run_support import PROJECT_ROOT


DEFAULT_RESULT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "p_mpc_local_formal_v1"
    / "full_fallback_20260726"
)
SCENE_CONFIG = {
    "peak": {
        "label": "调峰",
        "steps": 1280,
        "horizon": 60,
        "color": "#0072B2",
    },
    "freq": {
        "label": "调频",
        "steps": 720,
        "horizon": 45,
        "color": "#D55E00",
    },
}
TARGET_TEMP_C = 25.0
CONTROL_INTERVAL_S = 5.0
COMPRESSOR_OFF_MAX_RPM = 301.0
COMPRESSOR_CONTINUOUS_MIN_RPM = 1000.0
COMPRESSOR_LOW_SPEED_MAX_RPM = 2000.0
COMPRESSOR_SATURATION_MIN_RPM = 5990.0


def configure_style():
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
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.dpi": 180,
            "savefig.dpi": 350,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.65,
            "lines.linewidth": 1.7,
        }
    )


def case_csv_paths(result_root):
    result_root = Path(result_root)
    return {
        scene: (
            result_root
            / scene
            / "mpc"
            / (
                f"{scene}_physics_p_steps{config['steps']}_"
                f"horizon{config['horizon']}.csv"
            )
        )
        for scene, config in SCENE_CONFIG.items()
    }


def _numeric(frame, column):
    if column not in frame.columns:
        raise KeyError(f"missing required result column: {column}")
    return pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)


def _boolean(frame, column):
    if column not in frame.columns:
        raise KeyError(f"missing required result column: {column}")
    return frame[column].fillna(False).astype(str).str.lower().isin(("true", "1"))


def summarize_case(frame, scene):
    temperature = _numeric(frame, "Average temperature")
    compressor = _numeric(frame, "Compressor command")
    pump = _numeric(frame, "Pump command")
    solve_time = _numeric(frame, "MPC solve time")
    energy = _numeric(frame, "Cumulative energy consumption")
    solved = _boolean(frame, "MPC_Solved")
    domain_valid = _boolean(frame, "MPC prediction domain valid")
    recovery = _boolean(frame, "MPC solve recovery used")
    predicted_coolant_minimum = _numeric(
        frame, "MPC predicted minimum coolant temperature"
    )
    domain_violation = _numeric(frame, "MPC coolant prediction domain violation")
    finite_predicted_minimum = predicted_coolant_minimum[
        np.isfinite(predicted_coolant_minimum)
    ]
    finite_domain_violation = domain_violation[np.isfinite(domain_violation)]
    return {
        "scene": scene,
        "scene_label": SCENE_CONFIG[scene]["label"],
        "steps": int(len(frame)),
        "horizon_steps": int(SCENE_CONFIG[scene]["horizon"]),
        "temperature_mae_c": float(np.mean(np.abs(temperature - TARGET_TEMP_C))),
        "temperature_min_c": float(np.min(temperature)),
        "temperature_max_c": float(np.max(temperature)),
        "temperature_final_c": float(temperature[-1]),
        "energy_kwh": float(energy[-1]),
        "mean_compressor_rpm": float(np.mean(compressor)),
        "mean_pump_rpm": float(np.mean(pump)),
        "compressor_off_rate": float(
            np.mean(compressor <= COMPRESSOR_OFF_MAX_RPM)
        ),
        "compressor_startup_transition_rate": float(
            np.mean(
                (compressor > COMPRESSOR_OFF_MAX_RPM)
                & (compressor < COMPRESSOR_CONTINUOUS_MIN_RPM)
            )
        ),
        "compressor_low_speed_rate": float(
            np.mean(
                (compressor >= COMPRESSOR_CONTINUOUS_MIN_RPM)
                & (compressor < COMPRESSOR_LOW_SPEED_MAX_RPM)
            )
        ),
        "compressor_saturation_rate": float(
            np.mean(compressor >= COMPRESSOR_SATURATION_MIN_RPM)
        ),
        "solve_success_rate": float(np.mean(solved)),
        "prediction_domain_valid_rate": float(np.mean(domain_valid)),
        "solve_recovery_count": int(np.count_nonzero(recovery)),
        "predicted_coolant_min_c": (
            float(np.min(finite_predicted_minimum))
            if finite_predicted_minimum.size
            else float("nan")
        ),
        "coolant_domain_violation_max_c": (
            float(np.max(finite_domain_violation))
            if finite_domain_violation.size
            else float("nan")
        ),
        "solve_time_mean_s": float(np.mean(solve_time)),
        "solve_time_p95_s": float(np.percentile(solve_time, 95)),
        "solve_time_max_s": float(np.max(solve_time)),
        "deadline_exceed_rate": float(np.mean(solve_time > CONTROL_INTERVAL_S)),
    }


def _read_status(result_root):
    status_path = Path(result_root) / "local_formal_status.json"
    if not status_path.is_file():
        return {"status": "MISSING", "error": f"missing status file: {status_path}"}
    return json.loads(status_path.read_text(encoding="utf-8"))


def _write_plot_status(result_root, *, status, error="", outputs=()):
    payload = {
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "error": error,
        "outputs": [str(path) for path in outputs],
    }
    path = Path(result_root) / "formal_plot_status.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def wait_for_run(result_root, poll_seconds=30.0):
    result_root = Path(result_root)
    _write_plot_status(result_root, status="WAITING")
    while True:
        status = _read_status(result_root)
        run_status = str(status.get("status", "MISSING")).upper()
        if run_status == "COMPLETE":
            return status
        if run_status == "FAILED":
            raise RuntimeError(
                "formal run failed before plotting: "
                + str(status.get("error", "unknown error"))
            )
        print(
            f"PLOT_WAIT run_status={run_status} "
            f"active_scene={status.get('active_scene')}",
            flush=True,
        )
        time.sleep(max(1.0, float(poll_seconds)))


def load_results(result_root):
    frames = {}
    for scene, path in case_csv_paths(result_root).items():
        if not path.is_file():
            raise FileNotFoundError(f"missing completed case CSV: {path}")
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        expected = SCENE_CONFIG[scene]["steps"]
        if len(frame) != expected:
            raise ValueError(
                f"{scene} result has {len(frame)} rows; expected {expected}"
            )
        frames[scene] = frame
    return frames


def plot_overview(frames, output_dir):
    configure_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0), constrained_layout=True)

    for row, scene in enumerate(("peak", "freq")):
        frame = frames[scene]
        config = SCENE_CONFIG[scene]
        minutes = _numeric(frame, "Time") / 60.0
        temperature = _numeric(frame, "Average temperature")
        compressor = _numeric(frame, "Compressor command")
        pump = _numeric(frame, "Pump command")
        solve_time = _numeric(frame, "MPC solve time")
        summary = summarize_case(frame, scene)

        ax = axes[row, 0]
        ax.plot(minutes, temperature, color=config["color"], label="电池平均温度")
        ax.axhline(TARGET_TEMP_C, color="#666666", linestyle="--", linewidth=1.1, label="目标温度 25°C")
        ax.set_ylabel("温度 (°C)")
        ax.set_title(
            f"{config['label']}：电池温度（MAE {summary['temperature_mae_c']:.3f}°C）"
        )
        ax.legend(loc="best")

        ax = axes[row, 1]
        ax.plot(minutes, compressor, color="#D55E00", label="压缩机指令")
        ax.plot(minutes, pump, color="#56B4E9", label="泵指令", alpha=0.9)
        ax.set_ylabel("转速指令 (rpm)")
        ax.set_title(
            f"执行器指令（压缩机饱和率 {summary['compressor_saturation_rate']:.1%}）"
        )
        ax.legend(loc="best")

        ax = axes[row, 2]
        ax.plot(minutes, solve_time, color="#009E73", linewidth=1.2, label="MPC求解时间")
        exceeded = solve_time > CONTROL_INTERVAL_S
        if np.any(exceeded):
            ax.scatter(
                minutes[exceeded],
                solve_time[exceeded],
                s=11,
                color="#D55E00",
                label="超过5秒",
                zorder=3,
            )
        ax.axhline(CONTROL_INTERVAL_S, color="#CC0000", linestyle="--", linewidth=1.1, label="5秒控制周期")
        ax.set_ylabel("求解时间 (s)")
        ax.set_title(
            f"求解性能（均值 {summary['solve_time_mean_s']:.2f}s，P95 {summary['solve_time_p95_s']:.2f}s）"
        )
        ax.legend(loc="best")

        for col in range(3):
            axes[row, col].set_xlabel("时间 (min)")
            axes[row, col].text(
                0.01,
                0.96,
                f"({chr(ord('a') + row * 3 + col)})",
                transform=axes[row, col].transAxes,
                va="top",
                fontweight="bold",
            )

    png_path = output_dir / "fig_p_mpc_local_formal_overview.png"
    pdf_path = output_dir / "fig_p_mpc_local_formal_overview.pdf"
    fig.savefig(png_path, dpi=350)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def generate(result_root=DEFAULT_RESULT_ROOT):
    result_root = Path(result_root)
    _write_plot_status(result_root, status="RUNNING")
    try:
        frames = load_results(result_root)
        output_dir = result_root / "figures"
        png_path, pdf_path = plot_overview(frames, output_dir)
        summary_path = output_dir / "formal_plot_summary.csv"
        pd.DataFrame(
            [summarize_case(frames[scene], scene) for scene in ("peak", "freq")]
        ).to_csv(summary_path, index=False, encoding="utf-8-sig")
        _write_plot_status(
            result_root,
            status="COMPLETE",
            outputs=(png_path, pdf_path, summary_path),
        )
        return png_path, pdf_path, summary_path
    except Exception as exc:
        _write_plot_status(
            result_root,
            status="FAILED",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.wait:
        wait_for_run(args.result_root, poll_seconds=args.poll_seconds)
    outputs = generate(args.result_root)
    for path in outputs:
        print(f"PLOT_OUTPUT {path}", flush=True)


if __name__ == "__main__":
    main()
