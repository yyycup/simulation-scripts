"""Run the final fair controller comparison and MPC flow comparison.

Unique run matrix (twelve cases):
  - peak/frequency x On-off, PID, MPC x single/bidirectional flow

The single- and bidirectional MPC cases use the same selected MPC parameters
and the same single-predictive-delta-T controller; only flow reversal is
enabled in the bidirectional cases.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from btms_runtime import ensure_env_library_bin_on_path

ensure_env_library_bin_on_path()


PROJECT_ROOT = Path(__file__).resolve().parent
GEKKO_TEMP = Path.home() / "btms_gekko_tmp"
GEKKO_TEMP.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(GEKKO_TEMP)
os.environ["TEMP"] = str(GEKKO_TEMP)
tempfile.tempdir = str(GEKKO_TEMP)

from thermal_batch_config import CASES, FREQ_PID_PARAMS, PEAK_PID_PARAMS  # noqa: E402
from thermal_case_simulator import simulate_case  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "输出结果" / "最终控制器完整对比" / "完整仿真"
PID_SELECTION_PATH = (
    PROJECT_ROOT / "输出结果" / "最终控制器完整对比" / "PID局部优化" / "PID最终参数.json"
)
TARGET_TEMP_C = 25.0

SCENES = {"peak": "调峰", "freq": "调频"}
FLOWS = {"single": "单向", "double": "双向"}
CONTROLS = ("on-off", "pid", "mpc")


def build_run_matrix():
    matrix = []
    for scene_key, scene in SCENES.items():
        for control in CONTROLS:
            for flow_key, flow in FLOWS.items():
                matrix.append(
                    {
                        "scene_key": scene_key,
                        "scene": scene,
                        "control": control,
                        "flow_key": flow_key,
                        "flow": flow,
                        "mpc_flow_mode": "single_predictive_delta_t" if control == "mpc" else "standard",
                    }
                )
    return matrix


def load_pid_selection(path=PID_SELECTION_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"PID parameter file not found: {path}\n"
            "Run run_pid_local_refinement.py before the full comparison."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    selection = {}
    for scene_key in ("peak", "freq"):
        values = tuple(float(value) for value in data[scene_key])
        if len(values) != 3 or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid {scene_key} PID parameters in {path}: {values}")
        selection[scene_key] = values
    return selection


def configured_pid_selection():
    return {
        "peak": tuple(float(value) for value in PEAK_PID_PARAMS),
        "freq": tuple(float(value) for value in FREQ_PID_PARAMS),
    }


def _pick_column(df, names):
    for name in names:
        if name in df.columns:
            return df[name]
    raise KeyError(f"None of the required columns exists: {names}")


def summarize_dataframe(df, target_temp_c=TARGET_TEMP_C):
    time = _pick_column(df, ("Time", "Time (s)" )).astype(float).to_numpy()
    temperature = _pick_column(df, ("Average temperature", "Battery Temp (C)")).astype(float).to_numpy()
    delta_t = _pick_column(df, ("Maximum temperature difference", "Max_Delta_T")).astype(float).to_numpy()
    energy = _pick_column(
        df, ("Cumulative energy consumption", "Cumulative Energy (kWh)")
    ).astype(float).to_numpy()
    power = _pick_column(df, ("Total power", "Total Power (kW)")).astype(float).to_numpy()
    direction = _pick_column(df, ("Flow direction d", "Flow Direction d")).astype(float).to_numpy()
    error = temperature - float(target_temp_c)
    reversals = int(np.count_nonzero(np.diff(direction) != 0.0)) if direction.size > 1 else 0
    return {
        "rows": int(len(df)),
        "duration_s": float(time[-1] - time[0]) if time.size > 1 else 0.0,
        "temperature_MAE_C": float(np.mean(np.abs(error))),
        "temperature_RMSE_C": float(np.sqrt(np.mean(error**2))),
        "temperature_min_C": float(np.min(temperature)),
        "temperature_max_C": float(np.max(temperature)),
        "temperature_final_C": float(temperature[-1]),
        "mean_delta_T_C": float(np.mean(delta_t)),
        "max_delta_T_C": float(np.max(delta_t)),
        "energy_kWh": float(energy[-1]),
        "mean_power_kW": float(np.mean(power)),
        "reversal_count": reversals,
    }


def common_source_for_scene(scene):
    matches = [
        case
        for case in CASES
        if case.scene == scene and case.control == "mpc" and case.flow == FLOWS["single"]
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one common source case for {scene}, got {len(matches)}")
    return matches[0].source_csv


def _case_names(case, max_steps):
    suffix = "完整" if max_steps is None else f"冒烟{max_steps}步"
    stem = f"{case['scene']}_{case['flow']}_{case['control']}_{suffix}"
    return f"{stem}.csv", f"{stem}_温度快照.csv", stem


def run_case(case, pid_selection, output_root, max_steps=None, skip_existing=False):
    main_name, snap_name, result_tag = _case_names(case, max_steps)
    log_path = Path(output_root) / "运行日志" / f"{result_tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message):
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(str(message) + "\n")

    pid_params = pid_selection[case["scene_key"]] if case["control"] == "pid" else None
    result = simulate_case(
        case["control"],
        case["scene"],
        case["flow"],
        common_source_for_scene(case["scene"]),
        main_name,
        snap_name,
        output_root=output_root,
        force=not skip_existing,
        max_steps=max_steps,
        target_temp_c=TARGET_TEMP_C,
        mpc_flow_mode=case["mpc_flow_mode"],
        result_tag=result_tag,
        pid_params=pid_params,
        log_func=log,
    )
    out_csv = Path(result["out_csv"])
    df = pd.read_csv(out_csv, encoding="utf-8-sig")
    summary = summarize_dataframe(df, target_temp_c=TARGET_TEMP_C)
    summary.update(
        {
            "scene_key": case["scene_key"],
            "scene": case["scene"],
            "control": case["control"],
            "flow_key": case["flow_key"],
            "flow": case["flow"],
            "mpc_flow_mode": case["mpc_flow_mode"],
            "pid_params": str(pid_params) if pid_params is not None else "",
            "out_csv": str(out_csv),
        }
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="最终控制器完整对比：2场景 x 3控制器 x 2流向")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--pid-params", type=Path, default=PID_SELECTION_PATH)
    parser.add_argument("--pid-source", choices=("configured", "json"), default="configured")
    parser.add_argument("--max-steps", type=int, default=None, help="仅用于冒烟检查；正式仿真请省略")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    pid_selection = configured_pid_selection() if args.pid_source == "configured" else load_pid_selection(args.pid_params)
    suffix = "完整" if args.max_steps is None else f"冒烟{args.max_steps}步"
    summary_path = args.output_root / f"最终控制器指标汇总_{suffix}.csv"
    rows = []

    matrix = build_run_matrix()
    for index, case in enumerate(matrix, start=1):
        print(
            f"RUN [{index}/{len(matrix)}] scene={case['scene_key']} "
            f"control={case['control']} flow={case['flow_key']}",
            flush=True,
        )
        row = run_case(
            case,
            pid_selection,
            args.output_root,
            max_steps=args.max_steps,
            skip_existing=args.skip_existing,
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(
            f"DONE [{index}/{len(matrix)}] MAE={row['temperature_MAE_C']:.4f} ℃ "
            f"ΔTmax={row['max_delta_T_C']:.4f} ℃ energy={row['energy_kWh']:.6f} kWh "
            f"reversals={row['reversal_count']}",
            flush=True,
        )

    print(f"完整仿真汇总: {summary_path}", flush=True)
    print(f"原始CSV目录: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
