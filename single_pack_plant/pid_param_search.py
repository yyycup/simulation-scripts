"""PID candidate evaluation for full-profile particle-swarm tuning.

This module does not change the PID controller or plant.  It supplies the
server PSO driver with the current single-Pack peak/frequency cases, evaluates
each PID tuple on both flow directions, and returns comparable diagnostics.
The simulation entry point remains ``thermal_case_simulator.simulate_case``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from single_pack_plant.simulation.config import pid_target_temp_for_scene
from single_pack_plant.simulation.case import simulate_case


DEFAULT_PARAM_RANGES = {
    "kp": (0.2, 5.0),
    "ki": (0.0, 0.08),
    "kd": (0.0, 1.0),
}

PID_W_MAE = 1.0
PID_W_RMSE = 0.5
PID_W_OSC = 2.0
PID_MAX_TEMP_C = 28.0
PID_PEAK_MIN_TEMP_C = 24.0
PID_FREQ_MIN_TEMP_C = 23.0


class PidSearchCase(NamedTuple):
    control: str
    scene: str
    flow: str
    source_csv: Path
    main_name: str
    snap_name: str


def normalize_scene_arg(scene):
    aliases = {
        "peak": "调峰",
        "freq": "调频",
        "frequency": "调频",
        "reg": "调频",
    }
    text = str(scene)
    return aliases.get(text.lower(), text)


def _pick_column(frame, *names):
    for name in names:
        if name in frame.columns:
            return frame[name]
    raise KeyError(f"None of these columns exists: {names}")


def pid_tracking_metrics(frame, t_ref_c=25.0):
    temperature = _pick_column(
        frame,
        "Average temperature",
        "Battery Temp (C)",
        "T_avg_C",
    ).astype(float)
    energy = _pick_column(
        frame,
        "Cumulative energy consumption",
        "Cumulative Energy (kWh)",
    ).astype(float)
    compressor = _pick_column(
        frame,
        "Compressor Speed",
        "Compressor Speed (RPM)",
        "Compressor command",
    ).astype(float)

    temperatures = temperature.to_numpy(dtype=float)
    errors = temperatures - float(t_ref_c)
    compressor_delta = np.diff(compressor.to_numpy(dtype=float))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    oscillation = float(np.mean(np.abs(np.diff(temperatures)))) if len(temperatures) > 1 else 0.0
    action_metric = float(np.mean(compressor_delta**2)) if compressor_delta.size else 0.0
    score = PID_W_MAE * mae + PID_W_RMSE * rmse + PID_W_OSC * oscillation

    return {
        "score": float(score),
        "pid_temp_score": float(score),
        "MAE": mae,
        "RMSE": rmse,
        "OSC": oscillation,
        "T_avg_mean_error": float(np.mean(errors**2)),
        "T_avg_mae": mae,
        "T_avg_min": float(np.min(temperatures)),
        "T_avg_max": float(np.max(temperatures)),
        "T_final": float(temperatures[-1]),
        "below_24p8_count": int(np.count_nonzero(temperatures < 24.8)),
        "above_26_count": int(np.count_nonzero(temperatures > 26.0)),
        "below_23_count": int(np.count_nonzero(temperatures < 23.0)),
        "above_27_count": int(np.count_nonzero(temperatures > 27.0)),
        "energy": float(energy.iloc[-1]),
        "energy_kWh": float(energy.iloc[-1]),
        "compressor_action_count": action_metric,
    }


def pid_objective_score(candidate):
    value = candidate.get("pid_temp_score", candidate.get("score"))
    if value is None or pd.isna(value):
        value = candidate["T_avg_mean_error"]
    return float(value)


def candidate_passes_constraints(candidate, scene, max_energy=None, max_action_metric=None):
    normalized_scene = normalize_scene_arg(scene)
    minimum_temperature = (
        PID_FREQ_MIN_TEMP_C if normalized_scene == "调频" else PID_PEAK_MIN_TEMP_C
    )
    if float(candidate["T_avg_max"]) > PID_MAX_TEMP_C:
        return False
    if float(candidate["T_avg_min"]) < minimum_temperature:
        return False
    if max_energy is not None and float(candidate["energy"]) > float(max_energy):
        return False
    if (
        max_action_metric is not None
        and float(candidate["compressor_action_count"]) > float(max_action_metric)
    ):
        return False
    return True


def infer_constraint_limits(candidates, max_energy=None, max_action_metric=None):
    successful = [candidate for candidate in candidates if not candidate.get("failed", False)]
    inferred_energy = max_energy
    inferred_action = max_action_metric
    if inferred_energy is None and successful:
        inferred_energy = float(np.median([float(item["energy"]) for item in successful]) * 1.5)
    if inferred_action is None and successful:
        inferred_action = float(
            np.median([float(item["compressor_action_count"]) for item in successful]) * 2.0
        )
    return inferred_energy, inferred_action


def _default_source_root():
    configured = os.environ.get("SINGLE_PACK_PSO_SOURCE_ROOT")
    if configured:
        return Path(configured)
    return (
        Path(__file__).resolve().parent
        / "outputs"
        / "full12_physicsP_20260823_114038"
        / "_time_axis_source"
    )


def selected_pid_cases(scene, flow=None, source_root=None):
    normalized_scene = normalize_scene_arg(scene)
    scene_key = "freq" if normalized_scene == "调频" else "peak"
    source_root = Path(source_root) if source_root is not None else _default_source_root()
    source_name = "调频输出单向mpc.csv" if scene_key == "freq" else "调峰输出单向mpc.csv"
    selected_flows = ("单向", "双向") if flow is None else (str(flow),)
    return [
        PidSearchCase(
            "pid",
            normalized_scene,
            flow_name,
            source_root / "mpc" / source_name,
            f"{normalized_scene}输出{flow_name}pid.csv",
            f"{normalized_scene}温度快照{flow_name}pid.csv",
        )
        for flow_name in selected_flows
    ]


def _merge_case_metrics(params, case_metrics):
    return {
        "params": tuple(float(value) for value in params),
        "kp": float(params[0]),
        "ki": float(params[1]),
        "kd": float(params[2]),
        "score": float(np.mean([item["score"] for item in case_metrics])),
        "pid_temp_score": float(np.mean([item["pid_temp_score"] for item in case_metrics])),
        "MAE": float(np.mean([item["MAE"] for item in case_metrics])),
        "RMSE": float(np.mean([item["RMSE"] for item in case_metrics])),
        "OSC": float(np.mean([item["OSC"] for item in case_metrics])),
        "T_avg_mean_error": float(np.mean([item["T_avg_mean_error"] for item in case_metrics])),
        "T_avg_mae": float(np.mean([item["T_avg_mae"] for item in case_metrics])),
        "T_avg_min": float(min(item["T_avg_min"] for item in case_metrics)),
        "T_avg_max": float(max(item["T_avg_max"] for item in case_metrics)),
        "T_final": float(np.mean([item["T_final"] for item in case_metrics])),
        "below_24p8_count": int(sum(item["below_24p8_count"] for item in case_metrics)),
        "above_26_count": int(sum(item["above_26_count"] for item in case_metrics)),
        "below_23_count": int(sum(item["below_23_count"] for item in case_metrics)),
        "above_27_count": int(sum(item["above_27_count"] for item in case_metrics)),
        "energy": float(sum(item["energy"] for item in case_metrics)),
        "energy_kWh": float(sum(item["energy_kWh"] for item in case_metrics)),
        "compressor_action_count": float(
            np.mean([item["compressor_action_count"] for item in case_metrics])
        ),
    }


def evaluate_pid_candidate(
    params,
    cases,
    scene,
    output_root,
    max_steps=None,
    start_time_s=None,
    duration_s=None,
    log_func=None,
):
    normalized_scene = normalize_scene_arg(scene)
    target_temp_c = float(pid_target_temp_for_scene(normalized_scene))
    case_metrics = []
    for case in cases:
        if not Path(case.source_csv).is_file():
            raise FileNotFoundError(f"Missing PID PSO time-axis input: {case.source_csv}")
        result = simulate_case(
            *case,
            output_root=Path(output_root),
            force=True,
            max_steps=max_steps,
            start_time_s=start_time_s,
            duration_s=duration_s,
            target_temp_c=target_temp_c,
            mpc_flow_mode="standard",
            pid_params=tuple(float(value) for value in params),
            progress_interval_steps=100,
            log_func=log_func,
        )
        frame = pd.read_csv(result["out_csv"], encoding="utf-8-sig")
        case_metrics.append(pid_tracking_metrics(frame, t_ref_c=target_temp_c))
    return _merge_case_metrics(params, case_metrics)


def _log_factory(log_file):
    log_file = Path(log_file)

    def log(message):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        safe_message = str(message).encode("unicode_escape").decode("ascii")
        text = f"[{stamp}] {safe_message}"
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(text, flush=True)

    return log
