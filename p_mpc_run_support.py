"""Shared data lookup and summary helpers for root-level MPC runners."""

from pathlib import Path

import numpy as np
import pandas as pd

from thermal_batch_config import CASES


PROJECT_ROOT = Path(__file__).resolve().parent
SCENES = {
    "peak": "调峰",
    "freq": "调频",
}


def source_csv_for_scene(scene):
    matches = [
        case.source_csv
        for case in CASES
        if case.control == "mpc" and case.scene == scene and case.flow == "单向"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one MPC single-flow source for {scene}, got {len(matches)}")
    return matches[0]


def _column(frame, *names):
    for name in names:
        if name in frame.columns:
            return frame[name]
    raise KeyError(f"Missing required columns: {names}")


def summarize_run(frame, *, scene_key, predictor, horizon, artifact_path, out_csv):
    temperature = _column(
        frame,
        "Average temperature",
        "Battery Temp (C)",
    ).astype(float)
    energy = _column(
        frame,
        "Cumulative energy consumption",
        "Cumulative Energy (kWh)",
    ).astype(float)
    power = _column(frame, "Total power", "Total Power (kW)").astype(float)
    solve_time = _column(frame, "MPC solve time", "MPC_Solve_Time_S").astype(float)
    solved = _column(frame, "MPC_Solved").astype(str).str.lower().isin(("true", "1"))
    n_comp = _column(
        frame,
        "Compressor command",
        "Compressor Command (RPM)",
    ).astype(float)
    n_pump = _column(frame, "Pump command", "Pump Command (RPM)").astype(float)
    target_error = temperature - 25.0
    finite_solve_time = solve_time[np.isfinite(solve_time)]
    if "MPC solve error" in frame.columns:
        solve_errors = frame["MPC solve error"].fillna("").astype(str).str.strip()
    elif "MPC_Solve_Error" in frame.columns:
        solve_errors = frame["MPC_Solve_Error"].fillna("").astype(str).str.strip()
    else:
        solve_errors = pd.Series([""] * len(frame), index=frame.index, dtype=str)
    unique_solve_errors = list(dict.fromkeys(solve_errors[solve_errors != ""]))
    recovery_used_column = next(
        (
            name
            for name in ("MPC solve recovery used", "MPC_Solve_Recovery_Used")
            if name in frame.columns
        ),
        None,
    )
    if recovery_used_column is not None:
        recovery_used = (
            frame[recovery_used_column]
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin(("true", "1"))
        )
    else:
        recovery_used = pd.Series(False, index=frame.index, dtype=bool)
    recovery_reason_column = next(
        (
            name
            for name in ("MPC solve recovery reason", "MPC_Solve_Recovery_Reason")
            if name in frame.columns
        ),
        None,
    )
    if recovery_reason_column is not None:
        recovery_reasons = (
            frame[recovery_reason_column].fillna("").astype(str).str.strip()
        )
    else:
        recovery_reasons = pd.Series([""] * len(frame), index=frame.index, dtype=str)
    unique_recovery_reasons = list(
        dict.fromkeys(recovery_reasons[recovery_reasons != ""])
    )
    if "MPC_Strict_Predictor_Ablation" in frame.columns:
        strict_predictor_ablation = bool(
            frame["MPC_Strict_Predictor_Ablation"]
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin(("true", "1"))
            .all()
        )
    else:
        strict_predictor_ablation = False
    return {
        "scene": scene_key,
        "predictor": predictor,
        "strict_predictor_ablation": strict_predictor_ablation,
        "steps": int(len(frame)),
        "horizon_steps": int(horizon),
        "temperature_mae_c": float(np.mean(np.abs(target_error))),
        "temperature_max_c": float(np.max(temperature)),
        "temperature_final_c": float(temperature.iloc[-1]),
        "energy_kwh": float(energy.iloc[-1]),
        "mean_power_kw": float(np.mean(power)),
        "mean_n_comp_rpm": float(np.mean(n_comp)),
        "mean_n_pump_rpm": float(np.mean(n_pump)),
        "solve_success_rate": float(np.mean(solved)),
        "solve_time_mean_s": (
            float(np.mean(finite_solve_time)) if len(finite_solve_time) else np.nan
        ),
        "solve_time_p95_s": (
            float(np.percentile(finite_solve_time, 95))
            if len(finite_solve_time)
            else np.nan
        ),
        "solve_recovery_count": int(np.count_nonzero(recovery_used)),
        "solve_recovery_reasons": " | ".join(unique_recovery_reasons),
        "solve_error_count": int(np.count_nonzero(solve_errors != "")),
        "solve_error_messages": " | ".join(unique_solve_errors),
        "artifact": str(artifact_path) if artifact_path is not None else "",
        "out_csv": str(out_csv),
    }
