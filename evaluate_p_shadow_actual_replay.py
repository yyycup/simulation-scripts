"""Evaluate predictor P with retrospectively known executed commands.

Each forecast starts from a measured post-step plant state in a simulation CSV.
Only commands and heat loads from later rows are replayed.  Future temperature
and cooling targets are read after prediction and are never predictor inputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from mpc_physics_predictor import (
    initialize_physics_state,
    load_physics_artifact,
    step_physics_predictor,
    validate_physics_artifact,
)
from mpc_physics_shadow import battery_heat_generation_w
from thermal_batch_config import AMBIENT_TEMP_C


COLUMNS = {
    "time": "Time",
    "t_batt_c": "Average temperature",
    "t_cool_c": "Coolant temperature",
    "t_plate_c": "P_Shadow_T_Plate_Actual_C",
    "t_supply_c": "Supply pipe coolant temperature",
    "t_return_c": "Return pipe coolant temperature",
    "total_current_a": "Total current",
    "n_comp_cmd_rpm": "Compressor command",
    "n_pump_cmd_rpm": "Pump command",
    "n_comp_eff_rpm": "Compressor Speed",
    "n_pump_eff_rpm": "Pump Speed (RPM)",
    "q_evap_kw": "Evaporator cooling rate (kW)",
    "q_cond_kw": "Condenser heat rejection rate (kW)",
}

RESPONSE_FIELDS = (
    ("T_Batt", "t_batt_c", "t_batt_c", "C", 1.0),
    ("T_Cool", "t_cool_c", "t_cool_c", "C", 1.0),
    ("T_Plate", "t_plate_c", "t_plate_c", "C", 1.0),
    ("T_Supply", "t_supply_c", "t_supply_c", "C", 1.0),
    ("T_Return", "t_return_c", "t_return_c", "C", 1.0),
    ("Q_Evap", "q_evap_w", "q_evap_kw", "W", 1000.0),
)

Q_EVAP_ACTIVE_THRESHOLD_W = 500.0
FORMAL_BATTERY_HEAT_REMOVAL_COLUMN = "Battery heat removal rate (kW)"


def _selected_artifact(artifact: str | Path | Mapping) -> dict:
    if isinstance(artifact, Mapping):
        return validate_physics_artifact(copy.deepcopy(dict(artifact)))
    return load_physics_artifact(artifact, require_validated=True)


def _horizon_steps(dt_s: float, horizons_s: Sequence[float]) -> tuple[tuple[float, int], ...]:
    dt = float(dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be a positive finite number")
    if not horizons_s:
        raise ValueError("horizons_s must not be empty")
    result = []
    for raw_horizon in horizons_s:
        horizon = float(raw_horizon)
        steps = int(round(horizon / dt))
        if (
            not math.isfinite(horizon)
            or horizon <= 0.0
            or steps <= 0
            or not math.isclose(steps * dt, horizon, rel_tol=0.0, abs_tol=1e-9)
        ):
            raise ValueError("each horizon must be a positive multiple of dt_s")
        result.append((horizon, steps))
    if len({steps for _horizon, steps in result}) != len(result):
        raise ValueError("horizons_s must not contain duplicate step counts")
    return tuple(sorted(result, key=lambda item: item[1]))


def _numeric_frame(frame: pd.DataFrame, artifact: Mapping) -> pd.DataFrame:
    working = frame.copy()
    plate_column = COLUMNS["t_plate_c"]
    if (
        plate_column not in working.columns
        and FORMAL_BATTERY_HEAT_REMOVAL_COLUMN in working.columns
    ):
        conductance = float(
            artifact["thermal"]["battery_plate_conductance_w_k"]
        )
        working[plate_column] = (
            pd.to_numeric(
                working[COLUMNS["t_batt_c"]], errors="raise"
            )
            - 1000.0
            * pd.to_numeric(
                working[FORMAL_BATTERY_HEAT_REMOVAL_COLUMN], errors="raise"
            )
            / conductance
        )
    missing = [column for column in COLUMNS.values() if column not in working.columns]
    if missing:
        raise ValueError(f"input CSV is missing required columns: {missing}")
    selected = working.loc[:, list(COLUMNS.values())].copy()
    for column in selected.columns:
        selected[column] = pd.to_numeric(selected[column], errors="raise")
    if not np.isfinite(selected.to_numpy(dtype=float)).all():
        raise ValueError("required replay columns must contain only finite values")
    return selected


def _initial_state(row: pd.Series):
    return initialize_physics_state(
        n_comp_eff_rpm=row[COLUMNS["n_comp_eff_rpm"]],
        n_pump_eff_rpm=row[COLUMNS["n_pump_eff_rpm"]],
        q_cond_w=row[COLUMNS["q_cond_kw"]] * 1000.0,
        q_evap_w=row[COLUMNS["q_evap_kw"]] * 1000.0,
        t_supply_c=row[COLUMNS["t_supply_c"]],
        t_plate_c=row[COLUMNS["t_plate_c"]],
        t_return_c=row[COLUMNS["t_return_c"]],
        t_batt_c=row[COLUMNS["t_batt_c"]],
        t_cool_c=row[COLUMNS["t_cool_c"]],
    )


def _summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (horizon_s, state_name, unit), group in details.groupby(
        ["horizon_s", "state", "unit"], sort=True
    ):
        error = group["error"].to_numpy(dtype=float)
        actual = group["actual"].to_numpy(dtype=float)
        active_mape_percent = np.nan
        active_n = 0
        if state_name == "Q_Evap":
            active = np.abs(actual) >= Q_EVAP_ACTIVE_THRESHOLD_W
            active_n = int(np.count_nonzero(active))
            if active_n:
                active_mape_percent = float(
                    100.0 * np.mean(np.abs(error[active] / actual[active]))
                )
        rows.append(
            {
                "horizon_s": float(horizon_s),
                "state": state_name,
                "unit": unit,
                "n": int(len(error)),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "bias": float(np.mean(error)),
                "max_abs_error": float(np.max(np.abs(error))),
                "mean_actual": float(np.mean(actual)),
                "active_threshold_w": (
                    Q_EVAP_ACTIVE_THRESHOLD_W if state_name == "Q_Evap" else np.nan
                ),
                "active_n": active_n,
                "active_mape_percent": active_mape_percent,
            }
        )
    return pd.DataFrame(rows)


def evaluate_actual_command_replay(
    frame: pd.DataFrame,
    artifact: str | Path | Mapping,
    *,
    dt_s: float,
    horizons_s: Sequence[float] = (50.0, 100.0, 300.0),
    t_ambient_c: float = AMBIENT_TEMP_C,
    origin_stride: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-origin errors and aggregate metrics for actual-command replay."""
    if isinstance(origin_stride, bool) or int(origin_stride) != origin_stride or origin_stride < 1:
        raise ValueError("origin_stride must be a positive integer")
    selected_artifact = _selected_artifact(artifact)
    selected = _numeric_frame(frame, selected_artifact)
    horizons = _horizon_steps(dt_s, horizons_s)
    if max(steps for _horizon, steps in horizons) >= len(selected):
        raise ValueError("maximum horizon must be shorter than the available data")

    rows: list[dict[str, object]] = []
    horizons_by_step = {steps: horizon for horizon, steps in horizons}
    max_steps = max(horizons_by_step)
    for origin_index in range(
        0,
        len(selected) - min(steps for _horizon, steps in horizons),
        int(origin_stride),
    ):
        state = _initial_state(selected.iloc[origin_index])
        remaining_steps = len(selected) - 1 - origin_index
        for offset in range(1, min(max_steps, remaining_steps) + 1):
            input_row = selected.iloc[origin_index + offset]
            state = step_physics_predictor(
                state,
                n_comp_cmd_rpm=input_row[COLUMNS["n_comp_cmd_rpm"]],
                n_pump_cmd_rpm=input_row[COLUMNS["n_pump_cmd_rpm"]],
                q_gen_w=battery_heat_generation_w(
                    input_row[COLUMNS["total_current_a"]]
                ),
                t_ambient_c=t_ambient_c,
                dt_s=dt_s,
                artifact=selected_artifact,
            )
            if offset not in horizons_by_step:
                continue
            target_index = origin_index + offset
            target_row = selected.iloc[target_index]
            for state_name, prediction_field, actual_field, unit, actual_scale in RESPONSE_FIELDS:
                prediction = float(getattr(state, prediction_field))
                actual = float(target_row[COLUMNS[actual_field]]) * actual_scale
                error = prediction - actual
                rows.append(
                    {
                        "origin_index": origin_index,
                        "target_index": target_index,
                        "origin_time_s": float(selected.iloc[origin_index][COLUMNS["time"]]),
                        "target_time_s": float(target_row[COLUMNS["time"]]),
                        "horizon_s": float(horizons_by_step[offset]),
                        "state": state_name,
                        "prediction": prediction,
                        "actual": actual,
                        "error": error,
                        "abs_error": abs(error),
                        "unit": unit,
                        "command_source": "executed_future_commands",
                        "state_reset": "measured_at_origin",
                    }
                )
    details = pd.DataFrame(rows)
    return details, _summarize(details)


def _parse_horizons(raw: str) -> tuple[float, ...]:
    return tuple(float(value.strip()) for value in raw.split(",") if value.strip())


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用未来真实执行指令回放，评估P模型50/100/300秒本体精度"
    )
    parser.add_argument("--input-csv", type=Path, action="append", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dt-s", type=float, default=5.0)
    parser.add_argument("--horizons-s", default="50,100,300")
    parser.add_argument("--ambient-c", type=float, default=AMBIENT_TEMP_C)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_details = []
    all_summaries = []
    horizons = _parse_horizons(args.horizons_s)
    for input_csv in args.input_csv:
        frame = pd.read_csv(input_csv, encoding="utf-8-sig")
        details, summary = evaluate_actual_command_replay(
            frame,
            args.artifact,
            dt_s=args.dt_s,
            horizons_s=horizons,
            t_ambient_c=args.ambient_c,
        )
        case_name = input_csv.stem
        details.insert(0, "case", case_name)
        summary.insert(0, "case", case_name)
        all_details.append(details)
        all_summaries.append(summary)
    detail_frame = pd.concat(all_details, ignore_index=True)
    summary_frame = pd.concat(all_summaries, ignore_index=True)
    combined_summary = _summarize(detail_frame)
    combined_summary.insert(0, "case", "combined")
    summary_frame = pd.concat(
        [summary_frame, combined_summary],
        ignore_index=True,
    )
    detail_frame.to_csv(
        args.output_dir / "actual_command_replay_details.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_frame.to_csv(
        args.output_dir / "actual_command_replay_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "actual_command_replay_summary.json").write_text(
        json.dumps(summary_frame.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
