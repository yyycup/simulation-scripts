"""Evaluate physics-P artifacts on held-out identification trajectories."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mpc_physics_predictor import (
    initialize_physics_state,
    load_physics_artifact,
    step_physics_predictor,
)
from predictor_identification_data import validate_identification_frame


DEFAULT_HORIZONS_S = (5.0, 50.0, 100.0, 300.0)
Q_EVAP_ACTIVE_THRESHOLD_W = 500.0
RESPONSE_FIELDS = (
    ("N_Comp", "n_comp_eff_rpm", "n_comp_eff_rpm", "rpm"),
    ("N_Pump", "n_pump_eff_rpm", "n_pump_eff_rpm", "rpm"),
    ("Q_Evap", "q_evap_w", "q_evap_eff_w", "W"),
    ("T_Supply", "t_supply_c", "t_supply_c", "C"),
    ("T_Plate", "t_plate_c", "t_plate_c", "C"),
    ("T_Return", "t_return_c", "t_return_c", "C"),
    ("T_Batt", "t_batt_c", "t_batt_c", "C"),
    ("T_Cool", "t_cool_c", "t_cool_c", "C"),
)


def _initial_state(row):
    return initialize_physics_state(
        row["n_comp_eff_rpm"],
        row["n_pump_eff_rpm"],
        row["q_cond_eff_w"],
        row["q_evap_eff_w"],
        row["t_supply_c"],
        row["t_plate_c"],
        row["t_return_c"],
        row["t_batt_c"],
        row["t_cool_c"],
    )


def _horizon_steps(dt_s, horizons_s):
    steps = []
    for horizon_s in horizons_s:
        count = int(round(float(horizon_s) / float(dt_s)))
        if count <= 0 or not math.isclose(
            count * float(dt_s), float(horizon_s), abs_tol=1e-9
        ):
            raise ValueError("Each horizon must be a positive multiple of dt_s")
        steps.append((float(horizon_s), count))
    if len({count for _, count in steps}) != len(steps):
        raise ValueError("Horizon step counts must be unique")
    return tuple(sorted(steps, key=lambda item: item[1]))


def _step(state, row, artifact, timings_ns):
    started = time.perf_counter_ns()
    result = step_physics_predictor(
        state,
        row["n_comp_cmd_rpm"],
        row["n_pump_cmd_rpm"],
        row["q_gen_w"],
        row["t_ambient_c"],
        row["dt_s"],
        artifact,
    )
    timings_ns.append(time.perf_counter_ns() - started)
    values = np.array(
        [getattr(result, field) for _, field, _, _ in RESPONSE_FIELDS],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise RuntimeError("P rollout produced a nonfinite state")
    return result


def _detail_rows(model, scenario_id, origin_index, target_index, horizon_s, state, target):
    rows = []
    for state_name, prediction_field, actual_column, unit in RESPONSE_FIELDS:
        prediction = float(getattr(state, prediction_field))
        actual = float(target[actual_column])
        error = prediction - actual
        rows.append(
            {
                "model": model,
                "scenario_id": scenario_id,
                "origin_index": origin_index,
                "target_index": target_index,
                "horizon_s": horizon_s,
                "state": state_name,
                "unit": unit,
                "prediction": prediction,
                "actual": actual,
                "error": error,
                "abs_error": abs(error),
            }
        )
    return rows


def _summary(details):
    rows = []
    for keys, group in details.groupby(
        ["model", "horizon_s", "state", "unit"], sort=True
    ):
        model, horizon_s, state_name, unit = keys
        error = group["error"].to_numpy(float)
        actual = group["actual"].to_numpy(float)
        active = np.abs(actual) >= Q_EVAP_ACTIVE_THRESHOLD_W
        rows.append(
            {
                "model": model,
                "horizon_s": horizon_s,
                "state": state_name,
                "unit": unit,
                "n": len(group),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "bias": float(np.mean(error)),
                "p95_abs": float(np.quantile(np.abs(error), 0.95)),
                "max_abs": float(np.max(np.abs(error))),
                "active_n": int(np.sum(active)) if state_name == "Q_Evap" else 0,
                "active_mape_percent": (
                    float(100.0 * np.mean(np.abs(error[active] / actual[active])))
                    if state_name == "Q_Evap" and np.any(active)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_artifact(frame, model, artifact, horizons_s=DEFAULT_HORIZONS_S):
    selected = frame.loc[frame["split"] == "test"].copy()
    if selected.empty:
        raise ValueError("Dynamic data must contain test scenarios")
    dt_values = selected["dt_s"].drop_duplicates().to_numpy(float)
    if len(dt_values) != 1:
        raise ValueError("Test trajectories must use one common dt_s")
    horizons = _horizon_steps(dt_values[0], horizons_s)
    horizon_by_step = {steps: horizon for horizon, steps in horizons}
    max_steps = max(horizon_by_step)
    timings_ns = []
    detail_rows = []
    stability_rows = []

    for scenario_id, scenario in selected.groupby("scenario_id", sort=True):
        ordered = scenario.sort_values("time_s", kind="stable").reset_index(drop=True)
        for origin_index in range(len(ordered) - min(horizon_by_step)):
            state = _initial_state(ordered.iloc[origin_index])
            remaining = len(ordered) - origin_index - 1
            for offset in range(1, min(max_steps, remaining) + 1):
                target = ordered.iloc[origin_index + offset]
                state = _step(state, target, artifact, timings_ns)
                if offset in horizon_by_step:
                    detail_rows.extend(
                        _detail_rows(
                            model,
                            scenario_id,
                            origin_index,
                            origin_index + offset,
                            horizon_by_step[offset],
                            state,
                            target,
                        )
                    )

        state = _initial_state(ordered.iloc[0])
        maximum_absolute_state = 0.0
        for index in range(1, len(ordered)):
            state = _step(state, ordered.iloc[index], artifact, timings_ns)
            maximum_absolute_state = max(
                maximum_absolute_state,
                max(abs(float(getattr(state, field))) for _, field, _, _ in RESPONSE_FIELDS),
            )
        stability_rows.append(
            {
                "model": model,
                "scenario_id": scenario_id,
                "steps": len(ordered) - 1,
                "finite": True,
                "maximum_absolute_state": maximum_absolute_state,
            }
        )

    details = pd.DataFrame(detail_rows)
    timings_us = np.asarray(timings_ns, dtype=float) / 1000.0
    timing = {
        "model": model,
        "step_calls": len(timings_us),
        "median_step_time_us": float(np.median(timings_us)),
        "p95_step_time_us": float(np.quantile(timings_us, 0.95)),
        "max_step_time_us": float(np.max(timings_us)),
    }
    return details, _summary(details), pd.DataFrame(stability_rows), timing


def _parse_artifact(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Artifact must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Artifact must be LABEL=PATH")
    return label.strip(), Path(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic-csv", type=Path, required=True)
    parser.add_argument("--artifact", type=_parse_artifact, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.dynamic_csv, encoding="utf-8-sig")
    validate_identification_frame(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_details = []
    all_summaries = []
    all_stability = []
    timings = []
    metadata = {"dynamic_csv": str(args.dynamic_csv), "artifacts": {}}
    for label, artifact_path in args.artifact:
        artifact = load_physics_artifact(artifact_path, require_validated=True)
        details, summary, stability, timing = evaluate_artifact(
            frame, label, artifact
        )
        all_details.append(details)
        all_summaries.append(summary)
        all_stability.append(stability)
        timings.append(timing)
        metadata["artifacts"][label] = {
            "path": str(artifact_path),
            "minimum_active_rpm": artifact["capacity"].get("minimum_active_rpm"),
            "capacity_validation_target_met": artifact.get("fit", {}).get(
                "validation_target_met"
            ),
        }

    details = pd.concat(all_details, ignore_index=True)
    summary = pd.concat(all_summaries, ignore_index=True)
    stability = pd.concat(all_stability, ignore_index=True)
    timing_frame = pd.DataFrame(timings)
    details.to_csv(args.output_dir / "rollout_details.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "rollout_summary.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(args.output_dir / "stability_summary.csv", index=False, encoding="utf-8-sig")
    timing_frame.to_csv(args.output_dir / "timing_summary.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(timing_frame.to_string(index=False))


if __name__ == "__main__":
    main()
