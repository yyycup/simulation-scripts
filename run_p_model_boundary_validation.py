"""Validate Physics-P compressor boundaries against the detailed plant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from btms_runtime import ensure_env_library_bin_on_path

ensure_env_library_bin_on_path()

from mpc_flow_direction_strategies import (
    physics_p_operating_compressor_power_value,
)
from mpc_physics_predictor import (
    evaluate_physics_capacity,
    evaluate_physics_operating_capacity,
    load_physics_artifact,
    physics_p_startup_fraction_value,
)
from run_p_mpc_short_comparison import DEFAULT_P_ARTIFACT
from thermal_loop import staged_fan_speed
from thermal_system import (
    clear_refrigeration_cycle_cache,
    pump_model,
    run_refrigeration_cycle,
)


DEFAULT_OUTPUT_ROOT = Path("outputs/p_model_boundary_validation_v1")
DEFAULT_SPEEDS_RPM = (300.0, 650.0, 999.0, 1000.0, 1500.0, 1999.0, 2000.0, 6000.0)


def build_boundary_table(
    *,
    artifact_path=DEFAULT_P_ARTIFACT,
    speeds_rpm=DEFAULT_SPEEDS_RPM,
    pump_rpm=1600.0,
    coolant_c=25.0,
    ambient_c=35.0,
):
    artifact = load_physics_artifact(artifact_path, require_validated=True)
    minimum_active = float(artifact["capacity"]["minimum_active_rpm"])
    mass_flow, _ = pump_model(float(pump_rpm))
    rows = []
    for speed in speeds_rpm:
        speed = float(speed)
        clear_refrigeration_cycle_cache()
        plant = run_refrigeration_cycle(
            speed,
            staged_fan_speed(speed),
            float(coolant_c) + 273.15,
            mass_flow,
            float(ambient_c) + 273.15,
        )
        p_active_w = evaluate_physics_capacity(
            speed, pump_rpm, coolant_c, ambient_c, artifact=artifact
        )
        p_operating_w = evaluate_physics_operating_capacity(
            speed, pump_rpm, coolant_c, ambient_c, artifact=artifact
        )
        plant_q_w = float(plant["Q_evap"])
        plant_power_w = float(plant["W_comp"])
        p_power_w = physics_p_operating_compressor_power_value(
            speed,
            coolant_c,
            ambient_c,
            minimum_active_rpm=minimum_active,
        )
        rows.append(
            {
                "n_comp_cmd_rpm": speed,
                "startup_fraction": physics_p_startup_fraction_value(
                    speed, minimum_active
                ),
                "p_static_capacity_w": p_active_w,
                "p_operating_capacity_w": p_operating_w,
                "plant_capacity_w": plant_q_w,
                "capacity_error_w": p_operating_w - plant_q_w,
                "capacity_relative_error": (
                    abs(p_operating_w - plant_q_w) / plant_q_w
                    if plant_q_w > 1e-9
                    else 0.0
                ),
                "p_operating_power_w": p_power_w,
                "plant_compressor_power_w": plant_power_w,
            }
        )
    return pd.DataFrame(rows)


def summarize_boundary_table(frame):
    indexed = frame.set_index("n_comp_cmd_rpm")
    required = set(DEFAULT_SPEEDS_RPM)
    if not required.issubset(indexed.index):
        missing = sorted(required - set(indexed.index))
        raise ValueError(f"boundary table missing speeds: {missing}")
    active = frame.loc[frame["n_comp_cmd_rpm"] >= 1000.0]
    finite = bool(
        np.isfinite(
            frame[
                [
                    "p_operating_capacity_w",
                    "plant_capacity_w",
                    "p_operating_power_w",
                    "plant_compressor_power_w",
                ]
            ].to_numpy(dtype=float)
        ).all()
    )
    p_continuity = abs(
        float(indexed.loc[999.0, "p_operating_capacity_w"])
        - float(indexed.loc[1000.0, "p_operating_capacity_w"])
    ) / max(float(indexed.loc[1000.0, "p_operating_capacity_w"]), 1e-9)
    plant_continuity = abs(
        float(indexed.loc[999.0, "plant_capacity_w"])
        - float(indexed.loc[1000.0, "plant_capacity_w"])
    ) / max(float(indexed.loc[1000.0, "plant_capacity_w"]), 1e-9)
    summary = {
        "finite": finite,
        "p_off_capacity_w": float(indexed.loc[300.0, "p_operating_capacity_w"]),
        "plant_off_capacity_w": float(indexed.loc[300.0, "plant_capacity_w"]),
        "p_off_power_w": float(indexed.loc[300.0, "p_operating_power_w"]),
        "plant_off_power_w": float(indexed.loc[300.0, "plant_compressor_power_w"]),
        "p_999_to_1000_relative_jump": float(p_continuity),
        "plant_999_to_1000_relative_jump": float(plant_continuity),
        "active_capacity_mape": float(
            np.mean(active["capacity_relative_error"].to_numpy(dtype=float))
        ),
    }
    summary["qualified"] = bool(
        finite
        and summary["p_off_capacity_w"] <= 1e-6
        and summary["plant_off_capacity_w"] <= 1e-6
        and summary["p_off_power_w"] <= 1e-6
        and summary["plant_off_power_w"] <= 1e-6
        and p_continuity <= 0.01
        and plant_continuity <= 0.01
        and bool((active["p_operating_capacity_w"] > 0.0).all())
        and bool((active["plant_capacity_w"] > 0.0).all())
    )
    return summary


def run_boundary_validation(*, artifact_path, output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    frame = build_boundary_table(artifact_path=artifact_path)
    summary = summarize_boundary_table(frame)
    csv_path = output_root / "p_model_boundary_validation.csv"
    summary_path = output_root / "p_model_boundary_summary.json"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return frame, summary, csv_path, summary_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_P_ARTIFACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    frame, summary, csv_path, summary_path = run_boundary_validation(
        artifact_path=args.artifact,
        output_root=args.output_root,
    )
    print(frame.to_string(index=False), flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"BOUNDARY_CSV {csv_path}", flush=True)
    print(f"BOUNDARY_SUMMARY {summary_path}", flush=True)
    if not summary["qualified"]:
        raise SystemExit("Physics-P boundary validation did not qualify")


if __name__ == "__main__":
    main()
