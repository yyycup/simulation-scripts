"""Run one concise peak-shaving short-loop validation of State-Space QP-MPC."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from single_pack_plant.simulation.case import simulate_case
    from single_pack_plant.controllers.fixed_qp.mpc import QPMPCWeights
else:
    from ..simulation.case import simulate_case
    from ..controllers.fixed_qp.mpc import QPMPCWeights


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "model_data" / "physics_p_operational_v1.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "state_space_qp_short_validation"


def summarize_validation_dataframe(data: pd.DataFrame) -> dict:
    commands = data[["Compressor command", "Pump command"]].to_numpy(dtype=float)
    actual_speeds = data[["Compressor Speed", "Pump Speed (RPM)"]].to_numpy(
        dtype=float
    )
    pump_moves = np.diff(commands[:, 1])
    actual_pump_moves = np.diff(actual_speeds[:, 1])
    nonzero_pump_moves = pump_moves[np.abs(pump_moves) > 1.0e-9]
    pump_direction_changes = int(
        np.count_nonzero(
            np.sign(nonzero_pump_moves[1:]) != np.sign(nonzero_pump_moves[:-1])
        )
    )
    nonzero_actual_pump_moves = actual_pump_moves[
        np.abs(actual_pump_moves) > 1.0e-9
    ]
    actual_pump_direction_changes = int(
        np.count_nonzero(
            np.sign(nonzero_actual_pump_moves[1:])
            != np.sign(nonzero_actual_pump_moves[:-1])
        )
    )
    previous = np.array([1500.0, 3000.0])
    deltas = np.diff(np.vstack([previous, commands]), axis=0)
    solved = data["MPC_Solved"].astype(bool).to_numpy()
    solve_times = data["MPC solve time"].to_numpy(dtype=float)
    finite_columns = data[
        [
            "Average temperature",
            "T_cell_max_C",
            "Coolant temperature",
            "Compressor command",
            "Pump command",
            "Compressor Speed",
            "Pump Speed (RPM)",
        ]
    ].to_numpy(dtype=float)
    return {
        "steps": len(data),
        "all_core_values_finite": bool(np.all(np.isfinite(finite_columns))),
        "solved_rate": float(np.mean(solved)),
        "fallback_count": int(np.count_nonzero(~solved)),
        "t_avg_max_c": float(data["Average temperature"].max()),
        "t_max_c": float(data["T_cell_max_C"].max()),
        "delta_t_max_c": float(data["Delta_T_cell_C"].max()),
        "coolant_min_c": float(data["Coolant temperature"].min()),
        "coolant_max_c": float(data["Coolant temperature"].max()),
        "n_comp_mean_rpm": float(np.mean(commands[:, 0])),
        "n_comp_min_rpm": float(commands[:, 0].min()),
        "n_comp_max_rpm": float(commands[:, 0].max()),
        "compressor_above_5800_ratio": float(np.mean(commands[:, 0] > 5800.0)),
        "actual_comp_mean_rpm": float(np.mean(actual_speeds[:, 0])),
        "actual_comp_above_5800_ratio": float(
            np.mean(actual_speeds[:, 0] > 5800.0)
        ),
        "n_pump_mean_rpm": float(np.mean(commands[:, 1])),
        "n_pump_min_rpm": float(commands[:, 1].min()),
        "n_pump_max_rpm": float(commands[:, 1].max()),
        "pump_total_variation_rpm": float(np.sum(np.abs(pump_moves))),
        "pump_direction_changes": pump_direction_changes,
        "actual_pump_mean_rpm": float(np.mean(actual_speeds[:, 1])),
        "actual_pump_total_variation_rpm": float(
            np.sum(np.abs(actual_pump_moves))
        ),
        "actual_pump_direction_changes": actual_pump_direction_changes,
        "pump_at_minimum_ratio": float(np.mean(commands[:, 1] <= 1601.0)),
        "max_abs_dcomp_rpm": float(np.max(np.abs(deltas[:, 0]))),
        "max_abs_dpump_rpm": float(np.max(np.abs(deltas[:, 1]))),
        "solve_time_median_s": float(np.nanmedian(solve_times)),
        "solve_time_max_s": float(np.nanmax(solve_times)),
        "energy_kwh": float(data["Cumulative energy consumption"].iloc[-1]),
    }


def run_short_validation(
    duration_s: float,
    output_root: Path,
    weights: QPMPCWeights | dict | None = None,
    case_label: str = "baseline",
    linear_model_bank: Path | None = None,
) -> dict:
    duration = float(duration_s)
    if duration <= 0.0 or duration % 5.0:
        raise ValueError("duration_s must be a positive multiple of 5 s")
    steps = int(duration / 5.0)
    label_text = str(case_label).strip()
    if not label_text or not all(
        character.isalnum() or character == "_" for character in label_text
    ):
        raise ValueError("case_label must contain only letters, numbers, or underscores")
    label = f"peak_forward_{int(duration)}s_{label_text}"
    result = simulate_case(
        control="state_space_qp",
        scene="peak",
        flow="单向",
        source_csv=ROOT / "_state_space_qp_generated_time_axis.csv",
        main_name=f"{label}.csv",
        snap_name=f"{label}_snapshots.csv",
        output_root=output_root,
        duration_s=duration,
        current_profile_override=np.full(steps, 560.0),
        mpc_flow_mode="standard",
        mpc_predictor="physics_p",
        mpc_predictor_artifact=DEFAULT_ARTIFACT,
        state_space_qp_weights=weights,
        state_space_qp_linear_model_bank=linear_model_bank,
        force=True,
        progress_interval_steps=max(steps, 1),
        log_func=lambda _message: None,
    )
    data = pd.read_csv(result["out_csv"], encoding="utf-8-sig")
    selected_weights = (
        QPMPCWeights()
        if weights is None
        else QPMPCWeights(**weights)
        if isinstance(weights, dict)
        else weights
    )
    summary = summarize_validation_dataframe(data)
    summary.update(
        {
            "duration_s": duration,
            "case_label": label_text,
            "weights": asdict(selected_weights),
            "csv": str(result["out_csv"]),
        }
    )
    summary_path = output_root / "state_space_qp" / f"{label}_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--linear-model-bank", type=Path, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            run_short_validation(
                args.duration_s,
                args.output_root,
                linear_model_bank=args.linear_model_bank,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
