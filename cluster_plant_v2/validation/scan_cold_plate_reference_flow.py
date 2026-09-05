"""Stage 8D2 cold-plate reference-flow scan on the frozen Step A chiller."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import cluster_plant_v2.thermal.cold_plate_rom as cold_plate_module
from cluster_plant_v2.parameters import (
    BATTERY_PLATE_AREA_M2,
    BATTERY_PLATE_NOMINAL_HTC_W_M2_K,
    COOLANT_DENSITY_KG_M3,
)
from cluster_plant_v2.validation.audit_cluster_equipment_sizing import _trajectory


REFERENCE_FLOWS_L_MIN = (5.0, 6.5, 8.0)
TOTAL_FLOWS_L_MIN = (20.0, 25.2, 30.0)
LOADS_A = (560.0, 1120.0)


def mass_flow_from_l_min(flow_l_min: float) -> float:
    return flow_l_min / 1000.0 / 60.0 * COOLANT_DENSITY_KG_M3


def _run_case(task: tuple[float, float, float]):
    reference_l_min, total_flow_l_min, current_a = task
    original_reference = cold_plate_module.m_dot_nominal
    reference_mass_flow = mass_flow_from_l_min(reference_l_min)
    try:
        cold_plate_module.m_dot_nominal = reference_mass_flow
        frame, summary = _trajectory(
            experiment="Stage8D2_cold_plate_reference_l_min",
            case_value=reference_l_min,
            current_a=current_a,
            target_flow_l_min=total_flow_l_min,
        )
    finally:
        cold_plate_module.m_dot_nominal = original_reference
    branch_mass_flow = mass_flow_from_l_min(
        float(summary["branch_flow_mean_l_min"])
    )
    h_value = BATTERY_PLATE_NOMINAL_HTC_W_M2_K * (
        branch_mass_flow / reference_mass_flow
    ) ** 0.8
    summary.update(
        {
            "reference_flow_l_min_per_pack": reference_l_min,
            "reference_mass_flow_kg_s": reference_mass_flow,
            "target_total_flow_l_min": total_flow_l_min,
            "h_at_mean_branch_flow_w_m2_k": h_value,
            "ua_at_mean_branch_flow_w_k": h_value * BATTERY_PLATE_AREA_M2,
        }
    )
    frame["reference_flow_l_min_per_pack"] = reference_l_min
    frame["target_total_flow_l_min"] = total_flow_l_min
    return frame, summary


def formula_table() -> pd.DataFrame:
    rows = []
    for reference_l_min in REFERENCE_FLOWS_L_MIN:
        reference_mass_flow = mass_flow_from_l_min(reference_l_min)
        for branch_l_min in (4.0, 5.0, 6.0):
            branch_mass_flow = mass_flow_from_l_min(branch_l_min)
            h_value = BATTERY_PLATE_NOMINAL_HTC_W_M2_K * (
                branch_mass_flow / reference_mass_flow
            ) ** 0.8
            rows.append(
                {
                    "reference_flow_l_min_per_pack": reference_l_min,
                    "reference_mass_flow_kg_s": reference_mass_flow,
                    "branch_flow_l_min_per_pack": branch_l_min,
                    "branch_mass_flow_kg_s": branch_mass_flow,
                    "h_w_m2_k": h_value,
                    "ua_w_k": h_value * BATTERY_PLATE_AREA_M2,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "results"
            / "stage8d2_cluster_plant_rescale"
        ),
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    tasks = [
        (reference, total_flow, current)
        for reference in REFERENCE_FLOWS_L_MIN
        for total_flow in TOTAL_FLOWS_L_MIN
        for current in LOADS_A
    ]
    frames = []
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run_case, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            frame, summary = future.result()
            frames.append(frame)
            summaries.append(summary)
            print(f"completed {index}/{len(tasks)}", flush=True)
    summary_frame = pd.DataFrame(summaries).sort_values(
        ["reference_flow_l_min_per_pack", "target_total_flow_l_min", "current_a"]
    )
    timeseries = pd.concat(frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(
        args.output_dir / "cold_plate_reference_scan.csv", index=False
    )
    timeseries.to_csv(
        args.output_dir / "cold_plate_reference_scan_timeseries.csv", index=False
    )
    formula_table().to_csv(
        args.output_dir / "cold_plate_reference_formula.csv", index=False
    )
    columns = [
        "reference_flow_l_min_per_pack",
        "target_total_flow_l_min",
        "current_a",
        "h_at_mean_branch_flow_w_m2_k",
        "ua_at_mean_branch_flow_w_k",
        "return_temp_final_c",
        "battery_tavg_final_c",
        "battery_tmax_c",
        "delta_t_inter_max_k",
        "q_plate_to_fluid_last60_mean_w",
    ]
    print(summary_frame[columns].to_string(index=False))


if __name__ == "__main__":
    main()
