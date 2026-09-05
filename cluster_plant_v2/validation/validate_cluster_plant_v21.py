"""Validate the Stage 8D2 Cluster-Sized Plant V2.1 baseline."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_plant_v2.parameters import BATTERY_PLATE_AREA_M2
from cluster_plant_v2.validation.validate_final_cluster_plant import (
    AMBIENT_TEMPERATURE_K,
    DT_S,
    FAN_SPEED_RPM,
    PUMP_SPEED_RPM,
    build_final_plant,
    load_regd_case_currents,
)


@dataclass(frozen=True)
class V21Case:
    case_id: str
    duration_s: float
    current_kind: str
    current_a: float | None
    compressor_initial_rpm: float
    compressor_final_rpm: float
    compressor_step_s: float = 100.0
    direction_kind: str = "forward"
    direction_step_s: float = 300.0


def build_v21_cases() -> list[V21Case]:
    return [
        V21Case("T0_constant", 600.0, "constant", 560.0, 4000.0, 4000.0),
        V21Case("T1_compressor_step", 600.0, "constant", 560.0, 2000.0, 4000.0),
        V21Case("T2_regd", 600.0, "regd", None, 4000.0, 4000.0),
        V21Case("T3_560a_thermal_balance", 600.0, "constant", 560.0, 4000.0, 4000.0),
        V21Case("T4_1120a_stress", 600.0, "constant", 1120.0, 6000.0, 6000.0),
        V21Case(
            "D_reverse",
            600.0,
            "constant",
            560.0,
            4000.0,
            4000.0,
            direction_kind="reverse",
        ),
        V21Case(
            "D_forward_to_reverse",
            600.0,
            "constant",
            560.0,
            4000.0,
            4000.0,
            direction_kind="forward_to_reverse",
        ),
    ]


def _direction(case: V21Case, time_s: float) -> str:
    if case.direction_kind != "forward_to_reverse":
        return case.direction_kind
    return "forward" if time_s < case.direction_step_s else "reverse"


def _compressor_command(case: V21Case, time_s: float) -> float:
    if time_s < case.compressor_step_s:
        return case.compressor_initial_rpm
    return case.compressor_final_rpm


def _temperature_slope_c_per_hour(frame: pd.DataFrame, tail_s: float) -> float:
    tail = frame[frame["time_s"] > frame["time_s"].iloc[-1] - tail_s]
    if len(tail) < 2:
        return np.nan
    slope_c_per_s = np.polyfit(
        tail["time_s"].to_numpy(dtype=float),
        tail["battery_tavg_c"].to_numpy(dtype=float),
        1,
    )[0]
    return float(slope_c_per_s * 3600.0)


def run_v21_case(case: V21Case) -> tuple[pd.DataFrame, dict[str, object]]:
    steps = int(round(case.duration_s / DT_S))
    if case.current_kind == "regd":
        _, currents = load_regd_case_currents(duration_s=case.duration_s, dt_s=DT_S)
    else:
        currents = np.full(steps, float(case.current_a), dtype=float)
    plant = build_final_plant(
        case.compressor_initial_rpm,
        initial_cluster_current_a=float(currents[0]),
        direction=_direction(case, 0.0),
    )
    rows: list[dict[str, object]] = []
    failure_type = ""
    failure_message = ""
    for index in range(steps):
        time_before = index * DT_S
        direction = _direction(case, time_before)
        try:
            result = plant.step(
                dt_s=DT_S,
                cluster_current_a=float(currents[index]),
                pump_speed_rpm=PUMP_SPEED_RPM,
                compressor_speed_command_rpm=_compressor_command(case, time_before),
                fan_speed_rpm=FAN_SPEED_RPM,
                ambient_temperature_k=AMBIENT_TEMPERATURE_K,
                direction=direction,
            )
        except Exception as exc:
            failure_type = type(exc).__name__
            failure_message = str(exc)
            break
        refrigeration = result["refrigeration_result"]
        cluster = result["cluster_result"]
        q_air_w = float(
            sum(pack.battery.get_air_heat_loss().sum() for pack in plant.cluster.packs)
        )
        h_values = np.array(
            [pack.cold_plate.h_dynamic for pack in plant.cluster.packs], dtype=float
        )
        q_load_w = float(result["cluster_q_gen_total_w"] - q_air_w)
        q_evap_w = float(result["q_evap_applied_w"])
        shaft_power_w = float(result["compressor_shaft_power_w"])
        rows.append(
            {
                "case_id": case.case_id,
                "time_s": (index + 1) * DT_S,
                "current_a": float(currents[index]),
                "direction": direction,
                "compressor_command_rpm": _compressor_command(case, time_before),
                "compressor_actual_rpm": float(result["compressor_speed_rpm"]),
                "battery_tavg_c": float(
                    np.mean(cluster["pack_battery_average_temperatures_k"]) - 273.15
                ),
                "battery_tmax_c": float(cluster["cluster_max_temperature_k"] - 273.15),
                "delta_t_cluster_k": float(cluster["cluster_delta_temperature_k"]),
                "delta_t_inter_k": float(cluster["inter_pack_delta_temperature_k"]),
                "supply_temp_c": float(result["cluster_supply_temperature_k"] - 273.15),
                "return_temp_c": float(result["cluster_return_temperature_k"] - 273.15),
                "tank_temp_c": float(result["tank_temperature_after_k"] - 273.15),
                "q_gen_w": float(result["cluster_q_gen_total_w"]),
                "q_air_w": q_air_w,
                "q_load_w": q_load_w,
                "q_evap_cycle_w": float(result["q_evap_cycle_w"]),
                "q_evap_applied_w": q_evap_w,
                "thermal_deficit_w": q_load_w - q_evap_w,
                "q_condenser_w": float(result["q_condenser_cycle_w"]),
                "wref_w": float(result["refrigerant_compression_power_w"]),
                "wshaft_w": shaft_power_w,
                "cop_shaft": float(result["q_evap_cycle_w"]) / shaft_power_w,
                "refrigerant_mass_flow_kg_s": float(
                    refrigeration["refrigerant_mass_flow_kg_s"]
                ),
                "tevap_c": float(
                    refrigeration["evaporating_saturation_temperature_k"] - 273.15
                ),
                "tcond_c": float(
                    refrigeration["condensing_saturation_temperature_k"] - 273.15
                ),
                "cold_plate_h_mean_w_m2_k": float(h_values.mean()),
                "cold_plate_ua_mean_w_k": float(
                    h_values.mean() * BATTERY_PLATE_AREA_M2
                ),
                "cycle_mass_relative_residual": float(
                    result["cycle_mass_relative_residual"]
                ),
                "cycle_evaporator_relative_residual": float(
                    result["cycle_evaporator_relative_residual"]
                ),
                "cycle_condenser_relative_residual": float(
                    result["cycle_condenser_relative_residual"]
                ),
                "cycle_energy_residual_w": float(result["cycle_energy_residual_w"]),
                "evaporator_coolant_residual_w": float(
                    result["evaporator_coolant_residual_w"]
                ),
                "cluster_fluid_residual_w": float(result["cluster_fluid_residual_w"]),
                "tank_energy_residual_j": float(result["tank_energy_residual_j"]),
                "all_states_finite": bool(result["all_states_finite"]),
                "solver_success": bool(result["refrigeration_solver_success"]),
            }
        )
    frame = pd.DataFrame(rows)
    completed = len(frame) == steps
    if frame.empty:
        return frame, {
            "case_id": case.case_id,
            "steps_requested": steps,
            "steps_completed": 0,
            "run_completed": False,
            "all_gates_pass": False,
            "failure_type": failure_type,
            "failure_message": failure_message,
        }
    tail_s = min(600.0, case.duration_s)
    tail = frame[frame["time_s"] > case.duration_s - tail_s]
    max_residuals = {
        "max_mass_relative_residual": float(frame["cycle_mass_relative_residual"].abs().max()),
        "max_evaporator_relative_residual": float(
            frame["cycle_evaporator_relative_residual"].abs().max()
        ),
        "max_condenser_relative_residual": float(
            frame["cycle_condenser_relative_residual"].abs().max()
        ),
        "max_cycle_energy_residual_w": float(frame["cycle_energy_residual_w"].abs().max()),
        "max_evaporator_coolant_residual_w": float(
            frame["evaporator_coolant_residual_w"].abs().max()
        ),
        "max_cluster_fluid_residual_w": float(
            frame["cluster_fluid_residual_w"].abs().max()
        ),
        "max_tank_energy_residual_j": float(frame["tank_energy_residual_j"].abs().max()),
    }
    summary: dict[str, object] = {
        "case_id": case.case_id,
        "duration_s": case.duration_s,
        "steps_requested": steps,
        "steps_completed": len(frame),
        "run_completed": completed,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "final_battery_tavg_c": float(frame["battery_tavg_c"].iloc[-1]),
        "maximum_battery_tmax_c": float(frame["battery_tmax_c"].max()),
        "maximum_delta_t_cluster_k": float(frame["delta_t_cluster_k"].max()),
        "maximum_delta_t_inter_k": float(frame["delta_t_inter_k"].max()),
        "final_supply_temp_c": float(frame["supply_temp_c"].iloc[-1]),
        "final_return_temp_c": float(frame["return_temp_c"].iloc[-1]),
        "tail_qload_mean_w": float(tail["q_load_w"].mean()),
        "tail_qevap_applied_mean_w": float(tail["q_evap_applied_w"].mean()),
        "tail_thermal_deficit_mean_w": float(tail["thermal_deficit_w"].mean()),
        "tail_temperature_slope_c_per_hour": _temperature_slope_c_per_hour(
            frame, tail_s
        ),
        "tail_wref_mean_w": float(tail["wref_w"].mean()),
        "tail_wshaft_mean_w": float(tail["wshaft_w"].mean()),
        "tail_cop_mean": float(tail["cop_shaft"].mean()),
        "tail_refrigerant_mass_flow_mean_kg_s": float(
            tail["refrigerant_mass_flow_kg_s"].mean()
        ),
        "tail_tevap_mean_c": float(tail["tevap_c"].mean()),
        "tail_tcond_mean_c": float(tail["tcond_c"].mean()),
        "cold_plate_h_mean_w_m2_k": float(frame["cold_plate_h_mean_w_m2_k"].mean()),
        "cold_plate_ua_mean_w_k": float(frame["cold_plate_ua_mean_w_k"].mean()),
        "thermal_balance_available": bool(
            float(tail["q_evap_applied_w"].mean())
            >= float(tail["q_load_w"].mean())
        ),
        **max_residuals,
    }
    summary["all_gates_pass"] = bool(
        completed
        and frame["all_states_finite"].all()
        and frame["solver_success"].all()
        and max_residuals["max_mass_relative_residual"] < 1e-3
        and max_residuals["max_evaporator_relative_residual"] < 5e-3
        and max_residuals["max_condenser_relative_residual"] < 5e-3
        and max_residuals["max_cycle_energy_residual_w"] <= 1e-8
        and max_residuals["max_evaporator_coolant_residual_w"] <= 1e-8
        and max_residuals["max_cluster_fluid_residual_w"] <= 1e-8
        and max_residuals["max_tank_energy_residual_j"] <= 1e-8
    )
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "results"
            / "stage8d2_cluster_plant_rescale"
            / "v21_validation"
        ),
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_v21_cases()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_v21_case, case): case for case in cases}
        for future in as_completed(futures):
            case = futures[future]
            frame, summary = future.result()
            results.append(summary)
            frame.to_csv(args.output_dir / f"{case.case_id}_timeseries.csv", index=False)
            print(
                f"{case.case_id}: {summary['steps_completed']}/"
                f"{summary['steps_requested']} pass={summary['all_gates_pass']}",
                flush=True,
            )
    summary_frame = pd.DataFrame(results).sort_values("case_id")
    summary_frame.to_csv(args.output_dir / "v21_summary.csv", index=False)
    by_case = summary_frame.set_index("case_id")
    metadata = {
        "baseline": "Cluster Plant V2.1 / Cluster-Sized Baseline",
        "dt_s": DT_S,
        "tank_l": 3.0,
        "pump_rpm": PUMP_SPEED_RPM,
        "branch_delta_p_pa": 20000.0,
        "all_cases_pass": bool(summary_frame["all_gates_pass"].all()),
        "t3_560a_thermal_balance_available": bool(
            by_case.loc[
                "T3_560a_thermal_balance", "thermal_balance_available"
            ]
        ),
        "t4_1120a_expected_thermal_deficit_w": float(
            by_case.loc[
                "T4_1120a_stress", "tail_thermal_deficit_mean_w"
            ]
        ),
    }
    metadata["formal_acceptance_pass"] = bool(
        metadata["all_cases_pass"]
        and metadata["t3_560a_thermal_balance_available"]
        and metadata["t4_1120a_expected_thermal_deficit_w"] > 0.0
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
