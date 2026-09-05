"""Validate Stage 8C2 evaporator thermal dynamics in the cluster plant."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from cluster_plant_v2.refrigeration import CompressorSpeedActuator
from cluster_plant_v2.validation.regression.dynamic_refrigerated_cluster_cooling_loop import (
    DynamicRefrigeratedClusterCoolingLoop,
)
from cluster_plant_v2.refrigeration import EvaporatorThermalDynamics
from cluster_plant_v2.validation.regression.thermally_dynamic_refrigerated_cluster_cooling_loop import (
    ThermallyDynamicRefrigeratedClusterCoolingLoop,
)
from cluster_plant_v2.validation.validate_refrigerated_cluster_cooling_loop import (
    AMBIENT_TEMPERATURE_K,
    DT_S,
    PUMP_SPEED_RPM,
    build_loop,
)


DURATION_S = 600.0
CLUSTER_CURRENT_A = 560.0
FAN_SPEED_RPM = 1200.0
COMPRESSOR_TIME_CONSTANT_S = 5.0
EVAPORATOR_TIME_CONSTANT_S = 45.0


@dataclass(frozen=True)
class ThermalDynamicsCaseSpec:
    case_id: str
    initial_compressor_speed_rpm: float
    final_compressor_speed_rpm: float
    step_time_s: float = 100.0
    direction: str = "forward"


def build_case_specs() -> list[ThermalDynamicsCaseSpec]:
    return [
        ThermalDynamicsCaseSpec("H0_equilibrium", 4000.0, 4000.0),
        ThermalDynamicsCaseSpec("H1_step_up", 2000.0, 4000.0),
        ThermalDynamicsCaseSpec("H2_step_down", 4000.0, 2000.0),
    ]


def compressor_command_at_time(
    case: ThermalDynamicsCaseSpec, time_s: float
) -> float:
    if time_s < case.step_time_s:
        return case.initial_compressor_speed_rpm
    return case.final_compressor_speed_rpm


def build_stage8c1_loop(
    initial_compressor_speed_rpm: float,
) -> DynamicRefrigeratedClusterCoolingLoop:
    return DynamicRefrigeratedClusterCoolingLoop(
        cooling_loop=build_loop(),
        compressor_actuator=CompressorSpeedActuator(
            initial_speed_rpm=initial_compressor_speed_rpm,
            time_constant_s=COMPRESSOR_TIME_CONSTANT_S,
        ),
    )


def build_thermally_dynamic_loop(
    initial_compressor_speed_rpm: float,
) -> ThermallyDynamicRefrigeratedClusterCoolingLoop:
    return ThermallyDynamicRefrigeratedClusterCoolingLoop.from_equilibrium(
        dynamic_cooling_loop=build_stage8c1_loop(
            initial_compressor_speed_rpm
        ),
        pump_speed_rpm=PUMP_SPEED_RPM,
        fan_speed_rpm=FAN_SPEED_RPM,
        ambient_temperature_k=AMBIENT_TEMPERATURE_K,
        evaporator_time_constant_s=EVAPORATOR_TIME_CONSTANT_S,
    )


def _constant_target_response(dt_s: float, duration_s: float) -> float:
    dynamics = EvaporatorThermalDynamics(
        initial_q_evap_applied_w=1000.0,
        time_constant_s=EVAPORATOR_TIME_CONSTANT_S,
    )
    for _ in range(int(round(duration_s / dt_s))):
        result = dynamics.step(dt_s=dt_s, q_evap_cycle_w=2000.0)
    return float(result["q_evap_applied_w"])


def run_component_gates() -> dict[str, object]:
    dynamics = EvaporatorThermalDynamics(
        initial_q_evap_applied_w=1000.0,
        time_constant_s=EVAPORATOR_TIME_CONSTANT_S,
    )
    upward = {}
    max_dynamic_residual = 0.0
    for step in range(1, 46):
        result = dynamics.step(dt_s=5.0, q_evap_cycle_w=2000.0)
        if step in (1, 9, 27, 45):
            upward[step * 5] = result["q_evap_applied_w"]
        max_dynamic_residual = max(
            max_dynamic_residual,
            abs(result["evaporator_dynamic_energy_residual_j"]),
        )

    downward_dynamics = EvaporatorThermalDynamics(
        initial_q_evap_applied_w=2000.0,
        time_constant_s=EVAPORATOR_TIME_CONSTANT_S,
    )
    downward = np.asarray(
        [
            downward_dynamics.step(dt_s=5.0, q_evap_cycle_w=1000.0)[
                "q_evap_applied_w"
            ]
            for _ in range(45)
        ],
        dtype=float,
    )
    dt_values = np.asarray(
        [
            _constant_target_response(dt_s, 225.0)
            for dt_s in (5.0, 2.5, 1.0)
        ],
        dtype=float,
    )
    expected = {
        time_s: 2000.0 - 1000.0 * math.exp(-time_s / 45.0)
        for time_s in (5, 45, 135, 225)
    }
    result = {
        "up_5s_w": upward[5],
        "up_45s_w": upward[45],
        "up_135s_w": upward[135],
        "up_225s_w": upward[225],
        "down_5s_w": downward[0],
        "down_45s_w": downward[8],
        "down_135s_w": downward[26],
        "down_225s_w": downward[44],
        "dt_5s_final_w": dt_values[0],
        "dt_2p5s_final_w": dt_values[1],
        "dt_1s_final_w": dt_values[2],
        "dt_independence_spread_w": float(np.ptp(dt_values)),
        "max_dynamic_energy_residual_j": max_dynamic_residual,
        "downward_monotonic_without_overshoot": bool(
            np.all(np.diff(downward) < 0.0)
            and np.all(downward > 1000.0)
        ),
    }
    result["all_gates_pass"] = bool(
        all(abs(upward[t] - expected[t]) <= 1e-9 for t in expected)
        and result["dt_independence_spread_w"] <= 1e-9
        and result["max_dynamic_energy_residual_j"] <= 1e-9
        and result["downward_monotonic_without_overshoot"]
    )
    return result


def _result_passes_gates(result: dict[str, object]) -> bool:
    return bool(
        result["all_states_finite"]
        and result["refrigeration_solver_success"]
        and result["q_evap_cycle_w"] >= 0.0
        and result["q_evap_applied_w"] >= 0.0
        and result["supply_temperature_k"] > 0.0
        and abs(result["cycle_evaporator_relative_residual"]) <= 1e-9
        and abs(result["cycle_condenser_relative_residual"]) <= 1e-9
        and abs(result["cycle_energy_residual_w"]) <= 1e-8
        and abs(result["evaporator_coolant_residual_w"]) <= 1e-8
        and abs(result["evaporator_dynamic_energy_residual_j"]) <= 1e-9
        and abs(result["cluster_fluid_residual_w"]) <= 1e-8
        and abs(result["thermal_chain_residual_w"]) <= 1e-8
        and abs(result["bpt_energy_residual_j"]) <= 1e-5
        and abs(result["bpt_evaporator_energy_residual_j"]) <= 1e-5
        and abs(
            result["cluster_result"][
                "mass_flow_conservation_residual_kg_s"
            ]
        )
        <= 1e-10
        and result["cluster_result"]["max_abs_pack_coupling_residual_w"]
        <= 1e-8
    )


def run_one_step_gate() -> dict[str, object]:
    stage8c1 = build_stage8c1_loop(4000.0)
    loop = ThermallyDynamicRefrigeratedClusterCoolingLoop(
        dynamic_cooling_loop=stage8c1,
        evaporator_dynamics=EvaporatorThermalDynamics(
            initial_q_evap_applied_w=1000.0,
            time_constant_s=EVAPORATOR_TIME_CONSTANT_S,
        ),
    )
    result = loop.step(
        dt_s=DT_S,
        cluster_current_a=CLUSTER_CURRENT_A,
        pump_speed_rpm=PUMP_SPEED_RPM,
        compressor_speed_command_rpm=4000.0,
        fan_speed_rpm=FAN_SPEED_RPM,
        ambient_temperature_k=AMBIENT_TEMPERATURE_K,
        direction="forward",
    )
    return {**result, "dt_s": DT_S, "all_gates_pass": _result_passes_gates(result)}


def _timeseries_row(
    case: ThermalDynamicsCaseSpec,
    *,
    time_s: float,
    result: dict[str, object],
    stage8c1_reference: dict[str, object] | None,
) -> dict[str, object]:
    cluster = result["cluster_result"]
    battery_average_c = float(
        np.mean(cluster["pack_battery_average_temperatures_k"]) - 273.15
    )
    if stage8c1_reference is None:
        reference_q_difference = None
        reference_supply_difference = None
        reference_tank_difference = None
        reference_battery_difference = None
    else:
        reference_cluster = stage8c1_reference["cluster_result"]
        reference_battery_average_c = float(
            np.mean(reference_cluster["pack_battery_average_temperatures_k"])
            - 273.15
        )
        reference_q_difference = (
            result["q_evap_applied_w"]
            - stage8c1_reference["q_evaporator_w"]
        )
        reference_supply_difference = (
            result["supply_temperature_k"]
            - stage8c1_reference["supply_temperature_k"]
        )
        reference_tank_difference = (
            result["tank_temperature_after_k"]
            - stage8c1_reference["tank_temperature_after_k"]
        )
        reference_battery_difference = (
            battery_average_c - reference_battery_average_c
        )

    return {
        "case_id": case.case_id,
        "time_s": time_s,
        "compressor_command_rpm": result["compressor_speed_command_rpm"],
        "compressor_actual_rpm": result["compressor_speed_rpm"],
        "q_evap_cycle_w": result["q_evap_cycle_w"],
        "q_evap_applied_w": result["q_evap_applied_w"],
        "q_cond_cycle_w": result["q_condenser_cycle_w"],
        "w_ref_w": result["refrigerant_compression_power_w"],
        "w_shaft_w": result["compressor_shaft_power_w"],
        "evaporator_buffer_energy_j": result[
            "evaporator_buffer_energy_j"
        ],
        "tank_temp_c": result["tank_temperature_after_k"] - 273.15,
        "supply_temp_c": result["supply_temperature_k"] - 273.15,
        "return_temp_c": result["return_temperature_k"] - 273.15,
        "battery_avg_temp_c": battery_average_c,
        "battery_max_temp_c": cluster["cluster_max_temperature_k"] - 273.15,
        "cycle_evaporator_relative_residual": result[
            "cycle_evaporator_relative_residual"
        ],
        "cycle_condenser_relative_residual": result[
            "cycle_condenser_relative_residual"
        ],
        "cycle_energy_residual_w": result["cycle_energy_residual_w"],
        "evaporator_coolant_residual_w": result[
            "evaporator_coolant_residual_w"
        ],
        "evaporator_dynamic_energy_residual_j": result[
            "evaporator_dynamic_energy_residual_j"
        ],
        "cluster_fluid_residual_w": result["cluster_fluid_residual_w"],
        "thermal_chain_residual_w": result["thermal_chain_residual_w"],
        "bpt_energy_residual_j": result["bpt_energy_residual_j"],
        "bpt_evaporator_energy_residual_j": result[
            "bpt_evaporator_energy_residual_j"
        ],
        "stage8c1_q_evap_difference_w": reference_q_difference,
        "stage8c1_supply_temperature_difference_k": (
            reference_supply_difference
        ),
        "stage8c1_tank_temperature_difference_k": reference_tank_difference,
        "stage8c1_battery_average_temperature_difference_k": (
            reference_battery_difference
        ),
        "all_states_finite": result["all_states_finite"],
        "all_step_gates_pass": _result_passes_gates(result),
    }


def run_case(
    case: ThermalDynamicsCaseSpec,
    *,
    duration_s: float = DURATION_S,
    dt_s: float = DT_S,
) -> dict[str, object]:
    steps_float = float(duration_s) / float(dt_s)
    steps = int(round(steps_float))
    if steps < 1 or not np.isclose(steps_float, steps):
        raise ValueError("duration_s must be a positive integer multiple of dt_s")

    loop = build_thermally_dynamic_loop(
        case.initial_compressor_speed_rpm
    )
    stage8c1_reference_loop = (
        build_stage8c1_loop(case.initial_compressor_speed_rpm)
        if case.case_id == "H0_equilibrium"
        else None
    )
    rows: list[dict[str, object]] = []
    solver_failure_count = 0
    failure_message = ""
    start = perf_counter()
    for step_index in range(steps):
        time_before = step_index * dt_s
        time_after = (step_index + 1) * dt_s
        command = compressor_command_at_time(case, time_before)
        try:
            result = loop.step(
                dt_s=dt_s,
                cluster_current_a=CLUSTER_CURRENT_A,
                pump_speed_rpm=PUMP_SPEED_RPM,
                compressor_speed_command_rpm=command,
                fan_speed_rpm=FAN_SPEED_RPM,
                ambient_temperature_k=AMBIENT_TEMPERATURE_K,
                direction=case.direction,
            )
            reference = None
            if stage8c1_reference_loop is not None:
                reference = stage8c1_reference_loop.step(
                    dt_s=dt_s,
                    cluster_current_a=CLUSTER_CURRENT_A,
                    pump_speed_rpm=PUMP_SPEED_RPM,
                    compressor_speed_command_rpm=command,
                    fan_speed_rpm=FAN_SPEED_RPM,
                    ambient_temperature_k=AMBIENT_TEMPERATURE_K,
                    direction=case.direction,
                )
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            solver_failure_count += 1
            failure_message = str(exc)
            break
        rows.append(
            _timeseries_row(
                case,
                time_s=time_after,
                result=result,
                stage8c1_reference=reference,
            )
        )
    runtime_s = perf_counter() - start

    if not rows:
        return {
            "summary": {
                "case_id": case.case_id,
                "steps_requested": steps,
                "steps_completed": 0,
                "solver_failure_count": solver_failure_count,
                "failure_message": failure_message,
                "all_states_finite": False,
                "all_gates_pass": False,
                "runtime_s": runtime_s,
            },
            "timeseries": [],
        }

    frame = pd.DataFrame(rows)
    reference_columns = [
        "stage8c1_q_evap_difference_w",
        "stage8c1_supply_temperature_difference_k",
        "stage8c1_tank_temperature_difference_k",
        "stage8c1_battery_average_temperature_difference_k",
    ]
    numeric = frame.drop(columns=reference_columns).select_dtypes(
        include=[np.number]
    ).to_numpy(dtype=float)
    all_states_finite = bool(np.all(np.isfinite(numeric)))
    stage8c1_reference_run = stage8c1_reference_loop is not None
    if stage8c1_reference_run:
        stage8c1_differences = {
            "max_abs_stage8c1_q_evap_difference_w": frame[
                "stage8c1_q_evap_difference_w"
            ].abs().max(),
            "max_abs_stage8c1_supply_temperature_difference_k": frame[
                "stage8c1_supply_temperature_difference_k"
            ].abs().max(),
            "max_abs_stage8c1_tank_temperature_difference_k": frame[
                "stage8c1_tank_temperature_difference_k"
            ].abs().max(),
            "max_abs_stage8c1_battery_average_temperature_difference_k": frame[
                "stage8c1_battery_average_temperature_difference_k"
            ].abs().max(),
        }
    else:
        stage8c1_differences = {
            "max_abs_stage8c1_q_evap_difference_w": None,
            "max_abs_stage8c1_supply_temperature_difference_k": None,
            "max_abs_stage8c1_tank_temperature_difference_k": None,
            "max_abs_stage8c1_battery_average_temperature_difference_k": None,
        }
    summary = {
        "case_id": case.case_id,
        "steps_requested": steps,
        "steps_completed": len(rows),
        "initial_compressor_speed_rpm": case.initial_compressor_speed_rpm,
        "final_compressor_speed_command_rpm": case.final_compressor_speed_rpm,
        "step_time_s": case.step_time_s,
        "max_compressor_command_actual_difference_rpm": (
            frame["compressor_command_rpm"]
            - frame["compressor_actual_rpm"]
        ).abs().max(),
        "max_q_evap_cycle_applied_difference_w": (
            frame["q_evap_cycle_w"] - frame["q_evap_applied_w"]
        ).abs().max(),
        "minimum_supply_temperature_c": frame["supply_temp_c"].min(),
        "maximum_supply_temperature_c": frame["supply_temp_c"].max(),
        "minimum_tank_temperature_c": frame["tank_temp_c"].min(),
        "maximum_tank_temperature_c": frame["tank_temp_c"].max(),
        "final_tank_temperature_c": frame["tank_temp_c"].iloc[-1],
        "final_battery_average_temperature_c": frame[
            "battery_avg_temp_c"
        ].iloc[-1],
        "maximum_battery_temperature_c": frame[
            "battery_max_temp_c"
        ].max(),
        "max_abs_cycle_energy_residual_w": frame[
            "cycle_energy_residual_w"
        ].abs().max(),
        "max_abs_evaporator_coolant_residual_w": frame[
            "evaporator_coolant_residual_w"
        ].abs().max(),
        "max_abs_evaporator_dynamic_energy_residual_j": frame[
            "evaporator_dynamic_energy_residual_j"
        ].abs().max(),
        "max_abs_bpt_energy_residual_j": frame[
            "bpt_energy_residual_j"
        ].abs().max(),
        "max_abs_bpt_evaporator_energy_residual_j": frame[
            "bpt_evaporator_energy_residual_j"
        ].abs().max(),
        "stage8c1_reference_run": stage8c1_reference_run,
        **stage8c1_differences,
        "solver_failure_count": solver_failure_count,
        "failure_message": failure_message,
        "all_states_finite": all_states_finite,
        "runtime_s": runtime_s,
    }
    summary["all_gates_pass"] = bool(
        len(rows) == steps
        and solver_failure_count == 0
        and all_states_finite
        and bool(np.all(frame["all_step_gates_pass"]))
    )
    return {"summary": summary, "timeseries": rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Stage 8C2 evaporator thermal dynamics."
    )
    parser.add_argument("--duration-s", type=float, default=DURATION_S)
    parser.add_argument("--dt-s", type=float, default=DT_S)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "cluster_plant_v2/validation/results/stage8c2_validation_20260818"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    component = run_component_gates()
    pd.DataFrame([component]).to_csv(
        args.output_dir / "stage8c2_component_gates.csv", index=False
    )
    one_step = run_one_step_gate()
    pd.DataFrame(
        [{key: value for key, value in one_step.items() if np.isscalar(value)}]
    ).to_csv(args.output_dir / "stage8c2_one_step_gate.csv", index=False)

    results = []
    for case in build_case_specs():
        print(f"running {case.case_id}")
        result = run_case(case, duration_s=args.duration_s, dt_s=args.dt_s)
        results.append(result)
        pd.DataFrame(result["timeseries"]).to_csv(
            args.output_dir / f"case_{case.case_id}_timeseries.csv", index=False
        )
        summary = result["summary"]
        print(
            f"  steps={summary['steps_completed']}/{summary['steps_requested']}, "
            f"pass={summary['all_gates_pass']}"
        )

    pd.DataFrame([result["summary"] for result in results]).to_csv(
        args.output_dir / "stage8c2_summary.csv", index=False
    )
    print(f"Stage 8C2 results: {args.output_dir}")


if __name__ == "__main__":
    main()
