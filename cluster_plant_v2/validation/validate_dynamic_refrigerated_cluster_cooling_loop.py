"""Validate the Stage 8C1 compressor-speed state and energy closure."""

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
from cluster_plant_v2.validation.validate_refrigerated_cluster_cooling_loop import (
    AMBIENT_TEMPERATURE_K,
    DT_S,
    PUMP_SPEED_RPM,
    build_loop,
)


DURATION_S = 600.0
TIME_CONSTANT_S = 5.0
CLUSTER_CURRENT_A = 560.0
FAN_SPEED_RPM = 1200.0


@dataclass(frozen=True)
class DynamicCompressorCaseSpec:
    case_id: str
    initial_speed_rpm: float
    speed_command_rpm: float
    direction: str = "forward"


def build_case_specs() -> list[DynamicCompressorCaseSpec]:
    return [
        DynamicCompressorCaseSpec("C1_hold_4000", 4000.0, 4000.0),
        DynamicCompressorCaseSpec("C2_step_up_2000_to_4000", 2000.0, 4000.0),
        DynamicCompressorCaseSpec("C3_step_down_4000_to_2000", 4000.0, 2000.0),
    ]


def build_dynamic_loop(initial_speed_rpm: float) -> DynamicRefrigeratedClusterCoolingLoop:
    return DynamicRefrigeratedClusterCoolingLoop(
        cooling_loop=build_loop(),
        compressor_actuator=CompressorSpeedActuator(
            initial_speed_rpm=initial_speed_rpm,
            time_constant_s=TIME_CONSTANT_S,
        ),
    )


def _timeseries_row(
    case: DynamicCompressorCaseSpec,
    *,
    time_s: float,
    result: dict[str, object],
) -> dict[str, object]:
    expected_speed = case.speed_command_rpm + (
        case.initial_speed_rpm - case.speed_command_rpm
    ) * math.exp(-time_s / TIME_CONSTANT_S)
    return {
        "case_id": case.case_id,
        "time_s": time_s,
        "compressor_speed_command_rpm": result[
            "compressor_speed_command_rpm"
        ],
        "compressor_speed_before_rpm": result[
            "compressor_speed_before_rpm"
        ],
        "compressor_speed_rpm": result["compressor_speed_rpm"],
        "expected_compressor_speed_rpm": expected_speed,
        "speed_model_residual_rpm": result["compressor_speed_rpm"]
        - expected_speed,
        "q_evaporator_w": result["q_evaporator_w"],
        "q_condenser_w": result["q_condenser_w"],
        "compressor_shaft_power_w": result["compressor_shaft_power_w"],
        "tank_temperature_c": result["tank_temperature_after_k"] - 273.15,
        "supply_temperature_c": result["supply_temperature_k"] - 273.15,
        "return_temperature_c": result["return_temperature_k"] - 273.15,
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
        "cluster_fluid_residual_w": result["cluster_fluid_residual_w"],
        "thermal_chain_residual_w": result["thermal_chain_residual_w"],
        "total_energy_residual_j": result["total_energy_residual_j"],
        "mass_flow_residual_kg_s": result["cluster_result"][
            "mass_flow_conservation_residual_kg_s"
        ],
        "all_states_finite": result["all_states_finite"],
        "refrigeration_solver_success": result[
            "refrigeration_solver_success"
        ],
    }


def _rows_pass_gates(rows: list[dict[str, object]]) -> bool:
    frame = pd.DataFrame(rows)
    return bool(
        np.all(frame["all_states_finite"])
        and np.all(frame["refrigeration_solver_success"])
        and frame["speed_model_residual_rpm"].abs().max() <= 1e-9
        and frame["cycle_evaporator_relative_residual"].abs().max() <= 1e-9
        and frame["cycle_condenser_relative_residual"].abs().max() <= 1e-9
        and frame["cycle_energy_residual_w"].abs().max() <= 1e-8
        and frame["evaporator_coolant_residual_w"].abs().max() <= 1e-8
        and frame["cluster_fluid_residual_w"].abs().max() <= 1e-8
        and frame["thermal_chain_residual_w"].abs().max() <= 1e-8
        and frame["total_energy_residual_j"].abs().max() <= 1e-5
        and frame["mass_flow_residual_kg_s"].abs().max() <= 1e-10
    )


def run_one_step_gate() -> dict[str, object]:
    case = build_case_specs()[1]
    result = build_dynamic_loop(case.initial_speed_rpm).step(
        dt_s=DT_S,
        cluster_current_a=CLUSTER_CURRENT_A,
        pump_speed_rpm=PUMP_SPEED_RPM,
        compressor_speed_command_rpm=case.speed_command_rpm,
        fan_speed_rpm=FAN_SPEED_RPM,
        ambient_temperature_k=AMBIENT_TEMPERATURE_K,
        direction=case.direction,
    )
    row = _timeseries_row(case, time_s=DT_S, result=result)
    return {**result, **row, "all_gates_pass": _rows_pass_gates([row])}


def run_case(
    case: DynamicCompressorCaseSpec,
    *,
    duration_s: float = DURATION_S,
    dt_s: float = DT_S,
) -> dict[str, object]:
    steps_float = float(duration_s) / float(dt_s)
    steps = int(round(steps_float))
    if steps < 1 or not np.isclose(steps_float, steps):
        raise ValueError("duration_s must be a positive integer multiple of dt_s")

    loop = build_dynamic_loop(case.initial_speed_rpm)
    rows: list[dict[str, object]] = []
    solver_failure_count = 0
    failure_message = ""
    start = perf_counter()
    for step_index in range(steps):
        try:
            result = loop.step(
                dt_s=dt_s,
                cluster_current_a=CLUSTER_CURRENT_A,
                pump_speed_rpm=PUMP_SPEED_RPM,
                compressor_speed_command_rpm=case.speed_command_rpm,
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
                time_s=(step_index + 1) * dt_s,
                result=result,
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
                "runtime_s": runtime_s,
                "all_gates_pass": False,
            },
            "timeseries": [],
        }

    frame = pd.DataFrame(rows)
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    all_states_finite = bool(np.all(np.isfinite(numeric)))
    speeds = frame["compressor_speed_rpm"].to_numpy(dtype=float)
    speed_differences = np.diff(speeds)
    if case.speed_command_rpm > case.initial_speed_rpm:
        monotonic_without_overshoot = bool(
            np.all(speed_differences >= 0.0)
            and np.all(speeds <= case.speed_command_rpm)
        )
    elif case.speed_command_rpm < case.initial_speed_rpm:
        monotonic_without_overshoot = bool(
            np.all(speed_differences <= 0.0)
            and np.all(speeds >= case.speed_command_rpm)
        )
    else:
        monotonic_without_overshoot = bool(
            np.all(speeds == case.initial_speed_rpm)
        )

    summary = {
        "case_id": case.case_id,
        "initial_speed_rpm": case.initial_speed_rpm,
        "speed_command_rpm": case.speed_command_rpm,
        "time_constant_s": TIME_CONSTANT_S,
        "steps_requested": steps,
        "steps_completed": len(rows),
        "final_compressor_speed_rpm": speeds[-1],
        "final_speed_error_rpm": speeds[-1] - case.speed_command_rpm,
        "max_abs_speed_model_residual_rpm": frame[
            "speed_model_residual_rpm"
        ].abs().max(),
        "monotonic_without_overshoot": monotonic_without_overshoot,
        "max_cycle_energy_residual_w": frame[
            "cycle_energy_residual_w"
        ].abs().max(),
        "max_total_energy_residual_j": frame[
            "total_energy_residual_j"
        ].abs().max(),
        "final_q_evaporator_w": frame["q_evaporator_w"].iloc[-1],
        "final_compressor_shaft_power_w": frame[
            "compressor_shaft_power_w"
        ].iloc[-1],
        "final_tank_temperature_c": frame["tank_temperature_c"].iloc[-1],
        "solver_failure_count": solver_failure_count,
        "failure_message": failure_message,
        "all_states_finite": all_states_finite,
        "runtime_s": runtime_s,
    }
    summary["all_gates_pass"] = bool(
        len(rows) == steps
        and solver_failure_count == 0
        and all_states_finite
        and monotonic_without_overshoot
        and _rows_pass_gates(rows)
    )
    return {"summary": summary, "timeseries": rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Stage 8C1 compressor actuator dynamics."
    )
    parser.add_argument("--duration-s", type=float, default=DURATION_S)
    parser.add_argument("--dt-s", type=float, default=DT_S)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "cluster_plant_v2/validation/results/stage8c1_validation_20260818"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    one_step = run_one_step_gate()
    pd.DataFrame(
        [{key: value for key, value in one_step.items() if np.isscalar(value)}]
    ).to_csv(args.output_dir / "stage8c1_one_step_gate.csv", index=False)

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
            f"speed={summary.get('final_compressor_speed_rpm', np.nan):.6f} rpm, "
            f"pass={summary['all_gates_pass']}"
        )

    pd.DataFrame([result["summary"] for result in results]).to_csv(
        args.output_dir / "stage8c1_summary.csv", index=False
    )
    print(f"Stage 8C1 results: {args.output_dir}")


if __name__ == "__main__":
    main()
