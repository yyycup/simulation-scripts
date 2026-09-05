"""Validate the Stage 8C3 final cluster plant and transport delays."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from cluster_plant_v2.thermal.cluster import ReducedCluster
from cluster_plant_v2.hydraulics import CoolantTank
from cluster_plant_v2.parameters import (
    COMPRESSOR_TIME_CONSTANT_S,
    EVAPORATOR_TIME_CONSTANT_S,
)
from cluster_plant_v2.plant import ClusterPlant
from cluster_plant_v2.refrigeration import (
    ClosedR134aCycle,
    CompressorSpeedActuator,
    CoolantTransportDelay,
)
from cluster_plant_v2.validation.compare_legacy_reference import (
    load_regd_profile,
)
from cluster_plant_v2.validation.validate_refrigerated_cluster_cooling_loop import (
    AMBIENT_TEMPERATURE_K,
    DT_S,
    INITIAL_SOC,
    INITIAL_TEMPERATURE_K,
    PUMP_SPEED_RPM,
    build_engineering_pump,
    build_medium_header_network,
)
from cluster_plant_v2.profiles import AGC_DATA_FILE


DURATION_S = 600.0
FAN_SPEED_RPM = 1200.0


@dataclass(frozen=True)
class FinalPlantCaseSpec:
    case_id: str
    current_kind: str
    current_a: float | None
    initial_compressor_speed_rpm: float
    final_compressor_speed_rpm: float
    step_time_s: float = 100.0
    direction: str = "forward"


def build_case_specs() -> list[FinalPlantCaseSpec]:
    return [
        FinalPlantCaseSpec(
            "T0_constant", "constant", 560.0, 4000.0, 4000.0
        ),
        FinalPlantCaseSpec(
            "T1_compressor_step", "constant", 560.0, 2000.0, 4000.0
        ),
        FinalPlantCaseSpec(
            "T2_regd", "regd", None, 4000.0, 4000.0
        ),
    ]


def compressor_command_at_time(case: FinalPlantCaseSpec, time_s: float) -> float:
    if time_s < case.step_time_s:
        return case.initial_compressor_speed_rpm
    return case.final_compressor_speed_rpm


def load_regd_case_currents(
    *, duration_s: float = DURATION_S, dt_s: float = DT_S
) -> tuple[np.ndarray, np.ndarray]:
    return load_regd_profile(
        AGC_DATA_FILE,
        duration_s=duration_s,
        dt=dt_s,
    )


def run_delay_gates() -> dict[str, object]:
    supply = CoolantTransportDelay(
        delay_s=15.0, dt_s=5.0, initial_value=25.0
    )
    supply_outputs = [supply.step(20.0) for _ in range(5)]
    supply_change_index = next(
        index
        for index, value in enumerate(supply_outputs)
        if value == 20.0
    )
    return_delay = CoolantTransportDelay(
        delay_s=20.0, dt_s=5.0, initial_value=25.0
    )
    return_outputs = [return_delay.step(20.0) for _ in range(6)]
    return_change_index = next(
        index
        for index, value in enumerate(return_outputs)
        if value == 20.0
    )
    result = {
        "supply_delay_steps": supply.delay_steps,
        "return_delay_steps": return_delay.delay_steps,
        "measured_supply_delay_s": supply_change_index * 5.0,
        "measured_return_delay_s": return_change_index * 5.0,
        "supply_outputs": "|".join(map(str, supply_outputs)),
        "return_outputs": "|".join(map(str, return_outputs)),
    }
    result["all_gates_pass"] = bool(
        result["supply_delay_steps"] == 3
        and result["return_delay_steps"] == 4
        and result["measured_supply_delay_s"] == 15.0
        and result["measured_return_delay_s"] == 20.0
    )
    return result


def build_final_plant(
    initial_compressor_speed_rpm: float,
    *,
    initial_cluster_current_a: float,
    direction: str,
) -> ClusterPlant:
    network = build_medium_header_network()
    cluster = ReducedCluster(
        n_packs=5,
        hydraulic_mode="header_network",
        hydraulic_network=network,
        pack_config={
            "initial_soc": INITIAL_SOC,
            "initial_battery_temperature_k": INITIAL_TEMPERATURE_K,
            "initial_plate_temperature_k": INITIAL_TEMPERATURE_K,
        },
    )
    pump = build_engineering_pump(network)
    tank = CoolantTank(initial_temperature_k=INITIAL_TEMPERATURE_K)
    cycle = ClosedR134aCycle()
    actuator = CompressorSpeedActuator(
        initial_speed_rpm=initial_compressor_speed_rpm,
        time_constant_s=COMPRESSOR_TIME_CONSTANT_S,
    )
    return ClusterPlant.from_equilibrium(
        cluster=cluster,
        hydraulic_network=network,
        pump=pump,
        tank=tank,
        refrigeration_cycle=cycle,
        compressor_actuator=actuator,
        dt_s=DT_S,
        pump_speed_rpm=PUMP_SPEED_RPM,
        fan_speed_rpm=FAN_SPEED_RPM,
        initial_cluster_current_a=initial_cluster_current_a,
        ambient_temperature_k=AMBIENT_TEMPERATURE_K,
        direction=direction,
        evaporator_time_constant_s=EVAPORATOR_TIME_CONSTANT_S,
    )


def _result_passes_local_gates(result: dict[str, object]) -> bool:
    return bool(
        result["all_states_finite"]
        and result["refrigeration_solver_success"]
        and result["evaporator_outlet_temperature_k"] > 0.0
        and result["cluster_supply_temperature_k"] > 0.0
        and result["tank_return_temperature_k"] > 0.0
        and abs(result["cycle_evaporator_relative_residual"]) <= 1e-9
        and abs(result["cycle_condenser_relative_residual"]) <= 1e-9
        and abs(result["cycle_energy_residual_w"]) <= 1e-8
        and abs(result["evaporator_coolant_residual_w"]) <= 1e-8
        and abs(result["evaporator_dynamic_energy_residual_j"]) <= 1e-9
        and abs(result["cluster_fluid_residual_w"]) <= 1e-8
        and abs(result["tank_energy_residual_j"]) <= 1e-8
        and abs(
            result["cluster_result"][
                "mass_flow_conservation_residual_kg_s"
            ]
        )
        <= 1e-10
        and result["cluster_result"]["max_abs_pack_coupling_residual_w"]
        <= 1e-8
    )


def _timeseries_row(
    case: FinalPlantCaseSpec,
    *,
    time_s: float,
    current_a: float,
    expected_supply_k: float,
    expected_return_k: float,
    result: dict[str, object],
) -> dict[str, object]:
    cluster = result["cluster_result"]
    pack_flows = np.asarray(cluster["pack_mass_flows_kg_s"], dtype=float)
    row = {
        "case_id": case.case_id,
        "time_s": time_s,
        "cluster_current_a": current_a,
        "compressor_command_rpm": result["compressor_speed_command_rpm"],
        "compressor_actual_rpm": result["compressor_speed_rpm"],
        "q_evap_cycle_w": result["q_evap_cycle_w"],
        "q_evap_applied_w": result["q_evap_applied_w"],
        "evaporator_outlet_temp_c": (
            result["evaporator_outlet_temperature_k"] - 273.15
        ),
        "cluster_supply_temp_c": (
            result["cluster_supply_temperature_k"] - 273.15
        ),
        "cluster_return_temp_c": (
            result["cluster_return_temperature_k"] - 273.15
        ),
        "tank_return_temp_c": result["tank_return_temperature_k"] - 273.15,
        "tank_temp_c": result["tank_temperature_after_k"] - 273.15,
        "battery_avg_temp_c": float(
            np.mean(cluster["pack_battery_average_temperatures_k"])
            - 273.15
        ),
        "battery_max_temp_c": cluster["cluster_max_temperature_k"] - 273.15,
        "delta_t_inter_k": cluster["inter_pack_delta_temperature_k"],
        "delta_t_cluster_k": cluster["cluster_delta_temperature_k"],
        "total_mass_flow_kg_s": result["total_mass_flow_kg_s"],
        "q_gen_w": result["cluster_q_gen_total_w"],
        "q_evap_w": result["q_evap_applied_w"],
        "pump_power_w": result["pump_power_w"],
        "compressor_power_w": result["compressor_shaft_power_w"],
        "supply_delay_residual_k": (
            result["cluster_supply_temperature_k"] - expected_supply_k
        ),
        "return_delay_residual_k": (
            result["tank_return_temperature_k"] - expected_return_k
        ),
        "cycle_energy_residual_w": result["cycle_energy_residual_w"],
        "evaporator_coolant_residual_w": result[
            "evaporator_coolant_residual_w"
        ],
        "evaporator_dynamic_energy_residual_j": result[
            "evaporator_dynamic_energy_residual_j"
        ],
        "cluster_fluid_residual_w": result["cluster_fluid_residual_w"],
        "tank_energy_residual_j": result["tank_energy_residual_j"],
        "all_states_finite": result["all_states_finite"],
        "all_local_gates_pass": _result_passes_local_gates(result),
    }
    for index, flow in enumerate(pack_flows, start=1):
        row[f"pack_{index}_mass_flow_kg_s"] = flow
    return row


def run_case(
    case: FinalPlantCaseSpec,
    *,
    duration_s: float = DURATION_S,
    dt_s: float = DT_S,
) -> dict[str, object]:
    steps_float = float(duration_s) / float(dt_s)
    steps = int(round(steps_float))
    if steps < 1 or not np.isclose(steps_float, steps):
        raise ValueError("duration_s must be a positive integer multiple of dt_s")
    if not np.isclose(dt_s, DT_S, rtol=0.0, atol=1e-12):
        raise ValueError("Stage 8C3 validation uses the frozen 5 s step")

    if case.current_kind == "regd":
        _, currents = load_regd_case_currents(
            duration_s=duration_s, dt_s=dt_s
        )
    else:
        currents = np.full(steps, float(case.current_a), dtype=float)

    plant = build_final_plant(
        case.initial_compressor_speed_rpm,
        initial_cluster_current_a=float(currents[0]),
        direction=case.direction,
    )
    initial_supply_queue = plant.supply_delay.queue_values
    initial_return_queue = plant.return_delay.queue_values
    evaporator_history: list[float] = []
    cluster_return_history: list[float] = []
    rows: list[dict[str, object]] = []
    solver_failure_count = 0
    failure_message = ""
    start = perf_counter()
    for step_index in range(steps):
        time_before = step_index * dt_s
        command = compressor_command_at_time(case, time_before)
        expected_supply = (
            initial_supply_queue[step_index]
            if step_index < plant.supply_delay.delay_steps
            else evaporator_history[
                step_index - plant.supply_delay.delay_steps
            ]
        )
        expected_return = (
            initial_return_queue[step_index]
            if step_index < plant.return_delay.delay_steps
            else cluster_return_history[
                step_index - plant.return_delay.delay_steps
            ]
        )
        try:
            result = plant.step(
                dt_s=dt_s,
                cluster_current_a=float(currents[step_index]),
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
        evaporator_history.append(result["evaporator_outlet_temperature_k"])
        cluster_return_history.append(result["cluster_return_temperature_k"])
        rows.append(
            _timeseries_row(
                case,
                time_s=(step_index + 1) * dt_s,
                current_a=float(currents[step_index]),
                expected_supply_k=float(expected_supply),
                expected_return_k=float(expected_return),
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
                "all_gates_pass": False,
                "cross_delay_energy_balance_is_modeled": False,
                "runtime_s": runtime_s,
            },
            "timeseries": [],
        }

    frame = pd.DataFrame(rows)
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    all_states_finite = bool(np.all(np.isfinite(numeric)))

    def max_abs_step(column: str) -> float:
        values = frame[column].to_numpy(dtype=float)
        return float(np.max(np.abs(np.diff(values)))) if len(values) > 1 else 0.0

    summary = {
        "case_id": case.case_id,
        "current_kind": case.current_kind,
        "steps_requested": steps,
        "steps_completed": len(rows),
        "current_min_a": frame["cluster_current_a"].min(),
        "current_max_a": frame["cluster_current_a"].max(),
        "supply_delay_steps": plant.supply_delay.delay_steps,
        "return_delay_steps": plant.return_delay.delay_steps,
        "measured_supply_delay_s": plant.supply_delay.delay_steps * dt_s,
        "measured_return_delay_s": plant.return_delay.delay_steps * dt_s,
        "max_abs_supply_delay_residual_k": frame[
            "supply_delay_residual_k"
        ].abs().max(),
        "max_abs_return_delay_residual_k": frame[
            "return_delay_residual_k"
        ].abs().max(),
        "minimum_evaporator_outlet_temperature_c": frame[
            "evaporator_outlet_temp_c"
        ].min(),
        "maximum_evaporator_outlet_temperature_c": frame[
            "evaporator_outlet_temp_c"
        ].max(),
        "minimum_cluster_supply_temperature_c": frame[
            "cluster_supply_temp_c"
        ].min(),
        "maximum_cluster_supply_temperature_c": frame[
            "cluster_supply_temp_c"
        ].max(),
        "minimum_tank_return_temperature_c": frame[
            "tank_return_temp_c"
        ].min(),
        "maximum_tank_return_temperature_c": frame[
            "tank_return_temp_c"
        ].max(),
        "minimum_tank_temperature_c": frame["tank_temp_c"].min(),
        "maximum_tank_temperature_c": frame["tank_temp_c"].max(),
        "final_battery_average_temperature_c": frame[
            "battery_avg_temp_c"
        ].iloc[-1],
        "maximum_battery_temperature_c": frame[
            "battery_max_temp_c"
        ].max(),
        "maximum_inter_pack_delta_temperature_k": frame[
            "delta_t_inter_k"
        ].max(),
        "maximum_cluster_delta_temperature_k": frame[
            "delta_t_cluster_k"
        ].max(),
        "minimum_q_evap_applied_w": frame["q_evap_applied_w"].min(),
        "maximum_q_evap_applied_w": frame["q_evap_applied_w"].max(),
        "minimum_pump_power_w": frame["pump_power_w"].min(),
        "maximum_pump_power_w": frame["pump_power_w"].max(),
        "minimum_compressor_power_w": frame["compressor_power_w"].min(),
        "maximum_compressor_power_w": frame["compressor_power_w"].max(),
        "max_abs_cluster_supply_step_k": max_abs_step(
            "cluster_supply_temp_c"
        ),
        "max_abs_tank_temperature_step_k": max_abs_step("tank_temp_c"),
        "max_abs_battery_average_step_k": max_abs_step(
            "battery_avg_temp_c"
        ),
        "max_abs_cycle_energy_residual_w": frame[
            "cycle_energy_residual_w"
        ].abs().max(),
        "max_abs_evaporator_coolant_residual_w": frame[
            "evaporator_coolant_residual_w"
        ].abs().max(),
        "max_abs_evaporator_dynamic_energy_residual_j": frame[
            "evaporator_dynamic_energy_residual_j"
        ].abs().max(),
        "max_abs_cluster_fluid_residual_w": frame[
            "cluster_fluid_residual_w"
        ].abs().max(),
        "max_abs_tank_energy_residual_j": frame[
            "tank_energy_residual_j"
        ].abs().max(),
        "solver_failure_count": solver_failure_count,
        "failure_message": failure_message,
        "all_states_finite": all_states_finite,
        "cross_delay_energy_balance_is_modeled": False,
        "runtime_s": runtime_s,
    }
    summary["all_gates_pass"] = bool(
        len(rows) == steps
        and solver_failure_count == 0
        and all_states_finite
        and bool(np.all(frame["all_local_gates_pass"]))
        and summary["max_abs_supply_delay_residual_k"] <= 1e-12
        and summary["max_abs_return_delay_residual_k"] <= 1e-12
    )
    return {"summary": summary, "timeseries": rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the final Stage 8C3 Cluster Plant V2."
    )
    parser.add_argument("--duration-s", type=float, default=DURATION_S)
    parser.add_argument("--dt-s", type=float, default=DT_S)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "cluster_plant_v2/validation/results/stage8c3_validation_20260818"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    delay_gates = run_delay_gates()
    pd.DataFrame([delay_gates]).to_csv(
        args.output_dir / "stage8c3_delay_gates.csv", index=False
    )
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
        args.output_dir / "stage8c3_summary.csv", index=False
    )
    print(f"Stage 8C3 results: {args.output_dir}")


if __name__ == "__main__":
    main()
