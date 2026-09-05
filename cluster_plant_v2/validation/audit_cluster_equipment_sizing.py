"""Stage 8D single-factor equipment-sizing audit for Cluster Plant V2.

This script imports the frozen Plant as-is.  The chiller-capacity and out-of-range
pump cases are explicitly screening-only surrogates and are labelled as such in
the saved results.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import pandas as pd

import cluster_plant_v2.thermal.cold_plate_rom as cold_plate_module
from cluster_plant_v2.thermal.cluster import ReducedCluster
from cluster_plant_v2.hydraulics import CoolantPump, CoolantTank
from cluster_plant_v2.thermal.pack_rom import ReducedBatteryPack
from cluster_plant_v2.parameters import (
    COMPRESSOR_TIME_CONSTANT_S,
    COOLANT_DENSITY_KG_M3,
    COOLANT_SPECIFIC_HEAT_J_KG_K,
    EVAPORATOR_TIME_CONSTANT_S,
    LEGACY_REFERENCE_FLOW_L_MIN,
    LEGACY_REFERENCE_POWER_W,
    LEGACY_REFERENCE_SPEED_RPM,
    MAX_PUMP_SPEED_RPM,
    MIN_PUMP_SPEED_RPM,
)
from cluster_plant_v2.plant import ClusterPlant, ClusterPlantInputs
from cluster_plant_v2.thermal.reduced_pack import ReducedPack
from cluster_plant_v2.refrigeration import (
    ClosedR134aCycle,
    CompressorSpeedActuator,
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


DURATION_S = 600.0
BASE_COMPRESSOR_RPM = 4000.0
FAN_SPEED_RPM = 1200.0
CURRENT_SCAN_A = (0.0, 280.0, 560.0, 840.0, 1120.0)
COMPRESSOR_SCAN_RPM = (2000.0, 3000.0, 4000.0, 5000.0, 6000.0)
LOAD_CASES_A = (560.0, 1120.0)
TANK_VOLUMES_L = (3.0, 5.0, 10.0, 15.0, 30.0)
FLOW_TARGETS_L_MIN = (20.0, 25.0, 30.0, 35.0, 40.0, 50.0)
REFERENCE_FLOW_CANDIDATES_KG_S = (
    3.0 / 1000.0 / 60.0 * COOLANT_DENSITY_KG_M3,
    5.0 / 1000.0 / 60.0 * COOLANT_DENSITY_KG_M3,
    8.0 / 1000.0 / 60.0 * COOLANT_DENSITY_KG_M3,
    10.0 / 1000.0 / 60.0 * COOLANT_DENSITY_KG_M3,
    1.2,
)


class CapacityScreeningCycle:
    """Scale extensive cycle outputs for screening; never a final Plant model."""

    _EXTENSIVE_KEYS = (
        "refrigerant_mass_flow_kg_s",
        "compressor_mass_flow_kg_s",
        "mass_flow_residual_kg_s",
        "compressor_refrigerant_power_w",
        "compressor_shaft_power_w",
        "compressor_mechanical_loss_w",
        "q_evaporator_refrigerant_w",
        "q_evaporator_ntu_w",
        "evaporator_residual_w",
        "q_condenser_refrigerant_w",
        "q_condenser_ntu_w",
        "condenser_residual_w",
        "cycle_energy_residual_w",
        "evaporator_ua_w_k",
        "condenser_ua_w_k",
        "q_evaporator_w",
        "q_condenser_w",
    )

    dynamic_state_count = 0

    def __init__(self, factor: float) -> None:
        self.base = ClosedR134aCycle()
        self.factor = float(factor)

    def solve(self, **kwargs) -> dict[str, object]:
        result = self.base.solve(**kwargs)
        scaled = dict(result)
        for key in self._EXTENSIVE_KEYS:
            scaled[key] = float(result[key]) * self.factor
        coolant_capacity_rate = (
            float(kwargs["coolant_mass_flow_kg_s"])
            * COOLANT_SPECIFIC_HEAT_J_KG_K
        )
        scaled["coolant_outlet_temperature_k"] = (
            float(kwargs["coolant_inlet_temperature_k"])
            - float(scaled["q_evaporator_w"]) / coolant_capacity_rate
        )
        return scaled


def _pump_for_target_flow(network, target_flow_l_min: float) -> tuple[CoolantPump, float, str]:
    reference_pump = build_engineering_pump(network)
    speed_rpm = LEGACY_REFERENCE_SPEED_RPM * (
        float(target_flow_l_min) / LEGACY_REFERENCE_FLOW_L_MIN
    )
    if MIN_PUMP_SPEED_RPM <= speed_rpm <= MAX_PUMP_SPEED_RPM:
        return reference_pump, speed_rpm, "within_frozen_pump_bounds"
    extended_pump = CoolantPump(
        reference_operating_delta_p_pa=(
            reference_pump.reference_operating_delta_p_pa
        ),
        reference_speed_rpm=LEGACY_REFERENCE_SPEED_RPM,
        reference_volume_flow_l_min=LEGACY_REFERENCE_FLOW_L_MIN,
        reference_power_w=LEGACY_REFERENCE_POWER_W,
        maximum_speed_rpm=max(speed_rpm, MAX_PUMP_SPEED_RPM) + 1.0,
    )
    return extended_pump, speed_rpm, "screening_only_affinity_extrapolation"


def _build_plant(
    *,
    current_a: float,
    tank_volume_l: float = 3.0,
    target_flow_l_min: float = 25.2,
    capacity_factor: float = 1.0,
) -> tuple[ClusterPlant, float, str]:
    network = build_medium_header_network()
    cluster = ReducedCluster(
        n_packs=5,
        hydraulic_mode="header_network",
        hydraulic_network=network,
        pack_config={
            "initial_soc": INITIAL_SOC,
            "initial_temp_c": INITIAL_TEMPERATURE_K - 273.15,
        },
    )
    pump, pump_rpm, flow_status = _pump_for_target_flow(
        network, target_flow_l_min
    )
    tank = CoolantTank(
        initial_temperature_k=INITIAL_TEMPERATURE_K,
        volume_l=tank_volume_l,
    )
    cycle = (
        ClosedR134aCycle()
        if np.isclose(capacity_factor, 1.0)
        else CapacityScreeningCycle(capacity_factor)
    )
    plant = ClusterPlant.from_equilibrium(
        cluster=cluster,
        hydraulic_network=network,
        pump=pump,
        tank=tank,
        refrigeration_cycle=cycle,
        compressor_actuator=CompressorSpeedActuator(
            initial_speed_rpm=BASE_COMPRESSOR_RPM,
            time_constant_s=COMPRESSOR_TIME_CONSTANT_S,
        ),
        dt_s=DT_S,
        pump_speed_rpm=pump_rpm,
        fan_speed_rpm=FAN_SPEED_RPM,
        initial_cluster_current_a=current_a,
        ambient_temperature_k=AMBIENT_TEMPERATURE_K,
        evaporator_time_constant_s=EVAPORATOR_TIME_CONSTANT_S,
    )
    return plant, pump_rpm, flow_status


def _trajectory(
    *,
    experiment: str,
    case_value: float,
    current_a: float,
    tank_volume_l: float = 3.0,
    target_flow_l_min: float = 25.2,
    capacity_factor: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    plant, pump_rpm, flow_status = _build_plant(
        current_a=current_a,
        tank_volume_l=tank_volume_l,
        target_flow_l_min=target_flow_l_min,
        capacity_factor=capacity_factor,
    )
    rows = []
    failure_type = ""
    failure_message = ""
    failure_step = -1
    for step in range(int(DURATION_S / DT_S)):
        try:
            result = plant.step(
                ClusterPlantInputs(
                    cluster_current_a=current_a,
                    compressor_command_rpm=BASE_COMPRESSOR_RPM,
                    pump_rpm=pump_rpm,
                    fan_rpm=FAN_SPEED_RPM,
                    ambient_temperature_k=AMBIENT_TEMPERATURE_K,
                ),
                dt_s=DT_S,
            )
        except Exception as exc:
            failure_type = type(exc).__name__
            failure_message = str(exc)
            failure_step = step + 1
            break
        cluster = result["cluster_result"]
        q_air_w = float(
            sum(pack.battery.get_air_heat_loss().sum() for pack in plant.cluster.packs)
        )
        rows.append(
            {
                "experiment": experiment,
                "case_value": case_value,
                "current_a": current_a,
                "time_s": (step + 1) * DT_S,
                "tank_temp_c": result.tank_temp_c,
                "supply_temp_c": result.supply_temp_c,
                "return_temp_c": result.return_temp_c,
                "battery_tavg_c": result.battery_avg_temp_c,
                "battery_tmax_c": result.battery_max_temp_c,
                "delta_t_inter_k": result.inter_pack_delta_t_k,
                "tank_delta_c_step": float(
                    result["tank_temperature_after_k"]
                    - result["tank_temperature_before_k"]
                ),
                "total_flow_l_min": float(result["total_volume_flow_l_min"]),
                "branch_flow_mean_l_min": float(
                    np.mean(result.pack_mass_flows_kg_s)
                    / COOLANT_DENSITY_KG_M3
                    * 1000.0
                    * 60.0
                ),
                "q_gen_w": float(result["cluster_q_gen_total_w"]),
                "q_air_loss_w": q_air_w,
                "q_air_adjusted_load_w": float(
                    result["cluster_q_gen_total_w"] - q_air_w
                ),
                "q_plate_to_fluid_w": float(result["q_cluster_to_fluid_w"]),
                "q_evap_w": result.q_evap_applied_w,
                "pump_power_w": result.pump_power_w,
                "compressor_power_w": result.compressor_power_w,
                "all_states_finite": bool(result["all_states_finite"]),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(
            f"trajectory failed before its first completed step: "
            f"{failure_type}: {failure_message}"
        )
    tail = frame[frame["time_s"] > DURATION_S - 60.0]
    initial_tank_c = INITIAL_TEMPERATURE_K - 273.15
    final_delta = float(frame["tank_temp_c"].iloc[-1] - initial_tank_c)
    response_time = np.nan
    if abs(final_delta) > 1e-9:
        threshold = initial_tank_c + 0.632 * final_delta
        reached = frame[
            frame["tank_temp_c"].ge(threshold)
            if final_delta > 0.0
            else frame["tank_temp_c"].le(threshold)
        ]
        if not reached.empty:
            response_time = float(reached["time_s"].iloc[0])
    summary = {
        "experiment": experiment,
        "case_value": case_value,
        "current_a": current_a,
        "pump_rpm": pump_rpm,
        "flow_status": flow_status,
        "tank_volume_l": tank_volume_l,
        "tank_thermal_capacity_j_k": (
            tank_volume_l
            / 1000.0
            * COOLANT_DENSITY_KG_M3
            * COOLANT_SPECIFIC_HEAT_J_KG_K
        ),
        "tank_turnover_time_s": (
            tank_volume_l / float(frame["total_flow_l_min"].mean()) * 60.0
        ),
        "tank_temp_min_c": float(frame["tank_temp_c"].min()),
        "tank_temp_max_c": float(frame["tank_temp_c"].max()),
        "tank_temp_final_c": float(frame["tank_temp_c"].iloc[-1]),
        "supply_temp_final_c": float(frame["supply_temp_c"].iloc[-1]),
        "return_temp_final_c": float(frame["return_temp_c"].iloc[-1]),
        "battery_tavg_final_c": float(frame["battery_tavg_c"].iloc[-1]),
        "battery_tmax_c": float(frame["battery_tmax_c"].max()),
        "delta_t_inter_max_k": float(frame["delta_t_inter_k"].max()),
        "max_abs_tank_delta_c_step": float(
            frame["tank_delta_c_step"].abs().max()
        ),
        "tank_63pct_response_time_s": response_time,
        "total_flow_mean_l_min": float(frame["total_flow_l_min"].mean()),
        "branch_flow_mean_l_min": float(frame["branch_flow_mean_l_min"].mean()),
        "q_gen_last60_mean_w": float(tail["q_gen_w"].mean()),
        "q_air_loss_last60_mean_w": float(tail["q_air_loss_w"].mean()),
        "q_air_adjusted_load_last60_mean_w": float(
            tail["q_air_adjusted_load_w"].mean()
        ),
        "q_plate_to_fluid_last60_mean_w": float(
            tail["q_plate_to_fluid_w"].mean()
        ),
        "q_evap_last60_mean_w": float(tail["q_evap_w"].mean()),
        "pump_power_mean_w": float(frame["pump_power_w"].mean()),
        "compressor_power_mean_w": float(frame["compressor_power_w"].mean()),
        "all_states_finite": bool(frame["all_states_finite"].all()),
        "completed_steps": int(len(frame)),
        "run_completed": bool(len(frame) == int(DURATION_S / DT_S)),
        "failure_step": failure_step,
        "failure_type": failure_type,
        "failure_message": failure_message,
    }
    return frame, summary


def _qgen_scan() -> pd.DataFrame:
    rows = []
    branch_flow = 5.0 / 1000.0 / 60.0 * COOLANT_DENSITY_KG_M3
    for current_a in CURRENT_SCAN_A:
        pack = ReducedPack(
            battery_config={"initial_soc": INITIAL_SOC, "initial_temp_c": 25.0},
            initial_plate_temperature_c=25.0,
        )
        qgen = []
        qair = []
        for _ in range(int(DURATION_S / DT_S)):
            result = pack.step(
                DT_S,
                current_a,
                INITIAL_TEMPERATURE_K,
                branch_flow,
                AMBIENT_TEMPERATURE_K,
            )
            qgen.append(float(result["q_gen_total"]))
            qair.append(float(result["q_air_total"]))
        rows.append(
            {
                "current_a": current_a,
                "qgen_pack_first_step_w": qgen[0],
                "qgen_pack_mean_600s_w": float(np.mean(qgen)),
                "qgen_pack_last_step_w": qgen[-1],
                "qgen_cluster_mean_600s_w": float(5.0 * np.mean(qgen)),
                "qair_pack_mean_600s_w": float(np.mean(qair)),
                "qair_adjusted_cluster_load_mean_600s_w": float(
                    5.0 * np.mean(np.asarray(qgen) - np.asarray(qair))
                ),
            }
        )
    return pd.DataFrame(rows)


def _cycle_capability(qgen: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    network = build_medium_header_network()
    pump = build_engineering_pump(network)
    operating_point = pump.solve_operating_point(PUMP_SPEED_RPM, network)
    flow_kg_s = float(operating_point["total_mass_flow_kg_s"])
    flow_l_min = float(operating_point["total_volume_flow_l_min"])
    cycle = ClosedR134aCycle()
    rows = []
    for speed_rpm in COMPRESSOR_SCAN_RPM:
        result = cycle.solve(
            compressor_speed_rpm=speed_rpm,
            fan_speed_rpm=FAN_SPEED_RPM,
            coolant_inlet_temperature_k=INITIAL_TEMPERATURE_K,
            coolant_mass_flow_kg_s=flow_kg_s,
            ambient_temperature_k=AMBIENT_TEMPERATURE_K,
        )
        for qgen_row in qgen.to_dict("records"):
            load = float(qgen_row["qgen_cluster_mean_600s_w"])
            rows.append(
                {
                    "current_a": qgen_row["current_a"],
                    "compressor_rpm": speed_rpm,
                    "qgen_pack_mean_600s_w": qgen_row["qgen_pack_mean_600s_w"],
                    "qgen_cluster_mean_600s_w": load,
                    "qevap_w": float(result["q_evaporator_w"]),
                    "qevap_to_qgen_ratio": (
                        float(result["q_evaporator_w"]) / load
                        if load > 0.0
                        else np.inf
                    ),
                    "compressor_power_w": float(result["compressor_shaft_power_w"]),
                    "cop_shaft": float(
                        result["q_evaporator_w"]
                        / result["compressor_shaft_power_w"]
                    ),
                }
            )
    return pd.DataFrame(rows), flow_kg_s, flow_l_min


def _cold_plate_formula_table() -> pd.DataFrame:
    rows = []
    for flow_l_min in (3.0, 5.0, 8.0, 10.0):
        mass_flow = flow_l_min / 1000.0 / 60.0 * COOLANT_DENSITY_KG_M3
        h_value = max(50.0, 2000.0 * (mass_flow / 1.2) ** 0.8)
        rows.append(
            {
                "flow_l_min_per_pack": flow_l_min,
                "mass_flow_kg_s": mass_flow,
                "mass_flow_to_reference_ratio": mass_flow / 1.2,
                "h_w_m2_k": h_value,
                "h_to_nominal_ratio": h_value / 2000.0,
                "wall_fluid_ua_w_k": h_value * 0.5,
            }
        )
    return pd.DataFrame(rows)


def _trajectory_task(task: dict[str, object]):
    reference_mass_flow = task.get("reference_mass_flow_kg_s")
    original_reference = cold_plate_module.m_dot_nominal
    try:
        if reference_mass_flow is not None:
            cold_plate_module.m_dot_nominal = float(reference_mass_flow)
        try:
            frame, summary = _trajectory(**task["trajectory_kwargs"])
        except Exception as exc:
            kwargs = task["trajectory_kwargs"]
            frame = pd.DataFrame()
            summary = {
                "experiment": kwargs["experiment"],
                "case_value": kwargs["case_value"],
                "current_a": kwargs["current_a"],
                "completed_steps": 0,
                "run_completed": False,
                "failure_step": 1,
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "all_states_finite": False,
            }
    finally:
        cold_plate_module.m_dot_nominal = original_reference
    summary.update(task.get("summary_extra", {}))
    return str(task["group"]), frame, summary


def _run_sensitivity(
    max_workers: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    qgen = _qgen_scan()
    capability, flow_kg_s, flow_l_min = _cycle_capability(qgen)
    base_qevap = float(
        capability[
            (capability["current_a"] == 0.0)
            & (capability["compressor_rpm"] == BASE_COMPRESSOR_RPM)
        ]["qevap_w"].iloc[0]
    )
    frames: dict[str, list[pd.DataFrame]] = {
        key: [] for key in ("s1", "s2", "s3", "s4")
    }
    summaries: dict[str, list[dict[str, object]]] = {key: [] for key in frames}
    tasks: list[dict[str, object]] = []

    for target_kw in (base_qevap / 1000.0, 3.0, 5.0, 8.0):
        factor = target_kw * 1000.0 / base_qevap
        for current_a in LOAD_CASES_A:
            tasks.append(
                {
                    "group": "s1",
                    "trajectory_kwargs": {
                        "experiment": "S1_chiller_capacity_kw",
                        "case_value": target_kw,
                        "current_a": current_a,
                        "capacity_factor": factor,
                    },
                    "summary_extra": {
                        "screening_capacity_factor": factor,
                        "screening_only_surrogate": not np.isclose(factor, 1.0),
                    },
                }
            )

    for volume_l in TANK_VOLUMES_L:
        for current_a in LOAD_CASES_A:
            tasks.append(
                {
                    "group": "s2",
                    "trajectory_kwargs": {
                        "experiment": "S2_tank_volume_l",
                        "case_value": volume_l,
                        "current_a": current_a,
                        "tank_volume_l": volume_l,
                    },
                }
            )

    for target_flow_l_min in FLOW_TARGETS_L_MIN:
        for current_a in LOAD_CASES_A:
            tasks.append(
                {
                    "group": "s3",
                    "trajectory_kwargs": {
                        "experiment": "S3_total_flow_l_min",
                        "case_value": target_flow_l_min,
                        "current_a": current_a,
                        "target_flow_l_min": target_flow_l_min,
                    },
                }
            )

    for reference_mass_flow in REFERENCE_FLOW_CANDIDATES_KG_S:
        reference_l_min = (
            reference_mass_flow
            / COOLANT_DENSITY_KG_M3
            * 1000.0
            * 60.0
        )
        for current_a in LOAD_CASES_A:
            tasks.append(
                {
                    "group": "s4",
                    "reference_mass_flow_kg_s": reference_mass_flow,
                    "trajectory_kwargs": {
                        "experiment": "S4_cold_plate_reference_flow_l_min",
                        "case_value": reference_l_min,
                        "current_a": current_a,
                    },
                    "summary_extra": {
                        "reference_mass_flow_kg_s": reference_mass_flow,
                    },
                }
            )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_trajectory_task, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            group, frame, summary = future.result()
            frames[group].append(frame)
            summaries[group].append(summary)
            status = "ok" if summary["run_completed"] else "failed"
            print(
                f"completed {completed}/{len(tasks)}: {group} {status}",
                flush=True,
            )

    tables = {
        "qgen_scan": qgen,
        "cycle_capability": capability,
        "cold_plate_formula": _cold_plate_formula_table(),
        "base_conditions": pd.DataFrame(
            [
                {
                    "coolant_mass_flow_kg_s": flow_kg_s,
                    "coolant_flow_l_min": flow_l_min,
                    "tank_volume_l": 3.0,
                    "tank_turnover_time_s": 3.0 / flow_l_min * 60.0,
                    "base_qevap_4000rpm_w": base_qevap,
                    "ambient_temperature_c": AMBIENT_TEMPERATURE_K - 273.15,
                    "coolant_inlet_temperature_c": INITIAL_TEMPERATURE_K - 273.15,
                    "fan_speed_rpm": FAN_SPEED_RPM,
                }
            ]
        ),
    }
    time_series = {
        key: pd.concat(value, ignore_index=True) for key, value in frames.items()
    }
    summary_tables = {
        key: pd.DataFrame(value) for key, value in summaries.items()
    }
    return {**tables, **summary_tables}, time_series


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "results"
            / "stage8d_equipment_sizing_audit"
        ),
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables, time_series = _run_sensitivity(max_workers=args.workers)
    for name, frame in tables.items():
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)
    for name, frame in time_series.items():
        frame.to_csv(args.output_dir / f"{name}_timeseries.csv", index=False)
    metadata = {
        "duration_s": DURATION_S,
        "dt_s": DT_S,
        "load_cases_a": LOAD_CASES_A,
        "frozen_plant_modified": False,
        "screening_caveats": [
            "S1 scales extensive cycle outputs only; it is not a final physical model.",
            "S3 above 33.6 L/min extrapolates the current pump affinity law beyond 4800 rpm.",
            "S4 changes only the imported cold-plate reference-flow constant at runtime.",
        ],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
