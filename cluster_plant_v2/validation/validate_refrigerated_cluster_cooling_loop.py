"""Validate Stage 8B conservative R134a-cycle and Cluster coupling."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from cluster_plant_v2.thermal.cluster import ReducedCluster
from cluster_plant_v2.hydraulics import (
    CoolantPump,
    CoolantTank,
    BASE_BRANCH_RESISTANCE_PA_PER_KG_S2,
    MEDIUM_HEADER_TO_BRANCH_RESISTANCE_RATIO,
    ParallelHeaderHydraulicNetwork,
)
from cluster_plant_v2.validation.regression.refrigerated_cluster_cooling_loop import (
    RefrigeratedClusterCoolingLoop,
)
from cluster_plant_v2.parameters import LEGACY_REFERENCE_POWER_W
from cluster_plant_v2.refrigeration import ClosedR134aCycle


DT_S = 5.0
DURATION_S = 600.0
INITIAL_SOC = 0.95
INITIAL_TEMPERATURE_K = 298.15
AMBIENT_TEMPERATURE_K = 308.15
PUMP_SPEED_RPM = 3600.0


@dataclass(frozen=True)
class RefrigeratedClusterCaseSpec:
    case_id: str
    cluster_current_a: float
    compressor_speed_rpm: float
    fan_speed_rpm: float
    direction: str


def build_case_specs() -> list[RefrigeratedClusterCaseSpec]:
    return [
        RefrigeratedClusterCaseSpec(
            "R1_low_compressor_forward", 560.0, 2000.0, 800.0, "forward"
        ),
        RefrigeratedClusterCaseSpec(
            "R2_medium_compressor_forward", 560.0, 3000.0, 800.0, "forward"
        ),
        RefrigeratedClusterCaseSpec(
            "R3_baseline_compressor_forward", 560.0, 4000.0, 1200.0, "forward"
        ),
        RefrigeratedClusterCaseSpec(
            "R4_high_heat_forward", 1120.0, 4000.0, 1200.0, "forward"
        ),
        RefrigeratedClusterCaseSpec(
            "R5_baseline_compressor_reverse", 560.0, 4000.0, 1200.0, "reverse"
        ),
    ]


def build_medium_header_network() -> ParallelHeaderHydraulicNetwork:
    header_resistance = (
        BASE_BRANCH_RESISTANCE_PA_PER_KG_S2
        * MEDIUM_HEADER_TO_BRANCH_RESISTANCE_RATIO
    )
    return ParallelHeaderHydraulicNetwork(
        n_packs=5,
        supply_segment_resistances=header_resistance,
        return_segment_resistances=header_resistance,
    )


def build_engineering_pump(
    network: ParallelHeaderHydraulicNetwork,
) -> CoolantPump:
    # Stage 8D4b: the chiller rescale to 72 cc (Stage 8D4) made the legacy
    # 28 L/min pump the bottleneck -- 18 kW through 0.45 kg/s of coolant is
    # a ~12 K single-pass temperature drop that starves the spike capacity.
    # Rescale the pump selection x2 (56 L/min at the reference speed) with
    # the operating head re-matched to the network at the new point; the
    # affinity-scaled curve then delivers ~0.9 kg/s at 3600 rpm.
    reference_mass_flow = 56.0 / 1000.0 / 60.0 * 1071.0
    reference_delta_p = float(
        network.solve(reference_mass_flow)["network_delta_p"]
    )
    # Stage 8D4b pump power: keep the legacy pump's implied total efficiency
    # at its own reference point (hydraulic power q*dP/rho = 16.96 W over
    # LEGACY_REFERENCE_POWER_W = 27.40 W -> 61.9%, a normal small-pump
    # value) and rate the rescaled pump at its new hydraulic power divided
    # by the same efficiency. The network is quadratic (dP ~ q^2), so the
    # reference power is exactly 8x the legacy 27.40 W = 219.18 W. Leaving
    # the legacy default would imply >100% pump efficiency.
    legacy_mass_flow = 28.0 / 1000.0 / 60.0 * 1071.0
    legacy_delta_p = float(network.solve(legacy_mass_flow)["network_delta_p"])
    legacy_hydraulic_w = legacy_mass_flow / 1071.0 * legacy_delta_p
    pump_total_efficiency = legacy_hydraulic_w / LEGACY_REFERENCE_POWER_W
    reference_power_w = (
        reference_mass_flow / 1071.0 * reference_delta_p
    ) / pump_total_efficiency
    return CoolantPump(
        reference_operating_delta_p_pa=reference_delta_p,
        reference_volume_flow_l_min=56.0,
        reference_power_w=reference_power_w,
    )


def build_loop() -> RefrigeratedClusterCoolingLoop:
    network = build_medium_header_network()
    return RefrigeratedClusterCoolingLoop(
        cluster=ReducedCluster(
            n_packs=5,
            hydraulic_mode="header_network",
            hydraulic_network=network,
            pack_config={
                "initial_soc": INITIAL_SOC,
                "initial_battery_temperature_k": INITIAL_TEMPERATURE_K,
                "initial_plate_temperature_k": INITIAL_TEMPERATURE_K,
            },
        ),
        pump=build_engineering_pump(network),
        tank=CoolantTank(initial_temperature_k=INITIAL_TEMPERATURE_K),
        refrigeration_cycle=ClosedR134aCycle(),
    )


def run_nominal_cycle_interface_regression() -> dict[str, object]:
    loop = build_loop()
    operating_point = loop.pump.solve_operating_point(
        PUMP_SPEED_RPM, loop.cluster.hydraulic_network
    )
    cycle = loop.refrigeration_cycle.solve(
        compressor_speed_rpm=4000.0,
        fan_speed_rpm=1200.0,
        coolant_inlet_temperature_k=loop.tank.temperature_k,
        coolant_mass_flow_kg_s=operating_point["total_mass_flow_kg_s"],
        ambient_temperature_k=AMBIENT_TEMPERATURE_K,
    )
    coolant_volume_flow = float(operating_point["total_volume_flow_l_min"])
    result = {
        "compressor_speed_rpm": 4000.0,
        "fan_speed_rpm": 1200.0,
        "coolant_volume_flow_l_min": coolant_volume_flow,
        "coolant_mass_flow_kg_s": operating_point["total_mass_flow_kg_s"],
        "q_evaporator_w": cycle["q_evaporator_w"],
        "q_condenser_w": cycle["q_condenser_w"],
        "refrigerant_compression_power_w": cycle[
            "compressor_refrigerant_power_w"
        ],
        "compressor_shaft_power_w": cycle["compressor_shaft_power_w"],
        "compressor_mechanical_loss_w": cycle[
            "compressor_mechanical_loss_w"
        ],
        "cop_refrigerant": (
            cycle["q_evaporator_w"]
            / cycle["compressor_refrigerant_power_w"]
        ),
        "cop_shaft": (
            cycle["q_evaporator_w"] / cycle["compressor_shaft_power_w"]
        ),
        "evaporating_saturation_temperature_c": (
            cycle["evaporating_saturation_temperature_k"] - 273.15
        ),
        "condensing_saturation_temperature_c": (
            cycle["condensing_saturation_temperature_k"] - 273.15
        ),
        "mass_flow_relative_residual": cycle[
            "mass_flow_relative_residual"
        ],
        "evaporator_relative_residual": cycle[
            "evaporator_relative_residual"
        ],
        "condenser_relative_residual": cycle[
            "condenser_relative_residual"
        ],
        "cycle_energy_relative_residual": cycle[
            "cycle_energy_relative_residual"
        ],
        "solver_success": cycle["solver_success"],
    }
    result["all_gates_pass"] = bool(
        cycle["solver_success"]
        # Stage 8D4b: pump selection rescaled x2 (56 L/min reference),
        # operating point 50.4 L/min at 3600 rpm; nominal cycle point
        # re-solved at the new flow. Supersedes the 8D4 point
        # (15595.39 / 18693.24 / 3097.86 / 3391.35 at 25.2 L/min).
        and abs(coolant_volume_flow - 50.4) <= 1e-9
        and abs(result["q_evaporator_w"] - 18009.272475241098) <= 1e-6
        and abs(result["q_condenser_w"] - 21210.502824009964) <= 1e-6
        and abs(
            result["refrigerant_compression_power_w"]
            - 3201.2303487688687
        )
        <= 1e-6
        and abs(result["compressor_shaft_power_w"] - 3504.5219203425813)
        <= 1e-6
    )
    return result


def _one_step_gates(result: dict[str, object]) -> bool:
    return bool(
        result["all_states_finite"]
        and result["refrigeration_solver_success"]
        and result["q_evaporator_w"] > 0.0
        and result["q_condenser_w"] > result["q_evaporator_w"]
        and result["refrigerant_compression_power_w"] > 0.0
        and result["compressor_shaft_power_w"]
        >= result["refrigerant_compression_power_w"]
        and result["supply_temperature_k"]
        < result["tank_temperature_before_k"]
        and abs(result["pressure_balance_residual_pa"]) <= 1e-6
        and abs(result["cycle_mass_relative_residual"]) <= 1e-3
        and abs(result["cycle_evaporator_relative_residual"]) <= 1e-9
        and abs(result["cycle_condenser_relative_residual"]) <= 1e-9
        and abs(result["cycle_energy_residual_w"]) <= 1e-8
        and abs(result["evaporator_coolant_residual_w"]) <= 1e-8
        and abs(result["cluster_fluid_residual_w"]) <= 1e-8
        and abs(result["thermal_chain_residual_w"]) <= 1e-8
        and abs(result["tank_energy_residual_j"]) <= 1e-8
        and abs(result["total_energy_residual_j"]) <= 1e-5
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
    case = build_case_specs()[2]
    result = build_loop().step(
        dt_s=DT_S,
        cluster_current_a=case.cluster_current_a,
        pump_speed_rpm=PUMP_SPEED_RPM,
        compressor_speed_rpm=case.compressor_speed_rpm,
        fan_speed_rpm=case.fan_speed_rpm,
        ambient_temperature_k=AMBIENT_TEMPERATURE_K,
        direction=case.direction,
    )
    return {**result, "dt_s": DT_S, "all_gates_pass": _one_step_gates(result)}


def _timeseries_row(
    case: RefrigeratedClusterCaseSpec,
    time_s: float,
    result: dict[str, object],
) -> dict[str, object]:
    cluster = result["cluster_result"]
    pack_flows = np.asarray(cluster["pack_mass_flows_kg_s"], dtype=float)
    pack_outlets = np.asarray(
        cluster["pack_coolant_outlet_temperatures_k"], dtype=float
    )
    return_recomputed = float(
        np.sum(pack_flows * pack_outlets) / pack_flows.sum()
    )
    row = {
        "case_id": case.case_id,
        "time_s": time_s,
        "cluster_current_a": case.cluster_current_a,
        "direction": case.direction,
        "pump_speed_rpm": result["pump_speed_rpm"],
        "compressor_speed_rpm": result["compressor_speed_rpm"],
        "fan_speed_rpm": result["fan_speed_rpm"],
        "total_mass_flow_kg_s": result["total_mass_flow_kg_s"],
        "total_volume_flow_l_min": result["total_volume_flow_l_min"],
        "pump_power_w": result["pump_power_w"],
        "pump_delta_p_pa": result["pump_delta_p_pa"],
        "network_delta_p_pa": result["network_delta_p_pa"],
        "pressure_balance_residual_pa": result[
            "pressure_balance_residual_pa"
        ],
        "refrigerant_mass_flow_kg_s": result[
            "refrigerant_mass_flow_kg_s"
        ],
        "evaporating_saturation_temperature_c": result[
            "evaporating_saturation_temperature_k"
        ]
        - 273.15,
        "condensing_saturation_temperature_c": result[
            "condensing_saturation_temperature_k"
        ]
        - 273.15,
        "q_evaporator_w": result["q_evaporator_w"],
        "q_condenser_w": result["q_condenser_w"],
        "refrigerant_compression_power_w": result[
            "refrigerant_compression_power_w"
        ],
        "compressor_shaft_power_w": result["compressor_shaft_power_w"],
        "compressor_mechanical_loss_w": result[
            "compressor_mechanical_loss_w"
        ],
        "cop_refrigerant": result["cop_refrigerant"],
        "cop_shaft": result["cop_shaft"],
        "cycle_mass_residual_kg_s": result["cycle_mass_residual_kg_s"],
        "cycle_mass_relative_residual": result[
            "cycle_mass_relative_residual"
        ],
        "cycle_evaporator_residual_w": result[
            "cycle_evaporator_residual_w"
        ],
        "cycle_evaporator_relative_residual": result[
            "cycle_evaporator_relative_residual"
        ],
        "cycle_condenser_residual_w": result[
            "cycle_condenser_residual_w"
        ],
        "cycle_condenser_relative_residual": result[
            "cycle_condenser_relative_residual"
        ],
        "cycle_energy_residual_w": result["cycle_energy_residual_w"],
        "refrigeration_solver_success": result[
            "refrigeration_solver_success"
        ],
        "tank_temperature_before_c": result["tank_temperature_before_k"]
        - 273.15,
        "tank_temperature_after_c": result["tank_temperature_after_k"]
        - 273.15,
        "supply_temperature_c": result["supply_temperature_k"] - 273.15,
        "return_temperature_c": result["return_temperature_k"] - 273.15,
        "q_evaporator_coolant_w": result["q_evaporator_coolant_w"],
        "q_cluster_to_fluid_w": result["q_cluster_to_fluid_w"],
        "q_tank_w": result["q_tank_w"],
        "evaporator_coolant_residual_w": result[
            "evaporator_coolant_residual_w"
        ],
        "cluster_fluid_residual_w": result["cluster_fluid_residual_w"],
        "thermal_chain_residual_w": result["thermal_chain_residual_w"],
        "tank_energy_residual_j": result["tank_energy_residual_j"],
        "total_energy_residual_j": result["total_energy_residual_j"],
        "cluster_q_gen_total_w": result["cluster_q_gen_total_w"],
        "cluster_q_air_total_w": result["cluster_q_air_total_w"],
        "return_mixing_residual_k": (
            result["return_temperature_k"] - return_recomputed
        ),
        "mass_flow_residual_kg_s": cluster[
            "mass_flow_conservation_residual_kg_s"
        ],
        "flow_cv": result["flow_cv"],
        "flow_nonuniformity": result["flow_nonuniformity"],
        "cluster_battery_average_temperature_c": float(
            np.mean(cluster["pack_battery_average_temperatures_k"])
            - 273.15
        ),
        "cluster_battery_max_temperature_c": (
            cluster["cluster_max_temperature_k"] - 273.15
        ),
        "cluster_battery_min_temperature_c": (
            cluster["cluster_min_temperature_k"] - 273.15
        ),
        "inter_pack_delta_temperature_k": cluster[
            "inter_pack_delta_temperature_k"
        ],
        "cluster_delta_temperature_k": cluster[
            "cluster_delta_temperature_k"
        ],
        "max_abs_pack_coupling_residual_w": cluster[
            "max_abs_pack_coupling_residual_w"
        ],
        "max_abs_pack_energy_residual_j": cluster[
            "max_abs_pack_energy_residual_j"
        ],
    }
    for index in range(5):
        pack = index + 1
        row[f"pack_{pack}_mass_flow_kg_s"] = pack_flows[index]
        row[f"pack_{pack}_coolant_outlet_temperature_c"] = (
            pack_outlets[index] - 273.15
        )
        row[f"pack_{pack}_battery_average_temperature_c"] = (
            cluster["pack_battery_average_temperatures_k"][index] - 273.15
        )
        row[f"pack_{pack}_battery_max_temperature_c"] = (
            cluster["pack_battery_max_temperatures_k"][index] - 273.15
        )
        row[f"pack_{pack}_battery_min_temperature_c"] = (
            cluster["pack_battery_min_temperatures_k"][index] - 273.15
        )
        row[f"pack_{pack}_intra_delta_temperature_k"] = cluster[
            "pack_battery_delta_temperatures_k"
        ][index]
        row[f"pack_{pack}_soc_average"] = float(
            np.mean(cluster["pack_soc"][index])
        )
        row[f"pack_{pack}_q_gen_w"] = cluster["pack_q_gen_total_w"][index]
        row[f"pack_{pack}_q_battery_to_plate_w"] = cluster[
            "pack_q_battery_to_plate_total_w"
        ][index]
    return row


def run_case(
    case: RefrigeratedClusterCaseSpec,
    *,
    duration_s: float = DURATION_S,
    dt_s: float = DT_S,
) -> dict[str, object]:
    steps_float = float(duration_s) / float(dt_s)
    steps = int(round(steps_float))
    if steps < 1 or not np.isclose(steps_float, steps):
        raise ValueError("duration_s must be a positive integer multiple of dt_s")

    loop = build_loop()
    rows: list[dict[str, object]] = []
    solver_failure_count = 0
    failure_message = ""
    start = perf_counter()
    for step_index in range(steps):
        try:
            result = loop.step(
                dt_s=dt_s,
                cluster_current_a=case.cluster_current_a,
                pump_speed_rpm=PUMP_SPEED_RPM,
                compressor_speed_rpm=case.compressor_speed_rpm,
                fan_speed_rpm=case.fan_speed_rpm,
                ambient_temperature_k=AMBIENT_TEMPERATURE_K,
                direction=case.direction,
            )
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            solver_failure_count += 1
            failure_message = str(exc)
            break
        rows.append(_timeseries_row(case, (step_index + 1) * dt_s, result))
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
            "pack_summary": [],
            "timeseries": [],
        }

    frame = pd.DataFrame(rows)
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    all_states_finite = bool(np.all(np.isfinite(numeric)))
    summary = {
        "case_id": case.case_id,
        "cluster_current_a": case.cluster_current_a,
        "pump_speed_rpm": PUMP_SPEED_RPM,
        "compressor_speed_rpm": case.compressor_speed_rpm,
        "fan_speed_rpm": case.fan_speed_rpm,
        "direction": case.direction,
        "steps_requested": steps,
        "steps_completed": len(rows),
        "total_volume_flow_l_min": frame["total_volume_flow_l_min"].iloc[-1],
        "final_refrigerant_mass_flow_kg_s": frame[
            "refrigerant_mass_flow_kg_s"
        ].iloc[-1],
        "final_q_evaporator_w": frame["q_evaporator_w"].iloc[-1],
        "final_q_condenser_w": frame["q_condenser_w"].iloc[-1],
        "final_refrigerant_compression_power_w": frame[
            "refrigerant_compression_power_w"
        ].iloc[-1],
        "final_compressor_shaft_power_w": frame[
            "compressor_shaft_power_w"
        ].iloc[-1],
        "final_compressor_mechanical_loss_w": frame[
            "compressor_mechanical_loss_w"
        ].iloc[-1],
        "final_cop_shaft": frame["cop_shaft"].iloc[-1],
        "final_evaporating_saturation_temperature_c": frame[
            "evaporating_saturation_temperature_c"
        ].iloc[-1],
        "final_condensing_saturation_temperature_c": frame[
            "condensing_saturation_temperature_c"
        ].iloc[-1],
        "initial_tank_temperature_c": INITIAL_TEMPERATURE_K - 273.15,
        "final_tank_temperature_c": frame["tank_temperature_after_c"].iloc[-1],
        "minimum_tank_temperature_c": frame["tank_temperature_after_c"].min(),
        "maximum_tank_temperature_c": frame["tank_temperature_after_c"].max(),
        "final_supply_temperature_c": frame["supply_temperature_c"].iloc[-1],
        "final_return_temperature_c": frame["return_temperature_c"].iloc[-1],
        "final_battery_average_temperature_c": frame[
            "cluster_battery_average_temperature_c"
        ].iloc[-1],
        "maximum_battery_temperature_c": frame[
            "cluster_battery_max_temperature_c"
        ].max(),
        "maximum_inter_pack_delta_temperature_k": frame[
            "inter_pack_delta_temperature_k"
        ].max(),
        "maximum_cluster_delta_temperature_k": frame[
            "cluster_delta_temperature_k"
        ].max(),
        "max_evaporator_coolant_residual_w": frame[
            "evaporator_coolant_residual_w"
        ].abs().max(),
        "max_cycle_evaporator_residual_w": frame[
            "cycle_evaporator_residual_w"
        ].abs().max(),
        "max_cycle_evaporator_relative_residual": frame[
            "cycle_evaporator_relative_residual"
        ].abs().max(),
        "max_cycle_condenser_residual_w": frame[
            "cycle_condenser_residual_w"
        ].abs().max(),
        "max_cycle_condenser_relative_residual": frame[
            "cycle_condenser_relative_residual"
        ].abs().max(),
        "max_cycle_energy_residual_w": frame[
            "cycle_energy_residual_w"
        ].abs().max(),
        "max_cluster_fluid_residual_w": frame[
            "cluster_fluid_residual_w"
        ].abs().max(),
        "max_thermal_chain_residual_w": frame[
            "thermal_chain_residual_w"
        ].abs().max(),
        "max_total_energy_residual_j": frame[
            "total_energy_residual_j"
        ].abs().max(),
        "max_return_mixing_residual_k": frame[
            "return_mixing_residual_k"
        ].abs().max(),
        "max_mass_flow_residual_kg_s": frame[
            "mass_flow_residual_kg_s"
        ].abs().max(),
        "max_abs_pack_coupling_residual_w": frame[
            "max_abs_pack_coupling_residual_w"
        ].max(),
        "max_abs_pack_energy_residual_j": frame[
            "max_abs_pack_energy_residual_j"
        ].max(),
        "solver_failure_count": solver_failure_count,
        "failure_message": failure_message,
        "all_states_finite": all_states_finite,
        "runtime_s": runtime_s,
    }
    summary["all_gates_pass"] = bool(
        all_states_finite
        and solver_failure_count == 0
        and len(rows) == steps
        and summary["max_evaporator_coolant_residual_w"] <= 1e-8
        and summary["max_cycle_evaporator_relative_residual"] <= 1e-9
        and summary["max_cycle_condenser_relative_residual"] <= 1e-9
        and summary["max_cycle_energy_residual_w"] <= 1e-8
        and summary["max_cluster_fluid_residual_w"] <= 1e-8
        and summary["max_thermal_chain_residual_w"] <= 1e-8
        and summary["max_total_energy_residual_j"] <= 1e-5
        and summary["max_return_mixing_residual_k"] <= 1e-12
        and summary["max_mass_flow_residual_kg_s"] <= 1e-10
        and summary["max_abs_pack_coupling_residual_w"] <= 1e-8
    )

    final = frame.iloc[-1]
    pack_summary = []
    for pack in range(1, 6):
        pack_summary.append(
            {
                "case_id": case.case_id,
                "pack_index": pack,
                "mass_flow_kg_s": final[f"pack_{pack}_mass_flow_kg_s"],
                "final_battery_average_temperature_c": final[
                    f"pack_{pack}_battery_average_temperature_c"
                ],
                "maximum_battery_temperature_c": frame[
                    f"pack_{pack}_battery_max_temperature_c"
                ].max(),
                "minimum_battery_temperature_c": frame[
                    f"pack_{pack}_battery_min_temperature_c"
                ].min(),
                "final_intra_delta_temperature_k": final[
                    f"pack_{pack}_intra_delta_temperature_k"
                ],
                "final_coolant_outlet_temperature_c": final[
                    f"pack_{pack}_coolant_outlet_temperature_c"
                ],
                "final_soc_average": final[f"pack_{pack}_soc_average"],
                "final_q_gen_w": final[f"pack_{pack}_q_gen_w"],
                "final_q_battery_to_plate_w": final[
                    f"pack_{pack}_q_battery_to_plate_w"
                ],
            }
        )
    return {
        "summary": summary,
        "pack_summary": pack_summary,
        "timeseries": rows,
    }


def _scalar_values(mapping: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in mapping.items()
        if np.isscalar(value) and not isinstance(value, str)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Stage 8B conservative refrigeration coupling."
    )
    parser.add_argument("--duration-s", type=float, default=DURATION_S)
    parser.add_argument("--dt-s", type=float, default=DT_S)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "cluster_plant_v2/validation/results/stage8b_validation_20260818"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nominal = run_nominal_cycle_interface_regression()
    pd.DataFrame([nominal]).to_csv(
        args.output_dir / "stage8b_ab_nominal_regression.csv", index=False
    )
    one_step = run_one_step_gate()
    pd.DataFrame([_scalar_values(one_step)]).to_csv(
        args.output_dir / "stage8b_one_step_gate.csv", index=False
    )

    results = []
    cases = build_case_specs()
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] running {case.case_id}")
        result = run_case(
            case, duration_s=args.duration_s, dt_s=args.dt_s
        )
        results.append(result)
        pd.DataFrame(result["timeseries"]).to_csv(
            args.output_dir / f"case_{case.case_id}_timeseries.csv", index=False
        )
        summary = result["summary"]
        print(
            f"  steps={summary['steps_completed']}/{summary['steps_requested']}, "
            f"tank_final={summary.get('final_tank_temperature_c', np.nan):.6f} C, "
            f"pass={summary['all_gates_pass']}"
        )

    pd.DataFrame([result["summary"] for result in results]).to_csv(
        args.output_dir / "stage8b_cluster_summary.csv", index=False
    )
    pd.DataFrame(
        [row for result in results for row in result["pack_summary"]]
    ).to_csv(args.output_dir / "stage8b_pack_summary.csv", index=False)
    print(f"Stage 8B results: {args.output_dir}")


if __name__ == "__main__":
    main()
