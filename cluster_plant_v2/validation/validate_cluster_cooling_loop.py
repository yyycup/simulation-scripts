"""Validate Stage 8A Pump-Tank-Cluster coolant-loop coupling."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from cluster_plant_v2.thermal.cluster import ReducedCluster
from cluster_plant_v2.validation.regression.cluster_cooling_loop import (
    ClusterCoolingLoop,
)
from cluster_plant_v2.hydraulics import (
    CoolantPump,
    CoolantTank,
    BASE_BRANCH_RESISTANCE_PA_PER_KG_S2,
    MEDIUM_HEADER_TO_BRANCH_RESISTANCE_RATIO,
    ParallelHeaderHydraulicNetwork,
)


DT_S = 5.0
DURATION_S = 600.0
INITIAL_SOC = 0.95
INITIAL_BATTERY_TEMPERATURE_K = 298.15
INITIAL_PLATE_TEMPERATURE_K = 298.15
INITIAL_TANK_TEMPERATURE_K = 293.15
AMBIENT_TEMPERATURE_K = 308.15


@dataclass(frozen=True)
class ClusterLoopCaseSpec:
    case_id: str
    current_a: float
    pump_speed_rpm: float
    direction: str


def build_pump_case_speeds() -> list[float]:
    return [1600.0, 2400.0, 3200.0, 4000.0, 4800.0]


def build_cluster_case_specs() -> list[ClusterLoopCaseSpec]:
    return [
        ClusterLoopCaseSpec("P1_low_pump_forward", 560.0, 2400.0, "forward"),
        ClusterLoopCaseSpec("P2_nominal_pump_forward", 560.0, 3600.0, "forward"),
        ClusterLoopCaseSpec("P3_high_pump_forward", 560.0, 4800.0, "forward"),
        ClusterLoopCaseSpec("P4_high_heat_forward", 1120.0, 3600.0, "forward"),
        ClusterLoopCaseSpec("P5_nominal_pump_reverse", 560.0, 3600.0, "reverse"),
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
    reference_mass_flow = 28.0 / 1000.0 / 60.0 * 1071.0
    reference_delta_p = float(
        network.solve(reference_mass_flow)["network_delta_p"]
    )
    return CoolantPump(reference_operating_delta_p_pa=reference_delta_p)


def build_loop() -> ClusterCoolingLoop:
    network = build_medium_header_network()
    cluster = ReducedCluster(
        n_packs=5,
        hydraulic_mode="header_network",
        hydraulic_network=network,
        pack_config={
            "initial_soc": INITIAL_SOC,
            "initial_battery_temperature_k": INITIAL_BATTERY_TEMPERATURE_K,
            "initial_plate_temperature_k": INITIAL_PLATE_TEMPERATURE_K,
        },
    )
    return ClusterCoolingLoop(
        cluster=cluster,
        pump=build_engineering_pump(network),
        tank=CoolantTank(initial_temperature_k=INITIAL_TANK_TEMPERATURE_K),
    )


def run_pump_case(speed_rpm: float) -> dict[str, object]:
    network = build_medium_header_network()
    pump = build_engineering_pump(network)
    result = pump.solve_operating_point(speed_rpm, network)
    return {
        **result,
        "reference_speed_rpm": pump.reference_speed_rpm,
        "reference_volume_flow_l_min": pump.reference_volume_flow_l_min,
        "reference_power_w": pump.reference_power_w,
        "reference_operating_delta_p_pa": pump.reference_operating_delta_p_pa,
        "reference_shutoff_delta_p_pa": pump.reference_shutoff_delta_p_pa,
        "shutoff_to_operating_pressure_ratio": (
            pump.shutoff_to_operating_pressure_ratio
        ),
        "all_gates_pass": bool(
            result["solver_success"]
            and abs(result["pressure_balance_residual_pa"]) <= 1e-6
        ),
    }


def run_tank_one_step() -> dict[str, float | bool]:
    tank = CoolantTank(initial_temperature_k=293.15)
    mass_flow = 0.44625
    return_temperature = 295.15
    dt_s = 5.0
    expected_delta = (
        mass_flow
        * (return_temperature - tank.temperature_k)
        * dt_s
        / (tank.coolant_density_kg_m3 * tank.volume_m3)
    )
    result = tank.step(dt_s, return_temperature, mass_flow)
    difference = (
        result["tank_temperature_after_k"] - (293.15 + expected_delta)
    )
    return {
        **result,
        "dt_s": dt_s,
        "tank_volume_l": tank.volume_m3 * 1000.0,
        "coolant_density_kg_m3": tank.coolant_density_kg_m3,
        "coolant_specific_heat_j_kg_k": tank.coolant_specific_heat_j_kg_k,
        "hand_calculated_delta_temperature_k": expected_delta,
        "temperature_difference_from_hand_k": float(difference),
        "all_gates_pass": bool(
            abs(difference) <= 1e-12
            and abs(result["tank_energy_residual_j"]) <= 1e-8
        ),
    }


def _timeseries_row(
    case: ClusterLoopCaseSpec,
    time_s: float,
    result: dict[str, object],
) -> dict[str, object]:
    cluster = result["cluster_result"]
    flows = np.asarray(cluster["pack_mass_flows_kg_s"], dtype=float)
    outlets = np.asarray(
        cluster["pack_coolant_outlet_temperatures_k"], dtype=float
    )
    return_recomputed = float(np.sum(flows * outlets) / flows.sum())
    row = {
        "case_id": case.case_id,
        "time_s": time_s,
        "cluster_current_a": case.current_a,
        "direction": case.direction,
        "pump_speed_rpm": result["pump_speed_rpm"],
        "total_mass_flow_kg_s": result["total_mass_flow_kg_s"],
        "total_volume_flow_l_min": result["total_volume_flow_l_min"],
        "pump_delta_p_pa": result["pump_delta_p_pa"],
        "network_delta_p_pa": result["network_delta_p_pa"],
        "pressure_balance_residual_pa": result[
            "pressure_balance_residual_pa"
        ],
        "pump_power_w": result["pump_power_w"],
        "supply_temperature_c": result["supply_temperature_k"] - 273.15,
        "return_temperature_c": result["return_temperature_k"] - 273.15,
        "tank_temperature_before_c": result["tank_temperature_before_k"]
        - 273.15,
        "tank_temperature_after_c": result["tank_temperature_after_k"]
        - 273.15,
        "return_mixing_residual_k": result["return_temperature_k"]
        - return_recomputed,
        "return_to_tank_heat_w": result["return_to_tank_heat_w"],
        "tank_energy_change_j": result["tank_energy_change_j"],
        "tank_energy_residual_j": result["tank_energy_residual_j"],
        "fluid_heat_gain_w": result["fluid_heat_gain_w"],
        "plate_to_fluid_heat_w": result["plate_to_fluid_heat_w"],
        "plate_to_fluid_residual_w": result["plate_to_fluid_residual_w"],
        "flow_cv": result["flow_cv"],
        "flow_nonuniformity": result["flow_nonuniformity"],
        "mass_flow_residual_kg_s": cluster[
            "mass_flow_conservation_residual_kg_s"
        ],
        "inter_pack_delta_temperature_k": cluster[
            "inter_pack_delta_temperature_k"
        ],
        "cluster_delta_temperature_k": cluster[
            "cluster_delta_temperature_k"
        ],
        "cluster_battery_average_temperature_c": float(
            np.mean(cluster["pack_battery_average_temperatures_k"])
            - 273.15
        ),
        "cluster_battery_max_temperature_c": cluster[
            "cluster_max_temperature_k"
        ]
        - 273.15,
        "cluster_battery_min_temperature_c": cluster[
            "cluster_min_temperature_k"
        ]
        - 273.15,
        "cluster_q_gen_total_w": float(np.sum(cluster["pack_q_gen_total_w"])),
        "cluster_q_battery_to_plate_total_w": float(
            np.sum(cluster["pack_q_battery_to_plate_total_w"])
        ),
        "max_abs_pack_coupling_residual_w": cluster[
            "max_abs_pack_coupling_residual_w"
        ],
        "max_abs_pack_energy_residual_j": cluster[
            "max_abs_pack_energy_residual_j"
        ],
    }
    for index in range(5):
        pack = index + 1
        row[f"pack_{pack}_mass_flow_kg_s"] = flows[index]
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
        row[f"pack_{pack}_coolant_outlet_temperature_c"] = outlets[index] - 273.15
        row[f"pack_{pack}_soc_average"] = float(
            np.mean(cluster["pack_soc"][index])
        )
        row[f"pack_{pack}_q_gen_w"] = cluster["pack_q_gen_total_w"][index]
        row[f"pack_{pack}_q_battery_to_plate_w"] = cluster[
            "pack_q_battery_to_plate_total_w"
        ][index]
    return row


def run_cluster_case(
    case: ClusterLoopCaseSpec,
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
    start = perf_counter()
    for step_index in range(steps):
        result = loop.step(
            dt_s=dt_s,
            cluster_current_a=case.current_a,
            pump_speed_rpm=case.pump_speed_rpm,
            ambient_temperature_k=AMBIENT_TEMPERATURE_K,
            direction=case.direction,
        )
        rows.append(
            _timeseries_row(case, (step_index + 1) * dt_s, result)
        )
    runtime_s = perf_counter() - start
    frame = pd.DataFrame(rows)
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    all_states_finite = bool(np.all(np.isfinite(numeric)))
    summary = {
        "case_id": case.case_id,
        "current_a": case.current_a,
        "pump_speed_rpm": case.pump_speed_rpm,
        "direction": case.direction,
        "steps": steps,
        "total_mass_flow_kg_s": frame["total_mass_flow_kg_s"].iloc[-1],
        "total_volume_flow_l_min": frame["total_volume_flow_l_min"].iloc[-1],
        "network_delta_p_pa": frame["network_delta_p_pa"].iloc[-1],
        "pump_power_w": frame["pump_power_w"].iloc[-1],
        "flow_cv": frame["flow_cv"].iloc[-1],
        "flow_nonuniformity": frame["flow_nonuniformity"].iloc[-1],
        "initial_tank_temperature_c": INITIAL_TANK_TEMPERATURE_K - 273.15,
        "final_tank_temperature_c": frame["tank_temperature_after_c"].iloc[-1],
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
        "max_pressure_residual_pa": frame[
            "pressure_balance_residual_pa"
        ].abs().max(),
        "max_mass_flow_residual_kg_s": frame[
            "mass_flow_residual_kg_s"
        ].abs().max(),
        "max_return_mixing_residual_k": frame[
            "return_mixing_residual_k"
        ].abs().max(),
        "max_plate_to_fluid_residual_w": frame[
            "plate_to_fluid_residual_w"
        ].abs().max(),
        "max_tank_energy_residual_j": frame[
            "tank_energy_residual_j"
        ].abs().max(),
        "max_abs_pack_coupling_residual_w": frame[
            "max_abs_pack_coupling_residual_w"
        ].max(),
        "max_abs_pack_energy_residual_j": frame[
            "max_abs_pack_energy_residual_j"
        ].max(),
        "solver_failure_count": 0,
        "all_states_finite": all_states_finite,
        "runtime_s": runtime_s,
    }
    summary["all_gates_pass"] = bool(
        all_states_finite
        and summary["max_pressure_residual_pa"] <= 1e-6
        and summary["max_mass_flow_residual_kg_s"] <= 1e-10
        and summary["max_return_mixing_residual_k"] <= 1e-12
        and summary["max_plate_to_fluid_residual_w"] <= 1e-8
        and summary["max_tank_energy_residual_j"] <= 1e-8
    )

    final_row = frame.iloc[-1]
    pack_rows = []
    for pack in range(1, 6):
        pack_rows.append(
            {
                "case_id": case.case_id,
                "pack_index": pack,
                "mass_flow_kg_s": final_row[f"pack_{pack}_mass_flow_kg_s"],
                "final_battery_average_temperature_c": final_row[
                    f"pack_{pack}_battery_average_temperature_c"
                ],
                "maximum_battery_temperature_c": frame[
                    f"pack_{pack}_battery_max_temperature_c"
                ].max(),
                "minimum_battery_temperature_c": frame[
                    f"pack_{pack}_battery_min_temperature_c"
                ].min(),
                "final_intra_delta_temperature_k": final_row[
                    f"pack_{pack}_intra_delta_temperature_k"
                ],
                "final_coolant_outlet_temperature_c": final_row[
                    f"pack_{pack}_coolant_outlet_temperature_c"
                ],
                "final_soc_average": final_row[f"pack_{pack}_soc_average"],
                "final_q_gen_w": final_row[f"pack_{pack}_q_gen_w"],
                "final_q_battery_to_plate_w": final_row[
                    f"pack_{pack}_q_battery_to_plate_w"
                ],
            }
        )
    return {
        "summary": summary,
        "pack_summary": pack_rows,
        "timeseries": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Stage 8A Pump-Tank-Cluster coolant loop."
    )
    parser.add_argument("--duration-s", type=float, default=DURATION_S)
    parser.add_argument("--dt-s", type=float, default=DT_S)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "cluster_plant_v2/validation/results/stage8a_validation_20260818"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pump_rows = [run_pump_case(speed) for speed in build_pump_case_speeds()]
    pd.DataFrame(pump_rows).to_csv(
        args.output_dir / "pump_operating_points.csv", index=False
    )
    pd.DataFrame([run_tank_one_step()]).to_csv(
        args.output_dir / "tank_one_step.csv", index=False
    )

    case_results = []
    for index, case in enumerate(build_cluster_case_specs(), start=1):
        print(f"[{index}/5] running {case.case_id}")
        result = run_cluster_case(
            case, duration_s=args.duration_s, dt_s=args.dt_s
        )
        case_results.append(result)
        pd.DataFrame(result["timeseries"]).to_csv(
            args.output_dir / f"case_{case.case_id}_timeseries.csv", index=False
        )
        summary = result["summary"]
        print(
            "  flow="
            f"{summary['total_volume_flow_l_min']:.6f} L/min, "
            f"tank_final={summary['final_tank_temperature_c']:.6f} C, "
            f"pass={summary['all_gates_pass']}"
        )

    pd.DataFrame([result["summary"] for result in case_results]).to_csv(
        args.output_dir / "stage8a_cluster_summary.csv", index=False
    )
    pd.DataFrame(
        [row for result in case_results for row in result["pack_summary"]]
    ).to_csv(args.output_dir / "stage8a_pack_summary.csv", index=False)
    print(f"Stage 8A results: {args.output_dir}")


if __name__ == "__main__":
    main()
