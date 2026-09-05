"""Validate the Stage 7B reversed-return (homoverse) parallel-header hydraulic network."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_plant_v2.thermal.cluster import ReducedCluster
from cluster_plant_v2.hydraulics import (
    BASE_BRANCH_RESISTANCE_PA_PER_KG_S2,
    BRANCH_DESIGN_DELTA_P_PA,
    DESIGN_PACK_MASS_FLOW_KG_S,
    ParallelHeaderHydraulicNetwork,
)


N_PACKS = 5
TOTAL_MASS_FLOW_KG_S = N_PACKS * DESIGN_PACK_MASS_FLOW_KG_S
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
SUPPLY_TEMPERATURE_K = 20.0 + 273.15
AMBIENT_TEMPERATURE_K = 35.0 + 273.15


@dataclass(frozen=True)
class NetworkCaseSpec:
    case_id: str
    branch_factors: tuple[float, ...]
    header_ratio: float


@dataclass(frozen=True)
class ClusterCaseSpec:
    case_id: str
    branch_factors: tuple[float, ...]
    header_ratio: float
    direction: str


def build_network_case_specs() -> list[NetworkCaseSpec]:
    uniform = (1.0, 1.0, 1.0, 1.0, 1.0)
    return [
        NetworkCaseSpec("H0_zero_header_equal", uniform, 0.0),
        NetworkCaseSpec(
            "H1_zero_header_branch_nonuniform",
            (1.0, 1.05, 0.95, 1.10, 0.90),
            0.0,
        ),
        NetworkCaseSpec("H2_small_header_equal", uniform, 0.001),
        NetworkCaseSpec("H3_medium_header_equal", uniform, 0.005),
    ]


def build_cluster_case_specs() -> list[ClusterCaseSpec]:
    uniform = (1.0, 1.0, 1.0, 1.0, 1.0)
    nonuniform = (1.0, 1.05, 0.95, 1.10, 0.90)
    return [
        ClusterCaseSpec("B0_zero_header_equal_forward", uniform, 0.0, "forward"),
        ClusterCaseSpec(
            "B1_zero_header_branch_nonuniform_forward",
            nonuniform,
            0.0,
            "forward",
        ),
        ClusterCaseSpec("B2_small_header_equal_forward", uniform, 0.001, "forward"),
        ClusterCaseSpec("B3_medium_header_equal_forward", uniform, 0.005, "forward"),
        ClusterCaseSpec("B4_medium_header_equal_reverse", uniform, 0.005, "reverse"),
    ]


def _stage7a_analytic_flows(
    total_mass_flow_kg_s: float, branch_factors: tuple[float, ...]
) -> np.ndarray:
    factors = np.asarray(branch_factors, dtype=float)
    weights = 1.0 / np.sqrt(factors)
    return float(total_mass_flow_kg_s) * weights / weights.sum()


def _network_for_case(case: NetworkCaseSpec) -> ParallelHeaderHydraulicNetwork:
    header_resistance = (
        BASE_BRANCH_RESISTANCE_PA_PER_KG_S2 * case.header_ratio
    )
    return ParallelHeaderHydraulicNetwork(
        n_packs=N_PACKS,
        branch_resistances=(
            BASE_BRANCH_RESISTANCE_PA_PER_KG_S2
            * np.asarray(case.branch_factors, dtype=float)
        ),
        supply_segment_resistances=np.full(N_PACKS, header_resistance),
        return_segment_resistances=np.full(N_PACKS, header_resistance),
    )


def run_network_case(
    case: NetworkCaseSpec, total_mass_flow_kg_s: float = TOTAL_MASS_FLOW_KG_S
) -> dict:
    network = _network_for_case(case)
    solved = network.solve(total_mass_flow_kg_s)
    flows = np.asarray(solved["pack_mass_flows"], dtype=float)
    stage7a_flows = _stage7a_analytic_flows(
        total_mass_flow_kg_s, case.branch_factors
    )
    mean_flow = float(flows.mean())
    stage7a_difference = float(np.max(np.abs(flows - stage7a_flows)))
    max_node_residual = float(
        np.max(np.abs(solved["node_mass_balance_residuals"]))
    )
    max_path_residual = float(
        np.max(np.abs(solved["path_pressure_residuals"]))
    )
    summary = {
        "case_id": case.case_id,
        "n_packs": N_PACKS,
        "total_mass_flow_kg_s": total_mass_flow_kg_s,
        "branch_design_delta_p_pa": BRANCH_DESIGN_DELTA_P_PA,
        "base_branch_resistance_pa_per_kg_s2": (
            BASE_BRANCH_RESISTANCE_PA_PER_KG_S2
        ),
        "header_to_branch_resistance_ratio": case.header_ratio,
        "header_characteristic_delta_p_pa": (
            case.header_ratio * BRANCH_DESIGN_DELTA_P_PA
        ),
        "network_delta_p_pa": solved["network_delta_p"],
        "total_mass_balance_residual_kg_s": solved[
            "total_mass_balance_residual"
        ],
        "max_node_mass_residual_kg_s": max_node_residual,
        "max_path_pressure_residual_pa": max_path_residual,
        "stage7a_max_flow_difference_kg_s": stage7a_difference,
        "flow_cv": float(np.std(flows) / mean_flow),
        "flow_nonuniformity": float(
            np.max(np.abs((flows - mean_flow) / mean_flow))
        ),
        "solver_success": solved["solver_success"],
        "solver_iterations": solved["solver_iterations"],
    }
    degeneracy_pass = (
        case.header_ratio != 0.0 or stage7a_difference <= 1e-10
    )
    summary["all_gates_pass"] = bool(
        summary["solver_success"]
        and abs(summary["total_mass_balance_residual_kg_s"]) <= 1e-12
        and max_node_residual <= 1e-12
        and max_path_residual <= 1e-3
        and degeneracy_pass
        and np.all(flows > 0.0)
    )

    supply_path = np.cumsum(solved["supply_segment_delta_p"])
    return_path = np.cumsum(solved["return_segment_delta_p"][::-1])[::-1]
    pack_rows = [
        {
            "case_id": case.case_id,
            "pack_index": index + 1,
            "branch_resistance_factor": case.branch_factors[index],
            "mass_flow_kg_s": flows[index],
            "stage7a_analytic_mass_flow_kg_s": stage7a_flows[index],
            "supply_path_delta_p_pa": supply_path[index],
            "branch_delta_p_pa": solved["branch_delta_p"][index],
            "return_path_delta_p_pa": return_path[index],
            "pack_path_delta_p_pa": solved["pack_path_delta_p"][index],
            "path_pressure_residual_pa": solved["path_pressure_residuals"][index],
        }
        for index in range(N_PACKS)
    ]
    segment_rows = []
    for index in range(N_PACKS):
        segment_rows.extend(
            (
                {
                    "case_id": case.case_id,
                    "header": "supply",
                    "segment_index": index + 1,
                    "resistance_pa_per_kg_s2": network.supply_segment_resistances[index],
                    "mass_flow_kg_s": solved["supply_segment_flows"][index],
                    "delta_p_pa": solved["supply_segment_delta_p"][index],
                    "node_mass_residual_kg_s": solved["node_mass_balance_residuals"][index],
                },
                {
                    "case_id": case.case_id,
                    "header": "return",
                    "segment_index": index + 1,
                    "resistance_pa_per_kg_s2": network.return_segment_resistances[index],
                    "mass_flow_kg_s": solved["return_segment_flows"][index],
                    "delta_p_pa": solved["return_segment_delta_p"][index],
                    "node_mass_residual_kg_s": solved["node_mass_balance_residuals"][N_PACKS + index],
                },
            )
        )
    return {"summary": summary, "packs": pack_rows, "segments": segment_rows}


def _network_for_cluster_case(
    case: ClusterCaseSpec,
) -> ParallelHeaderHydraulicNetwork:
    return _network_for_case(
        NetworkCaseSpec(case.case_id, case.branch_factors, case.header_ratio)
    )


def _max_cluster_difference(header: dict, lumped: dict) -> float:
    keys = (
        "pack_currents_a",
        "pack_mass_flows_kg_s",
        "pack_battery_average_temperatures_k",
        "pack_battery_max_temperatures_k",
        "pack_battery_min_temperatures_k",
        "pack_battery_delta_temperatures_k",
        "pack_battery_zone_temperatures_k",
        "pack_plate_average_temperatures_k",
        "pack_plate_temperatures_k",
        "pack_coolant_outlet_temperatures_k",
        "pack_soc",
        "pack_branch_currents_a",
        "pack_q_gen_total_w",
        "pack_q_battery_to_plate_total_w",
        "pack_q_plate_to_fluid_total_w",
    )
    return float(
        max(
            np.max(np.abs(np.asarray(header[key]) - np.asarray(lumped[key])))
            for key in keys
        )
    )


def _hydraulic_result_is_finite(hydraulic: dict[str, object]) -> bool:
    return all(
        np.all(np.isfinite(value))
        for key, value in hydraulic.items()
        if key != "solver_message"
    )


def _cluster_timeseries_row(
    time_s: float,
    case: ClusterCaseSpec,
    cluster_result: dict,
    hydraulic: dict[str, object],
    stage7a_difference: float | None,
) -> dict:
    mass_flows = np.asarray(cluster_result["pack_mass_flows_kg_s"])
    outlets = np.asarray(cluster_result["pack_coolant_outlet_temperatures_k"])
    manual_return = float(np.sum(mass_flows * outlets) / mass_flows.sum())
    mean_flow = float(mass_flows.mean())
    row = {
        "time_s": time_s,
        "cluster_current_a": 560.0,
        "cold_plate_direction": case.direction,
        "header_to_branch_resistance_ratio": case.header_ratio,
        "supply_temperature_c": cluster_result["supply_temperature_k"] - 273.15,
        "return_temperature_c": cluster_result["return_temperature_k"] - 273.15,
        "return_mixing_residual_k": cluster_result["return_temperature_k"] - manual_return,
        "network_delta_p_pa": hydraulic["network_delta_p"],
        "total_mass_balance_residual_kg_s": hydraulic["total_mass_balance_residual"],
        "max_node_mass_residual_kg_s": float(
            np.max(np.abs(hydraulic["node_mass_balance_residuals"]))
        ),
        "max_path_pressure_residual_pa": float(
            np.max(np.abs(hydraulic["path_pressure_residuals"]))
        ),
        "solver_success": hydraulic["solver_success"],
        "solver_iterations": hydraulic["solver_iterations"],
        "flow_cv": float(np.std(mass_flows) / mean_flow),
        "flow_nonuniformity": float(
            np.max(np.abs((mass_flows - mean_flow) / mean_flow))
        ),
        "inter_pack_delta_temperature_k": cluster_result[
            "inter_pack_delta_temperature_k"
        ],
        "cluster_delta_temperature_k": cluster_result[
            "cluster_delta_temperature_k"
        ],
        "max_abs_pack_coupling_residual_w": cluster_result[
            "max_abs_pack_coupling_residual_w"
        ],
        "max_abs_pack_energy_residual_j": cluster_result[
            "max_abs_pack_energy_residual_j"
        ],
        "stage7a_regression_difference": stage7a_difference,
    }
    for pack in range(N_PACKS):
        row[f"pack_{pack + 1}_branch_factor"] = case.branch_factors[pack]
        row[f"pack_{pack + 1}_mass_flow_kg_s"] = mass_flows[pack]
        row[f"pack_{pack + 1}_battery_avg_temp_c"] = cluster_result[
            "pack_battery_average_temperatures_k"
        ][pack] - 273.15
        row[f"pack_{pack + 1}_battery_delta_temp_k"] = cluster_result[
            "pack_battery_delta_temperatures_k"
        ][pack]
        row[f"pack_{pack + 1}_coolant_outlet_temp_c"] = outlets[pack] - 273.15
        row[f"pack_{pack + 1}_path_delta_p_pa"] = hydraulic[
            "pack_path_delta_p"
        ][pack]
        row[f"pack_{pack + 1}_path_pressure_residual_pa"] = hydraulic[
            "path_pressure_residuals"
        ][pack]
        for segment in range(N_PACKS):
            if pack == 0:
                row[f"supply_segment_{segment + 1}_flow_kg_s"] = hydraulic[
                    "supply_segment_flows"
                ][segment]
                row[f"supply_segment_{segment + 1}_delta_p_pa"] = hydraulic[
                    "supply_segment_delta_p"
                ][segment]
                row[f"return_segment_{segment + 1}_flow_kg_s"] = hydraulic[
                    "return_segment_flows"
                ][segment]
                row[f"return_segment_{segment + 1}_delta_p_pa"] = hydraulic[
                    "return_segment_delta_p"
                ][segment]
    return row


def run_cluster_case(
    case: ClusterCaseSpec,
    *,
    duration_s: float,
    dt_s: float,
    output_dir: Path,
) -> dict:
    if (
        duration_s <= 0.0
        or dt_s <= 0.0
        or not np.isclose(duration_s / dt_s, round(duration_s / dt_s))
    ):
        raise ValueError("duration_s must be a positive integer multiple of dt_s")
    header_cluster = ReducedCluster(
        n_packs=N_PACKS,
        hydraulic_mode="header_network",
        hydraulic_network=_network_for_cluster_case(case),
    )
    check_stage7a = case.header_ratio == 0.0
    lumped_cluster = (
        ReducedCluster(branch_resistance_factors=case.branch_factors)
        if check_stage7a
        else None
    )
    rows = []
    started = time.perf_counter()
    for step_index in range(int(round(duration_s / dt_s))):
        header_result = header_cluster.step(
            dt_s,
            560.0,
            SUPPLY_TEMPERATURE_K,
            TOTAL_MASS_FLOW_KG_S,
            AMBIENT_TEMPERATURE_K,
            case.direction,
        )
        hydraulic = header_cluster.last_hydraulic_result
        if hydraulic is None:
            raise RuntimeError("header_network mode did not expose a hydraulic result")
        if lumped_cluster is not None:
            lumped_result = lumped_cluster.step(
                dt_s,
                560.0,
                SUPPLY_TEMPERATURE_K,
                TOTAL_MASS_FLOW_KG_S,
                AMBIENT_TEMPERATURE_K,
                case.direction,
            )
            stage7a_difference = _max_cluster_difference(
                header_result, lumped_result
            )
        else:
            stage7a_difference = None
        if not _hydraulic_result_is_finite(hydraulic):
            raise FloatingPointError("hydraulic network produced a non-finite result")
        rows.append(
            _cluster_timeseries_row(
                (step_index + 1) * dt_s,
                case,
                header_result,
                hydraulic,
                stage7a_difference,
            )
        )
    runtime_s = time.perf_counter() - started
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / f"case_{case.case_id}_timeseries.csv", index=False)

    optional_columns = ["stage7a_regression_difference"]
    required_numeric = frame.drop(columns=optional_columns).select_dtypes(
        include=[np.number]
    )
    solver_failure_count = int((~frame["solver_success"]).sum())
    summary = {
        "case_id": case.case_id,
        "direction": case.direction,
        "header_to_branch_resistance_ratio": case.header_ratio,
        "steps": len(frame),
        "network_delta_p_pa": float(frame["network_delta_p_pa"].iloc[-1]),
        "max_total_mass_residual_kg_s": float(
            frame["total_mass_balance_residual_kg_s"].abs().max()
        ),
        "max_node_mass_residual_kg_s": float(
            frame["max_node_mass_residual_kg_s"].max()
        ),
        "max_path_pressure_residual_pa": float(
            frame["max_path_pressure_residual_pa"].max()
        ),
        "max_return_mixing_residual_k": float(
            frame["return_mixing_residual_k"].abs().max()
        ),
        "flow_cv": float(frame["flow_cv"].iloc[-1]),
        "flow_nonuniformity": float(frame["flow_nonuniformity"].iloc[-1]),
        "max_inter_pack_delta_temperature_k": float(
            frame["inter_pack_delta_temperature_k"].max()
        ),
        "final_inter_pack_delta_temperature_k": float(
            frame["inter_pack_delta_temperature_k"].iloc[-1]
        ),
        "max_cluster_delta_temperature_k": float(
            frame["cluster_delta_temperature_k"].max()
        ),
        "final_cluster_delta_temperature_k": float(
            frame["cluster_delta_temperature_k"].iloc[-1]
        ),
        "max_stage7a_regression_difference": (
            float(frame["stage7a_regression_difference"].max())
            if check_stage7a
            else None
        ),
        "solver_failure_count": solver_failure_count,
        "max_solver_iterations": int(frame["solver_iterations"].max()),
        "max_abs_pack_coupling_residual_w": float(
            frame["max_abs_pack_coupling_residual_w"].max()
        ),
        "max_abs_pack_energy_residual_j": float(
            frame["max_abs_pack_energy_residual_j"].max()
        ),
        "all_states_finite": bool(
            np.all(np.isfinite(required_numeric.to_numpy()))
        ),
        "runtime_s": runtime_s,
    }
    summary["all_gates_pass"] = bool(
        summary["max_total_mass_residual_kg_s"] <= 1e-10
        and summary["max_node_mass_residual_kg_s"] <= 1e-12
        and summary["max_path_pressure_residual_pa"] <= 1e-3
        and summary["max_return_mixing_residual_k"] <= 1e-12
        and summary["solver_failure_count"] == 0
        and summary["max_abs_pack_coupling_residual_w"] <= 1e-12
        and summary["all_states_finite"]
        and (
            not check_stage7a
            or summary["max_stage7a_regression_difference"] <= 1e-8
        )
    )
    final = frame.iloc[-1]
    pack_rows = [
        {
            "case_id": case.case_id,
            "pack_index": pack + 1,
            "branch_factor": case.branch_factors[pack],
            "mass_flow_kg_s": final[f"pack_{pack + 1}_mass_flow_kg_s"],
            "final_battery_avg_temp_c": final[
                f"pack_{pack + 1}_battery_avg_temp_c"
            ],
            "max_battery_avg_temp_c": frame[
                f"pack_{pack + 1}_battery_avg_temp_c"
            ].max(),
            "final_intra_pack_delta_temperature_k": final[
                f"pack_{pack + 1}_battery_delta_temp_k"
            ],
            "final_coolant_outlet_temp_c": final[
                f"pack_{pack + 1}_coolant_outlet_temp_c"
            ],
            "pack_path_delta_p_pa": final[
                f"pack_{pack + 1}_path_delta_p_pa"
            ],
        }
        for pack in range(N_PACKS)
    ]
    segment_rows = []
    for segment in range(N_PACKS):
        for header in ("supply", "return"):
            segment_rows.append(
                {
                    "case_id": case.case_id,
                    "header": header,
                    "segment_index": segment + 1,
                    "mass_flow_kg_s": final[
                        f"{header}_segment_{segment + 1}_flow_kg_s"
                    ],
                    "delta_p_pa": final[
                        f"{header}_segment_{segment + 1}_delta_p_pa"
                    ],
                }
            )
    return {
        "timeseries": frame,
        "summary": summary,
        "packs": pack_rows,
        "segments": segment_rows,
    }


def _write_network_results(output_dir: Path, results: list[dict]) -> None:
    pd.DataFrame([result["summary"] for result in results]).to_csv(
        output_dir / "hydraulic_network_summary.csv", index=False
    )
    pd.DataFrame([row for result in results for row in result["packs"]]).to_csv(
        output_dir / "hydraulic_network_pack_paths.csv", index=False
    )
    pd.DataFrame(
        [row for result in results for row in result["segments"]]
    ).to_csv(output_dir / "hydraulic_network_segments.csv", index=False)


def _write_cluster_results(output_dir: Path, results: list[dict]) -> None:
    pd.DataFrame([result["summary"] for result in results]).to_csv(
        output_dir / "stage7b_cluster_summary.csv", index=False
    )
    pd.DataFrame([row for result in results for row in result["packs"]]).to_csv(
        output_dir / "stage7b_cluster_pack_summary.csv", index=False
    )
    pd.DataFrame(
        [row for result in results for row in result["segments"]]
    ).to_csv(output_dir / "stage7b_cluster_segments.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument("--dt-s", type=float, default=5.0)
    args = parser.parse_args()
    output_dir = args.output_dir or RESULTS_ROOT / (
        f"hydraulic_network_validation_{datetime.now():%Y%m%d}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for case in build_network_case_specs():
        result = run_network_case(case)
        results.append(result)
        _write_network_results(output_dir, results)
        print(
            f"{case.case_id}: flow CV={result['summary']['flow_cv']:.6%}, "
            f"path residual={result['summary']['max_path_pressure_residual_pa']:.3g} Pa, "
            f"pass={result['summary']['all_gates_pass']}",
            flush=True,
        )
    cluster_results = []
    for index, case in enumerate(build_cluster_case_specs(), start=1):
        print(f"[{index}/5] running {case.case_id}", flush=True)
        result = run_cluster_case(
            case,
            duration_s=args.duration_s,
            dt_s=args.dt_s,
            output_dir=output_dir,
        )
        cluster_results.append(result)
        _write_cluster_results(output_dir, cluster_results)
        print(
            f"  flow CV={result['summary']['flow_cv']:.6%}, "
            f"inter-pack max={result['summary']['max_inter_pack_delta_temperature_k']:.6g} K, "
            f"pass={result['summary']['all_gates_pass']}",
            flush=True,
        )
    print(f"Stage 7B results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
