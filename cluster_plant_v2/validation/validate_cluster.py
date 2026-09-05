"""Validate Stage 7A five-Pack series-electric, parallel-hydraulic assembly."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_plant_v2.thermal.cluster import ReducedCluster
from cluster_plant_v2.thermal.reduced_pack import ReducedPack
from cluster_plant_v2.validation.validate_cold_plate_rom import (
    liters_per_minute_to_mass_flow,
)


DEFAULT_DT_S = 5.0
DEFAULT_DURATION_S = 600.0
N_PACKS = 5
TOTAL_FLOW_L_MIN = 25.0
TOTAL_MASS_FLOW_KG_S = liters_per_minute_to_mass_flow(TOTAL_FLOW_L_MIN)
SUPPLY_TEMPERATURE_K = 20.0 + 273.15
AMBIENT_TEMPERATURE_K = 35.0 + 273.15
RESULTS_ROOT = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    current_a: float
    direction: str
    resistance_factors: tuple[float, ...]

    @property
    def symmetric(self) -> bool:
        return bool(np.all(np.asarray(self.resistance_factors) == self.resistance_factors[0]))


def build_case_specs() -> list[CaseSpec]:
    uniform = (1.0, 1.0, 1.0, 1.0, 1.0)
    return [
        CaseSpec("560a_25lpm_forward_uniform", 560.0, "forward", uniform),
        CaseSpec("280a_25lpm_forward_uniform", 280.0, "forward", uniform),
        CaseSpec("1120a_25lpm_forward_uniform", 1120.0, "forward", uniform),
        CaseSpec("560a_25lpm_reverse_uniform", 560.0, "reverse", uniform),
        CaseSpec(
            "560a_25lpm_forward_nonuniform",
            560.0,
            "forward",
            (1.0, 1.05, 0.95, 1.10, 0.90),
        ),
    ]


def _max_symmetric_pack_difference(result: dict) -> float:
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
            np.max(np.abs(np.asarray(result[key]) - np.asarray(result[key])[0]))
            for key in keys
        )
    )


def _max_stage6_pack_difference(cluster: dict, standalone: dict) -> float:
    pairs = (
        (cluster["pack_battery_average_temperatures_k"][0], standalone["battery_temperature_average"]),
        (cluster["pack_battery_max_temperatures_k"][0], standalone["battery_temperature_max_zone"]),
        (cluster["pack_battery_min_temperatures_k"][0], standalone["battery_temperature_min_zone"]),
        (cluster["pack_battery_delta_temperatures_k"][0], standalone["battery_temperature_delta_zone"]),
        (cluster["pack_battery_zone_temperatures_k"][0], standalone["battery_zone_temperatures"]),
        (cluster["pack_plate_temperatures_k"][0], standalone["plate_temperatures"]),
        (cluster["pack_coolant_outlet_temperatures_k"][0], standalone["coolant_outlet_temperature"]),
        (cluster["pack_soc"][0], standalone["soc"]),
        (cluster["pack_branch_currents_a"][0], standalone["branch_currents"]),
        (cluster["pack_q_gen_total_w"][0], standalone["q_gen_total"]),
        (cluster["pack_q_battery_to_plate_total_w"][0], standalone["q_battery_to_plate_total"]),
        (cluster["pack_q_plate_to_fluid_total_w"][0], standalone["q_plate_to_fluid_total"]),
    )
    return float(
        max(np.max(np.abs(np.asarray(cluster_value) - np.asarray(pack_value)))
            for cluster_value, pack_value in pairs)
    )


def _all_finite(result: dict) -> bool:
    return all(np.all(np.isfinite(value)) for value in result.values())


def compare_one_step() -> dict[str, float | bool]:
    cluster = ReducedCluster()
    standalone = ReducedPack()
    cluster_result = cluster.step(
        5.0,
        560.0,
        SUPPLY_TEMPERATURE_K,
        TOTAL_MASS_FLOW_KG_S,
        AMBIENT_TEMPERATURE_K,
        "forward",
    )
    standalone_result = standalone.step(
        5.0,
        560.0,
        SUPPLY_TEMPERATURE_K,
        TOTAL_MASS_FLOW_KG_S / N_PACKS,
        AMBIENT_TEMPERATURE_K,
        1,
    )
    manual_return = float(
        np.sum(
            cluster_result["pack_mass_flows_kg_s"]
            * cluster_result["pack_coolant_outlet_temperatures_k"]
        )
        / TOTAL_MASS_FLOW_KG_S
    )
    return {
        "n_packs": N_PACKS,
        "cluster_current_a": 560.0,
        "total_flow_l_min": TOTAL_FLOW_L_MIN,
        "total_mass_flow_kg_s": TOTAL_MASS_FLOW_KG_S,
        "pack_mass_flow_kg_s": cluster_result["pack_mass_flows_kg_s"][0],
        "mass_flow_conservation_residual_kg_s": cluster_result[
            "mass_flow_conservation_residual_kg_s"
        ],
        "return_temperature_c": cluster_result["return_temperature_k"] - 273.15,
        "return_mixing_residual_k": cluster_result["return_temperature_k"] - manual_return,
        "inter_pack_delta_temperature_k": cluster_result[
            "inter_pack_delta_temperature_k"
        ],
        "flow_nonuniformity": cluster_result["flow_nonuniformity"],
        "max_symmetric_pack_difference": _max_symmetric_pack_difference(
            cluster_result
        ),
        "max_stage6_pack_regression_difference": _max_stage6_pack_difference(
            cluster_result, standalone_result
        ),
        "max_abs_pack_coupling_residual_w": cluster_result[
            "max_abs_pack_coupling_residual_w"
        ],
        "all_states_finite": _all_finite(cluster_result),
    }


def _hydraulic_order_is_correct(
    resistance_factors: np.ndarray, mass_flows: np.ndarray
) -> bool:
    for left in range(len(resistance_factors)):
        for right in range(len(resistance_factors)):
            if resistance_factors[left] > resistance_factors[right]:
                if not mass_flows[left] < mass_flows[right]:
                    return False
            elif resistance_factors[left] == resistance_factors[right]:
                if mass_flows[left] != mass_flows[right]:
                    return False
    return True


def _timeseries_row(
    time_s: float,
    case: CaseSpec,
    result: dict,
    symmetry_difference: float | None,
    stage6_difference: float | None,
) -> dict:
    manual_return = float(
        np.sum(
            result["pack_mass_flows_kg_s"]
            * result["pack_coolant_outlet_temperatures_k"]
        )
        / result["total_mass_flow_kg_s"]
    )
    row = {
        "time_s": time_s,
        "cluster_current_a": case.current_a,
        "direction": case.direction,
        "supply_temperature_c": result["supply_temperature_k"] - 273.15,
        "return_temperature_c": result["return_temperature_k"] - 273.15,
        "total_mass_flow_kg_s": result["total_mass_flow_kg_s"],
        "mass_flow_conservation_residual_kg_s": result[
            "mass_flow_conservation_residual_kg_s"
        ],
        "return_mixing_residual_k": result["return_temperature_k"] - manual_return,
        "flow_nonuniformity": result["flow_nonuniformity"],
        "cluster_max_temperature_c": result["cluster_max_temperature_k"] - 273.15,
        "cluster_min_temperature_c": result["cluster_min_temperature_k"] - 273.15,
        "cluster_delta_temperature_k": result["cluster_delta_temperature_k"],
        "inter_pack_delta_temperature_k": result[
            "inter_pack_delta_temperature_k"
        ],
        "max_symmetric_pack_difference": symmetry_difference,
        "stage6_pack_regression_difference": stage6_difference,
        "max_abs_pack_coupling_residual_w": result[
            "max_abs_pack_coupling_residual_w"
        ],
        "max_abs_pack_energy_residual_j": result[
            "max_abs_pack_energy_residual_j"
        ],
    }
    for pack in range(N_PACKS):
        row[f"pack_{pack + 1}_resistance_factor"] = case.resistance_factors[pack]
        row[f"pack_{pack + 1}_mass_flow_kg_s"] = result["pack_mass_flows_kg_s"][pack]
        row[f"pack_{pack + 1}_battery_avg_temp_c"] = result[
            "pack_battery_average_temperatures_k"
        ][pack] - 273.15
        row[f"pack_{pack + 1}_battery_max_temp_c"] = result[
            "pack_battery_max_temperatures_k"
        ][pack] - 273.15
        row[f"pack_{pack + 1}_battery_min_temp_c"] = result[
            "pack_battery_min_temperatures_k"
        ][pack] - 273.15
        row[f"pack_{pack + 1}_battery_delta_temp_k"] = result[
            "pack_battery_delta_temperatures_k"
        ][pack]
        row[f"pack_{pack + 1}_plate_avg_temp_c"] = result[
            "pack_plate_average_temperatures_k"
        ][pack] - 273.15
        row[f"pack_{pack + 1}_coolant_outlet_temp_c"] = result[
            "pack_coolant_outlet_temperatures_k"
        ][pack] - 273.15
        row[f"pack_{pack + 1}_q_gen_w"] = result["pack_q_gen_total_w"][pack]
        row[f"pack_{pack + 1}_q_battery_to_plate_w"] = result[
            "pack_q_battery_to_plate_total_w"
        ][pack]
        row[f"pack_{pack + 1}_q_plate_to_fluid_w"] = result[
            "pack_q_plate_to_fluid_total_w"
        ][pack]
        for branch in range(4):
            row[f"pack_{pack + 1}_soc_branch_{branch + 1}"] = result["pack_soc"][pack, branch]
            row[f"pack_{pack + 1}_current_branch_{branch + 1}_a"] = result[
                "pack_branch_currents_a"
            ][pack, branch]
            for zone in range(3):
                row[f"pack_{pack + 1}_battery_b{branch}_z{zone}_temp_c"] = result[
                    "pack_battery_zone_temperatures_k"
                ][pack, branch, zone] - 273.15
        for zone in range(3):
            row[f"pack_{pack + 1}_plate_zone_{zone}_temp_c"] = result[
                "pack_plate_temperatures_k"
            ][pack, zone] - 273.15
    return row


def run_case(
    case: CaseSpec,
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
    cluster = ReducedCluster(
        n_packs=N_PACKS,
        branch_resistance_factors=case.resistance_factors,
    )
    standalone = ReducedPack() if case.symmetric else None
    rows = []
    started = time.perf_counter()
    for step_index in range(int(round(duration_s / dt_s))):
        result = cluster.step(
            dt_s,
            case.current_a,
            SUPPLY_TEMPERATURE_K,
            TOTAL_MASS_FLOW_KG_S,
            AMBIENT_TEMPERATURE_K,
            case.direction,
        )
        if standalone is not None:
            standalone_result = standalone.step(
                dt_s,
                case.current_a,
                SUPPLY_TEMPERATURE_K,
                TOTAL_MASS_FLOW_KG_S / N_PACKS,
                AMBIENT_TEMPERATURE_K,
                1 if case.direction == "forward" else -1,
            )
            symmetry_difference = _max_symmetric_pack_difference(result)
            stage6_difference = _max_stage6_pack_difference(
                result, standalone_result
            )
        else:
            symmetry_difference = None
            stage6_difference = None
        rows.append(
            _timeseries_row(
                (step_index + 1) * dt_s,
                case,
                result,
                symmetry_difference,
                stage6_difference,
            )
        )
    runtime_s = time.perf_counter() - started
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / f"case_{case.case_id}_timeseries.csv", index=False)

    resistance = np.asarray(case.resistance_factors, dtype=float)
    final_flows = np.array(
        [frame[f"pack_{pack + 1}_mass_flow_kg_s"].iloc[-1] for pack in range(N_PACKS)]
    )
    summary = {
        "case_id": case.case_id,
        "current_a": case.current_a,
        "direction": case.direction,
        "symmetric": case.symmetric,
        "steps": len(frame),
        "max_abs_mass_flow_residual_kg_s": float(
            frame["mass_flow_conservation_residual_kg_s"].abs().max()
        ),
        "max_abs_return_mixing_residual_k": float(
            frame["return_mixing_residual_k"].abs().max()
        ),
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
        "max_flow_nonuniformity": float(frame["flow_nonuniformity"].max()),
        "max_symmetric_pack_difference": (
            float(frame["max_symmetric_pack_difference"].max())
            if case.symmetric
            else None
        ),
        "max_stage6_pack_regression_difference": (
            float(frame["stage6_pack_regression_difference"].max())
            if case.symmetric
            else None
        ),
        "max_abs_pack_coupling_residual_w": float(
            frame["max_abs_pack_coupling_residual_w"].max()
        ),
        "max_abs_pack_energy_residual_j": float(
            frame["max_abs_pack_energy_residual_j"].max()
        ),
        "hydraulic_order_pass": _hydraulic_order_is_correct(
            resistance, final_flows
        ),
        "all_states_finite": bool(
            np.all(np.isfinite(frame.select_dtypes(include=[np.number]).dropna(axis=1).to_numpy()))
        ),
        "runtime_s": runtime_s,
    }
    summary["all_gates_pass"] = bool(
        summary["max_abs_mass_flow_residual_kg_s"] <= 1e-15
        and summary["max_abs_return_mixing_residual_k"] <= 1e-12
        and summary["max_abs_pack_coupling_residual_w"] <= 1e-12
        and summary["hydraulic_order_pass"]
        and summary["all_states_finite"]
        and (
            not case.symmetric
            or (
                summary["max_symmetric_pack_difference"] <= 1e-12
                and summary["max_stage6_pack_regression_difference"] <= 1e-12
                and summary["max_inter_pack_delta_temperature_k"] <= 1e-12
            )
        )
    )

    pack_rows = []
    hydraulic_rows = []
    for pack in range(N_PACKS):
        pack_rows.append(
            {
                "case_id": case.case_id,
                "pack_index": pack + 1,
                "final_battery_avg_temp_c": frame[f"pack_{pack + 1}_battery_avg_temp_c"].iloc[-1],
                "max_battery_avg_temp_c": frame[f"pack_{pack + 1}_battery_avg_temp_c"].max(),
                "final_battery_max_temp_c": frame[f"pack_{pack + 1}_battery_max_temp_c"].iloc[-1],
                "final_battery_min_temp_c": frame[f"pack_{pack + 1}_battery_min_temp_c"].iloc[-1],
                "final_intra_pack_delta_temperature_k": frame[f"pack_{pack + 1}_battery_delta_temp_k"].iloc[-1],
                "max_intra_pack_delta_temperature_k": frame[f"pack_{pack + 1}_battery_delta_temp_k"].max(),
                "final_plate_avg_temp_c": frame[f"pack_{pack + 1}_plate_avg_temp_c"].iloc[-1],
                "final_coolant_outlet_temp_c": frame[f"pack_{pack + 1}_coolant_outlet_temp_c"].iloc[-1],
                "final_return_temperature_c": frame["return_temperature_c"].iloc[-1],
            }
        )
        hydraulic_rows.append(
            {
                "case_id": case.case_id,
                "pack_index": pack + 1,
                "resistance_factor": case.resistance_factors[pack],
                "mean_mass_flow_kg_s": frame[f"pack_{pack + 1}_mass_flow_kg_s"].mean(),
                "flow_share": frame[f"pack_{pack + 1}_mass_flow_kg_s"].mean() / TOTAL_MASS_FLOW_KG_S,
            }
        )
    return {
        "timeseries": frame,
        "summary": summary,
        "packs": pack_rows,
        "hydraulics": hydraulic_rows,
    }


def _write_aggregate_results(output_dir: Path, results: list[dict]) -> None:
    pd.DataFrame([result["summary"] for result in results]).to_csv(
        output_dir / "cluster_summary.csv", index=False
    )
    pd.DataFrame([row for result in results for row in result["packs"]]).to_csv(
        output_dir / "cluster_pack_summary.csv", index=False
    )
    pd.DataFrame(
        [row for result in results for row in result["hydraulics"]]
    ).to_csv(output_dir / "cluster_hydraulics_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--dt-s", type=float, default=DEFAULT_DT_S)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--case-id", action="append")
    args = parser.parse_args()

    output_dir = args.output_dir or RESULTS_ROOT / (
        f"cluster_validation_{datetime.now():%Y%m%d}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    gate = compare_one_step()
    pd.DataFrame([gate]).to_csv(output_dir / "cluster_one_step.csv", index=False)
    if not (
        gate["mass_flow_conservation_residual_kg_s"] == 0.0
        and gate["return_mixing_residual_k"] == 0.0
        and gate["inter_pack_delta_temperature_k"] == 0.0
        and gate["max_symmetric_pack_difference"] == 0.0
        and gate["max_stage6_pack_regression_difference"] == 0.0
        and gate["max_abs_pack_coupling_residual_w"] == 0.0
        and gate["all_states_finite"]
    ):
        raise RuntimeError("Stage 7A one-step gate failed; formal cases stopped")

    selected = set(args.case_id or [])
    cases = [
        case
        for case in build_case_specs()
        if not selected or case.case_id in selected
    ]
    unknown = selected.difference(case.case_id for case in cases)
    if unknown:
        raise ValueError(f"unknown case ids: {sorted(unknown)}")
    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] running {case.case_id}", flush=True)
        result = run_case(
            case,
            duration_s=args.duration_s,
            dt_s=args.dt_s,
            output_dir=output_dir,
        )
        results.append(result)
        _write_aggregate_results(output_dir, results)
        summary = result["summary"]
        print(
            f"  inter-pack max={summary['max_inter_pack_delta_temperature_k']:.9g} K, "
            f"flow residual={summary['max_abs_mass_flow_residual_kg_s']:.3g} kg/s, "
            f"pass={summary['all_gates_pass']}",
            flush=True,
        )
    print(f"cluster results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
