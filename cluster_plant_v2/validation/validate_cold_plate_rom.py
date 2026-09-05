"""Validate the isolated three-zone cold-plate ROM against the 13-node model."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_plant_v2.thermal.cold_plate_rom import ReducedColdPlate, ZONE_COLUMN_COUNTS
from cluster_plant_v2.parameters import PLATE_NODE_HEAT_CAPACITY_TOTAL
from cluster_plant_v2.validation.legacy_cold_plate_reference import (
    A_bp_seg,
    cold_plate_fluid_exchange,
    cp_cool,
    h_bp_nominal,
    m_dot_nominal,
    rho_cool,
)


ZONE_GROUPS = ((0, 1, 2, 3), (4, 5, 6, 7, 8), (9, 10, 11, 12))
DEFAULT_DT_S = 5.0
DEFAULT_DURATION_S = 600.0
COOLANT_INLET_K = 20.0 + 273.15
Q_RELATIVE_EPS_W = 1.0
RESULTS_ROOT = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    flow_L_min: float
    heat_kind: str
    flow_direction: int

    @property
    def direction_name(self) -> str:
        return "forward" if self.flow_direction == 1 else "reverse"


def liters_per_minute_to_mass_flow(flow_L_min: float) -> float:
    flow = float(flow_L_min)
    if not np.isfinite(flow) or flow <= 0.0:
        raise ValueError("flow_L_min must be positive and finite")
    return flow / 1000.0 / 60.0 * rho_cool


def aggregate_node_temperatures(node_temperatures: np.ndarray) -> np.ndarray:
    nodes = np.asarray(node_temperatures, dtype=float).reshape(13)
    return np.array([nodes[list(group)].mean() for group in ZONE_GROUPS])


def aggregate_node_heat(node_heat: np.ndarray) -> np.ndarray:
    nodes = np.asarray(node_heat, dtype=float).reshape(13)
    return np.array([nodes[list(group)].sum() for group in ZONE_GROUPS])


def build_case_specs() -> list[CaseSpec]:
    cases = [
        CaseSpec(f"{flow:g}lpm_uniform_forward", flow, "uniform", 1)
        for flow in (4.0, 5.0, 6.0, 10.0, 21.0, 37.0)
    ]
    cases.extend(
        CaseSpec(f"21lpm_{heat_kind}_{direction_name}", 21.0, heat_kind, direction)
        for heat_kind in ("uniform", "forward", "reverse", "dynamic")
        for direction_name, direction in (("forward", 1), ("reverse", -1))
        if not (heat_kind == "uniform" and direction == 1)
    )
    return cases


def battery_heat_profile(heat_kind: str, time_s: float) -> np.ndarray:
    if heat_kind == "uniform":
        return np.full(13, 100.0)
    if heat_kind == "forward":
        return np.linspace(50.0, 150.0, 13)
    if heat_kind == "reverse":
        return np.linspace(150.0, 50.0, 13)
    if heat_kind == "dynamic":
        if time_s < 200.0:
            heat_per_node = 50.0
        elif time_s < 400.0:
            heat_per_node = 50.0 + 100.0 * (time_s - 200.0) / 200.0
        else:
            heat_per_node = 150.0
        return np.full(13, heat_per_node)
    raise ValueError(f"unknown battery heat profile: {heat_kind}")


class DetailedColdPlateAdapter:
    """Validation-only adapter reproducing the current 13-node main path."""

    def __init__(self, initial_temperature_c: float = 25.0) -> None:
        self.plate_temperatures = np.full(13, float(initial_temperature_c) + 273.15)
        self.node_heat_capacity = PLATE_NODE_HEAT_CAPACITY_TOTAL / 13.0

    def step(
        self,
        dt: float,
        coolant_inlet_temperature: float,
        coolant_mass_flow: float,
        q_battery_to_plate_nodes: np.ndarray,
        flow_direction: int = 1,
    ) -> dict[str, float | np.ndarray]:
        if flow_direction not in (-1, 1):
            raise ValueError("flow_direction must be 1 or -1")
        battery_heat = np.asarray(q_battery_to_plate_nodes, dtype=float)
        if battery_heat.shape != (13,):
            raise ValueError("q_battery_to_plate_nodes must have shape (13,)")
        old_temperatures = self.plate_temperatures.copy()
        traversal_temperatures = (
            old_temperatures if flow_direction == 1 else old_temperatures[::-1]
        )
        exchange = cold_plate_fluid_exchange(
            traversal_temperatures,
            float(coolant_inlet_temperature),
            float(coolant_mass_flow),
        )
        fluid_means = np.asarray(exchange["T_fluid_profile"], dtype=float)
        if flow_direction == -1:
            fluid_means = fluid_means[::-1]
        h_dynamic = max(
            50.0,
            h_bp_nominal * (float(coolant_mass_flow) / m_dot_nominal) ** 0.8,
        )
        plate_state_heat_loss = h_dynamic * A_bp_seg * (
            old_temperatures - fluid_means
        )
        self.plate_temperatures = old_temperatures + float(dt) * (
            battery_heat - plate_state_heat_loss
        ) / self.node_heat_capacity
        coolant_heat_gain = float(exchange["Q_total"])
        return {
            "plate_temperatures": self.plate_temperatures.copy(),
            "coolant_outlet_temperature": float(exchange["T_out"]),
            "coolant_mean_temperatures": fluid_means,
            "q_plate_to_fluid_total": coolant_heat_gain,
            "q_plate_state_loss_total": float(plate_state_heat_loss.sum()),
            "energy_closure_error_W": float(plate_state_heat_loss.sum() - coolant_heat_gain),
            "h_dynamic": h_dynamic,
        }


def compare_one_step(flow_L_min: float = 21.0, flow_direction: int = 1) -> dict[str, float]:
    mass_flow = liters_per_minute_to_mass_flow(flow_L_min)
    node_heat = battery_heat_profile("uniform", 0.0)
    zone_heat = aggregate_node_heat(node_heat)
    reference = DetailedColdPlateAdapter()
    rom = ReducedColdPlate()
    ref = reference.step(5.0, COOLANT_INLET_K, mass_flow, node_heat, flow_direction)
    reduced = rom.step(5.0, COOLANT_INLET_K, mass_flow, zone_heat, flow_direction)
    ref_average = float(np.mean(reference.plate_temperatures))
    rom_average = float(
        np.sum(rom.plate_temperatures * ZONE_COLUMN_COUNTS) / 13.0
    )
    q_error = abs(
        reduced["q_plate_to_fluid_total"] - ref["q_plate_to_fluid_total"]
    )
    rom_coolant_gain = mass_flow * cp_cool * (
        reduced["coolant_outlet_temperature"] - COOLANT_INLET_K
    )
    return {
        "flow_L_min": float(flow_L_min),
        "flow_direction": int(flow_direction),
        "Tout_abs_error_C": abs(
            reduced["coolant_outlet_temperature"] - ref["coolant_outlet_temperature"]
        ),
        "weighted_plate_avg_abs_error_C": abs(rom_average - ref_average),
        "Q_total_abs_error_W": q_error,
        "Q_total_relative_error": q_error / max(abs(ref["q_plate_to_fluid_total"]), Q_RELATIVE_EPS_W),
        "reference_energy_closure_error_W": abs(ref["energy_closure_error_W"]),
        "rom_energy_closure_abs_error_W": abs(
            rom_coolant_gain - reduced["q_plate_to_fluid_total"]
        ),
    }


def _run_reference(
    case: CaseSpec, times: np.ndarray, dt: float, mass_flow: float
) -> tuple[list[dict], float]:
    started = time.perf_counter()
    plate = DetailedColdPlateAdapter()
    records = []
    for time_s in times:
        node_heat = battery_heat_profile(case.heat_kind, float(time_s))
        result = plate.step(
            dt, COOLANT_INLET_K, mass_flow, node_heat, case.flow_direction
        )
        records.append({**result, "battery_heat": node_heat})
    return records, time.perf_counter() - started


def _run_rom(
    case: CaseSpec, times: np.ndarray, dt: float, mass_flow: float
) -> tuple[list[dict], float]:
    started = time.perf_counter()
    plate = ReducedColdPlate()
    records = []
    for time_s in times:
        zone_heat = aggregate_node_heat(
            battery_heat_profile(case.heat_kind, float(time_s))
        )
        result = plate.step(
            dt, COOLANT_INLET_K, mass_flow, zone_heat, case.flow_direction
        )
        records.append({**result, "battery_heat": zone_heat})
    return records, time.perf_counter() - started


def _build_timeseries(
    case: CaseSpec,
    times: np.ndarray,
    dt: float,
    mass_flow: float,
    reference_records: list[dict],
    rom_records: list[dict],
) -> pd.DataFrame:
    rows = []
    for time_s, reference, rom in zip(times, reference_records, rom_records):
        ref_plate = np.asarray(reference["plate_temperatures"], dtype=float)
        rom_plate = np.asarray(rom["plate_temperatures"], dtype=float)
        ref_zones = aggregate_node_temperatures(ref_plate)
        zone_heat = np.asarray(rom["battery_heat"], dtype=float)
        row = {
            "time_s": float(time_s + dt),
            "flow_direction": case.flow_direction,
            "coolant_flow_L_min": case.flow_L_min,
            "coolant_mass_flow_kg_s": mass_flow,
            "coolant_inlet_temperature_C": COOLANT_INLET_K - 273.15,
            "Ref_Tout_C": float(reference["coolant_outlet_temperature"] - 273.15),
            "ROM_Tout_C": float(rom["coolant_outlet_temperature"] - 273.15),
            "Ref_Tplate_avg_C": float(ref_plate.mean() - 273.15),
            "ROM_Tplate_avg_C": float(
                np.sum((rom_plate - 273.15) * ZONE_COLUMN_COUNTS) / 13.0
            ),
            "Ref_Q_plate_to_fluid_total_W": float(reference["q_plate_to_fluid_total"]),
            "ROM_Q_plate_to_fluid_total_W": float(rom["q_plate_to_fluid_total"]),
            "Ref_plate_state_loss_total_W": float(reference["q_plate_state_loss_total"]),
            "Ref_energy_closure_error_W": float(reference["energy_closure_error_W"]),
            "ROM_energy_closure_error_W": float(
                mass_flow * cp_cool * (
                    rom["coolant_outlet_temperature"] - COOLANT_INLET_K
                ) - rom["q_plate_to_fluid_total"]
            ),
        }
        for zone in range(3):
            row[f"Ref_Tplate_zone{zone}_C"] = float(ref_zones[zone] - 273.15)
            row[f"ROM_Tplate_zone{zone}_C"] = float(rom_plate[zone] - 273.15)
            row[f"battery_side_Q_zone{zone}_W"] = float(zone_heat[zone])
            row[f"ROM_Tfluid_zone{zone}_C"] = float(
                rom["coolant_mean_temperatures"][zone] - 273.15
            )
        for node in range(13):
            row[f"Ref_Tfluid_node{node}_C"] = float(
                reference["coolant_mean_temperatures"][node] - 273.15
            )
        rows.append(row)
    return pd.DataFrame(rows)


def error_metrics(reference: np.ndarray, rom: np.ndarray) -> dict[str, float]:
    error = np.asarray(rom, dtype=float) - np.asarray(reference, dtype=float)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(np.max(np.abs(error))),
        "final_error": float(error[-1]),
    }


def _metric_row(
    case_id: str,
    metric: str,
    component: str,
    unit: str,
    reference: np.ndarray,
    rom: np.ndarray,
) -> dict:
    return {
        "case_id": case_id,
        "metric": metric,
        "component": component,
        "unit": unit,
        **error_metrics(reference, rom),
    }


def _calculate_metrics(case: CaseSpec, frame: pd.DataFrame) -> tuple[list[dict], list[dict], dict]:
    metrics = []
    for metric, unit, ref_column, rom_column in (
        ("Tout", "degC", "Ref_Tout_C", "ROM_Tout_C"),
        ("plate_average", "degC", "Ref_Tplate_avg_C", "ROM_Tplate_avg_C"),
        ("Q_plate_to_fluid", "W", "Ref_Q_plate_to_fluid_total_W", "ROM_Q_plate_to_fluid_total_W"),
    ):
        metrics.append(
            _metric_row(
                case.case_id, metric, "plate", unit,
                frame[ref_column].to_numpy(), frame[rom_column].to_numpy(),
            )
        )
    zone_metrics = []
    for zone, location in enumerate(("inlet", "middle", "outlet")):
        row = _metric_row(
            case.case_id,
            "plate_zone_temperature",
            f"zone_{zone}",
            "degC",
            frame[f"Ref_Tplate_zone{zone}_C"].to_numpy(),
            frame[f"ROM_Tplate_zone{zone}_C"].to_numpy(),
        )
        row.update({"zone": zone, "zone_location": location})
        zone_metrics.append(row)
        metrics.append(dict(row))

    indexed = {(row["metric"], row["component"]): row for row in metrics}
    q_reference = frame["Ref_Q_plate_to_fluid_total_W"].to_numpy()
    q_error = np.abs(frame["ROM_Q_plate_to_fluid_total_W"].to_numpy() - q_reference)
    q_relative = q_error / np.maximum(np.abs(q_reference), Q_RELATIVE_EPS_W)
    worst_zone = max(zone_metrics, key=lambda row: row["mae"])
    plate_average_error = frame["ROM_Tplate_avg_C"] - frame["Ref_Tplate_avg_C"]
    drift = 0.0
    if len(frame) > 1:
        drift = float(np.polyfit(frame["time_s"], plate_average_error, 1)[0] * 600.0)
    summary = {
        "case_id": case.case_id,
        "flow_L_min": case.flow_L_min,
        "coolant_mass_flow_kg_s": float(frame["coolant_mass_flow_kg_s"].iloc[0]),
        "heat_kind": case.heat_kind,
        "flow_direction": case.flow_direction,
        "direction_name": case.direction_name,
        "steps": len(frame),
        "Tout_MAE_C": indexed[("Tout", "plate")]["mae"],
        "Tout_max_error_C": indexed[("Tout", "plate")]["max_abs_error"],
        "Tout_final_error_C": indexed[("Tout", "plate")]["final_error"],
        "plate_avg_MAE_C": indexed[("plate_average", "plate")]["mae"],
        "plate_avg_max_error_C": indexed[("plate_average", "plate")]["max_abs_error"],
        "plate_avg_final_error_C": indexed[("plate_average", "plate")]["final_error"],
        "plate_avg_error_drift_per_600s_C": drift,
        "worst_zone": worst_zone["component"],
        "worst_zone_location": worst_zone["zone_location"],
        "worst_zone_MAE_C": worst_zone["mae"],
        "worst_zone_max_error_C": worst_zone["max_abs_error"],
        "Q_mean_relative_error": float(q_relative.mean()),
        "Q_max_relative_error": float(q_relative.max()),
        "reference_energy_closure_max_abs_W": float(
            frame["Ref_energy_closure_error_W"].abs().max()
        ),
        "rom_energy_closure_max_abs_W": float(
            frame["ROM_energy_closure_error_W"].abs().max()
        ),
    }
    summary.update(
        {
            "pass_Tout": summary["Tout_MAE_C"] <= 0.20,
            "pass_plate_zone": summary["worst_zone_MAE_C"] <= 0.30,
            "pass_plate_average": summary["plate_avg_MAE_C"] <= 0.20,
            "pass_Q": summary["Q_mean_relative_error"] <= 0.05,
        }
    )
    summary["all_numeric_gates_pass"] = all(
        summary[key] for key in ("pass_Tout", "pass_plate_zone", "pass_plate_average", "pass_Q")
    )
    return metrics, zone_metrics, summary


def _plot_case(case: CaseSpec, frame: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "legend.frameon": False,
            "savefig.dpi": 300,
        }
    )
    blue, orange = "#0072B2", "#D55E00"
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    axes[0].plot(frame["time_s"], frame["Ref_Tout_C"], color=blue, label="13-node Reference")
    axes[0].plot(frame["time_s"], frame["ROM_Tout_C"], color=orange, linestyle="--", label="3-node ROM")
    axes[0].set_title("Coolant outlet")
    axes[0].set_ylabel("Temperature (°C)")
    for zone, color in enumerate(("#009E73", "#E69F00", "#CC79A7")):
        axes[1].plot(frame["time_s"], frame[f"Ref_Tplate_zone{zone}_C"], color=color, label=f"Ref zone {zone}")
        axes[1].plot(frame["time_s"], frame[f"ROM_Tplate_zone{zone}_C"], color=color, linestyle="--", label=f"ROM zone {zone}")
    axes[1].set_title("Plate zone temperatures")
    axes[1].set_ylabel("Temperature (°C)")
    axes[2].plot(frame["time_s"], frame["Ref_Q_plate_to_fluid_total_W"], color=blue, label="13-node Reference")
    axes[2].plot(frame["time_s"], frame["ROM_Q_plate_to_fluid_total_W"], color=orange, linestyle="--", label="3-node ROM")
    axes[2].set_title("Plate-to-fluid heat")
    axes[2].set_ylabel("Heat transfer (W)")
    for axis in axes:
        axis.set_xlabel("Time (s)")
        axis.legend(fontsize=7)
    fig.suptitle(f"{case.flow_L_min:g} L/min, {case.heat_kind} heat, {case.direction_name} flow")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_case(
    case: CaseSpec,
    *,
    duration_s: float,
    dt: float,
    output_dir: Path,
    make_plot: bool = True,
) -> dict:
    if duration_s <= 0.0 or dt <= 0.0 or not np.isclose(duration_s / dt, round(duration_s / dt)):
        raise ValueError("duration_s must be a positive integer multiple of dt")
    times = np.arange(0.0, duration_s, dt)
    mass_flow = liters_per_minute_to_mass_flow(case.flow_L_min)
    reference, reference_runtime = _run_reference(case, times, dt, mass_flow)
    rom, rom_runtime = _run_rom(case, times, dt, mass_flow)
    frame = _build_timeseries(case, times, dt, mass_flow, reference, rom)
    frame.to_csv(output_dir / f"case_{case.case_id}_timeseries.csv", index=False)
    metrics, zone_metrics, summary = _calculate_metrics(case, frame)
    runtime = {
        "case_id": case.case_id,
        "reference_runtime_s": reference_runtime,
        "rom_runtime_s": rom_runtime,
        "speedup": reference_runtime / rom_runtime,
        "runtime_scope": "model construction plus all cold-plate steps",
    }
    representative = {
        "21lpm_uniform_forward",
        "21lpm_uniform_reverse",
        "21lpm_forward_forward",
        "21lpm_dynamic_forward",
        "21lpm_dynamic_reverse",
    }
    if make_plot and case.case_id in representative:
        _plot_case(case, frame, output_dir / f"case_{case.case_id}_comparison.png")
    return {
        "timeseries": frame,
        "metrics": metrics,
        "zone_metrics": zone_metrics,
        "summary": summary,
        "runtime": runtime,
    }


def _write_aggregate_results(output_dir: Path, results: list[dict]) -> None:
    pd.DataFrame([result["summary"] for result in results]).to_csv(
        output_dir / "cold_plate_rom_summary.csv", index=False
    )
    pd.DataFrame([row for result in results for row in result["metrics"]]).to_csv(
        output_dir / "cold_plate_rom_metrics.csv", index=False
    )
    pd.DataFrame([row for result in results for row in result["zone_metrics"]]).to_csv(
        output_dir / "cold_plate_zone_metrics.csv", index=False
    )
    runtime_rows = [result["runtime"] for result in results]
    reference_total = sum(row["reference_runtime_s"] for row in runtime_rows)
    rom_total = sum(row["rom_runtime_s"] for row in runtime_rows)
    runtime_rows.append(
        {
            "case_id": "OVERALL",
            "reference_runtime_s": reference_total,
            "rom_runtime_s": rom_total,
            "speedup": reference_total / rom_total,
            "runtime_scope": "sum across completed formal cases",
        }
    )
    pd.DataFrame(runtime_rows).to_csv(
        output_dir / "cold_plate_runtime_summary.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT_S)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or RESULTS_ROOT / f"cold_plate_rom_validation_{datetime.now():%Y%m%d}"
    output_dir.mkdir(parents=True, exist_ok=False)
    selected = set(args.case_id or [])
    cases = [case for case in build_case_specs() if not selected or case.case_id in selected]
    unknown = selected.difference(case.case_id for case in cases)
    if unknown:
        raise ValueError(f"unknown case ids: {sorted(unknown)}")
    one_step_rows = [
        compare_one_step(flow, direction)
        for flow in (4.0, 5.0, 6.0, 10.0, 21.0, 37.0)
        for direction in (1, -1)
    ]
    pd.DataFrame(one_step_rows).to_csv(
        output_dir / "cold_plate_rom_one_step.csv", index=False
    )

    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] running {case.case_id}", flush=True)
        result = run_case(
            case,
            duration_s=args.duration_s,
            dt=args.dt,
            output_dir=output_dir,
            make_plot=not args.no_plots,
        )
        results.append(result)
        _write_aggregate_results(output_dir, results)
        summary = result["summary"]
        print(
            f"  Tout MAE={summary['Tout_MAE_C']:.6f} C, "
            f"worst zone MAE={summary['worst_zone_MAE_C']:.6f} C, "
            f"Q mean rel={summary['Q_mean_relative_error']:.6%}",
            flush=True,
        )
    print(f"cold-plate ROM results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
