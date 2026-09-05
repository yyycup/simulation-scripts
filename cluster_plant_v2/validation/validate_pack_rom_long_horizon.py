"""Validate the frozen 4-by-3 battery ROM against ReferenceBatteryPack V2."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_plant_v2.pack_reference import ReferenceBatteryPack
from cluster_plant_v2.thermal.pack_rom import ReducedBatteryPack, ZONE_CELL_COUNTS, ZONE_GROUPS
from cluster_plant_v2.profiles import AGC_DATA_FILE, load_regd_profile


DEFAULT_DT_S = 5.0
DEFAULT_DURATION_S = 600.0
AMBIENT_TEMPERATURE_K = 35.0 + 273.15
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
QGEN_RELATIVE_EPS_W = 1.0


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    current_kind: str
    current_A: float | None
    boundary_kind: str

    @property
    def current_condition(self) -> str:
        return "RegD [0, 600) s" if self.current_kind == "regd" else f"{self.current_A:g} A"


def build_case_specs() -> list[CaseSpec]:
    cases = []
    currents = (("280a", "constant", 280.0), ("560a", "constant", 560.0),
                ("1120a", "constant", 1120.0), ("regd", "regd", None))
    for boundary_kind in ("uniform", "forward", "reverse"):
        for current_label, current_kind, current_A in currents:
            cases.append(
                CaseSpec(
                    case_id=f"{current_label}_{boundary_kind}",
                    current_kind=current_kind,
                    current_A=current_A,
                    boundary_kind=boundary_kind,
                )
            )
    return cases


def build_plate_boundaries(boundary_kind: str) -> tuple[np.ndarray, np.ndarray]:
    if boundary_kind == "uniform":
        reference_c = np.full(13, 25.0)
    elif boundary_kind == "forward":
        reference_c = np.linspace(22.0, 25.0, 13)
    elif boundary_kind == "reverse":
        reference_c = np.linspace(25.0, 22.0, 13)
    else:
        raise ValueError(f"unknown plate boundary: {boundary_kind}")
    rom_c = np.array(
        [reference_c[list(columns)].mean() for columns in ZONE_GROUPS], dtype=float
    )
    return reference_c + 273.15, rom_c + 273.15


def aggregate_reference_zones(cell_values: np.ndarray) -> np.ndarray:
    cells = np.asarray(cell_values, dtype=float).reshape(4, 13)
    return np.array(
        [[cells[branch, list(columns)].mean() for columns in ZONE_GROUPS]
         for branch in range(4)],
        dtype=float,
    )


def weighted_zone_average(zone_values: np.ndarray) -> float:
    zones = np.asarray(zone_values, dtype=float).reshape(4, 3)
    return float(np.sum(zones * ZONE_CELL_COUNTS[None, :]) / 52.0)


def error_metrics(reference: np.ndarray, rom: np.ndarray) -> dict[str, float]:
    error = np.asarray(rom, dtype=float) - np.asarray(reference, dtype=float)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(np.max(np.abs(error))),
        "final_error": float(error[-1]),
    }


def normalized_rmse(reference: np.ndarray, rom: np.ndarray) -> float:
    reference_values = np.asarray(reference, dtype=float)
    error = np.asarray(rom, dtype=float) - reference_values
    scale = float(np.sqrt(np.mean(reference_values**2)))
    if scale <= 1e-12:
        return 0.0 if np.allclose(error, 0.0, atol=1e-12) else float("inf")
    return float(np.sqrt(np.mean(error**2)) / scale)


def _current_profile(
    case: CaseSpec, *, duration_s: float, dt: float, regd_file: Path
) -> np.ndarray:
    if case.current_kind == "regd":
        _, currents = load_regd_profile(regd_file, duration_s=duration_s, dt=dt)
        return currents
    return np.full(int(round(duration_s / dt)), float(case.current_A), dtype=float)


def _pack_config(initial_current_A: float) -> dict[str, float | bool]:
    return {
        "capacity": 280.0,
        "initial_soc": 0.95,
        "initial_temp_c": 25.0,
        "total_current": float(initial_current_A),
        "cell_thermal_mass": 4747.0,
        "k_intercell": 0.5,
        "h_plate_convection": 10.0,
        "h_air_convection_edge": 0.1,
        "uniform_air_convection": False,
        "dynamic_resistance_update": True,
    }


def _run_reference(
    currents: np.ndarray, dt: float, plate_k: np.ndarray
) -> tuple[list[dict[str, np.ndarray | float]], dict[str, float]]:
    init_started = time.perf_counter()
    pack = ReferenceBatteryPack(_pack_config(float(currents[0])))
    init_runtime = time.perf_counter() - init_started
    records = []
    step_runtimes = []
    for current in currents:
        started = time.perf_counter()
        pack.step(dt, float(current), plate_k, AMBIENT_TEMPERATURE_K)
        step_runtimes.append(time.perf_counter() - started)
        cell_temperatures_c = pack.temps.reshape(4, 13) - 273.15
        records.append(
            {
                "cells_c": cell_temperatures_c.copy(),
                "zones_c": aggregate_reference_zones(cell_temperatures_c),
                "soc": pack.socs.reshape(4, 13).mean(axis=1),
                "branch_currents": pack.branch_currents.copy(),
                "qgen_total": float(pack.q_gen_cells.sum()),
            }
        )
    stepping_runtime = float(sum(step_runtimes))
    return records, {
        "initialization_s": init_runtime,
        "first_step_s": step_runtimes[0],
        "steps_s": stepping_runtime,
        "excluding_first_step_s": float(sum(step_runtimes[1:])),
        "total_s": init_runtime + stepping_runtime,
    }


def _run_rom(
    currents: np.ndarray, dt: float, plate_k: np.ndarray
) -> tuple[list[dict[str, np.ndarray | float]], dict[str, float]]:
    init_started = time.perf_counter()
    pack = ReducedBatteryPack(_pack_config(float(currents[0])))
    init_runtime = time.perf_counter() - init_started
    records = []
    step_runtimes = []
    for current in currents:
        started = time.perf_counter()
        pack.step(dt, float(current), plate_k, AMBIENT_TEMPERATURE_K)
        step_runtimes.append(time.perf_counter() - started)
        records.append(
            {
                "zones_c": pack.temps.copy() - 273.15,
                "soc": pack.soc_branch.copy(),
                "branch_currents": pack.branch_currents.copy(),
                "qgen_total": float(pack.q_gen_total),
            }
        )
    stepping_runtime = float(sum(step_runtimes))
    return records, {
        "initialization_s": init_runtime,
        "first_step_s": step_runtimes[0],
        "steps_s": stepping_runtime,
        "excluding_first_step_s": float(sum(step_runtimes[1:])),
        "total_s": init_runtime + stepping_runtime,
    }


def _build_timeseries(
    currents: np.ndarray,
    dt: float,
    reference_records: list[dict[str, np.ndarray | float]],
    rom_records: list[dict[str, np.ndarray | float]],
) -> pd.DataFrame:
    records = []
    for index, (current, reference, rom) in enumerate(
        zip(currents, reference_records, rom_records), start=1
    ):
        ref_cells = np.asarray(reference["cells_c"], dtype=float)
        ref_zones = np.asarray(reference["zones_c"], dtype=float)
        rom_zones = np.asarray(rom["zones_c"], dtype=float)
        ref_zone_max = float(ref_zones.max())
        rom_max = float(rom_zones.max())
        record = {
            "time_s": index * dt,
            "current_A": float(current),
            "Ref_Tavg_C": float(ref_cells.mean()),
            "ROM_Tavg_C": weighted_zone_average(rom_zones),
            "Ref_zone_Tmax_C": ref_zone_max,
            "ROM_Tmax_C": rom_max,
            "Ref_zone_Tmin_C": float(ref_zones.min()),
            "ROM_Tmin_C": float(rom_zones.min()),
            "Ref_zone_DeltaT_C": float(np.ptp(ref_zones)),
            "ROM_DeltaT_C": float(np.ptp(rom_zones)),
            "Ref_cell_Tmax_C": float(ref_cells.max()),
            "Ref_cell_Tmin_C": float(ref_cells.min()),
            "Ref_cell_DeltaT_C": float(np.ptp(ref_cells)),
            "hotspot_loss_C": float(ref_cells.max() - rom_max),
            "Ref_Qgen_total_W": float(reference["qgen_total"]),
            "ROM_Qgen_total_W": float(rom["qgen_total"]),
        }
        for branch in range(4):
            record[f"Ref_SOC_branch_{branch + 1}"] = float(reference["soc"][branch])
            record[f"ROM_SOC_branch_{branch + 1}"] = float(rom["soc"][branch])
            record[f"Ref_Ibranch_{branch + 1}_A"] = float(reference["branch_currents"][branch])
            record[f"ROM_Ibranch_{branch + 1}_A"] = float(rom["branch_currents"][branch])
            for zone in range(3):
                record[f"Ref_T_b{branch}_z{zone}_C"] = float(ref_zones[branch, zone])
                record[f"ROM_T_b{branch}_z{zone}_C"] = float(rom_zones[branch, zone])
        records.append(record)
    return pd.DataFrame(records)


def _metric_row(
    case_id: str,
    metric: str,
    component: str,
    unit: str,
    reference: np.ndarray,
    rom: np.ndarray,
) -> dict[str, str | float]:
    return {
        "case_id": case_id,
        "metric": metric,
        "component": component,
        "unit": unit,
        **error_metrics(reference, rom),
    }


def _calculate_metrics(
    case: CaseSpec, frame: pd.DataFrame
) -> tuple[list[dict], list[dict], dict]:
    metric_specs = (
        ("Tavg", "pack", "degC", "Ref_Tavg_C", "ROM_Tavg_C"),
        ("zone_Tmax", "pack", "degC", "Ref_zone_Tmax_C", "ROM_Tmax_C"),
        ("zone_Tmin", "pack", "degC", "Ref_zone_Tmin_C", "ROM_Tmin_C"),
        ("zone_DeltaT", "pack", "degC", "Ref_zone_DeltaT_C", "ROM_DeltaT_C"),
        ("Qgen_total", "pack", "W", "Ref_Qgen_total_W", "ROM_Qgen_total_W"),
    )
    metrics = [
        _metric_row(case.case_id, metric, component, unit,
                    frame[reference].to_numpy(), frame[rom].to_numpy())
        for metric, component, unit, reference, rom in metric_specs
    ]
    for branch in range(1, 5):
        current_reference = frame[f"Ref_Ibranch_{branch}_A"].to_numpy()
        current_rom = frame[f"ROM_Ibranch_{branch}_A"].to_numpy()
        current_row = _metric_row(
            case.case_id, "branch_current", f"branch_{branch}", "A",
            current_reference, current_rom,
        )
        current_row["nrmse"] = normalized_rmse(current_reference, current_rom)
        metrics.append(current_row)
        metrics.append(
            _metric_row(
                case.case_id, "branch_SOC", f"branch_{branch}", "fraction",
                frame[f"Ref_SOC_branch_{branch}"].to_numpy(),
                frame[f"ROM_SOC_branch_{branch}"].to_numpy(),
            )
        )

    zone_metrics = []
    location_names = ("inlet", "middle", "outlet")
    for branch in range(4):
        for zone in range(3):
            zone_row = _metric_row(
                case.case_id,
                "zone_temperature",
                f"b{branch}_z{zone}",
                "degC",
                frame[f"Ref_T_b{branch}_z{zone}_C"].to_numpy(),
                frame[f"ROM_T_b{branch}_z{zone}_C"].to_numpy(),
            )
            zone_row.update({"branch": branch, "zone": zone, "zone_location": location_names[zone]})
            zone_metrics.append(zone_row)
            metrics.append(dict(zone_row))

    by_name = {(row["metric"], row["component"]): row for row in metrics}
    q_reference = frame["Ref_Qgen_total_W"].to_numpy()
    q_error = np.abs(frame["ROM_Qgen_total_W"].to_numpy() - q_reference)
    q_relative = q_error / np.maximum(np.abs(q_reference), QGEN_RELATIVE_EPS_W)
    worst_zone = max(zone_metrics, key=lambda row: row["mae"])
    tavg_error = frame["ROM_Tavg_C"].to_numpy() - frame["Ref_Tavg_C"].to_numpy()
    drift_per_600 = 0.0
    if len(frame) > 1:
        drift_per_600 = float(np.polyfit(frame["time_s"], tavg_error, 1)[0] * 600.0)
    branch_nrmse = [
        by_name[("branch_current", f"branch_{branch}")]["nrmse"]
        for branch in range(1, 5)
    ]
    summary = {
        "case_id": case.case_id,
        "current_condition": case.current_condition,
        "boundary_kind": case.boundary_kind,
        "steps": len(frame),
        "Tavg_MAE_C": by_name[("Tavg", "pack")]["mae"],
        "Tavg_max_error_C": by_name[("Tavg", "pack")]["max_abs_error"],
        "Tavg_final_error_C": by_name[("Tavg", "pack")]["final_error"],
        "Tavg_error_drift_per_600s_C": drift_per_600,
        "zone_Tmax_MAE_C": by_name[("zone_Tmax", "pack")]["mae"],
        "zone_Tmax_max_error_C": by_name[("zone_Tmax", "pack")]["max_abs_error"],
        "zone_Tmax_mean_signed_error_C": float(
            np.mean(frame["ROM_Tmax_C"] - frame["Ref_zone_Tmax_C"])
        ),
        "zone_DeltaT_MAE_C": by_name[("zone_DeltaT", "pack")]["mae"],
        "zone_DeltaT_max_error_C": by_name[("zone_DeltaT", "pack")]["max_abs_error"],
        "worst_zone": worst_zone["component"],
        "worst_zone_location": worst_zone["zone_location"],
        "worst_zone_MAE_C": worst_zone["mae"],
        "worst_zone_max_error_C": worst_zone["max_abs_error"],
        "hotspot_loss_mean_C": float(frame["hotspot_loss_C"].mean()),
        "hotspot_loss_max_C": float(frame["hotspot_loss_C"].max()),
        "hotspot_loss_final_C": float(frame["hotspot_loss_C"].iloc[-1]),
        "Qgen_mean_relative_error": float(q_relative.mean()),
        "Qgen_max_relative_error": float(q_relative.max()),
        "branch_current_max_NRMSE": float(max(branch_nrmse)),
        "branch_SOC_max_MAE": float(
            max(by_name[("branch_SOC", f"branch_{branch}")]["mae"] for branch in range(1, 5))
        ),
        "branch_SOC_max_abs_error": float(
            max(by_name[("branch_SOC", f"branch_{branch}")]["max_abs_error"] for branch in range(1, 5))
        ),
    }
    summary.update(
        {
            "pass_Tavg": summary["Tavg_MAE_C"] <= 0.20,
            "pass_zone_temperature": summary["worst_zone_MAE_C"] <= 0.30,
            "pass_zone_Tmax": summary["zone_Tmax_MAE_C"] <= 0.30,
            "pass_zone_DeltaT": summary["zone_DeltaT_MAE_C"] <= 0.30,
            "pass_Qgen": summary["Qgen_mean_relative_error"] <= 0.05,
            "pass_branch_current": summary["branch_current_max_NRMSE"] <= 0.05,
        }
    )
    summary["all_numeric_gates_pass"] = all(
        summary[key] for key in (
            "pass_Tavg", "pass_zone_temperature", "pass_zone_Tmax",
            "pass_zone_DeltaT", "pass_Qgen", "pass_branch_current",
        )
    )
    return metrics, zone_metrics, summary


def _plot_comparison(case: CaseSpec, frame: pd.DataFrame, output_path: Path) -> None:
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
    colors = {"Reference": "#0072B2", "ROM": "#D55E00"}
    panels = (
        ("Ref_Tavg_C", "ROM_Tavg_C", "Average temperature", "Temperature (°C)"),
        ("Ref_zone_Tmax_C", "ROM_Tmax_C", "Zone maximum temperature", "Temperature (°C)"),
        ("Ref_zone_DeltaT_C", "ROM_DeltaT_C", "Zone temperature spread", "ΔT (°C)"),
        ("Ref_Qgen_total_W", "ROM_Qgen_total_W", "Total heat generation", "Heat (W)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), sharex=True)
    for axis, (ref_column, rom_column, title, ylabel) in zip(axes.ravel(), panels):
        axis.plot(frame["time_s"], frame[ref_column], label="Reference", color=colors["Reference"])
        axis.plot(frame["time_s"], frame[rom_column], label="4×3 ROM", color=colors["ROM"], linestyle="--")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.legend()
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    fig.suptitle(f"{case.current_condition}, {case.boundary_kind} plate boundary")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_zone_errors(case: CaseSpec, frame: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    errors = np.vstack(
        [
            frame[f"ROM_T_b{branch}_z{zone}_C"].to_numpy()
            - frame[f"Ref_T_b{branch}_z{zone}_C"].to_numpy()
            for branch in range(4)
            for zone in range(3)
        ]
    )
    limit = max(float(np.max(np.abs(errors))), 1e-6)
    fig, axis = plt.subplots(figsize=(9.0, 4.2))
    image = axis.imshow(
        errors,
        aspect="auto",
        origin="lower",
        extent=[frame["time_s"].iloc[0], frame["time_s"].iloc[-1], -0.5, 11.5],
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    axis.set_yticks(range(12))
    axis.set_yticklabels([f"b{branch} z{zone}" for branch in range(4) for zone in range(3)])
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("ROM zone")
    axis.set_title(f"Zone temperature error: {case.current_condition}, {case.boundary_kind}")
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("ROM − Reference (°C)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_case(
    case: CaseSpec,
    *,
    duration_s: float,
    dt: float,
    regd_file: Path,
    output_dir: Path,
    make_plot: bool = True,
) -> dict:
    if duration_s <= 0.0 or dt <= 0.0 or not np.isclose(duration_s / dt, round(duration_s / dt)):
        raise ValueError("duration_s must be a positive integer multiple of dt")
    currents = _current_profile(case, duration_s=duration_s, dt=dt, regd_file=regd_file)
    if currents.size == 0:
        raise ValueError("validation case has no time steps")
    reference_plate, rom_plate = build_plate_boundaries(case.boundary_kind)
    reference_records, reference_runtime = _run_reference(currents, dt, reference_plate)
    rom_records, rom_runtime = _run_rom(currents, dt, rom_plate)
    frame = _build_timeseries(currents, dt, reference_records, rom_records)
    frame.to_csv(output_dir / f"case_{case.case_id}_timeseries.csv", index=False)
    metrics, zone_metrics, summary = _calculate_metrics(case, frame)
    steady_speedup = (
        reference_runtime["excluding_first_step_s"] / rom_runtime["excluding_first_step_s"]
        if rom_runtime["excluding_first_step_s"] > 0.0 else np.nan
    )
    runtime = {
        "case_id": case.case_id,
        **{f"reference_{key}": value for key, value in reference_runtime.items()},
        **{f"rom_{key}": value for key, value in rom_runtime.items()},
        "total_speedup": reference_runtime["total_s"] / rom_runtime["total_s"],
        "excluding_first_step_speedup": steady_speedup,
        "runtime_scope": "model construction plus 120 steps for total; steps 2-120 for excluding-first-step",
    }
    if make_plot:
        _plot_comparison(case, frame, output_dir / f"case_{case.case_id}_comparison.png")
        if case.case_id in {"560a_forward", "regd_forward"}:
            _plot_zone_errors(case, frame, output_dir / f"case_{case.case_id}_zone_errors.png")
    return {
        "timeseries": frame,
        "metrics": metrics,
        "zone_metrics": zone_metrics,
        "summary": summary,
        "runtime": runtime,
    }


def _write_aggregate_results(output_dir: Path, results: list[dict]) -> None:
    pd.DataFrame([result["summary"] for result in results]).to_csv(
        output_dir / "pack_rom_long_horizon_summary.csv", index=False
    )
    pd.DataFrame([row for result in results for row in result["metrics"]]).to_csv(
        output_dir / "pack_rom_long_horizon_metrics.csv", index=False
    )
    pd.DataFrame([row for result in results for row in result["zone_metrics"]]).to_csv(
        output_dir / "pack_rom_zone_metrics.csv", index=False
    )
    runtime_rows = [result["runtime"] for result in results]
    reference_total = sum(row["reference_total_s"] for row in runtime_rows)
    rom_total = sum(row["rom_total_s"] for row in runtime_rows)
    reference_steady = sum(row["reference_excluding_first_step_s"] for row in runtime_rows)
    rom_steady = sum(row["rom_excluding_first_step_s"] for row in runtime_rows)
    runtime_rows.append(
        {
            "case_id": "OVERALL",
            "reference_total_s": reference_total,
            "rom_total_s": rom_total,
            "total_speedup": reference_total / rom_total,
            "reference_excluding_first_step_s": reference_steady,
            "rom_excluding_first_step_s": rom_steady,
            "excluding_first_step_speedup": reference_steady / rom_steady,
            "runtime_scope": "sum across completed formal cases",
        }
    )
    pd.DataFrame(runtime_rows).to_csv(output_dir / "runtime_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT_S)
    parser.add_argument("--regd-file", type=Path, default=AGC_DATA_FILE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or RESULTS_ROOT / f"pack_rom_long_horizon_{datetime.now():%Y%m%d}"
    output_dir.mkdir(parents=True, exist_ok=False)
    selected = set(args.case_id or [])
    cases = [case for case in build_case_specs() if not selected or case.case_id in selected]
    unknown = selected.difference(case.case_id for case in cases)
    if unknown:
        raise ValueError(f"unknown case ids: {sorted(unknown)}")

    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] running {case.case_id}", flush=True)
        result = run_case(
            case,
            duration_s=args.duration_s,
            dt=args.dt,
            regd_file=args.regd_file,
            output_dir=output_dir,
            make_plot=not args.no_plots,
        )
        results.append(result)
        _write_aggregate_results(output_dir, results)
        summary = result["summary"]
        print(
            f"  Tavg MAE={summary['Tavg_MAE_C']:.6f} C, "
            f"worst zone MAE={summary['worst_zone_MAE_C']:.6f} C, "
            f"Qgen mean rel={summary['Qgen_mean_relative_error']:.6%}",
            flush=True,
        )
    print(f"long-horizon results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
