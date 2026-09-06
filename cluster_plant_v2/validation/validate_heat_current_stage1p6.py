"""Stage 1.6: freeze the cold-plate model and the validation caliber.

Scope guard: this stage touches the cold plate only. No evaporator, no
Cluster integration, no parameter fitting.

What is being frozen
--------------------
1. **Validation caliber.** The 13-node reference is the new
   ``HarmonizedColdPlateAdapter``, whose HTC correlation is anchored at
   ``COLD_PLATE_REFERENCE_MASS_FLOW_KG_S`` = 0.1428 kg/s -- the same per-plate
   design flow both ROMs use. That value is a *correlation anchor*, not the
   operating flow. ``legacy_cold_plate_reference`` is left untouched.

2. **Heat-transfer error metric.** The reference keeps its historical
   ``q_plate_to_fluid_reported`` (single-guess LMTD) for continuity, but every
   error metric here compares against ``q_plate_to_fluid_energy`` -- the heat
   actually removed from the wall states. The two differ by the reference's
   own internal closure gap (0.5-1.2 % at steady state, Stage 1.5).

3. **Formal configuration.** ``ColdPlateHeatCurrent`` stays at three zones
   ``[4, 5, 4]``. The n = 1/3/5/7/13 sweep is sensitivity evidence only and is
   reported in a separate appendix table.

4. **Time base.** External interface stays at 5 s. ``ColdPlateHeatCurrent``
   sub-divides internally (default 1 s) and reports outer-step averages, so
   both energy paths close exactly across the 5 s interface step. The plant
   and MPC time base is untouched.

Series compared
---------------
* ``existing_rom``      -- ReducedColdPlate at a native 5 s step (as deployed)
* ``existing_rom_sub1`` -- ReducedColdPlate driven with 1 s sub-steps by a
                           validation-side wrapper (numerics-free baseline;
                           ``ReducedColdPlate`` itself is not modified)
* ``heat_current``      -- ColdPlateHeatCurrent with its built-in 1 s sub-step
                           (**the deliverable**)
* ``heat_current_sub5`` -- same model with the sub-step disabled
                           (``internal_dt_s=5``) to show why it is needed

Outputs land in validation/results/heat_current_stage1p6_20260905/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cluster_plant_v2.parameters import (
    COOLANT_SPECIFIC_HEAT_J_KG_K as CP_COOL,
    NOMINAL_COOLANT_MASS_FLOW_KG_S,
)
from cluster_plant_v2.thermal.cold_plate_heat_current import ColdPlateHeatCurrent
from cluster_plant_v2.thermal.cold_plate_rom import ReducedColdPlate
from cluster_plant_v2.validation.harmonized_cold_plate_reference import (
    DEFAULT_REFERENCE_MASS_FLOW_KG_S,
    HarmonizedColdPlateAdapter,
)
from cluster_plant_v2.validation.validate_cold_plate_rom import (
    COOLANT_INLET_K,
    DEFAULT_DURATION_S,
    CaseSpec,
    battery_heat_profile,
    error_metrics,
    liters_per_minute_to_mass_flow,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "heat_current_stage1p6_20260905"

#: Frozen external interface step (plant / MPC time base).
INTERFACE_DT_S = 5.0
#: Internal Euler step used by the heat-current model and the reference.
INTERNAL_DT_S = 1.0

FLOW_SWEEP_L_MIN = (4.0, 5.0, 6.0, 10.0, 21.0, 37.0)
HEAT_KINDS = ("uniform", "forward", "reverse", "dynamic")
DIRECTIONS = (1, -1)

#: Frozen formal configuration -- three zones of 13 columns.
FORMAL_COUNTS: tuple[int, ...] = (4, 5, 4)
#: Sensitivity-only partitions; they do not change the shipped model.
SEGMENT_COUNTS: dict[int, tuple[int, ...]] = {
    1: (13,),
    3: (4, 5, 4),
    5: (2, 3, 3, 3, 2),
    7: (1, 2, 2, 3, 2, 2, 1),
    13: (1,) * 13,
}
SENSITIVITY_CASES = (
    CaseSpec("21lpm_uniform_forward", 21.0, "uniform", 1),
    CaseSpec("21lpm_forward_heat_reverse", 21.0, "forward", -1),
    CaseSpec("5lpm_uniform_forward", 5.0, "uniform", 1),
)
PRIMARY_CASE = CaseSpec("21lpm_uniform_forward", 21.0, "uniform", 1)

PLATE_BOUND_UPPER_C = 200.0
PLATE_BOUND_LOWER_C = -100.0


class SubsteppedModel:
    """Validation-side wrapper that sub-divides ``dt`` for a model.

    Lets ``ReducedColdPlate`` be driven with 1 s sub-steps without modifying
    it. Applies the same outer-step-average convention as the heat-current
    model so the two are compared on identical footing.
    """

    def __init__(self, model, internal_dt_s: float) -> None:
        self.model = model
        self.internal_dt_s = float(internal_dt_s)
        self.coolant_cp = model.coolant_cp

    def step(self, dt, inlet, mass_flow, q_battery_to_plate, flow_direction=1):
        n_steps = max(
            1, int(np.ceil(float(dt) / self.internal_dt_s - 1e-9))
        )
        sub_dt = float(dt) / n_steps
        outlet_sum = 0.0
        heat_sum = 0.0
        result = None
        for _ in range(n_steps):
            result = self.model.step(
                sub_dt, inlet, mass_flow, q_battery_to_plate, flow_direction
            )
            outlet_sum += result["coolant_outlet_temperature"]
            heat_sum += result["q_plate_to_fluid_total"]
        return {
            "plate_temperatures": np.asarray(self.model.plate_temperatures, dtype=float),
            "coolant_outlet_temperature": outlet_sum / n_steps,
            "q_plate_to_fluid_total": heat_sum / n_steps,
            "h_dynamic": result["h_dynamic"],
            "internal_steps": n_steps,
        }


def build_series() -> list[tuple[str, object, tuple[int, ...]]]:
    return [
        ("existing_rom", ReducedColdPlate(), FORMAL_COUNTS),
        ("existing_rom_sub1", SubsteppedModel(ReducedColdPlate(), INTERNAL_DT_S), FORMAL_COUNTS),
        ("heat_current", ColdPlateHeatCurrent(internal_dt_s=INTERNAL_DT_S), FORMAL_COUNTS),
        ("heat_current_sub5", ColdPlateHeatCurrent(internal_dt_s=INTERFACE_DT_S), FORMAL_COUNTS),
    ]


# --------------------------------------------------------------------------
# Discretization helpers
# --------------------------------------------------------------------------
def column_slices(counts: tuple[int, ...]) -> list[tuple[int, int]]:
    counts_array = np.asarray(counts, dtype=int)
    if counts_array.sum() != 13:
        raise ValueError(f"zone column counts must sum to 13, got {counts}")
    bounds = np.cumsum(np.concatenate(([0], counts_array)))
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(counts_array.size)]


def aggregate_heat_by_counts(node_heat: np.ndarray, counts: tuple[int, ...]) -> np.ndarray:
    nodes = np.asarray(node_heat, dtype=float).reshape(13)
    return np.array([nodes[start:stop].sum() for start, stop in column_slices(counts)])


def to_columns(zone_values: np.ndarray, counts: tuple[int, ...]) -> np.ndarray:
    """Expand n zone values onto the 13 physical columns."""
    return np.repeat(np.asarray(zone_values, dtype=float), np.asarray(counts, dtype=int))


def aggregate_to_counts(columns: np.ndarray, counts: tuple[int, ...]) -> np.ndarray:
    """Average 13 column values onto an arbitrary contiguous partition."""
    values = np.asarray(columns, dtype=float).reshape(13)
    return np.array([values[start:stop].mean() for start, stop in column_slices(counts)])


# --------------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------------
def run_reference(case: CaseSpec, times: np.ndarray, dt: float, mass_flow: float) -> list[dict]:
    plate = HarmonizedColdPlateAdapter(internal_dt_s=INTERNAL_DT_S)
    capacity_rate = mass_flow * CP_COOL
    records: list[dict] = []
    for time_s in times:
        node_heat = battery_heat_profile(case.heat_kind, float(time_s))
        result = plate.step(dt, COOLANT_INLET_K, mass_flow, node_heat, case.flow_direction)
        temperatures = np.asarray(result["plate_temperatures"], dtype=float) - 273.15
        q_energy = float(result["q_plate_to_fluid_energy"])
        q_reported = float(result["q_plate_to_fluid_reported"])
        records.append(
            {
                "T_out_C": float(result["coolant_outlet_temperature"] - 273.15),
                "T_out_energy_C": COOLANT_INLET_K - 273.15 + q_energy / capacity_rate,
                "Q_energy_W": q_energy,
                "Q_reported_W": q_reported,
                "closure_gap_W": q_energy - q_reported,
                "plate_avg_C": float(np.mean(temperatures)),
                "plate_max_C": float(np.max(temperatures)),
                # Resolution-matched hot spot: the reference's 13 nodes are
                # averaged onto [4,5,4] first, so a 3-zone model is not
                # penalised for lacking node-level resolution.
                "plate_max_z3_C": float(
                    aggregate_to_counts(temperatures, FORMAL_COUNTS).max()
                ),
                "plate_min_C": float(np.min(temperatures)),
                "axial_dT_C": float(np.max(temperatures) - np.min(temperatures)),
                "axial_dT_z3_C": float(
                    np.ptp(aggregate_to_counts(temperatures, FORMAL_COUNTS))
                ),
                "h_W_m2K": float(result["h_dynamic"]),
                "internal_steps": int(result["internal_steps"]),
                "plate_temperatures_C": temperatures,
            }
        )
    return records


def run_candidate(
    model,
    counts: tuple[int, ...],
    case: CaseSpec,
    times: np.ndarray,
    dt: float,
    mass_flow: float,
) -> list[dict]:
    counts_array = np.asarray(counts, dtype=int)
    capacity_rate = mass_flow * CP_COOL
    records: list[dict] = []
    for time_s in times:
        node_heat = battery_heat_profile(case.heat_kind, float(time_s))
        zone_heat = aggregate_heat_by_counts(node_heat, counts)
        result = model.step(dt, COOLANT_INLET_K, mass_flow, zone_heat, case.flow_direction)
        temperatures = np.asarray(result["plate_temperatures"], dtype=float) - 273.15
        q_energy = float(result["q_plate_to_fluid_total"])
        records.append(
            {
                "T_out_C": float(result["coolant_outlet_temperature"] - 273.15),
                "T_out_energy_C": COOLANT_INLET_K - 273.15 + q_energy / capacity_rate,
                "Q_energy_W": q_energy,
                "Q_reported_W": q_energy,
                "closure_gap_W": float(
                    capacity_rate
                    * (result["coolant_outlet_temperature"] - COOLANT_INLET_K)
                    - q_energy
                ),
                "plate_avg_C": float(
                    np.sum(temperatures * counts_array) / counts_array.sum()
                ),
                "plate_max_C": float(np.max(temperatures)),
                "plate_max_z3_C": float(
                    aggregate_to_counts(
                        to_columns(temperatures, counts), FORMAL_COUNTS
                    ).max()
                ),
                "plate_min_C": float(np.min(temperatures)),
                "axial_dT_C": float(np.max(temperatures) - np.min(temperatures)),
                "axial_dT_z3_C": float(
                    np.ptp(
                        aggregate_to_counts(
                            to_columns(temperatures, counts), FORMAL_COUNTS
                        )
                    )
                ),
                "h_W_m2K": float(result["h_dynamic"]),
                "internal_steps": int(result.get("internal_steps", 1)),
                "plate_temperatures_C": temperatures,
            }
        )
    return records


def series(records: list[dict], key: str) -> np.ndarray:
    return np.array([record[key] for record in records], dtype=float)


def is_bounded(records: list[dict]) -> bool:
    stack = np.concatenate([np.ravel(r["plate_temperatures_C"]) for r in records])
    outlet = series(records, "T_out_C")
    heat = series(records, "Q_energy_W")
    return bool(
        np.all(np.isfinite(stack))
        and np.all(np.isfinite(outlet))
        and np.all(np.isfinite(heat))
        and stack.max() < PLATE_BOUND_UPPER_C
        and stack.min() > PLATE_BOUND_LOWER_C
        and np.max(np.abs(outlet)) < PLATE_BOUND_UPPER_C
    )


METRICS: tuple[tuple[str, str, str], ...] = (
    ("plate_avg", "plate_avg_C", "degC"),
    ("plate_max", "plate_max_C", "degC"),
    ("plate_max_z3", "plate_max_z3_C", "degC"),
    ("axial_dT_z3", "axial_dT_z3_C", "degC"),
    ("T_out", "T_out_C", "degC"),
    ("T_out_energy", "T_out_energy_C", "degC"),
    ("Q_energy", "Q_energy_W", "W"),
)


def compare(
    case: CaseSpec,
    model_name: str,
    counts: tuple[int, ...],
    reference: list[dict],
    candidate: list[dict],
    study: str,
) -> dict:
    row = {
        "study": study,
        "case_id": case.case_id,
        "flow_L_min": case.flow_L_min,
        "heat_kind": case.heat_kind,
        "flow_direction": case.direction_name,
        "model": model_name,
        "n_zones": len(counts),
        "counts": "-".join(str(count) for count in counts),
        "stable": is_bounded(reference) and is_bounded(candidate),
        "Ref_h_W_m2K": series(reference, "h_W_m2K")[0],
        "model_h_W_m2K": series(candidate, "h_W_m2K")[0],
        "internal_steps": int(series(candidate, "internal_steps")[0]),
        "Ref_final_plate_avg_C": series(reference, "plate_avg_C")[-1],
        "Ref_final_plate_max_C": series(reference, "plate_max_C")[-1],
        "Ref_final_plate_max_z3_C": series(reference, "plate_max_z3_C")[-1],
        "Ref_final_axial_dT_z3_C": series(reference, "axial_dT_z3_C")[-1],
        "Ref_final_T_out_C": series(reference, "T_out_C")[-1],
        "Ref_final_Q_energy_W": series(reference, "Q_energy_W")[-1],
        "Ref_final_Q_reported_W": series(reference, "Q_reported_W")[-1],
        "Ref_closure_gap_max_abs_W": float(np.max(np.abs(series(reference, "closure_gap_W")))),
        "model_closure_max_abs_W": float(np.max(np.abs(series(candidate, "closure_gap_W")))),
    }
    row["h_ratio_model_over_ref"] = row["model_h_W_m2K"] / row["Ref_h_W_m2K"]
    for label, key, _unit in METRICS:
        stats = error_metrics(series(reference, key), series(candidate, key))
        for name, value in stats.items():
            row[f"{label}_{name}"] = value
    reference_q = series(reference, "Q_energy_W")
    row["Q_energy_rmse_rel_pct"] = (
        row["Q_energy_rmse"] / max(float(np.mean(np.abs(reference_q))), 1.0) * 100.0
    )
    return row


# --------------------------------------------------------------------------
# Studies
# --------------------------------------------------------------------------
def build_sweep(times: np.ndarray, dt: float) -> pd.DataFrame:
    rows = []
    for flow_L_min in FLOW_SWEEP_L_MIN:
        mass_flow = liters_per_minute_to_mass_flow(flow_L_min)
        for heat_kind in HEAT_KINDS:
            for direction in DIRECTIONS:
                case = CaseSpec(
                    f"{flow_L_min:g}lpm_{heat_kind}_{'forward' if direction == 1 else 'reverse'}",
                    flow_L_min,
                    heat_kind,
                    direction,
                )
                reference = run_reference(case, times, dt, mass_flow)
                for model_name, model, counts in build_series():
                    rows.append(
                        compare(
                            case,
                            model_name,
                            counts,
                            reference,
                            run_candidate(model, counts, case, times, dt, mass_flow),
                            "sweep",
                        )
                    )
    return pd.DataFrame(rows)


def build_sensitivity(times: np.ndarray, dt: float) -> pd.DataFrame:
    """n = 1/3/5/7/13 sensitivity evidence; the shipped model stays at [4,5,4]."""
    rows = []
    for case in SENSITIVITY_CASES:
        mass_flow = liters_per_minute_to_mass_flow(case.flow_L_min)
        reference = run_reference(case, times, dt, mass_flow)
        for n_zones, counts in SEGMENT_COUNTS.items():
            model = ColdPlateHeatCurrent(
                zone_column_counts=np.array(counts, dtype=int),
                internal_dt_s=INTERNAL_DT_S,
            )
            rows.append(
                compare(
                    case,
                    "heat_current",
                    counts,
                    reference,
                    run_candidate(model, counts, case, times, dt, mass_flow),
                    "sensitivity_n",
                )
            )
    return pd.DataFrame(rows)


def build_primary_timeseries(times: np.ndarray, dt: float) -> pd.DataFrame:
    case = PRIMARY_CASE
    mass_flow = liters_per_minute_to_mass_flow(case.flow_L_min)
    reference = run_reference(case, times, dt, mass_flow)
    runs = {"Ref": reference}
    for model_name, model, counts in build_series():
        runs[model_name] = run_candidate(model, counts, case, times, dt, mass_flow)
    frame = pd.DataFrame({"time_s": times + dt})
    for label, records in runs.items():
        frame[f"{label}_T_out_C"] = series(records, "T_out_C")
        frame[f"{label}_Q_energy_W"] = series(records, "Q_energy_W")
        frame[f"{label}_plate_max_C"] = series(records, "plate_max_C")
        frame[f"{label}_plate_avg_C"] = series(records, "plate_avg_C")
    frame["Ref_Q_reported_W"] = series(reference, "Q_reported_W")
    return frame


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
SERIES_STYLE = {
    "existing_rom": ("#185FA5", "o", "Existing ROM (native 5 s)"),
    "existing_rom_sub1": ("#85B7EB", "s", "Existing ROM (1 s sub-step)"),
    "heat_current": ("#0F6E56", "^", "Heat-Current (1 s sub-step)"),
    "heat_current_sub5": ("#BA7517", "v", "Heat-Current (no sub-step)"),
}


def _style(ax) -> None:
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, frameon=False)


def plot_metric_vs_flow(sweep: pd.DataFrame, column: str, ylabel: str, path: Path) -> None:
    stable = sweep[sweep["stable"]]
    subset = stable[
        (stable["heat_kind"] == "uniform") & (stable["flow_direction"] == "forward")
    ]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for model_name, (colour, marker, label) in SERIES_STYLE.items():
        group = subset[subset["model"] == model_name].sort_values("flow_L_min")
        if group.empty:
            continue
        ax.plot(
            group["flow_L_min"],
            group[column],
            marker=marker,
            color=colour,
            label=label,
        )
    ax.set_xscale("log")
    if column.endswith("rel_pct"):
        ax.set_yscale("log")
    ax.set_xlabel("coolant flow [L/min]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Uniform load, forward flow, dt = {INTERFACE_DT_S:g} s")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_plate_metrics(sweep: pd.DataFrame, path: Path) -> None:
    stable = sweep[sweep["stable"]]
    subset = stable[
        (stable["heat_kind"] == "uniform") & (stable["flow_direction"] == "forward")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for ax, column, ylabel in (
        (axes[0], "plate_max_z3_rmse", "plate_max RMSE [degC] (matched [4,5,4])"),
        (axes[1], "axial_dT_z3_rmse", "axial dT RMSE [degC] (matched [4,5,4])"),
    ):
        for model_name, (colour, marker, label) in SERIES_STYLE.items():
            group = subset[subset["model"] == model_name].sort_values("flow_L_min")
            if group.empty:
                continue
            ax.plot(group["flow_L_min"], group[column], marker=marker, color=colour, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("coolant flow [L/min]")
        ax.set_ylabel(ylabel)
        _style(ax)
    fig.suptitle(f"Plate temperature fidelity, dt = {INTERFACE_DT_S:g} s")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_primary_timeseries(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.6), sharex=True)
    for ax, suffix, ylabel in (
        (axes[0], "_T_out_C", "coolant outlet [degC]"),
        (axes[1], "_plate_max_C", "plate max [degC]"),
        (axes[2], "_Q_energy_W", "Q plate-to-fluid, energy [W]"),
    ):
        ax.plot(
            frame["time_s"],
            frame[f"Ref{suffix}"],
            "k-",
            lw=2.2,
            label="13-node harmonized reference",
        )
        for model_name, (colour, _marker, label) in SERIES_STYLE.items():
            ax.plot(
                frame["time_s"],
                frame[f"{model_name}{suffix}"],
                "--",
                color=colour,
                label=label,
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, frameon=False)
    axes[2].set_xlabel("time [s]")
    axes[0].set_title(f"{PRIMARY_CASE.case_id}, dt = {INTERFACE_DT_S:g} s")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sensitivity(sensitivity: pd.DataFrame, path: Path) -> None:
    stable = sensitivity[sensitivity["stable"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for ax, column, ylabel in (
        (axes[0], "Q_energy_rmse_rel_pct", "Q_energy RMSE [%]"),
        (axes[1], "plate_max_rmse", "plate_max RMSE [degC]"),
    ):
        for case_id, group in stable.groupby("case_id", sort=False):
            group = group.sort_values("n_zones")
            ax.plot(group["n_zones"], group[column], marker="o", label=case_id)
        ax.set_xscale("log")
        ax.set_xticks(sorted(SEGMENT_COUNTS))
        ax.set_xticklabels([str(n) for n in sorted(SEGMENT_COUNTS)])
        ax.set_xlabel("segments n (sensitivity only)")
        ax.set_ylabel(ylabel)
        _style(ax)
    fig.suptitle("Sensitivity: segmentation count (shipped model stays at n = 3)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> None:
    global INTERNAL_DT_S
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=INTERFACE_DT_S)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--internal-dt", type=float, default=INTERNAL_DT_S)
    args = parser.parse_args()

    INTERNAL_DT_S = float(args.internal_dt)

    dt = float(args.dt)
    times = np.arange(0.0, float(args.duration), dt)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"interface dt = {dt:g} s, internal dt = {INTERNAL_DT_S:g} s", flush=True)
    sweep = build_sweep(times, dt)
    sensitivity = build_sensitivity(times, dt)
    primary = build_primary_timeseries(times, dt)

    sweep.to_csv(RESULTS_DIR / "stage1p6_sweep.csv", index=False)
    sensitivity.to_csv(RESULTS_DIR / "stage1p6_sensitivity_n.csv", index=False)
    primary.to_csv(RESULTS_DIR / "stage1p6_primary_timeseries.csv", index=False)
    plot_metric_vs_flow(
        sweep, "Q_energy_rmse_rel_pct", "Q_energy RMSE [%]",
        RESULTS_DIR / "fig_stage1p6_q_energy_error.png",
    )
    plot_plate_metrics(sweep, RESULTS_DIR / "fig_stage1p6_plate_metrics.png")
    plot_primary_timeseries(primary, RESULTS_DIR / "fig_stage1p6_primary_timeseries.png")
    plot_sensitivity(sensitivity, RESULTS_DIR / "fig_stage1p6_sensitivity_n.png")

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 80)

    print("\n=== Divergent (explicit-Euler unstable) cases ===")
    bad = sweep[~sweep["stable"]]
    if bad.empty:
        print("none")
    else:
        print(
            bad.groupby(["flow_L_min", "model"]).size().rename("divergent_cases").to_string()
        )

    stable = sweep[sweep["stable"]]
    columns = [
        "Q_energy_rmse_rel_pct",
        "Q_energy_rmse",
        "plate_avg_rmse",
        "plate_max_rmse",
        "plate_max_z3_rmse",
        "axial_dT_z3_rmse",
        "T_out_rmse",
        "T_out_energy_rmse",
        "model_closure_max_abs_W",
    ]
    print("\n=== Sweep: mean over all stable cases (48 per series) ===")
    print(stable.groupby("model")[columns].mean().round(4).to_string())

    # Some series diverge at 37 L/min, so a plain mean compares different case
    # sets. Restrict to the cases every series survives for a like-for-like mean.
    n_series = sweep["model"].nunique()
    per_case = stable.groupby("case_id")["model"].nunique()
    common_ids = per_case[per_case == n_series].index
    common = stable[stable["case_id"].isin(common_ids)]
    print(
        f"\n=== Sweep: mean over the COMMON stable subset "
        f"({len(common_ids)} cases, every series stable) ==="
    )
    print(common.groupby("model")[columns].mean().round(4).to_string())

    print("\n=== plate_max RMSE [degC] by flow (common subset) ===")
    print(
        common.pivot_table(
            index="flow_L_min", columns="model", values="plate_max_rmse"
        ).round(4).to_string()
    )
    print("\n=== plate_max RMSE [degC], resolution-matched to [4,5,4] (common subset) ===")
    print(
        common.pivot_table(
            index="flow_L_min", columns="model", values="plate_max_z3_rmse"
        ).round(4).to_string()
    )
    print("\n=== axial dT RMSE [degC], matched [4,5,4] (common subset) ===")
    print(
        common.pivot_table(
            index="flow_L_min", columns="model", values="axial_dT_z3_rmse"
        ).round(4).to_string()
    )
    print("\n=== T_out RMSE [degC] / T_out_energy RMSE [degC] (common subset) ===")
    print(
        common.pivot_table(
            index="flow_L_min", columns="model", values="T_out_rmse"
        ).round(4).to_string()
    )
    print(
        common.pivot_table(
            index="flow_L_min", columns="model", values="T_out_energy_rmse"
        ).round(4).to_string()
    )
    print("\n=== plate_avg RMSE [degC] (common subset) ===")
    print(
        common.pivot_table(
            index="flow_L_min", columns="model", values="plate_avg_rmse"
        ).round(4).to_string()
    )

    print("\n=== Sweep: worst case per series ===")
    print(
        stable.loc[stable.groupby("model")["Q_energy_rmse_rel_pct"].idxmax()][
            ["model", "case_id", "Q_energy_rmse_rel_pct", "plate_max_rmse", "T_out_rmse"]
        ].round(4).to_string(index=False)
    )

    print("\n=== Sweep: Q_energy RMSE [%] by flow ===")
    print(
        stable.pivot_table(
            index="flow_L_min", columns="model", values="Q_energy_rmse_rel_pct"
        ).round(3).to_string()
    )

    print("\n=== Reference internal closure gap (Q_energy - Q_reported) ===")
    print(
        sweep.groupby("flow_L_min")["Ref_closure_gap_max_abs_W"].max().round(3).to_string()
    )

    print("\n=== Sensitivity: n segments (shipped model stays n = 3) ===")
    stable_sens = sensitivity[sensitivity["stable"]]
    print(
        stable_sens.pivot_table(
            index="case_id", columns="n_zones", values="Q_energy_rmse_rel_pct"
        ).round(3).to_string()
    )
    print(f"\nResults written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
