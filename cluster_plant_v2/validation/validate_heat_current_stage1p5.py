"""Stage 1.5 validation: Heat-Current ROM vs Existing ROM, 13-node reference.

Validation only. This script changes no model structure, calibrates no
parameter and touches no Cluster code. It drives

  * the frozen 13-node legacy path (DetailedColdPlateAdapter)  -> reference
  * the validated three-zone ROM (ReducedColdPlate)            -> incumbent
  * the new heat-current ROM (ColdPlateHeatCurrent)            -> candidate

with identical coolant inlet conditions, identical battery heat profiles and
identical frozen parameters, then reports

  1. a flow x heat-load x flow-direction dynamic sweep,
  2. an n = 1, 3, 5, 7, 13 segmentation convergence study of the heat-current
     ROM against the same 13-node reference,
  3. a dt-refinement probe separating explicit-Euler instability from model
     error.

Two口径 (caliber) facts dominate the numbers and are handled explicitly:

* The frozen 13-node reference normalises its HTC with
  ``NOMINAL_COOLANT_MASS_FLOW_KG_S`` (1.2 kg/s, a *cluster-level* nominal)
  while both ROMs use ``COLD_PLATE_REFERENCE_MASS_FLOW_KG_S`` (0.1428 kg/s,
  the *per-plate* design flow). As shipped the two sides therefore differ by
  a flow-independent factor (1.2 / 0.1428) ** 0.8 = 5.49 in h. Stage 1.5 must
  not recalibrate anything, so the whole study is repeated under three
  calibration variants; each variant only swaps module-level constants in
  memory, no file on disk is modified.

* All three models advance the wall states with explicit Euler. At dt = 5 s
  and 37 L/min the stability limit 2*C/(h*A) is violated, so the sweep runs at
  dt = 1.0 s where every case is stable under every variant; the dt probe
  quantifies the boundary.

Outputs land in validation/results/heat_current_stage1p5_20260905/.
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cluster_plant_v2 import validation
from cluster_plant_v2.parameters import (
    COLD_PLATE_REFERENCE_MASS_FLOW_KG_S,
    COOLANT_SPECIFIC_HEAT_J_KG_K as cp_cool,
    NOMINAL_COOLANT_MASS_FLOW_KG_S,
)
from cluster_plant_v2.thermal import cold_plate_heat_current, cold_plate_rom
from cluster_plant_v2.thermal.cold_plate_heat_current import ColdPlateHeatCurrent
from cluster_plant_v2.thermal.cold_plate_rom import ReducedColdPlate
from cluster_plant_v2.validation import legacy_cold_plate_reference, validate_cold_plate_rom
from cluster_plant_v2.validation.validate_cold_plate_rom import (
    COOLANT_INLET_K,
    DEFAULT_DURATION_S,
    CaseSpec,
    DetailedColdPlateAdapter,
    battery_heat_profile,
    error_metrics,
    liters_per_minute_to_mass_flow,
)

# --------------------------------------------------------------------------
# Calibration variants: (rom reference flow, legacy reference flow)
# --------------------------------------------------------------------------
CALIBRATION_VARIANTS: dict[str, tuple[float, float]] = {
    "as_is_rom0.1428_ref1.2": (
        COLD_PLATE_REFERENCE_MASS_FLOW_KG_S,
        NOMINAL_COOLANT_MASS_FLOW_KG_S,
    ),
    "harmonized_rom0.1428": (
        COLD_PLATE_REFERENCE_MASS_FLOW_KG_S,
        COLD_PLATE_REFERENCE_MASS_FLOW_KG_S,
    ),
    "harmonized_legacy1.2": (
        NOMINAL_COOLANT_MASS_FLOW_KG_S,
        NOMINAL_COOLANT_MASS_FLOW_KG_S,
    ),
}
# 0.1428 kg/s is the per-plate design flow, so h_nominal = 2000 W/m2K is
# anchored there; 1.2 kg/s is a cluster-level number misapplied per plate.
PRIMARY_CALIBRATION = "harmonized_rom0.1428"
BASELINE_CALIBRATION = "as_is_rom0.1428_ref1.2"


@contextlib.contextmanager
def calibration(rom_reference_flow: float, legacy_reference_flow: float):
    """Temporarily align the HTC normalisation flows used by both sides."""
    # cold_plate_rom imports the constant under the alias ``m_dot_nominal``,
    # cold_plate_heat_current keeps the original name.
    rom_attributes = (
        (cold_plate_rom, "m_dot_nominal"),
        (cold_plate_heat_current, "COLD_PLATE_REFERENCE_MASS_FLOW_KG_S"),
    )
    legacy_attributes = (
        (legacy_cold_plate_reference, "m_dot_nominal"),
        (validate_cold_plate_rom, "m_dot_nominal"),
    )
    saved = [
        (module, name, getattr(module, name))
        for module, name in rom_attributes + legacy_attributes
    ]
    for module, name in rom_attributes:
        setattr(module, name, rom_reference_flow)
    for module, name in legacy_attributes:
        setattr(module, name, legacy_reference_flow)
    try:
        yield
    finally:
        for module, name, value in saved:
            setattr(module, name, value)


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "heat_current_stage1p5_20260905"

# dt = 1.0 s keeps every case inside the explicit-Euler stability limit under
# every calibration variant; the existing cold-plate validation uses 5.0 s.
SWEEP_DT_S = 1.0
FLOW_SWEEP_L_MIN = (4.0, 5.0, 6.0, 10.0, 21.0, 37.0)
HEAT_KINDS = ("uniform", "forward", "reverse", "dynamic")
DIRECTIONS = (1, -1)

# Contiguous 13-column partitions used by the segmentation study.
SEGMENT_COUNTS: dict[int, tuple[int, ...]] = {
    1: (13,),
    3: (4, 5, 4),
    5: (2, 3, 3, 3, 2),
    7: (1, 2, 2, 3, 2, 2, 1),
    13: (1,) * 13,
}
INCUMBENT_COUNTS = SEGMENT_COUNTS[3]

SEGMENTATION_CASES = (
    CaseSpec("21lpm_forward_heat_forward", 21.0, "forward", 1),
    CaseSpec("21lpm_uniform_forward", 21.0, "uniform", 1),
    CaseSpec("5lpm_uniform_forward", 5.0, "uniform", 1),
)
PRIMARY_CASE = SEGMENTATION_CASES[0]

STABILITY_FLOWS_L_MIN = (21.0, 37.0)
STABILITY_DT_S = (5.0, 2.0, 1.0, 0.2)
PLATE_BOUND_UPPER_C = 200.0
PLATE_BOUND_LOWER_C = -100.0


# --------------------------------------------------------------------------
# Helpers
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


def make_model(model_name: str, counts: tuple[int, ...]):
    if model_name == "existing_rom":
        if counts != INCUMBENT_COUNTS:
            raise ValueError("the validated ROM is hard-wired to [4, 5, 4]")
        return ReducedColdPlate()
    if model_name == "heat_current":
        return ColdPlateHeatCurrent(zone_column_counts=np.array(counts, dtype=int))
    raise ValueError(f"unknown model {model_name}")


def run_reference(case: CaseSpec, times: np.ndarray, dt: float, mass_flow: float) -> list[dict]:
    plate = DetailedColdPlateAdapter()
    records: list[dict] = []
    for time_s in times:
        node_heat = battery_heat_profile(case.heat_kind, float(time_s))
        result = plate.step(dt, COOLANT_INLET_K, mass_flow, node_heat, case.flow_direction)
        temperatures = np.asarray(result["plate_temperatures"], dtype=float) - 273.15
        records.append(
            {
                "T_out_C": float(result["coolant_outlet_temperature"] - 273.15),
                "Q_total_W": float(result["q_plate_to_fluid_total"]),
                "plate_avg_C": float(np.mean(temperatures)),
                "plate_max_C": float(np.max(temperatures)),
                "closure_W": float(result["energy_closure_error_W"]),
                "h_W_m2K": float(result["h_dynamic"]),
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
    records: list[dict] = []
    for time_s in times:
        node_heat = battery_heat_profile(case.heat_kind, float(time_s))
        zone_heat = aggregate_heat_by_counts(node_heat, counts)
        result = model.step(dt, COOLANT_INLET_K, mass_flow, zone_heat, case.flow_direction)
        temperatures = np.asarray(result["plate_temperatures"], dtype=float) - 273.15
        coolant_gain = mass_flow * cp_cool * (
            result["coolant_outlet_temperature"] - COOLANT_INLET_K
        )
        records.append(
            {
                "T_out_C": float(result["coolant_outlet_temperature"] - 273.15),
                "Q_total_W": float(result["q_plate_to_fluid_total"]),
                "plate_avg_C": float(
                    np.sum(temperatures * counts_array) / counts_array.sum()
                ),
                "plate_max_C": float(np.max(temperatures)),
                "closure_W": float(coolant_gain - result["q_plate_to_fluid_total"]),
                "h_W_m2K": float(result["h_dynamic"]),
                "plate_temperatures_C": temperatures,
            }
        )
    return records


def series(records: list[dict], key: str) -> np.ndarray:
    return np.array([record[key] for record in records], dtype=float)


def is_bounded(records: list[dict]) -> bool:
    """Explicit-Euler divergence detector (values blow up to ~1e16 C)."""
    stack = np.concatenate([np.ravel(r["plate_temperatures_C"]) for r in records])
    outlet = series(records, "T_out_C")
    heat = series(records, "Q_total_W")
    return bool(
        np.all(np.isfinite(stack))
        and np.all(np.isfinite(outlet))
        and np.all(np.isfinite(heat))
        and stack.max() < PLATE_BOUND_UPPER_C
        and stack.min() > PLATE_BOUND_LOWER_C
        and np.max(np.abs(outlet)) < PLATE_BOUND_UPPER_C
    )


def compare(
    case: CaseSpec,
    model_name: str,
    counts: tuple[int, ...],
    reference: list[dict],
    candidate: list[dict],
    calibration_name: str,
    dt: float,
) -> dict:
    row = {
        "calibration": calibration_name,
        "dt_s": dt,
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
        "Ref_final_Tout_C": series(reference, "T_out_C")[-1],
        "Ref_final_Q_total_W": series(reference, "Q_total_W")[-1],
        "Ref_final_plate_avg_C": series(reference, "plate_avg_C")[-1],
        "Ref_final_plate_max_C": series(reference, "plate_max_C")[-1],
        "Ref_closure_max_abs_W": float(np.max(np.abs(series(reference, "closure_W")))),
        "model_closure_max_abs_W": float(np.max(np.abs(series(candidate, "closure_W")))),
    }
    row["h_ratio_model_over_ref"] = row["model_h_W_m2K"] / row["Ref_h_W_m2K"]
    for metric, key in (
        ("Tout", "T_out_C"),
        ("plate_avg", "plate_avg_C"),
        ("plate_max", "plate_max_C"),
        ("Q_total", "Q_total_W"),
    ):
        stats = error_metrics(series(reference, key), series(candidate, key))
        for name, value in stats.items():
            row[f"{metric}_{name}"] = value
    reference_q = series(reference, "Q_total_W")
    row["Q_total_rmse_rel_pct"] = (
        row["Q_total_rmse"] / max(float(np.mean(np.abs(reference_q))), 1.0) * 100.0
    )
    return row


# --------------------------------------------------------------------------
# Study builders
# --------------------------------------------------------------------------
def build_sweep(calibration_name: str, times: np.ndarray, dt: float) -> pd.DataFrame:
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
                for model_name in ("existing_rom", "heat_current"):
                    counts = INCUMBENT_COUNTS
                    rows.append(
                        compare(
                            case,
                            model_name,
                            counts,
                            reference,
                            run_candidate(
                                make_model(model_name, counts), counts, case, times, dt, mass_flow
                            ),
                            calibration_name,
                            dt,
                        )
                    )
    return pd.DataFrame(rows)


def build_segmentation(calibration_name: str, times: np.ndarray, dt: float) -> pd.DataFrame:
    rows = []
    for case in SEGMENTATION_CASES:
        mass_flow = liters_per_minute_to_mass_flow(case.flow_L_min)
        reference = run_reference(case, times, dt, mass_flow)
        for n_zones, counts in SEGMENT_COUNTS.items():
            rows.append(
                compare(
                    case,
                    "heat_current",
                    counts,
                    reference,
                    run_candidate(
                        make_model("heat_current", counts), counts, case, times, dt, mass_flow
                    ),
                    calibration_name,
                    dt,
                )
            )
        rows.append(
            compare(
                case,
                "existing_rom",
                INCUMBENT_COUNTS,
                reference,
                run_candidate(
                    ReducedColdPlate(), INCUMBENT_COUNTS, case, times, dt, mass_flow
                ),
                calibration_name,
                dt,
            )
        )
    return pd.DataFrame(rows)


def build_stability_probe() -> pd.DataFrame:
    """Separate explicit-Euler instability from model error."""
    rows = []
    for calibration_name, (rom_flow, legacy_flow) in CALIBRATION_VARIANTS.items():
        with calibration(rom_flow, legacy_flow):
            for flow_L_min in STABILITY_FLOWS_L_MIN:
                mass_flow = liters_per_minute_to_mass_flow(flow_L_min)
                for dt in STABILITY_DT_S:
                    times = np.arange(0.0, DEFAULT_DURATION_S, dt)
                    case = CaseSpec(
                        f"{flow_L_min:g}lpm_uniform_forward", flow_L_min, "uniform", 1
                    )
                    reference = run_reference(case, times, dt, mass_flow)
                    for model_name in ("existing_rom", "heat_current"):
                        counts = INCUMBENT_COUNTS
                        rows.append(
                            compare(
                                case,
                                model_name,
                                counts,
                                reference,
                                run_candidate(
                                    make_model(model_name, counts),
                                    counts,
                                    case,
                                    times,
                                    dt,
                                    mass_flow,
                                ),
                                calibration_name,
                                dt,
                            )
                        )
    return pd.DataFrame(rows)


def build_primary_timeseries(times: np.ndarray, dt: float) -> pd.DataFrame:
    case = PRIMARY_CASE
    mass_flow = liters_per_minute_to_mass_flow(case.flow_L_min)
    reference = run_reference(case, times, dt, mass_flow)
    runs = {
        "Ref": reference,
        "ExistingROM_n3": run_candidate(
            ReducedColdPlate(), INCUMBENT_COUNTS, case, times, dt, mass_flow
        ),
        "HeatCurrent_n3": run_candidate(
            make_model("heat_current", SEGMENT_COUNTS[3]),
            SEGMENT_COUNTS[3],
            case,
            times,
            dt,
            mass_flow,
        ),
        "HeatCurrent_n13": run_candidate(
            make_model("heat_current", SEGMENT_COUNTS[13]),
            SEGMENT_COUNTS[13],
            case,
            times,
            dt,
            mass_flow,
        ),
    }
    frame = pd.DataFrame({"time_s": times + dt})
    for label, records in runs.items():
        frame[f"{label}_Tout_C"] = series(records, "T_out_C")
        frame[f"{label}_Q_total_W"] = series(records, "Q_total_W")
        frame[f"{label}_plate_max_C"] = series(records, "plate_max_C")
    return frame


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _style_axis(ax) -> None:
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, frameon=False)


def plot_segmentation(segmentation: pd.DataFrame, path: Path) -> None:
    stable = segmentation[segmentation["stable"]]
    variants = [BASELINE_CALIBRATION, PRIMARY_CALIBRATION]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=False)
    for ax, variant in zip(axes, variants):
        subset = stable[stable["calibration"] == variant]
        for case_id, group in subset.groupby("case_id", sort=False):
            group = group.sort_values("n_zones")
            ax.plot(
                group["n_zones"],
                group["Q_total_rmse_rel_pct"],
                marker="o",
                label=case_id,
            )
        ax.set_xscale("log")
        ax.set_xticks(sorted(SEGMENT_COUNTS))
        ax.set_xticklabels([str(n) for n in sorted(SEGMENT_COUNTS)])
        ax.set_xlabel("segments n")
        ax.set_ylabel("Q_total RMSE vs 13-node reference [%]")
        ax.set_title(variant)
        _style_axis(ax)
    fig.suptitle("Heat-current ROM: segmentation convergence")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_flow_sweep(sweep: pd.DataFrame, path: Path) -> None:
    stable = sweep[sweep["stable"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for ax, variant in zip(axes, [BASELINE_CALIBRATION, PRIMARY_CALIBRATION]):
        subset = stable[
            (stable["calibration"] == variant)
            & (stable["heat_kind"] == "uniform")
            & (stable["flow_direction"] == "forward")
        ]
        for model, group in subset.groupby("model", sort=False):
            group = group.sort_values("flow_L_min")
            ax.plot(
                group["flow_L_min"],
                group["Q_total_rmse_rel_pct"],
                marker="s",
                label=model,
            )
        ax.set_xscale("log")
        ax.set_xlabel("coolant flow [L/min]")
        ax.set_ylabel("Q_total RMSE [%]")
        ax.set_title(f"{variant}\nuniform load, forward flow")
        _style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_primary_timeseries(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.0), sharex=True)
    pairs = (
        (axes[0], "_Tout_C", "coolant outlet [degC]"),
        (axes[1], "_Q_total_W", "Q plate-to-fluid [W]"),
    )
    styles = {
        "Ref": ("k-", "13-node reference"),
        "ExistingROM_n3": ("--", "Existing ROM n=3"),
        "HeatCurrent_n3": ("-.", "Heat-current n=3"),
        "HeatCurrent_n13": (":", "Heat-current n=13"),
    }
    for ax, suffix, ylabel in pairs:
        for label, (style, name) in styles.items():
            ax.plot(frame["time_s"], frame[f"{label}{suffix}"], style, label=name)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    axes[1].set_xlabel("time [s]")
    axes[0].set_title(f"{PRIMARY_CASE.case_id}  ({PRIMARY_CALIBRATION})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stability_probe(probe: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=False)
    for ax, variant in zip(axes, [BASELINE_CALIBRATION, PRIMARY_CALIBRATION]):
        subset = probe[probe["calibration"] == variant]
        for (model, flow), group in subset.groupby(["model", "flow_L_min"], sort=False):
            group = group.sort_values("dt_s", ascending=False)
            ax.plot(
                group["dt_s"],
                group["Q_total_rmse_rel_pct"],
                marker="o",
                label=f"{model} @ {flow:g} L/min",
            )
        ax.set_xscale("log")
        ax.set_xlabel("dt [s]")
        ax.set_ylabel("Q_total RMSE [%]")
        ax.set_title(variant)
        _style_axis(ax)
    fig.suptitle("dt refinement: explicit-Euler stability vs model error")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=SWEEP_DT_S)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S)
    args = parser.parse_args()

    dt = float(args.dt)
    times = np.arange(0.0, float(args.duration), dt)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sweeps, segmentations = [], []
    for calibration_name, (rom_flow, legacy_flow) in CALIBRATION_VARIANTS.items():
        with calibration(rom_flow, legacy_flow):
            print(f"[sweep] {calibration_name}", flush=True)
            sweeps.append(build_sweep(calibration_name, times, dt))
            segmentations.append(build_segmentation(calibration_name, times, dt))
            if calibration_name == PRIMARY_CALIBRATION:
                primary = build_primary_timeseries(times, dt)
    sweep = pd.concat(sweeps, ignore_index=True)
    segmentation = pd.concat(segmentations, ignore_index=True)

    print("[probe] dt refinement", flush=True)
    probe = build_stability_probe()

    sweep.to_csv(RESULTS_DIR / "stage1p5_flow_heat_direction_sweep.csv", index=False)
    segmentation.to_csv(RESULTS_DIR / "stage1p5_segmentation_convergence.csv", index=False)
    probe.to_csv(RESULTS_DIR / "stage1p5_dt_stability_probe.csv", index=False)
    primary.to_csv(RESULTS_DIR / "stage1p5_primary_timeseries.csv", index=False)
    plot_segmentation(segmentation, RESULTS_DIR / "fig_stage1p5_segmentation_convergence.png")
    plot_flow_sweep(sweep, RESULTS_DIR / "fig_stage1p5_flow_sweep_q_error.png")
    plot_primary_timeseries(primary, RESULTS_DIR / "fig_stage1p5_primary_timeseries.png")
    plot_stability_probe(probe, RESULTS_DIR / "fig_stage1p5_dt_stability_probe.png")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)

    stable_sweep = sweep[sweep["stable"]]
    print("\n=== Unstable (explicit-Euler divergent) cases ===")
    bad = sweep[~sweep["stable"]]
    if bad.empty:
        print("none")
    else:
        print(
            bad.groupby(["calibration", "flow_L_min", "model"])
            .size()
            .rename("unstable_cases")
            .to_string()
        )

    print("\n=== Sweep: mean over stable cases ===")
    print(
        stable_sweep.groupby(["calibration", "model"])[
            ["Q_total_rmse_rel_pct", "Tout_rmse", "plate_max_rmse", "h_ratio_model_over_ref"]
        ]
        .mean()
        .round(4)
    )

    print("\n=== Sweep: worst case per (calibration, model) ===")
    print(
        stable_sweep.loc[
            stable_sweep.groupby(["calibration", "model"])["Q_total_rmse_rel_pct"].idxmax()
        ][
            [
                "calibration",
                "model",
                "case_id",
                "Q_total_rmse_rel_pct",
                "Tout_rmse",
                "plate_max_max_abs_error",
            ]
        ].round(4)
    )

    print("\n=== Segmentation: Q_total RMSE [%] (stable only) ===")
    stable_seg = segmentation[segmentation["stable"]]
    print(
        stable_seg.pivot_table(
            index=["calibration", "case_id"], columns="n_zones", values="Q_total_rmse_rel_pct"
        ).round(3)
    )

    print("\n=== Segmentation: Tout RMSE [degC] (stable only) ===")
    print(
        stable_seg.pivot_table(
            index=["calibration", "case_id"], columns="n_zones", values="Tout_rmse"
        ).round(4)
    )

    print("\n=== dt stability probe: Q_total RMSE [%] ===")
    print(
        probe.pivot_table(
            index=["calibration", "flow_L_min", "model"],
            columns="dt_s",
            values="Q_total_rmse_rel_pct",
        ).round(3)
    )
    print(
        "\nstable flags (probe):",
        probe.groupby(["calibration", "dt_s"])["stable"].all().to_dict(),
    )
    print(f"\nResults written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
