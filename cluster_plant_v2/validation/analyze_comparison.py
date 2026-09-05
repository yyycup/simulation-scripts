"""Analyze and plot aligned Legacy-versus-Reference pack results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ("T_avg_C", "T_max_C", "T_min_C", "DeltaT_C", "SOC_avg")
BRANCH_COLUMNS = tuple(f"branch_current_{index}_A" for index in range(1, 5))


def error_metrics(legacy: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    legacy = np.asarray(legacy, dtype=float)
    reference = np.asarray(reference, dtype=float)
    valid = np.isfinite(legacy) & np.isfinite(reference)
    if not np.any(valid):
        return {name: np.nan for name in ("mae", "rmse", "max_abs_error", "final_error")}
    error = reference[valid] - legacy[valid]
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(np.max(np.abs(error))),
        "final_error": float(error[-1]),
    }


def branch_current_nrmse(legacy: np.ndarray, reference: np.ndarray) -> float:
    legacy = np.asarray(legacy, dtype=float)
    reference = np.asarray(reference, dtype=float)
    rmse = float(np.sqrt(np.mean((reference - legacy) ** 2)))
    scale = float(np.sqrt(np.mean(legacy**2)))
    if scale == 0.0:
        return 0.0 if rmse == 0.0 else np.nan
    return rmse / scale


def electrical_diagnostics(frame: pd.DataFrame, *, dt_s: float) -> dict[str, float]:
    legacy_heat = frame["legacy_Q_gen_total_W"].to_numpy(dtype=float)
    reference_heat = frame["reference_Q_gen_total_W"].to_numpy(dtype=float)
    heat_error = reference_heat - legacy_heat
    legacy_energy = float(np.sum(legacy_heat) * dt_s)
    reference_energy = float(np.sum(reference_heat) * dt_s)
    legacy_voltage = frame[
        [f"legacy_branch_voltage_{index}_V" for index in range(1, 5)]
    ].to_numpy(dtype=float)
    reference_voltage = frame[
        [f"reference_branch_voltage_{index}_V" for index in range(1, 5)]
    ].to_numpy(dtype=float)
    return {
        "qgen_mae_W": float(np.mean(np.abs(heat_error))),
        "qgen_max_abs_W": float(np.max(np.abs(heat_error))),
        "qgen_integral_legacy_J": legacy_energy,
        "qgen_integral_reference_J": reference_energy,
        "qgen_integral_difference_pct": (
            100.0 * (reference_energy / legacy_energy - 1.0)
            if legacy_energy != 0.0
            else np.nan
        ),
        "branch_voltage_max_abs_diff_V": float(
            np.max(np.abs(reference_voltage - legacy_voltage))
        ),
        "legacy_branch_voltage_spread_max_V": float(
            np.max(np.ptp(legacy_voltage, axis=1))
        ),
        "reference_branch_voltage_spread_max_V": float(
            np.max(np.ptp(reference_voltage, axis=1))
        ),
    }


def _plot_case(frame: pd.DataFrame, case_id: str, output_path: Path) -> None:
    time_s = frame["time_s"]
    figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    plot_specs = (
        ("T_avg_C", "Average cell temperature", "Temperature / degC"),
        ("T_max_C", "Maximum cell temperature", "Temperature / degC"),
        ("DeltaT_C", "Cell temperature spread", "Delta T / degC"),
        ("SOC_avg", "Average SOC", "SOC / 1"),
    )
    for axis, (column, title, ylabel) in zip(axes.flat[:4], plot_specs):
        axis.plot(time_s, frame[f"legacy_{column}"], label="Legacy")
        axis.plot(time_s, frame[f"reference_{column}"], label="Reference V2")
        if column in ("T_max_C", "DeltaT_C"):
            axis.plot(
                time_s,
                frame[f"reference_no_neighbor_{column}"],
                linestyle=":",
                label="Reference no neighbor",
            )
        axis.set(title=title, xlabel="Time / s", ylabel=ylabel)
        axis.grid(alpha=0.3)
        axis.legend()

    branch_axis = axes[2, 0]
    for index, column in enumerate(BRANCH_COLUMNS, start=1):
        branch_axis.plot(
            time_s,
            frame[f"legacy_{column}"],
            label=f"Legacy B{index}",
            linewidth=1.0,
        )
        branch_axis.plot(
            time_s,
            frame[f"reference_{column}"],
            linestyle="--",
            label=f"Reference B{index}",
            linewidth=1.0,
        )
    branch_axis.set(title="Branch currents", xlabel="Time / s", ylabel="Current / A")
    branch_axis.grid(alpha=0.3)
    branch_axis.legend(ncol=2, fontsize=8)

    heat_axis = axes[2, 1]
    heat_axis.plot(
        time_s, frame["legacy_Q_gen_total_W"], label="Legacy actual PyBaMM"
    )
    heat_axis.plot(
        time_s, frame["reference_Q_gen_total_W"], label="Reference V2 actual PyBaMM"
    )
    heat_axis.set(
        title="Total cell heat generation",
        xlabel="Time / s",
        ylabel="Heat generation / W",
    )
    heat_axis.grid(alpha=0.3)
    heat_axis.legend()
    figure.suptitle(case_id)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def analyze_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    metric_rows = []
    sanity_rows = []
    electrical_rows = []
    for csv_path in sorted(results_dir.glob("*_timeseries.csv")):
        case_id = csv_path.name.removesuffix("_timeseries.csv")
        metadata = json.loads(
            (results_dir / f"{case_id}_metadata.json").read_text(encoding="utf-8")
        )
        frame = pd.read_csv(csv_path)
        electrical_rows.append(
            {
                "case": case_id,
                **electrical_diagnostics(frame, dt_s=float(metadata["dt_s"])),
            }
        )
        for metric in METRICS + BRANCH_COLUMNS:
            values = error_metrics(frame[f"legacy_{metric}"], frame[f"reference_{metric}"])
            metric_rows.append({"case": case_id, "metric": metric, **values})

        legacy_branches = frame[[f"legacy_{column}" for column in BRANCH_COLUMNS]].to_numpy()
        reference_branches = frame[
            [f"reference_{column}" for column in BRANCH_COLUMNS]
        ].to_numpy()
        branch_nrmse = branch_current_nrmse(legacy_branches, reference_branches)
        summary_rows.append(
            {
                "case": case_id,
                "current_condition": metadata["current_condition"],
                "plate_temp_C": metadata["plate_temp_c"],
                "ambient_temp_C": metadata["ambient_temp_c"],
                "legacy_Tmax_peak_C": float(frame["legacy_T_max_C"].max()),
                "reference_Tmax_peak_C": float(frame["reference_T_max_C"].max()),
                "Tmax_peak_difference_C": float(
                    frame["reference_T_max_C"].max() - frame["legacy_T_max_C"].max()
                ),
                "legacy_Tavg_final_C": float(frame["legacy_T_avg_C"].iloc[-1]),
                "reference_Tavg_final_C": float(frame["reference_T_avg_C"].iloc[-1]),
                "Tavg_final_difference_C": float(
                    frame["reference_T_avg_C"].iloc[-1] - frame["legacy_T_avg_C"].iloc[-1]
                ),
                "legacy_DeltaT_final_C": float(frame["legacy_DeltaT_C"].iloc[-1]),
                "reference_DeltaT_final_C": float(frame["reference_DeltaT_C"].iloc[-1]),
                "DeltaT_final_difference_C": float(
                    frame["reference_DeltaT_C"].iloc[-1]
                    - frame["legacy_DeltaT_C"].iloc[-1]
                ),
                "Tavg_MAE_C": error_metrics(
                    frame["legacy_T_avg_C"], frame["reference_T_avg_C"]
                )["mae"],
                "Tavg_RMSE_C": error_metrics(
                    frame["legacy_T_avg_C"], frame["reference_T_avg_C"]
                )["rmse"],
                "SOC_avg_max_abs_difference": error_metrics(
                    frame["legacy_SOC_avg"], frame["reference_SOC_avg"]
                )["max_abs_error"],
                "branch_current_NRMSE": branch_nrmse,
                "max_abs_neighbor_sum_W": float(
                    frame["reference_Q_neighbor_sum_W"].abs().max()
                ),
                "neighbor_effect_Tmax_max_abs_C": float(
                    (frame["reference_T_max_C"] - frame["reference_no_neighbor_T_max_C"])
                    .abs()
                    .max()
                ),
                "neighbor_effect_DeltaT_max_abs_C": float(
                    (frame["reference_DeltaT_C"] - frame["reference_no_neighbor_DeltaT_C"])
                    .abs()
                    .max()
                ),
                "legacy_runtime_s": metadata["legacy_runtime_s"],
                "reference_runtime_s": metadata["reference_runtime_s"],
                "reference_over_legacy_runtime_ratio": metadata[
                    "reference_over_legacy_runtime_ratio"
                ],
            }
        )

        if (
            metadata["current_kind"] == "constant"
            and metadata["current_A"] == 560.0
            and metadata["plate_temp_c"] == 25.0
            and metadata["ambient_temp_c"] == 25.0
        ):
            first = frame.iloc[0]
            estimated = (
                first["reference_Q_gen_total_W"]
                * metadata["dt_s"]
                / (52.0 * 4747.0)
            )
            actual = first["reference_T_avg_C"] - metadata["initial_cell_temp_c"]
            sanity_rows.append(
                {
                    "case": case_id,
                    "Q_gen_total_first_step_W": first["reference_Q_gen_total_W"],
                    "DeltaT_est_C": estimated,
                    "DeltaT_actual_C": actual,
                    "absolute_difference_C": abs(actual - estimated),
                }
            )
        _plot_case(frame, case_id, results_dir / f"{case_id}_comparison.png")

    if not summary_rows:
        raise FileNotFoundError(f"no comparison time-series CSV files found in {results_dir}")
    summary = pd.DataFrame(summary_rows)
    metrics = pd.DataFrame(metric_rows)
    sanity = pd.DataFrame(sanity_rows)
    electrical = pd.DataFrame(electrical_rows)
    summary.to_csv(results_dir / "comparison_summary.csv", index=False)
    metrics.to_csv(results_dir / "comparison_metrics.csv", index=False)
    sanity.to_csv(results_dir / "sanity_checks.csv", index=False)
    electrical.to_csv(results_dir / "electrical_diagnostics_summary.csv", index=False)
    return summary, metrics, sanity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    summary, _, sanity = analyze_results(args.results_dir)
    print(summary.to_string(index=False), flush=True)
    if not sanity.empty:
        print("\nReference first-step heat sanity check", flush=True)
        print(sanity.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
