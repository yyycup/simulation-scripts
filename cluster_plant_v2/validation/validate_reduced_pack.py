"""Validate the coupled ReducedPack against Reference V2 plus 13-node plate."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_plant_v2.pack_reference import ReferenceBatteryPack
from cluster_plant_v2.thermal.pack_rom import ZONE_CELL_COUNTS, ZONE_GROUPS
from cluster_plant_v2.thermal.reduced_pack import ReducedPack
from cluster_plant_v2.profiles import AGC_DATA_FILE, load_regd_profile
from cluster_plant_v2.validation.validate_cold_plate_rom import (
    DetailedColdPlateAdapter,
    liters_per_minute_to_mass_flow,
)


DEFAULT_DT_S = 5.0
DEFAULT_DURATION_S = 600.0
COOLANT_INLET_K = 20.0 + 273.15
AMBIENT_TEMPERATURE_K = 35.0 + 273.15
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
Q_RELATIVE_EPS_W = 1.0


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    current_kind: str
    current_A: float | None
    flow_L_min: float
    flow_direction: int

    @property
    def current_condition(self) -> str:
        return "RegD [0, 600) s" if self.current_kind == "regd" else f"{self.current_A:g} A"

    @property
    def direction_name(self) -> str:
        return "forward" if self.flow_direction == 1 else "reverse"


def build_case_specs() -> list[CaseSpec]:
    return [
        CaseSpec("280a_5lpm_forward", "constant", 280.0, 5.0, 1),
        CaseSpec("560a_5lpm_forward", "constant", 560.0, 5.0, 1),
        CaseSpec("1120a_5lpm_forward", "constant", 1120.0, 5.0, 1),
        CaseSpec("regd_5lpm_forward", "regd", None, 5.0, 1),
        CaseSpec("560a_4lpm_forward", "constant", 560.0, 4.0, 1),
        CaseSpec("560a_6lpm_forward", "constant", 560.0, 6.0, 1),
        CaseSpec("560a_5lpm_reverse", "constant", 560.0, 5.0, -1),
    ]


def aggregate_reference_battery_zones(cell_values: np.ndarray) -> np.ndarray:
    cells = np.asarray(cell_values, dtype=float).reshape(4, 13)
    return np.array(
        [
            [cells[branch, list(columns)].mean() for columns in ZONE_GROUPS]
            for branch in range(4)
        ],
        dtype=float,
    )


def aggregate_reference_plate_zones(node_values: np.ndarray) -> np.ndarray:
    nodes = np.asarray(node_values, dtype=float).reshape(13)
    return np.array(
        [nodes[list(columns)].mean() for columns in ZONE_GROUPS], dtype=float
    )


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


class ReferenceCoupledPackAdapter:
    """Validation-only coupling of ReferenceBatteryPack V2 and the 13-node plate."""

    def __init__(self, initial_current_A: float = 560.0) -> None:
        self.battery = ReferenceBatteryPack(_pack_config(initial_current_A))
        self.cold_plate = DetailedColdPlateAdapter(initial_temperature_c=25.0)

    def _stored_thermal_energy(self) -> float:
        battery_energy = float(self.battery.config["cell_thermal_mass"]) * np.sum(
            self.battery.temps
        )
        plate_energy = self.cold_plate.node_heat_capacity * np.sum(
            self.cold_plate.plate_temperatures
        )
        return float(battery_energy + plate_energy)

    def step(
        self,
        dt: float,
        total_current: float,
        coolant_inlet_temperature: float,
        coolant_mass_flow: float,
        ambient_temperature: float,
        flow_direction: int = 1,
    ) -> dict[str, float | np.ndarray]:
        old_energy = self._stored_thermal_energy()
        old_cell_temperatures = self.battery.temps.copy()
        old_plate_temperatures = self.cold_plate.plate_temperatures.copy()
        cell_to_plate = float(self.battery.config["h_plate_convection"]) * (
            old_cell_temperatures.reshape(4, 13)
            - old_plate_temperatures[None, :]
        )
        q_battery_to_plate_nodes = cell_to_plate.sum(axis=0)
        q_air_cells = self.battery._air_convection_coefficients() * (
            old_cell_temperatures - float(ambient_temperature)
        )

        self.battery.step(
            dt,
            total_current,
            old_plate_temperatures,
            ambient_temperature,
        )
        plate_result = self.cold_plate.step(
            dt,
            coolant_inlet_temperature,
            coolant_mass_flow,
            q_battery_to_plate_nodes,
            flow_direction,
        )

        actual_delta_energy = self._stored_thermal_energy() - old_energy
        expected_delta_energy = float(dt) * (
            float(self.battery.q_gen_cells.sum())
            - float(q_air_cells.sum())
            - float(plate_result["q_plate_state_loss_total"])
        )
        battery_zones = aggregate_reference_battery_zones(self.battery.temps)
        plate_zones = aggregate_reference_plate_zones(
            self.cold_plate.plate_temperatures
        )
        result = {
            "battery_cell_temperatures": self.battery.temps.copy(),
            "battery_zone_temperatures": battery_zones,
            "battery_temperature_average": float(self.battery.temps.mean()),
            "battery_temperature_max_zone": float(battery_zones.max()),
            "battery_temperature_min_zone": float(battery_zones.min()),
            "battery_temperature_delta_zone": float(np.ptp(battery_zones)),
            "battery_cell_temperature_max": float(self.battery.temps.max()),
            "branch_currents": self.battery.branch_currents.copy(),
            "soc": self.battery.socs.reshape(4, 13).mean(axis=1),
            "q_gen_total": float(self.battery.q_gen_cells.sum()),
            "q_battery_to_plate_nodes": q_battery_to_plate_nodes.copy(),
            "q_battery_to_plate_zones": np.array(
                [q_battery_to_plate_nodes[list(columns)].sum() for columns in ZONE_GROUPS]
            ),
            "q_battery_to_plate_total": float(q_battery_to_plate_nodes.sum()),
            "plate_temperatures": self.cold_plate.plate_temperatures.copy(),
            "plate_zone_temperatures": plate_zones,
            "plate_temperature_average": float(
                self.cold_plate.plate_temperatures.mean()
            ),
            "coolant_outlet_temperature": float(
                plate_result["coolant_outlet_temperature"]
            ),
            "q_plate_to_fluid_total": float(
                plate_result["q_plate_to_fluid_total"]
            ),
            "q_plate_state_loss_total": float(
                plate_result["q_plate_state_loss_total"]
            ),
            "whole_pack_energy_residual_J": float(
                actual_delta_energy - expected_delta_energy
            ),
        }
        if not all(np.all(np.isfinite(value)) for value in result.values()):
            raise FloatingPointError(
                "ReferenceCoupledPackAdapter produced a non-finite output"
            )
        return result


def _current_profile(
    case: CaseSpec, *, duration_s: float, dt: float, regd_file: Path
) -> np.ndarray:
    if case.current_kind == "regd":
        _, currents = load_regd_profile(
            regd_file, duration_s=duration_s, dt=dt
        )
        return currents
    return np.full(
        int(round(duration_s / dt)), float(case.current_A), dtype=float
    )


def _run_reference(
    currents: np.ndarray,
    case: CaseSpec,
    dt: float,
    mass_flow: float,
) -> tuple[list[dict], dict[str, float]]:
    started = time.perf_counter()
    pack = ReferenceCoupledPackAdapter(float(currents[0]))
    initialization_s = time.perf_counter() - started
    records = []
    step_started = time.perf_counter()
    for current in currents:
        records.append(
            pack.step(
                dt,
                float(current),
                COOLANT_INLET_K,
                mass_flow,
                AMBIENT_TEMPERATURE_K,
                case.flow_direction,
            )
        )
    steps_s = time.perf_counter() - step_started
    return records, {
        "initialization_s": initialization_s,
        "steps_s": steps_s,
        "total_s": initialization_s + steps_s,
    }


def _run_reduced(
    currents: np.ndarray,
    case: CaseSpec,
    dt: float,
    mass_flow: float,
) -> tuple[list[dict], dict[str, float]]:
    started = time.perf_counter()
    pack = ReducedPack(_pack_config(float(currents[0])))
    initialization_s = time.perf_counter() - started
    records = []
    step_started = time.perf_counter()
    for current in currents:
        records.append(
            pack.step(
                dt,
                float(current),
                COOLANT_INLET_K,
                mass_flow,
                AMBIENT_TEMPERATURE_K,
                case.flow_direction,
            )
        )
    steps_s = time.perf_counter() - step_started
    return records, {
        "initialization_s": initialization_s,
        "steps_s": steps_s,
        "total_s": initialization_s + steps_s,
    }


def _weighted_battery_average(zone_values_k: np.ndarray) -> float:
    zones = np.asarray(zone_values_k, dtype=float).reshape(4, 3)
    return float(np.sum(zones * ZONE_CELL_COUNTS[None, :]) / 52.0)


def _weighted_plate_average(zone_values_k: np.ndarray) -> float:
    zones = np.asarray(zone_values_k, dtype=float).reshape(3)
    return float(np.sum(zones * ZONE_CELL_COUNTS) / 13.0)


def _build_timeseries(
    currents: np.ndarray,
    case: CaseSpec,
    dt: float,
    mass_flow: float,
    reference_records: list[dict],
    reduced_records: list[dict],
) -> pd.DataFrame:
    rows = []
    for index, (current, reference, reduced) in enumerate(
        zip(currents, reference_records, reduced_records), start=1
    ):
        ref_battery_zones = reference["battery_zone_temperatures"] - 273.15
        rom_battery_zones = reduced["battery_zone_temperatures"] - 273.15
        ref_plate_zones = reference["plate_zone_temperatures"] - 273.15
        rom_plate_zones = reduced["plate_temperatures"] - 273.15
        row = {
            "time_s": index * dt,
            "current_A": float(current),
            "flow_L_min": case.flow_L_min,
            "coolant_mass_flow_kg_s": mass_flow,
            "flow_direction": case.flow_direction,
            "Ref_Tbattery_avg_C": reference["battery_temperature_average"] - 273.15,
            "ROM_Tbattery_avg_C": _weighted_battery_average(
                reduced["battery_zone_temperatures"]
            ) - 273.15,
            "Ref_Tbattery_zone_max_C": float(ref_battery_zones.max()),
            "ROM_Tbattery_zone_max_C": float(rom_battery_zones.max()),
            "Ref_Tbattery_zone_min_C": float(ref_battery_zones.min()),
            "ROM_Tbattery_zone_min_C": float(rom_battery_zones.min()),
            "Ref_DeltaT_zone_C": float(np.ptp(ref_battery_zones)),
            "ROM_DeltaT_zone_C": float(np.ptp(rom_battery_zones)),
            "Ref_cell_Tmax_C": reference["battery_cell_temperature_max"] - 273.15,
            "hotspot_loss_C": (
                reference["battery_cell_temperature_max"] - 273.15
                - float(rom_battery_zones.max())
            ),
            "Ref_Tplate_avg_C": reference["plate_temperature_average"] - 273.15,
            "ROM_Tplate_avg_C": _weighted_plate_average(
                reduced["plate_temperatures"]
            ) - 273.15,
            "Ref_Tout_C": reference["coolant_outlet_temperature"] - 273.15,
            "ROM_Tout_C": reduced["coolant_outlet_temperature"] - 273.15,
            "Ref_Qgen_total_W": reference["q_gen_total"],
            "ROM_Qgen_total_W": reduced["q_gen_total"],
            "Ref_Qbattery_to_plate_total_W": reference["q_battery_to_plate_total"],
            "ROM_Qbattery_to_plate_total_W": reduced["q_battery_to_plate_total"],
            "Ref_Qplate_to_fluid_total_W": reference["q_plate_to_fluid_total"],
            "ROM_Qplate_to_fluid_total_W": reduced["q_plate_to_fluid_total"],
            "Ref_whole_pack_energy_residual_J": reference[
                "whole_pack_energy_residual_J"
            ],
            "ROM_whole_pack_energy_residual_J": reduced[
                "whole_pack_energy_residual_J"
            ],
            "ROM_whole_pack_energy_relative_error": reduced[
                "whole_pack_energy_relative_error"
            ],
            "ROM_max_abs_coupling_residual_W": reduced[
                "max_abs_coupling_residual_W"
            ],
            "ROM_coupling_residual_total_W": float(
                np.sum(reduced["coupling_energy_residual"])
            ),
        }
        for branch in range(4):
            row[f"Ref_Ibranch_{branch + 1}_A"] = reference["branch_currents"][branch]
            row[f"ROM_Ibranch_{branch + 1}_A"] = reduced["branch_currents"][branch]
            row[f"Ref_SOC_branch_{branch + 1}"] = reference["soc"][branch]
            row[f"ROM_SOC_branch_{branch + 1}"] = reduced["soc"][branch]
            for zone in range(3):
                row[f"Ref_Tbattery_b{branch}_z{zone}_C"] = ref_battery_zones[branch, zone]
                row[f"ROM_Tbattery_b{branch}_z{zone}_C"] = rom_battery_zones[branch, zone]
        for zone in range(3):
            row[f"Ref_Tplate_zone{zone}_C"] = ref_plate_zones[zone]
            row[f"ROM_Tplate_zone{zone}_C"] = rom_plate_zones[zone]
        rows.append(row)
    return pd.DataFrame(rows)


def _error_metrics(reference: np.ndarray, reduced: np.ndarray) -> dict[str, float]:
    error = np.asarray(reduced, dtype=float) - np.asarray(reference, dtype=float)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(np.max(np.abs(error))),
        "final_error": float(error[-1]),
    }


def _normalized_rmse(reference: np.ndarray, reduced: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    denominator = float(np.sqrt(np.mean(reference**2)))
    error_rmse = float(
        np.sqrt(np.mean((np.asarray(reduced, dtype=float) - reference) ** 2))
    )
    if denominator <= 1e-12:
        return 0.0 if error_rmse <= 1e-12 else float("inf")
    return error_rmse / denominator


def _metric_row(
    case_id: str,
    metric: str,
    component: str,
    unit: str,
    reference: np.ndarray,
    reduced: np.ndarray,
    *,
    relative: bool = False,
    nrmse: bool = False,
) -> dict:
    row = {
        "case_id": case_id,
        "metric": metric,
        "component": component,
        "unit": unit,
        **_error_metrics(reference, reduced),
    }
    if relative:
        absolute_error = np.abs(np.asarray(reduced) - np.asarray(reference))
        relative_error = absolute_error / np.maximum(
            np.abs(reference), Q_RELATIVE_EPS_W
        )
        row["mean_relative_error"] = float(np.mean(relative_error))
        row["max_relative_error"] = float(np.max(relative_error))
    if nrmse:
        row["nrmse"] = _normalized_rmse(reference, reduced)
    return row


def _drift_metrics(frame: pd.DataFrame, reference: str, reduced: str) -> dict[str, float]:
    error = frame[reduced].to_numpy() - frame[reference].to_numpy()
    early = error[frame["time_s"].to_numpy() <= 100.0]
    late = error[frame["time_s"].to_numpy() > frame["time_s"].max() - 100.0]
    trend = 0.0
    if len(frame) > 1:
        trend = float(np.polyfit(frame["time_s"], error, 1)[0] * 600.0)
    return {
        "first_100s_mean_error_C": float(np.mean(early)),
        "last_100s_mean_error_C": float(np.mean(late)),
        "error_trend_per_600s_C": trend,
    }


def _calculate_metrics(
    case: CaseSpec, frame: pd.DataFrame
) -> tuple[list[dict], list[dict], dict]:
    specs = (
        ("battery_average_temperature", "pack", "degC", "Ref_Tbattery_avg_C", "ROM_Tbattery_avg_C", False),
        ("battery_zone_maximum", "pack", "degC", "Ref_Tbattery_zone_max_C", "ROM_Tbattery_zone_max_C", False),
        ("battery_zone_delta", "pack", "degC", "Ref_DeltaT_zone_C", "ROM_DeltaT_zone_C", False),
        ("plate_average_temperature", "plate", "degC", "Ref_Tplate_avg_C", "ROM_Tplate_avg_C", False),
        ("coolant_outlet_temperature", "coolant", "degC", "Ref_Tout_C", "ROM_Tout_C", False),
        ("q_generation", "pack", "W", "Ref_Qgen_total_W", "ROM_Qgen_total_W", True),
        ("q_battery_to_plate", "coupling", "W", "Ref_Qbattery_to_plate_total_W", "ROM_Qbattery_to_plate_total_W", True),
        ("q_plate_to_fluid", "plate", "W", "Ref_Qplate_to_fluid_total_W", "ROM_Qplate_to_fluid_total_W", True),
    )
    metrics = [
        _metric_row(
            case.case_id,
            metric,
            component,
            unit,
            frame[ref_column].to_numpy(),
            frame[rom_column].to_numpy(),
            relative=relative,
        )
        for metric, component, unit, ref_column, rom_column, relative in specs
    ]
    zone_metrics = []
    for branch in range(4):
        for zone in range(3):
            row = _metric_row(
                case.case_id,
                "battery_zone_temperature",
                f"b{branch}_z{zone}",
                "degC",
                frame[f"Ref_Tbattery_b{branch}_z{zone}_C"].to_numpy(),
                frame[f"ROM_Tbattery_b{branch}_z{zone}_C"].to_numpy(),
            )
            row.update({"level": "battery", "branch": branch, "zone": zone})
            zone_metrics.append(row)
            metrics.append(dict(row))
    for zone in range(3):
        row = _metric_row(
            case.case_id,
            "plate_zone_temperature",
            f"zone_{zone}",
            "degC",
            frame[f"Ref_Tplate_zone{zone}_C"].to_numpy(),
            frame[f"ROM_Tplate_zone{zone}_C"].to_numpy(),
        )
        row.update({"level": "plate", "branch": np.nan, "zone": zone})
        zone_metrics.append(row)
        metrics.append(dict(row))
    for branch in range(1, 5):
        metrics.append(
            _metric_row(
                case.case_id,
                "branch_current",
                f"branch_{branch}",
                "A",
                frame[f"Ref_Ibranch_{branch}_A"].to_numpy(),
                frame[f"ROM_Ibranch_{branch}_A"].to_numpy(),
                nrmse=True,
            )
        )
        metrics.append(
            _metric_row(
                case.case_id,
                "branch_soc",
                f"branch_{branch}",
                "fraction",
                frame[f"Ref_SOC_branch_{branch}"].to_numpy(),
                frame[f"ROM_SOC_branch_{branch}"].to_numpy(),
            )
        )

    indexed = {(row["metric"], row["component"]): row for row in metrics}
    battery_zones = [row for row in zone_metrics if row["level"] == "battery"]
    plate_zones = [row for row in zone_metrics if row["level"] == "plate"]
    branch_rows = [row for row in metrics if row["metric"] == "branch_current"]
    soc_rows = [row for row in metrics if row["metric"] == "branch_soc"]
    battery_drift = _drift_metrics(
        frame, "Ref_Tbattery_avg_C", "ROM_Tbattery_avg_C"
    )
    plate_drift = _drift_metrics(frame, "Ref_Tplate_avg_C", "ROM_Tplate_avg_C")
    outlet_drift = _drift_metrics(frame, "Ref_Tout_C", "ROM_Tout_C")
    summary = {
        "case_id": case.case_id,
        "current_condition": case.current_condition,
        "flow_L_min": case.flow_L_min,
        "flow_direction": case.flow_direction,
        "direction_name": case.direction_name,
        "steps": len(frame),
        "battery_Tavg_MAE_C": indexed[("battery_average_temperature", "pack")]["mae"],
        "battery_zone_Tmax_MAE_C": indexed[("battery_zone_maximum", "pack")]["mae"],
        "battery_zone_DeltaT_MAE_C": indexed[("battery_zone_delta", "pack")]["mae"],
        "worst_battery_zone_MAE_C": max(row["mae"] for row in battery_zones),
        "plate_avg_MAE_C": indexed[("plate_average_temperature", "plate")]["mae"],
        "worst_plate_zone_MAE_C": max(row["mae"] for row in plate_zones),
        "Tout_MAE_C": indexed[("coolant_outlet_temperature", "coolant")]["mae"],
        "Qgen_mean_relative_error": indexed[("q_generation", "pack")]["mean_relative_error"],
        "Qbp_mean_relative_error": indexed[("q_battery_to_plate", "coupling")]["mean_relative_error"],
        "Qplate_fluid_mean_relative_error": indexed[("q_plate_to_fluid", "plate")]["mean_relative_error"],
        "branch_current_max_NRMSE": max(row["nrmse"] for row in branch_rows),
        "branch_SOC_max_MAE": max(row["mae"] for row in soc_rows),
        "hotspot_loss_mean_C": float(frame["hotspot_loss_C"].mean()),
        "hotspot_loss_max_C": float(frame["hotspot_loss_C"].max()),
        "battery_error_first_100s_C": battery_drift["first_100s_mean_error_C"],
        "battery_error_last_100s_C": battery_drift["last_100s_mean_error_C"],
        "battery_error_trend_per_600s_C": battery_drift["error_trend_per_600s_C"],
        "plate_error_first_100s_C": plate_drift["first_100s_mean_error_C"],
        "plate_error_last_100s_C": plate_drift["last_100s_mean_error_C"],
        "plate_error_trend_per_600s_C": plate_drift["error_trend_per_600s_C"],
        "Tout_error_first_100s_C": outlet_drift["first_100s_mean_error_C"],
        "Tout_error_last_100s_C": outlet_drift["last_100s_mean_error_C"],
        "Tout_error_trend_per_600s_C": outlet_drift["error_trend_per_600s_C"],
        "all_states_finite": bool(
            np.all(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()))
        ),
    }
    summary.update(
        {
            "pass_battery_Tavg": summary["battery_Tavg_MAE_C"] <= 0.20,
            "pass_battery_zone": summary["worst_battery_zone_MAE_C"] <= 0.30,
            "pass_battery_DeltaT": summary["battery_zone_DeltaT_MAE_C"] <= 0.30,
            "pass_plate_avg": summary["plate_avg_MAE_C"] <= 0.20,
            "pass_plate_zone": summary["worst_plate_zone_MAE_C"] <= 0.30,
            "pass_Tout": summary["Tout_MAE_C"] <= 0.20,
            "pass_Qgen": summary["Qgen_mean_relative_error"] <= 0.05,
            "pass_Qbp": summary["Qbp_mean_relative_error"] <= 0.05,
            "pass_Qplate_fluid": summary["Qplate_fluid_mean_relative_error"] <= 0.05,
            "pass_branch_current": summary["branch_current_max_NRMSE"] <= 0.05,
        }
    )
    summary["all_numeric_gates_pass"] = all(
        summary[key]
        for key in (
            "pass_battery_Tavg",
            "pass_battery_zone",
            "pass_battery_DeltaT",
            "pass_plate_avg",
            "pass_plate_zone",
            "pass_Tout",
            "pass_Qgen",
            "pass_Qbp",
            "pass_Qplate_fluid",
            "pass_branch_current",
            "all_states_finite",
        )
    )
    return metrics, zone_metrics, summary


def _energy_summary(case: CaseSpec, frame: pd.DataFrame, dt: float) -> dict:
    coupling = frame["ROM_coupling_residual_total_W"].to_numpy()
    return {
        "case_id": case.case_id,
        "max_abs_coupling_residual_W": float(
            frame["ROM_max_abs_coupling_residual_W"].max()
        ),
        "mean_abs_coupling_residual_W": float(np.mean(np.abs(coupling))),
        "cumulative_coupling_residual_J": float(dt * np.sum(coupling)),
        "max_abs_whole_pack_energy_residual_J": float(
            frame["ROM_whole_pack_energy_residual_J"].abs().max()
        ),
        "max_abs_whole_pack_energy_relative_error": float(
            frame["ROM_whole_pack_energy_relative_error"].abs().max()
        ),
        "reference_max_abs_whole_pack_energy_residual_J": float(
            frame["Ref_whole_pack_energy_residual_J"].abs().max()
        ),
    }


def _plot_case(case: CaseSpec, frame: pd.DataFrame, output_stem: Path) -> None:
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
            "grid.alpha": 0.18,
            "legend.frameon": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    ref_color, rom_color = "#0072B2", "#D55E00"
    zone_colors = ("#009E73", "#E69F00", "#CC79A7")
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.6), sharex=True)
    axes[0, 0].plot(frame["time_s"], frame["Ref_Tbattery_avg_C"], color=ref_color, label="Reference average")
    axes[0, 0].plot(frame["time_s"], frame["ROM_Tbattery_avg_C"], color=rom_color, linestyle="--", label="ROM average")
    axes[0, 0].plot(frame["time_s"], frame["Ref_Tbattery_zone_max_C"], color=ref_color, alpha=0.55, label="Reference zone max")
    axes[0, 0].plot(frame["time_s"], frame["ROM_Tbattery_zone_max_C"], color=rom_color, linestyle=":", label="ROM zone max")
    axes[0, 0].set_title("Battery temperatures")
    axes[0, 0].set_ylabel("Temperature (°C)")
    for zone, color in enumerate(zone_colors):
        axes[0, 1].plot(frame["time_s"], frame[f"Ref_Tplate_zone{zone}_C"], color=color, label=f"Ref zone {zone}")
        axes[0, 1].plot(frame["time_s"], frame[f"ROM_Tplate_zone{zone}_C"], color=color, linestyle="--", label=f"ROM zone {zone}")
    axes[0, 1].set_title("Cold-plate zone temperatures")
    axes[0, 1].set_ylabel("Temperature (°C)")
    axes[1, 0].plot(frame["time_s"], frame["Ref_Tout_C"], color=ref_color, label="Reference")
    axes[1, 0].plot(frame["time_s"], frame["ROM_Tout_C"], color=rom_color, linestyle="--", label="ROM")
    axes[1, 0].set_title("Coolant outlet")
    axes[1, 0].set_ylabel("Temperature (°C)")
    for ref_column, rom_column, label, color in (
        ("Ref_Qgen_total_W", "ROM_Qgen_total_W", "Qgen", "#009E73"),
        ("Ref_Qbattery_to_plate_total_W", "ROM_Qbattery_to_plate_total_W", "Qbp", "#E69F00"),
        ("Ref_Qplate_to_fluid_total_W", "ROM_Qplate_to_fluid_total_W", "Qplate-fluid", "#CC79A7"),
    ):
        axes[1, 1].plot(frame["time_s"], frame[ref_column], color=color, label=f"Ref {label}")
        axes[1, 1].plot(frame["time_s"], frame[rom_column], color=color, linestyle="--", label=f"ROM {label}")
    axes[1, 1].set_title("Heat-transfer rates")
    axes[1, 1].set_ylabel("Heat rate (W)")
    for axis in axes.ravel():
        axis.legend(fontsize=7, ncol=2)
    for axis in axes[1]:
        axis.set_xlabel("Time (s)")
    fig.suptitle(
        f"{case.current_condition}, {case.flow_L_min:g} L/min, {case.direction_name}"
    )
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
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
    if (
        duration_s <= 0.0
        or dt <= 0.0
        or not np.isclose(duration_s / dt, round(duration_s / dt))
    ):
        raise ValueError("duration_s must be a positive integer multiple of dt")
    currents = _current_profile(
        case, duration_s=duration_s, dt=dt, regd_file=regd_file
    )
    mass_flow = liters_per_minute_to_mass_flow(case.flow_L_min)
    reference_records, reference_runtime = _run_reference(
        currents, case, dt, mass_flow
    )
    reduced_records, reduced_runtime = _run_reduced(
        currents, case, dt, mass_flow
    )
    frame = _build_timeseries(
        currents,
        case,
        dt,
        mass_flow,
        reference_records,
        reduced_records,
    )
    frame.to_csv(output_dir / f"case_{case.case_id}_timeseries.csv", index=False)
    metrics, zone_metrics, summary = _calculate_metrics(case, frame)
    energy = _energy_summary(case, frame, dt)
    runtime = {
        "case_id": case.case_id,
        **{f"reference_{key}": value for key, value in reference_runtime.items()},
        **{f"reduced_{key}": value for key, value in reduced_runtime.items()},
        "speedup": reference_runtime["total_s"] / reduced_runtime["total_s"],
        "runtime_scope": "model construction plus all coupled steps",
    }
    if make_plot and case.case_id in {
        "560a_5lpm_forward",
        "regd_5lpm_forward",
        "560a_5lpm_reverse",
    }:
        _plot_case(
            case,
            frame,
            output_dir / f"case_{case.case_id}_comparison",
        )
    return {
        "timeseries": frame,
        "metrics": metrics,
        "zone_metrics": zone_metrics,
        "summary": summary,
        "energy": energy,
        "runtime": runtime,
    }


def compare_one_step() -> dict:
    case = CaseSpec("one_step_560a_5lpm_forward", "constant", 560.0, 5.0, 1)
    mass_flow = liters_per_minute_to_mass_flow(5.0)
    reference = ReferenceCoupledPackAdapter(560.0)
    reduced = ReducedPack(_pack_config(560.0))
    ref = reference.step(
        5.0, 560.0, COOLANT_INLET_K, mass_flow, AMBIENT_TEMPERATURE_K, 1
    )
    rom = reduced.step(
        5.0, 560.0, COOLANT_INLET_K, mass_flow, AMBIENT_TEMPERATURE_K, 1
    )
    return {
        "case_id": case.case_id,
        "dt_s": 5.0,
        "total_current_A": 560.0,
        "coolant_flow_L_min": 5.0,
        "coolant_mass_flow_kg_s": mass_flow,
        "branch_current_sum_A": float(rom["branch_currents"].sum()),
        "soc_initial": 0.95,
        "soc_final_max": float(np.max(rom["soc"])),
        "q_gen_total_W": rom["q_gen_total"],
        "q_battery_to_plate_total_W": rom["q_battery_to_plate_total"],
        "q_plate_to_fluid_total_W": rom["q_plate_to_fluid_total"],
        "coolant_inlet_temperature_C": COOLANT_INLET_K - 273.15,
        "coolant_outlet_temperature_C": rom["coolant_outlet_temperature"] - 273.15,
        "all_reduced_states_finite": bool(
            all(np.all(np.isfinite(value)) for value in rom.values())
        ),
        "max_abs_coupling_residual_W": rom["max_abs_coupling_residual_W"],
        "whole_pack_energy_residual_J": rom["whole_pack_energy_residual_J"],
        "whole_pack_energy_relative_error": rom["whole_pack_energy_relative_error"],
        "reference_whole_pack_energy_residual_J": ref[
            "whole_pack_energy_residual_J"
        ],
    }


def _write_aggregate_results(output_dir: Path, results: list[dict]) -> None:
    pd.DataFrame([result["summary"] for result in results]).to_csv(
        output_dir / "reduced_pack_summary.csv", index=False
    )
    pd.DataFrame(
        [row for result in results for row in result["metrics"]]
    ).to_csv(output_dir / "reduced_pack_metrics.csv", index=False)
    pd.DataFrame(
        [row for result in results for row in result["zone_metrics"]]
    ).to_csv(output_dir / "reduced_pack_zone_metrics.csv", index=False)
    pd.DataFrame([result["energy"] for result in results]).to_csv(
        output_dir / "reduced_pack_energy_balance.csv", index=False
    )
    runtime_rows = [result["runtime"] for result in results]
    reference_total = sum(row["reference_total_s"] for row in runtime_rows)
    reduced_total = sum(row["reduced_total_s"] for row in runtime_rows)
    runtime_rows.append(
        {
            "case_id": "OVERALL",
            "reference_total_s": reference_total,
            "reduced_total_s": reduced_total,
            "speedup": reference_total / reduced_total,
            "runtime_scope": "sum across completed formal cases",
        }
    )
    pd.DataFrame(runtime_rows).to_csv(
        output_dir / "reduced_pack_runtime_summary.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT_S)
    parser.add_argument("--regd-file", type=Path, default=AGC_DATA_FILE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or RESULTS_ROOT / (
        f"reduced_pack_validation_{datetime.now():%Y%m%d}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    one_step = compare_one_step()
    pd.DataFrame([one_step]).to_csv(
        output_dir / "reduced_pack_one_step.csv", index=False
    )
    if not (
        np.isclose(one_step["branch_current_sum_A"], 560.0)
        and one_step["soc_final_max"] < one_step["soc_initial"]
        and one_step["q_gen_total_W"] > 0.0
        and one_step["q_plate_to_fluid_total_W"] > 0.0
        and one_step["coolant_outlet_temperature_C"]
        > one_step["coolant_inlet_temperature_C"]
        and one_step["all_reduced_states_finite"]
        and one_step["max_abs_coupling_residual_W"] <= 1e-12
    ):
        raise RuntimeError("one-step coupled validation failed; formal cases stopped")

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
            dt=args.dt,
            regd_file=args.regd_file,
            output_dir=output_dir,
            make_plot=not args.no_plots,
        )
        results.append(result)
        _write_aggregate_results(output_dir, results)
        summary = result["summary"]
        print(
            f"  Tb avg MAE={summary['battery_Tavg_MAE_C']:.6f} C, "
            f"plate avg MAE={summary['plate_avg_MAE_C']:.6f} C, "
            f"Tout MAE={summary['Tout_MAE_C']:.6f} C, "
            f"pass={summary['all_numeric_gates_pass']}",
            flush=True,
        )
    print(f"reduced-pack results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
