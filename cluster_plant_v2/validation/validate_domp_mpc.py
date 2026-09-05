"""Closed-loop validation of the do-mpc cluster controller.

Compares the physics-ROM do-mpc controller against the frozen fixed-speed
baseline (compressor 4000 rpm / pump 3600 rpm) used in the Stage 9B
comparison. Cases:

- ``S0_constant``: 560 A constant discharge, 600 s
- ``S1_step``: 280 A -> 560 A (25 s) -> 1120 A (50 s), 600 s
- ``T2_regd``: PJM RegD profile, only when the external data file exists

The MPC receives perfect current preview (same information assumption as
the deleted Stage 9B GEKKO controller) and its moves are clipped by the
application-layer DMAX rate limits because do-mpc 5.x has no native
move-rate bounds.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cluster_plant_v2.control.chiller_surrogates import load_surrogates
from cluster_plant_v2.control.cluster_domp_model import (
    STATE_NAMES,
    build_cluster_model,
    derive_coefficients,
    extract_x0,
)
from cluster_plant_v2.control.cluster_domp_mpc import (
    ClusterDompcParameters,
    apply_rate_limits,
    build_mpc,
)
from cluster_plant_v2.parameters import (
    MAXIMUM_COMPRESSOR_SPEED_RPM,
    MAX_PUMP_SPEED_RPM,
    MINIMUM_COMPRESSOR_SPEED_RPM,
    MIN_PUMP_SPEED_RPM,
)
from cluster_plant_v2.profiles import AGC_DATA_FILE, load_regd_profile
from cluster_plant_v2.validation.validate_final_cluster_plant import (
    DURATION_S,
    FAN_SPEED_RPM,
    build_final_plant,
)
from cluster_plant_v2.validation.validate_refrigerated_cluster_cooling_loop import (
    AMBIENT_TEMPERATURE_K,
    DT_S,
    PUMP_SPEED_RPM,
)


BASELINE_COMPRESSOR_RPM = 4000.0
BASELINE_PUMP_RPM = float(PUMP_SPEED_RPM)
TARGET_TEMPERATURE_K = 298.15
BAND_HALF_WIDTH_K = 0.65


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    description: str


def case_currents(case: CaseSpec, steps: int, dt_s: float) -> np.ndarray:
    """Build the discharge-current schedule for a case."""
    times = np.arange(steps) * dt_s
    if case.case_id == "S0_constant":
        return np.full(steps, 560.0)
    if case.case_id == "S1_step":
        currents = np.where(times < 25.0, 280.0, 560.0)
        currents = np.where(times >= 50.0, 1120.0, currents)
        return currents
    if case.case_id == "T2_regd":
        _, currents = load_regd_profile(
            AGC_DATA_FILE, duration_s=steps * dt_s, dt=dt_s
        )
        return np.asarray(currents, dtype=float)[:steps]
    raise ValueError(f"unknown case {case.case_id}")


def initial_snapshot(plant) -> dict[str, object]:
    """Synthesize a ClusterPlantOutputs-shaped snapshot before any step."""
    packs = plant.cluster.packs
    battery_averages = [
        float(
            np.average(
                pack.battery.temps, weights=pack.battery.zone_heat_capacities
            )
        )
        for pack in packs
    ]
    plate_temperatures = np.stack(
        [pack.cold_plate.plate_temperatures for pack in packs]
    )
    return {
        "compressor_speed_rpm": float(plant.compressor_actuator.speed_rpm),
        "q_evap_applied_w": float(
            plant.evaporator_dynamics.q_evap_applied_w
        ),
        "supply_delay_queue_k": list(plant.supply_delay.queue_values),
        "return_delay_queue_k": list(plant.return_delay.queue_values),
        "tank_temperature_after_k": float(plant.tank.temperature_k),
        "cluster_result": {
            "pack_battery_average_temperatures_k": battery_averages,
            "pack_plate_temperatures_k": plate_temperatures,
        },
    }


def _record_row(
    *,
    controller: str,
    time_s: float,
    current_a: float,
    compressor_command_rpm: float,
    pump_command_rpm: float,
    result,
) -> dict[str, object]:
    cluster = result["cluster_result"]
    return {
        "controller": controller,
        "time_s": time_s,
        "current_a": current_a,
        "compressor_command_rpm": compressor_command_rpm,
        "compressor_actual_rpm": float(result["compressor_speed_rpm"]),
        "pump_command_rpm": pump_command_rpm,
        "battery_avg_temp_c": float(
            np.mean(cluster["pack_battery_average_temperatures_k"]) - 273.15
        ),
        "battery_max_temp_c": float(
            cluster["cluster_max_temperature_k"] - 273.15
        ),
        "tank_temp_c": float(result["tank_temperature_after_k"] - 273.15),
        "supply_temp_c": float(
            result["cluster_supply_temperature_k"] - 273.15
        ),
        "return_temp_c": float(
            result["cluster_return_temperature_k"] - 273.15
        ),
        "q_evap_applied_w": float(result["q_evap_applied_w"]),
        "compressor_power_w": float(result["compressor_shaft_power_w"]),
        "pump_power_w": float(result["pump_power_w"]),
    }


def run_baseline(
    case: CaseSpec, *, duration_s: float, dt_s: float
) -> pd.DataFrame:
    steps = int(round(duration_s / dt_s))
    currents = case_currents(case, steps, dt_s)
    plant = build_final_plant(
        BASELINE_COMPRESSOR_RPM,
        initial_cluster_current_a=float(currents[0]),
        direction="forward",
    )
    rows = []
    for step_index in range(steps):
        result = plant.step(
            dt_s=dt_s,
            cluster_current_a=float(currents[step_index]),
            pump_speed_rpm=BASELINE_PUMP_RPM,
            compressor_speed_command_rpm=BASELINE_COMPRESSOR_RPM,
            fan_speed_rpm=FAN_SPEED_RPM,
            ambient_temperature_k=AMBIENT_TEMPERATURE_K,
            direction="forward",
        )
        rows.append(
            _record_row(
                controller="baseline_fixed",
                time_s=(step_index + 1) * dt_s,
                current_a=float(currents[step_index]),
                compressor_command_rpm=BASELINE_COMPRESSOR_RPM,
                pump_command_rpm=BASELINE_PUMP_RPM,
                result=result,
            )
        )
    return pd.DataFrame(rows)


def run_mpc(
    case: CaseSpec,
    *,
    duration_s: float,
    dt_s: float,
    parameters: ClusterDompcParameters | None = None,
) -> pd.DataFrame:
    if parameters is None:
        parameters = ClusterDompcParameters()
    steps = int(round(duration_s / dt_s))
    currents = case_currents(case, steps, dt_s)
    plant = build_final_plant(
        BASELINE_COMPRESSOR_RPM,
        initial_cluster_current_a=float(currents[0]),
        direction="forward",
    )

    surrogates = load_surrogates()
    coefficients = derive_coefficients(
        plant,
        nominal_pump_speed_rpm=BASELINE_PUMP_RPM,
        surrogates=surrogates,
    )
    model = build_cluster_model(coefficients)

    def current_preview(time_s: float) -> float:
        index = min(max(int(float(time_s) / dt_s), 0), steps - 1)
        return float(currents[index])

    mpc = build_mpc(
        model,
        coefficients,
        parameters,
        current_preview=current_preview,
        ambient_temperature_k=AMBIENT_TEMPERATURE_K,
    )

    x0 = extract_x0(plant, initial_snapshot(plant))
    for name, value in zip(STATE_NAMES, x0):
        mpc.x0[name] = float(value)
    mpc.u0["comp_cmd_rpm"] = BASELINE_COMPRESSOR_RPM
    mpc.u0["pump_rpm"] = BASELINE_PUMP_RPM
    mpc.set_initial_guess()

    comp_previous = BASELINE_COMPRESSOR_RPM
    pump_previous = BASELINE_PUMP_RPM
    rows = []
    for step_index in range(steps):
        u_new = mpc.make_step(x0.reshape(-1, 1))
        comp_command = apply_rate_limits(
            float(u_new[0, 0]),
            comp_previous,
            parameters.compressor_dmax_rpm,
            MINIMUM_COMPRESSOR_SPEED_RPM,
            MAXIMUM_COMPRESSOR_SPEED_RPM,
        )
        pump_command = apply_rate_limits(
            float(u_new[1, 0]),
            pump_previous,
            parameters.pump_dmax_rpm,
            MIN_PUMP_SPEED_RPM,
            MAX_PUMP_SPEED_RPM,
        )
        result = plant.step(
            dt_s=dt_s,
            cluster_current_a=float(currents[step_index]),
            pump_speed_rpm=pump_command,
            compressor_speed_command_rpm=comp_command,
            fan_speed_rpm=FAN_SPEED_RPM,
            ambient_temperature_k=AMBIENT_TEMPERATURE_K,
            direction="forward",
        )
        rows.append(
            _record_row(
                controller="domp_mpc",
                time_s=(step_index + 1) * dt_s,
                current_a=float(currents[step_index]),
                compressor_command_rpm=comp_command,
                pump_command_rpm=pump_command,
                result=result,
            )
        )
        comp_previous = comp_command
        pump_previous = pump_command
        x0 = extract_x0(plant, result)
    return pd.DataFrame(rows)


def summarize(case_id: str, frame: pd.DataFrame, dt_s: float) -> dict:
    controller = str(frame["controller"].iloc[0])
    battery = frame["battery_avg_temp_c"].to_numpy()
    band_error = np.maximum(
        np.abs(battery - (TARGET_TEMPERATURE_K - 273.15)) - BAND_HALF_WIDTH_K,
        0.0,
    )
    comp_commands = frame["compressor_command_rpm"].to_numpy()
    pump_commands = frame["pump_command_rpm"].to_numpy()
    return {
        "case_id": case_id,
        "controller": controller,
        "final_battery_avg_temp_c": float(battery[-1]),
        "max_battery_temp_c": float(frame["battery_max_temp_c"].max()),
        "mean_band_error_k": float(np.mean(band_error)),
        "max_band_error_k": float(np.max(band_error)),
        "time_outside_band_s": float(np.count_nonzero(band_error) * dt_s),
        "compressor_energy_kwh": float(
            np.sum(frame["compressor_power_w"]) * dt_s / 3.6e6
        ),
        "pump_energy_kwh": float(
            np.sum(frame["pump_power_w"]) * dt_s / 3.6e6
        ),
        "compressor_movement_rpm": float(
            np.sum(np.abs(np.diff(comp_commands)))
        ),
        "pump_movement_rpm": float(np.sum(np.abs(np.diff(pump_commands)))),
    }


def plot_case(case_id: str, frame: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    target_c = TARGET_TEMPERATURE_K - 273.15
    for controller, group in frame.groupby("controller"):
        style = {
            "domp_mpc": "tab:blue",
            "baseline_fixed": "tab:red",
            "physics_p_nmpc": "tab:green",
        }
        color = style.get(controller, "tab:gray")
        axes[0].plot(
            group["time_s"], group["battery_avg_temp_c"],
            label=controller, color=color,
        )
        axes[1].plot(
            group["time_s"], group["compressor_command_rpm"],
            label=controller, color=color,
        )
        axes[1].plot(
            group["time_s"], group["pump_command_rpm"],
            color=color, linestyle="--",
        )
        axes[2].plot(
            group["time_s"],
            group["compressor_power_w"] + group["pump_power_w"],
            label=controller, color=color,
        )
    axes[0].axhline(target_c, color="k", linewidth=0.8)
    axes[0].axhspan(
        target_c - BAND_HALF_WIDTH_K, target_c + BAND_HALF_WIDTH_K,
        color="green", alpha=0.15,
    )
    axes[0].set_ylabel("battery avg temp (degC)")
    axes[1].set_ylabel("speed (rpm)")
    axes[2].set_ylabel("total power (W)")
    axes[2].set_xlabel("time (s)")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle(f"{case_id}: do-mpc vs fixed baseline")
    fig.tight_layout()
    fig.savefig(output_dir / f"{case_id}_comparison.png", dpi=150)
    plt.close(fig)


def build_cases() -> list[CaseSpec]:
    cases = [
        CaseSpec("S0_constant", "560 A constant discharge"),
        CaseSpec("S1_step", "280 -> 560 (25 s) -> 1120 A (50 s)"),
    ]
    if Path(AGC_DATA_FILE).exists():
        cases.append(CaseSpec("T2_regd", "PJM RegD profile"))
    else:
        print(f"RegD data missing, skipping T2_regd ({AGC_DATA_FILE})")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Closed-loop do-mpc MPC vs fixed-speed baseline."
    )
    parser.add_argument("--duration-s", type=float, default=DURATION_S)
    parser.add_argument("--dt-s", type=float, default=DT_S)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "cluster_plant_v2/validation/results/domp_mpc_v1_20260826"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for case in build_cases():
        print(f"running {case.case_id}: {case.description}")
        frame_parts = []
        start = perf_counter()
        baseline = run_baseline(case, duration_s=args.duration_s, dt_s=args.dt_s)
        print(f"  baseline done in {perf_counter() - start:.1f} s")
        frame_parts.append(baseline)
        start = perf_counter()
        mpc_frame = run_mpc(case, duration_s=args.duration_s, dt_s=args.dt_s)
        print(f"  domp-mpc done in {perf_counter() - start:.1f} s")
        frame_parts.append(mpc_frame)
        frame = pd.concat(frame_parts, ignore_index=True)
        frame.to_csv(
            args.output_dir / f"case_{case.case_id}_timeseries.csv",
            index=False,
        )
        plot_case(case.case_id, frame, args.output_dir)
        summaries.append(summarize(case.case_id, baseline, args.dt_s))
        summaries.append(summarize(case.case_id, mpc_frame, args.dt_s))

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(args.output_dir / "domp_mpc_summary.csv", index=False)
    (args.output_dir / "domp_mpc_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    with pd.option_context("display.width", 200):
        print(summary_frame.to_string(index=False))
    print(f"results: {args.output_dir}")


if __name__ == "__main__":
    main()
