"""Closed-loop validation of the frozen Physics-P discrete NMPC.

The full PyBaMM-backed cluster plant is the Plant; the frozen 14-state
Physics-P predictor is only the internal NMPC model (no re-identification,
no modification). do-mpc builds the NLP over the augmented 16-state discrete
model and IPOPT solves it; no linearization, QP, or GEKKO is involved.

Protocol:
- Plant step 5 s, horizon 60 steps (300 s), re-optimize every 15 s and hold
  the command for the two intermediate steps;
- perfect current preview ``I[k:k+N]`` feeds the TVP;
- compared against the frozen fixed-speed baseline (4000 / 3600 rpm).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from cluster_plant_v2.control.physics_p_nmpc_controller import (
    PhysicsPNmpcParameters,
    build_mpc,
)
from cluster_plant_v2.control.physics_p_nmpc_model import (
    PUMP_VIRTUAL_SPEED_SCALE,
    STATE_DIM,
    battery_heat_generation_preview_w,
    build_physics_p_model,
)
from cluster_plant_v2.validation.validate_domp_mpc import (
    BASELINE_COMPRESSOR_RPM,
    BASELINE_PUMP_RPM,
    CaseSpec,
    build_cases,
    case_currents,
    plot_case,
    run_baseline,
    summarize,
)
from cluster_plant_v2.validation.validate_final_cluster_plant import (
    DURATION_S,
    FAN_SPEED_RPM,
    build_final_plant,
)
from cluster_plant_v2.validation.validate_refrigerated_cluster_cooling_loop import (
    AMBIENT_TEMPERATURE_K,
    DT_S,
)

KELVIN_OFFSET = 273.15

# Stage 8D4d: thermostat cycling of the oversized chiller at low load.
# The 72 cc compressor floor capacity (~4.1-4.5 kW) exceeds the RegD
# valley loads, so continuous modulation cannot hold the lower band
# (8D4c ran with both actuators saturated and still drifted to 20.6 C).
# The supervisor cuts the compressor once the NMPC has exhausted its
# cooling authority (command at the floor) and the battery sags toward
# the band center, and restarts it when the battery recovers or an
# upcoming RegD spike needs the chiller ready.
COMPRESSOR_CYCLE_OFF_TEMP_C = 24.85
COMPRESSOR_CYCLE_ON_TEMP_C = 25.15
COMPRESSOR_CYCLE_SPIKE_LOOKAHEAD_S = 300.0
COMPRESSOR_CYCLE_SPIKE_THRESHOLD_W = 2500.0


def _battery_avg_temp_c(cluster_result) -> float:
    return float(
        np.mean(cluster_result["pack_battery_average_temperatures_k"])
        - KELVIN_OFFSET
    )


def _plate_representative_temp_c(cluster_result) -> float:
    return float(np.mean(cluster_result["pack_plate_temperatures_k"]) - KELVIN_OFFSET)


def initial_physics_state(plant) -> np.ndarray:
    """Map the zero-step Plant to the 14 Physics-P states (degC / rpm / W)."""
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
    return np.array(
        [
            float(np.mean(battery_averages) - KELVIN_OFFSET),
            float(plant.tank.temperature_k - KELVIN_OFFSET),
            float(np.mean(plate_temperatures) - KELVIN_OFFSET),
            float(plant.compressor_actuator.speed_rpm),
            # Stage 8D4c: the model lives in the virtual legacy-pump domain
            # (see PUMP_VIRTUAL_SPEED_SCALE); the plant starts at
            # BASELINE_PUMP_RPM on the rescaled pump.
            min(
                float(BASELINE_PUMP_RPM) * PUMP_VIRTUAL_SPEED_SCALE,
                4800.0,
            ),
            float(plant.evaporator_dynamics.q_evap_applied_w),
            min(
                float(BASELINE_PUMP_RPM) * PUMP_VIRTUAL_SPEED_SCALE,
                4800.0,
            ),
            *(np.asarray(plant.supply_delay.queue_values) - KELVIN_OFFSET),
            *(np.asarray(plant.return_delay.queue_values) - KELVIN_OFFSET),
        ],
        dtype=float,
    )


def physics_state_from_result(
    result, *, last_pump_command_rpm: float
) -> np.ndarray:
    """Map one Plant step output to the 14 Physics-P states."""
    cluster = result["cluster_result"]
    return np.array(
        [
            _battery_avg_temp_c(cluster),
            float(result["tank_temperature_after_k"] - KELVIN_OFFSET),
            _plate_representative_temp_c(cluster),
            float(result["compressor_speed_rpm"]),
            float(last_pump_command_rpm),
            float(result["q_evap_applied_w"]),
            float(last_pump_command_rpm),
            *(np.asarray(result["supply_delay_queue_k"]) - KELVIN_OFFSET),
            *(np.asarray(result["return_delay_queue_k"]) - KELVIN_OFFSET),
        ],
        dtype=float,
    )


def _clip_command(
    command: float, previous: float, dmax: float, lower: float, upper: float
) -> float:
    """Defensive application-layer guard; the hard bound lives in the NLP."""
    clipped = float(previous) + min(
        max(float(command) - float(previous), -float(dmax)), float(dmax)
    )
    return min(max(clipped, float(lower)), float(upper))


def run_nmpc(
    case: CaseSpec,
    *,
    duration_s: float,
    dt_s: float,
    parameters: PhysicsPNmpcParameters | None = None,
    heat_preview=None,
    compressor_thermostat_cycling: bool = False,
) -> pd.DataFrame:
    if parameters is None:
        parameters = PhysicsPNmpcParameters()
    steps = int(round(duration_s / dt_s))
    currents = case_currents(case, steps, dt_s)
    plant = build_final_plant(
        BASELINE_COMPRESSOR_RPM,
        initial_cluster_current_a=float(currents[0]),
        direction="forward",
    )

    model = build_physics_p_model()

    def current_preview(time_s: float) -> float:
        index = min(max(int(float(time_s) / dt_s), 0), steps - 1)
        return float(currents[index])

    if heat_preview is None:
        heat_preview = (
            lambda preview_time_s, current_a: battery_heat_generation_preview_w(
                current_a
            )
        )
    mpc = build_mpc(
        model,
        parameters,
        current_preview=current_preview,
        ambient_temperature_c=AMBIENT_TEMPERATURE_K - KELVIN_OFFSET,
        q_gen_preview=heat_preview,
    )

    # Stage 8D4c: the NMPC decides the pump input in the virtual
    # legacy-pump domain; the plant receives the rescaled-domain command
    # (virtual / PUMP_VIRTUAL_SPEED_SCALE).
    pump_virtual_rpm = min(
        BASELINE_PUMP_RPM * PUMP_VIRTUAL_SPEED_SCALE,
        parameters.pump_upper_rpm,
    )

    x0 = np.empty(STATE_DIM, dtype=float)
    x0[:14] = initial_physics_state(plant)
    x0[14] = BASELINE_COMPRESSOR_RPM
    x0[15] = pump_virtual_rpm
    mpc.x0["x"] = x0.reshape(-1, 1)
    mpc.u0["n_comp_cmd"] = BASELINE_COMPRESSOR_RPM
    mpc.u0["n_pump_cmd"] = pump_virtual_rpm
    mpc.set_initial_guess()

    comp_command = BASELINE_COMPRESSOR_RPM
    pump_command = BASELINE_PUMP_RPM
    result = None
    rows = []
    solve_times_s = []
    comp_running = True
    spike_lookahead_steps = int(COMPRESSOR_CYCLE_SPIKE_LOOKAHEAD_S / dt_s)
    if compressor_thermostat_cycling:
        heat_schedule_w = np.array(
            [float(heat_preview(float(t) * dt_s, 0.0)) for t in range(steps)]
        )
    for step_index in range(steps):
        if step_index % parameters.reoptimize_every_steps == 0:
            # Re-align the do-mpc clock: it only advances on make_step.
            mpc._t0 = step_index * dt_s
            if result is not None:
                x0[:14] = physics_state_from_result(
                    result,
                    last_pump_command_rpm=(
                        pump_command * PUMP_VIRTUAL_SPEED_SCALE
                    ),
                )
            x0[14] = comp_command
            x0[15] = pump_virtual_rpm
            start = perf_counter()
            u_new = mpc.make_step(x0.reshape(-1, 1))
            solve_times_s.append(perf_counter() - start)
            comp_command = _clip_command(
                float(u_new[0, 0]),
                comp_command,
                parameters.compressor_dmax_rpm,
                parameters.compressor_plant_lower_rpm,
                parameters.compressor_upper_rpm,
            )
            pump_virtual_rpm = _clip_command(
                float(u_new[1, 0]),
                pump_virtual_rpm,
                parameters.pump_dmax_rpm,
                parameters.pump_lower_rpm,
                parameters.pump_upper_rpm,
            )
            pump_command = pump_virtual_rpm / PUMP_VIRTUAL_SPEED_SCALE
        # Stage 8D4d thermostat cycling: only act when the NMPC itself has
        # exhausted its cooling authority (command at the plant floor); a
        # deliberate pre-cool (command above floor) is never overridden.
        applied_comp_command = comp_command
        if compressor_thermostat_cycling:
            battery_temp_c = (
                25.0
                if result is None
                else _battery_avg_temp_c(result["cluster_result"])
            )
            if (
                comp_running
                and comp_command
                <= parameters.compressor_plant_lower_rpm + 1e-6
                and battery_temp_c < COMPRESSOR_CYCLE_OFF_TEMP_C
            ):
                comp_running = False
            if not comp_running:
                applied_comp_command = 0.0
                # Park the pump at its ceiling: the warm battery return
                # is the only heat source left to recover the tank.
                pump_virtual_rpm = parameters.pump_upper_rpm
                pump_command = pump_virtual_rpm / PUMP_VIRTUAL_SPEED_SCALE
                lookahead_end = min(
                    step_index + spike_lookahead_steps, steps
                )
                spike_ahead = (
                    float(heat_schedule_w[step_index:lookahead_end].max())
                    > COMPRESSOR_CYCLE_SPIKE_THRESHOLD_W
                )
                if (
                    battery_temp_c > COMPRESSOR_CYCLE_ON_TEMP_C
                    or spike_ahead
                ):
                    comp_running = True
        result = plant.step(
            dt_s=dt_s,
            cluster_current_a=float(currents[step_index]),
            pump_speed_rpm=pump_command,
            compressor_speed_command_rpm=applied_comp_command,
            fan_speed_rpm=FAN_SPEED_RPM,
            ambient_temperature_k=AMBIENT_TEMPERATURE_K,
            direction="forward",
        )
        row = dict(
            controller="physics_p_nmpc",
            time_s=(step_index + 1) * dt_s,
            current_a=float(currents[step_index]),
            compressor_command_rpm=applied_comp_command,
            compressor_actual_rpm=float(result["compressor_speed_rpm"]),
            pump_command_rpm=pump_command,
            battery_avg_temp_c=float(
                np.mean(result["cluster_result"]["pack_battery_average_temperatures_k"])
                - KELVIN_OFFSET
            ),
            battery_max_temp_c=float(
                result["cluster_result"]["cluster_max_temperature_k"] - KELVIN_OFFSET
            ),
            tank_temp_c=float(result["tank_temperature_after_k"] - KELVIN_OFFSET),
            supply_temp_c=float(
                result["cluster_supply_temperature_k"] - KELVIN_OFFSET
            ),
            return_temp_c=float(
                result["cluster_return_temperature_k"] - KELVIN_OFFSET
            ),
            q_evap_applied_w=float(result["q_evap_applied_w"]),
            compressor_power_w=float(result["compressor_shaft_power_w"]),
            pump_power_w=float(result["pump_power_w"]),
        )
        rows.append(row)
    print(
        f"  nmpc solves: {len(solve_times_s)}, "
        f"mean {np.mean(solve_times_s):.2f} s, "
        f"max {np.max(solve_times_s):.2f} s"
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Closed-loop frozen Physics-P NMPC vs fixed baseline."
    )
    parser.add_argument("--duration-s", type=float, default=DURATION_S)
    parser.add_argument("--dt-s", type=float, default=DT_S)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "cluster_plant_v2/validation/results/physics_p_nmpc_20260826"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for case in build_cases():
        print(f"running {case.case_id}: {case.description}")
        start = perf_counter()
        baseline = run_baseline(case, duration_s=args.duration_s, dt_s=args.dt_s)
        print(f"  baseline done in {perf_counter() - start:.1f} s")
        start = perf_counter()
        nmpc_frame = run_nmpc(case, duration_s=args.duration_s, dt_s=args.dt_s)
        print(f"  physics-p nmpc done in {perf_counter() - start:.1f} s")
        frame = pd.concat([baseline, nmpc_frame], ignore_index=True)
        frame.to_csv(
            args.output_dir / f"case_{case.case_id}_timeseries.csv",
            index=False,
        )
        plot_case(case.case_id, frame, args.output_dir)
        summaries.append(summarize(case.case_id, baseline, args.dt_s))
        summaries.append(summarize(case.case_id, nmpc_frame, args.dt_s))

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(
        args.output_dir / "physics_p_nmpc_summary.csv", index=False
    )
    (args.output_dir / "physics_p_nmpc_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    with pd.option_context("display.width", 200):
        print(summary_frame.to_string(index=False))
    print(f"results: {args.output_dir}")


if __name__ == "__main__":
    main()
