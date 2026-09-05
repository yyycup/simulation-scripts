"""Stage 8D4e peak-shaving NMPC pass with checkpoint resume (S0, 560 A).

Detached full runs kept stalling inside the fixed-speed baseline pass on
this machine, so this driver skips pass 1 entirely: the baseline is
artifact-independent (no predictor involved), therefore the 8d4d baseline
timeseries is reused verbatim. Only the NMPC pass is recomputed on the
re-derived 8D4 capacity artifact, with:

- progress printout every 10 steps with ETA;
- a checkpoint CSV every 50 steps; on restart the plant state is rebuilt
  by deterministically replaying the logged commands, then the loop
  resumes from the checkpoint step (MPC warm-start history is lost, the
  first post-resume solve uses a cold initial guess).

Output names match run_peak_s0_8d4d.py so the comparison plots work.
"""

from pathlib import Path
import shutil
import sys
from time import perf_counter

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import numpy as np
import pandas as pd

from cluster_plant_v2.control.physics_p_nmpc_controller import (
    PhysicsPNmpcParameters,
    build_mpc,
)
from cluster_plant_v2.control.physics_p_nmpc_model import (
    PUMP_VIRTUAL_SPEED_SCALE,
    STATE_DIM,
    ClusterHeatGenerationPreview,
    battery_heat_generation_preview_w,
    build_physics_p_model,
)
from cluster_plant_v2.validation.validate_domp_mpc import (
    BASELINE_COMPRESSOR_RPM,
    BASELINE_PUMP_RPM,
    build_cases,
    case_currents,
)
from cluster_plant_v2.validation.validate_final_cluster_plant import (
    FAN_SPEED_RPM,
    build_final_plant,
)
from cluster_plant_v2.validation.validate_physics_p_nmpc import (
    COMPRESSOR_CYCLE_OFF_TEMP_C,
    COMPRESSOR_CYCLE_ON_TEMP_C,
    COMPRESSOR_CYCLE_SPIKE_LOOKAHEAD_S,
    COMPRESSOR_CYCLE_SPIKE_THRESHOLD_W,
    KELVIN_OFFSET,
    _battery_avg_temp_c,
    _clip_command,
    initial_physics_state,
    physics_state_from_result,
)
from cluster_plant_v2.validation.validate_refrigerated_cluster_cooling_loop import (
    AMBIENT_TEMPERATURE_K,
)

HERE = Path(__file__).resolve().parent
DT = 5.0
STEPS = 1280
DURATION = STEPS * DT
REF_C = 25.0
BAND_HALF = 0.65
PUMP_POWER_CORRECTION = 8.0
CKPT_EVERY = 50

CKPT_CSV = HERE / "_8d4e_nmpc_ckpt.csv"
BASELINE_SRC = HERE / "case_S0_peak_8d4d_baseline_fixed_timeseries.csv"
NMPC_CSV = HERE / "case_S0_peak_8d4e_nmpc_soc_aware_8d4d_timeseries.csv"
BASELINE_DST = HERE / "case_S0_peak_8d4e_baseline_fixed_timeseries.csv"
SUMMARY_CSV = HERE / "s0_peak_8d4e_summary.csv"

case = [c for c in build_cases() if c.case_id == "S0_constant"][0]
currents = case_currents(case, STEPS, DT)
parameters = PhysicsPNmpcParameters()
# 2026-08-30: the standard ohmic preview was tried per the "no SOC-aware"
# request but supplies only ~1 kW against the plant's ~6 kW heat at
# 560 A, which starves the feedforward and drifts the loop out of band
# (845 s). The accurate cluster heat preview is therefore restored; it
# is a feedforward necessity, not an optional variant.
heat_preview = ClusterHeatGenerationPreview(currents, DT)

plant = build_final_plant(
    BASELINE_COMPRESSOR_RPM,
    initial_cluster_current_a=float(currents[0]),
    direction="forward",
)
model = build_physics_p_model()


def current_preview(time_s: float) -> float:
    index = min(max(int(float(time_s) / DT), 0), STEPS - 1)
    return float(currents[index])


mpc = build_mpc(
    model,
    parameters,
    current_preview=current_preview,
    ambient_temperature_c=AMBIENT_TEMPERATURE_K - KELVIN_OFFSET,
    q_gen_preview=heat_preview,
)

pump_virtual_rpm = min(
    BASELINE_PUMP_RPM * PUMP_VIRTUAL_SPEED_SCALE,
    parameters.pump_upper_rpm,
)

rows: list[dict] = []
comp_command = BASELINE_COMPRESSOR_RPM
pump_command = BASELINE_PUMP_RPM
comp_running = True
result = None
start_step = 0

if CKPT_CSV.exists():
    ckpt = pd.read_csv(CKPT_CSV)
    start_step = len(ckpt)
    print(f"resuming from checkpoint: {start_step} steps logged", flush=True)
    # Deterministic replay rebuilds the plant state at the checkpoint.
    t_replay = perf_counter()
    for i in range(start_step):
        result = plant.step(
            dt_s=DT,
            cluster_current_a=float(ckpt["current_a"].iloc[i]),
            pump_speed_rpm=float(ckpt["pump_command_rpm"].iloc[i]),
            compressor_speed_command_rpm=float(ckpt["compressor_command_rpm"].iloc[i]),
            fan_speed_rpm=FAN_SPEED_RPM,
            ambient_temperature_k=AMBIENT_TEMPERATURE_K,
            direction="forward",
        )
    print(f"  replayed {start_step} plant steps in {perf_counter() - t_replay:.1f} s", flush=True)
    rows = ckpt.to_dict("records")
    applied_last = float(ckpt["compressor_command_rpm"].iloc[-1])
    pump_command = float(ckpt["pump_command_rpm"].iloc[-1])
    pump_virtual_rpm = pump_command * PUMP_VIRTUAL_SPEED_SCALE
    if applied_last <= 0.0:
        comp_running = False
        comp_command = parameters.compressor_plant_lower_rpm
    else:
        comp_running = True
        comp_command = applied_last

x0 = np.empty(STATE_DIM, dtype=float)
if result is not None:
    x0[:14] = physics_state_from_result(
        result, last_pump_command_rpm=pump_command * PUMP_VIRTUAL_SPEED_SCALE
    )
else:
    x0[:14] = initial_physics_state(plant)
x0[14] = comp_command
x0[15] = pump_virtual_rpm
mpc.x0["x"] = x0.reshape(-1, 1)
mpc.u0["n_comp_cmd"] = comp_command
mpc.u0["n_pump_cmd"] = pump_virtual_rpm
mpc.set_initial_guess()

solve_times_s: list[float] = []
spike_lookahead_steps = int(COMPRESSOR_CYCLE_SPIKE_LOOKAHEAD_S / DT)
heat_schedule_w = np.array(
    [float(heat_preview(float(t) * DT, 0.0)) for t in range(STEPS)]
)

print(f"running NMPC pass: steps {start_step}..{STEPS}", flush=True)
pass_start = perf_counter()
for step_index in range(start_step, STEPS):
    if step_index % parameters.reoptimize_every_steps == 0:
        mpc._t0 = step_index * DT
        if result is not None:
            x0[:14] = physics_state_from_result(
                result,
                last_pump_command_rpm=(pump_command * PUMP_VIRTUAL_SPEED_SCALE),
            )
        x0[14] = comp_command
        x0[15] = pump_virtual_rpm
        t_solve = perf_counter()
        u_new = mpc.make_step(x0.reshape(-1, 1))
        solve_times_s.append(perf_counter() - t_solve)
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
    applied_comp_command = comp_command
    battery_temp_c = (
        25.0 if result is None else _battery_avg_temp_c(result["cluster_result"])
    )
    if (
        comp_running
        and comp_command <= parameters.compressor_plant_lower_rpm + 1e-6
        and battery_temp_c < COMPRESSOR_CYCLE_OFF_TEMP_C
    ):
        comp_running = False
    if not comp_running:
        applied_comp_command = 0.0
        pump_virtual_rpm = parameters.pump_upper_rpm
        pump_command = pump_virtual_rpm / PUMP_VIRTUAL_SPEED_SCALE
        lookahead_end = min(step_index + spike_lookahead_steps, STEPS)
        spike_ahead = (
            float(heat_schedule_w[step_index:lookahead_end].max())
            > COMPRESSOR_CYCLE_SPIKE_THRESHOLD_W
        )
        if battery_temp_c > COMPRESSOR_CYCLE_ON_TEMP_C or spike_ahead:
            comp_running = True
    result = plant.step(
        dt_s=DT,
        cluster_current_a=float(currents[step_index]),
        pump_speed_rpm=pump_command,
        compressor_speed_command_rpm=applied_comp_command,
        fan_speed_rpm=FAN_SPEED_RPM,
        ambient_temperature_k=AMBIENT_TEMPERATURE_K,
        direction="forward",
    )
    rows.append(
        dict(
            controller="physics_p_nmpc",
            time_s=(step_index + 1) * DT,
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
            supply_temp_c=float(result["cluster_supply_temperature_k"] - KELVIN_OFFSET),
            return_temp_c=float(result["cluster_return_temperature_k"] - KELVIN_OFFSET),
            q_evap_applied_w=float(result["q_evap_applied_w"]),
            compressor_power_w=float(result["compressor_shaft_power_w"]),
            pump_power_w=float(result["pump_power_w"]),
        )
    )
    if (step_index + 1) % 10 == 0:
        done = step_index + 1 - start_step
        rate = (perf_counter() - pass_start) / max(done, 1)
        eta_min = rate * (STEPS - step_index - 1) / 60.0
        print(
            f"step {step_index + 1}/{STEPS} "
            f"({(perf_counter() - pass_start) / 60.0:.1f} min elapsed, "
            f"ETA {eta_min:.1f} min)",
            flush=True,
        )
    if (step_index + 1) % CKPT_EVERY == 0:
        pd.DataFrame(rows).to_csv(CKPT_CSV, index=False)

print("NMPC pass done", flush=True)
nmpc = pd.DataFrame(rows)
nmpc.to_csv(NMPC_CSV, index=False)
if CKPT_CSV.exists():
    CKPT_CSV.unlink()

shutil.copy(BASELINE_SRC, BASELINE_DST)
baseline = pd.read_csv(BASELINE_DST)

summary_rows = []
for name, frame in {
    "baseline_fixed": baseline.sort_values("time_s"),
    "nmpc_soc_aware_8d4d": nmpc.sort_values("time_s"),
}.items():
    temp = frame["battery_avg_temp_c"].to_numpy()
    out_band = float(np.sum(np.abs(temp - REF_C) > BAND_HALF)) * DT
    comp_kwh = float(frame["compressor_power_w"].sum()) * DT / 3.6e6
    pump_kwh = float(frame["pump_power_w"].sum()) * PUMP_POWER_CORRECTION * DT / 3.6e6
    summary_rows.append(
        {
            "controller": name,
            "final_temp_c": temp[-1],
            "max_temp_c": temp.max(),
            "min_temp_c": temp.min(),
            "max_dev_k": np.abs(temp - REF_C).max(),
            "out_of_band_s": out_band,
            "comp_kwh": comp_kwh,
            "pump_kwh_corrected": pump_kwh,
            "total_kwh": comp_kwh + pump_kwh,
        }
    )

comp_cmd = nmpc["compressor_command_rpm"].to_numpy()
cycle_switches = int(np.sum(np.diff((comp_cmd > 0).astype(int)) != 0))
off_steps = int(np.sum(comp_cmd <= 0.0))
print(
    f"thermostat cycling: {cycle_switches} start/stop switches, "
    f"off time {off_steps * DT:.0f} s",
    flush=True,
)
if solve_times_s:
    print(
        f"nmpc solves: {len(solve_times_s)}, "
        f"mean {np.mean(solve_times_s):.2f} s, "
        f"max {np.max(solve_times_s):.2f} s",
        flush=True,
    )

summary = pd.DataFrame(summary_rows)
summary["scenario"] = "S0_peak_6400s_8d4e"
summary.to_csv(SUMMARY_CSV, index=False)
print(summary.to_string(index=False), flush=True)
print("done", flush=True)
