"""Stage 8D4i peak-shaving NMPC pass with a soft start (S0, 560 A).

Per the user's directive the run must start at the 25 C setpoint (no
hot or cold bias) with the compressor ramping up gently from the floor
instead of starting on a pre-chilled 4000 rpm equilibrium (8D4g's
start-up dip) or a 27 C hot start (8D4h).

This variant builds the plant locally with:

- battery / cold plate / tank initial temperature 25.0 C (the setpoint);
- compressor actuator initialized at the 1000 rpm floor (the cycle
  solver rejects rpm < 1000), so the evaporator starts near its
  minimum charge and the loop is not pre-chilled;
- initial compressor command at the floor, so the NMPC ramps capacity
  up only as fast as the temperature demands it.

No thermostat cycling overlay (same as 8D4g/8D4h), only the band-floor
safety net. Output names are distinct so 8D4h results stay intact.
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
    BASELINE_PUMP_RPM,
    build_cases,
    case_currents,
)
from cluster_plant_v2.validation.validate_final_cluster_plant import (
    FAN_SPEED_RPM,
)
from cluster_plant_v2.validation.validate_physics_p_nmpc import (
    KELVIN_OFFSET,
    _battery_avg_temp_c,
    _clip_command,
    initial_physics_state,
    physics_state_from_result,
)
from cluster_plant_v2.validation.validate_refrigerated_cluster_cooling_loop import (
    AMBIENT_TEMPERATURE_K,
    DT_S,
    INITIAL_SOC,
    PUMP_SPEED_RPM,
    build_engineering_pump,
    build_medium_header_network,
)
from cluster_plant_v2.thermal.cluster import ReducedCluster
from cluster_plant_v2.hydraulics import CoolantTank
from cluster_plant_v2.parameters import (
    COMPRESSOR_TIME_CONSTANT_S,
    EVAPORATOR_TIME_CONSTANT_S,
)
from cluster_plant_v2.plant import ClusterPlant
from cluster_plant_v2.refrigeration import ClosedR134aCycle, CompressorSpeedActuator

HERE = Path(__file__).resolve().parent
DT = 5.0
STEPS = 1280
DURATION = STEPS * DT
REF_C = 25.0
BAND_HALF = 0.65
PUMP_POWER_CORRECTION = 8.0
CKPT_EVERY = 50

# Soft-start initial condition: everything at the 25 C setpoint.
SOFT_START_TEMP_C = 25.0
SOFT_START_TEMP_K = SOFT_START_TEMP_C + KELVIN_OFFSET

# Band-floor safety net only (25 - 0.65 = 24.35 is the band floor).
SAFETY_OFF_TEMP_C = 24.35
SAFETY_ON_TEMP_C = 24.60

CKPT_CSV = HERE / "_8d4i_nmpc_ckpt.csv"
BASELINE_SRC = HERE / "case_S0_peak_8d4d_baseline_fixed_timeseries.csv"
NMPC_CSV = HERE / "case_S0_peak_8d4i_soft_start_timeseries.csv"
BASELINE_DST = HERE / "case_S0_peak_8d4i_baseline_fixed_timeseries.csv"
SUMMARY_CSV = HERE / "s0_peak_8d4i_summary.csv"

case = [c for c in build_cases() if c.case_id == "S0_constant"][0]
currents = case_currents(case, STEPS, DT)
parameters = PhysicsPNmpcParameters()
# The accurate SOC-aware cluster heat preview is a feedforward necessity
# (see the 8D4e driver notes); kept unchanged here.
heat_preview = ClusterHeatGenerationPreview(currents, DT)


def build_soft_start_plant() -> ClusterPlant:
    """Same topology as build_final_plant but starting at 25 C un-chilled."""
    network = build_medium_header_network()
    cluster = ReducedCluster(
        n_packs=5,
        hydraulic_mode="header_network",
        hydraulic_network=network,
        pack_config={
            "initial_soc": INITIAL_SOC,
            # ReducedBatteryPack reads "initial_temp_c" (degC); the legacy
            # kelvin keys in build_final_plant are silently ignored.
            "initial_temp_c": SOFT_START_TEMP_C,
        },
    )
    # ReducedCluster never forwards a plate temperature, so set it directly.
    for pack in cluster.packs:
        pack.cold_plate.plate_temperatures[:] = SOFT_START_TEMP_K
        pack.cold_plate.initial_zone_energy_J = (
            pack.cold_plate.zone_heat_capacities
            * pack.cold_plate.plate_temperatures
        )
        pack.cold_plate.coolant_outlet_temperature = float(
            pack.cold_plate.plate_temperatures[0]
        )
        pack.cold_plate.coolant_mean_temperatures = (
            pack.cold_plate.plate_temperatures.copy()
        )
    pump = build_engineering_pump(network)
    tank = CoolantTank(initial_temperature_k=SOFT_START_TEMP_K)
    cycle = ClosedR134aCycle()
    # The cycle solver requires >= 1000 rpm; the floor keeps the
    # evaporator near its minimum charge (no 4000 rpm pre-chill).
    actuator = CompressorSpeedActuator(
        initial_speed_rpm=parameters.compressor_plant_lower_rpm,
        time_constant_s=COMPRESSOR_TIME_CONSTANT_S,
    )
    return ClusterPlant.from_equilibrium(
        cluster=cluster,
        hydraulic_network=network,
        pump=pump,
        tank=tank,
        refrigeration_cycle=cycle,
        compressor_actuator=actuator,
        dt_s=DT_S,
        pump_speed_rpm=PUMP_SPEED_RPM,
        fan_speed_rpm=FAN_SPEED_RPM,
        initial_cluster_current_a=float(currents[0]),
        ambient_temperature_k=AMBIENT_TEMPERATURE_K,
        direction="forward",
        evaporator_time_constant_s=EVAPORATOR_TIME_CONSTANT_S,
    )


plant = build_soft_start_plant()
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
comp_command = parameters.compressor_plant_lower_rpm
pump_command = BASELINE_PUMP_RPM
comp_running = True
safety_events = 0
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
        SOFT_START_TEMP_C
        if result is None
        else _battery_avg_temp_c(result["cluster_result"])
    )
    # Band-floor safety net only: no thermostat cycling overlay.
    if comp_running and battery_temp_c < SAFETY_OFF_TEMP_C:
        comp_running = False
        safety_events += 1
        print(f"  safety net fired at t={step_index * DT:.0f} s", flush=True)
    if not comp_running:
        applied_comp_command = 0.0
        pump_virtual_rpm = parameters.pump_upper_rpm
        pump_command = pump_virtual_rpm / PUMP_VIRTUAL_SPEED_SCALE
        if battery_temp_c > SAFETY_ON_TEMP_C:
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
            controller="physics_p_nmpc_8d4i_soft_start",
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
    "nmpc_8d4i_soft_start": nmpc.sort_values("time_s"),
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
    f"start/stop switches: {cycle_switches}, off time {off_steps * DT:.0f} s, "
    f"safety-net events: {safety_events}",
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
summary["scenario"] = "S0_peak_6400s_8d4i"
summary.to_csv(SUMMARY_CSV, index=False)
print(summary.to_string(index=False), flush=True)
print("done", flush=True)
