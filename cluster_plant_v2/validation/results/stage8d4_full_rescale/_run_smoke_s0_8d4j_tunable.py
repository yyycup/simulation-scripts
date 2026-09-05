"""Stage 8D4j closed-loop smoke: OnlineTunableNmpc facade on S0 (600 s).

Runs the soft-start plant under the horizon-bank facade and performs two
online interventions mid-loop:
- step 60: weight retune (w_comp 0.1 -> 0.2) with zero CasADi rebuild;
- step 60: horizon switch 60 -> 90 steps with state carry-over.

Before the intervention the trajectory must reproduce the finalized 8D4i
pass (same default weights, horizon 60); after it the loop must stay
finite and in band. Output names are distinct so 8D4i results stay intact.
"""

from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import numpy as np
import pandas as pd

from cluster_plant_v2.control.physics_p_nmpc_controller import (
    OnlineTunableNmpc,
    PhysicsPNmpcParameters,
)
from cluster_plant_v2.control.physics_p_nmpc_model import (
    PUMP_VIRTUAL_SPEED_SCALE,
    STATE_DIM,
    ClusterHeatGenerationPreview,
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
STEPS = 120  # 600 s smoke
REF_C = 25.0
BAND_HALF = 0.65
PUMP_POWER_CORRECTION = 8.0
INTERVENE_AT = 60

SOFT_START_TEMP_C = 25.0
SOFT_START_TEMP_K = SOFT_START_TEMP_C + KELVIN_OFFSET

TIMESERIES_CSV = HERE / "case_S0_smoke_8d4j_tunable_timeseries.csv"
SUMMARY_CSV = HERE / "s0_smoke_8d4j_tunable_summary.csv"
REF_CSV = HERE / "case_S0_peak_8d4i_soft_start_timeseries.csv"

case = [c for c in build_cases() if c.case_id == "S0_constant"][0]
currents = case_currents(case, STEPS, DT)
parameters = PhysicsPNmpcParameters()
heat_preview = ClusterHeatGenerationPreview(currents, DT)


def build_soft_start_plant() -> ClusterPlant:
    """Identical soft-start construction to the 8D4i driver."""
    network = build_medium_header_network()
    cluster = ReducedCluster(
        n_packs=5,
        hydraulic_mode="header_network",
        hydraulic_network=network,
        pack_config={"initial_soc": INITIAL_SOC, "initial_temp_c": SOFT_START_TEMP_C},
    )
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


def current_preview(time_s: float) -> float:
    index = min(max(int(float(time_s) / DT), 0), STEPS - 1)
    return float(currents[index])


t_build = perf_counter()
controller = OnlineTunableNmpc(
    parameters,
    horizon_bank=(30, 60, 90),
    current_preview=current_preview,
    ambient_temperature_c=AMBIENT_TEMPERATURE_K - KELVIN_OFFSET,
    q_gen_preview=heat_preview,
)
print(f"horizon bank build: {perf_counter() - t_build:.1f} s", flush=True)

pump_virtual_rpm = min(
    BASELINE_PUMP_RPM * PUMP_VIRTUAL_SPEED_SCALE,
    parameters.pump_upper_rpm,
)
comp_command = parameters.compressor_plant_lower_rpm

x0 = np.empty(STATE_DIM, dtype=float)
x0[:14] = initial_physics_state(plant)
x0[14] = comp_command
x0[15] = pump_virtual_rpm
controller.prepare(x0, comp_command, pump_virtual_rpm)

rows: list[dict] = []
solve_times_s: list[float] = []
result = None

print(f"running tunable smoke pass: {STEPS} steps", flush=True)
pass_start = perf_counter()
for step_index in range(STEPS):
    if step_index % parameters.reoptimize_every_steps == 0:
        controller.align_time(step_index * DT)
        if step_index == INTERVENE_AT:
            # Online interventions: zero-recompile weight retune plus a
            # horizon switch, both through the facade API.
            controller.set_weights(w_comp=0.2)
            switched = controller.switch_horizon(90)
            print(
                f"  intervention at t={step_index * DT:.0f} s: "
                f"w_comp->0.2, horizon->90 (switched={switched})",
                flush=True,
            )
        if result is not None:
            x0[:14] = physics_state_from_result(
                result, last_pump_command_rpm=pump_command * PUMP_VIRTUAL_SPEED_SCALE
            )
        x0[14] = comp_command
        x0[15] = pump_virtual_rpm
        t_solve = perf_counter()
        u_new = controller.make_step(x0.reshape(-1, 1))
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
    result = plant.step(
        dt_s=DT,
        cluster_current_a=float(currents[step_index]),
        pump_speed_rpm=pump_command,
        compressor_speed_command_rpm=comp_command,
        fan_speed_rpm=FAN_SPEED_RPM,
        ambient_temperature_k=AMBIENT_TEMPERATURE_K,
        direction="forward",
    )
    rows.append(
        dict(
            controller="physics_p_nmpc_8d4j_tunable_smoke",
            time_s=(step_index + 1) * DT,
            horizon_steps=controller.horizon_steps,
            w_comp=controller.weights["w_comp"],
            compressor_command_rpm=comp_command,
            pump_command_rpm=pump_command,
            battery_avg_temp_c=float(
                np.mean(result["cluster_result"]["pack_battery_average_temperatures_k"])
                - KELVIN_OFFSET
            ),
            tank_temp_c=float(result["tank_temperature_after_k"] - KELVIN_OFFSET),
            q_evap_applied_w=float(result["q_evap_applied_w"]),
            compressor_power_w=float(result["compressor_shaft_power_w"]),
            pump_power_w=float(result["pump_power_w"]),
        )
    )
    if (step_index + 1) % 30 == 0:
        print(
            f"step {step_index + 1}/{STEPS} "
            f"({(perf_counter() - pass_start) / 60.0:.1f} min elapsed)",
            flush=True,
        )

frame = pd.DataFrame(rows)
frame.to_csv(TIMESERIES_CSV, index=False)

# 1. pre-intervention trajectory must reproduce the finalized 8D4i pass
reference = pd.read_csv(REF_CSV)
pre = frame[frame["time_s"] <= INTERVENE_AT * DT]
ref = reference[reference["time_s"] <= INTERVENE_AT * DT]
temp_dev = float(np.max(np.abs(pre["battery_avg_temp_c"].to_numpy()
                               - ref["battery_avg_temp_c"].to_numpy())))
cmd_dev = float(np.max(np.abs(pre["compressor_command_rpm"].to_numpy()
                              - ref["compressor_command_rpm"].to_numpy())))
print(f"pre-intervention vs 8D4i: max temp dev {temp_dev:.3e} K, "
      f"max comp cmd dev {cmd_dev:.3e} rpm", flush=True)

# 2. post-intervention sanity: finite, in band, horizon/weight recorded
post = frame[frame["time_s"] > INTERVENE_AT * DT]
assert np.all(np.isfinite(post.drop(columns=["controller"]).to_numpy()))
assert post["horizon_steps"].eq(90).all()
assert np.allclose(post["w_comp"], 0.2)
max_dev = float(np.max(np.abs(frame["battery_avg_temp_c"] - REF_C)))
print(f"whole-run max deviation: {max_dev:.4f} K", flush=True)
assert max_dev < BAND_HALF, "smoke run must stay inside the band"

comp_kwh = float(frame["compressor_power_w"].sum()) * DT / 3.6e6
pump_kwh = float(frame["pump_power_w"].sum()) * PUMP_POWER_CORRECTION * DT / 3.6e6
summary = pd.DataFrame(
    [
        {
            "controller": "physics_p_nmpc_8d4j_tunable_smoke",
            "steps": STEPS,
            "max_dev_k": max_dev,
            "pre_intervention_temp_dev_vs_8d4i_k": temp_dev,
            "pre_intervention_cmd_dev_vs_8d4i_rpm": cmd_dev,
            "comp_kwh": comp_kwh,
            "pump_kwh_corrected": pump_kwh,
            "total_kwh": comp_kwh + pump_kwh,
            "mean_solve_s": float(np.mean(solve_times_s)),
            "scenario": "S0_smoke_600s_8d4j",
        }
    ]
)
summary.to_csv(SUMMARY_CSV, index=False)
print(summary.to_string(index=False), flush=True)
print("done", flush=True)
