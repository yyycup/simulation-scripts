"""Probe: sequential sweep heat anchors for the 8D4f fit.

Replicates the _probe_capacity_vs_plant_sweep protocol exactly (same
sequential settle chain, same commands, 20 min per point) while also
recording the plant's measured per-pack heat generation at the end of
each point. The fit's heat anchors are then consistent with the sweep
battery temperatures. Kept for reference.
"""

from pathlib import Path
import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import numpy as np

from cluster_plant_v2.control.physics_p_nmpc_model import PUMP_VIRTUAL_SPEED_SCALE
from cluster_plant_v2.validation.validate_domp_mpc import BASELINE_PUMP_RPM
from cluster_plant_v2.validation.validate_final_cluster_plant import (
    FAN_SPEED_RPM,
    build_final_plant,
)
from cluster_plant_v2.validation.validate_refrigerated_cluster_cooling_loop import (
    AMBIENT_TEMPERATURE_K,
)

HERE = Path(__file__).resolve().parent
DT = 5.0
SETTLE_STEPS = 240
SWEEP_RPM = (1000, 1300, 1600, 1900, 2200, 2500)

plant = build_final_plant(
    4000,
    initial_cluster_current_a=560.0,
    direction="forward",
)

rows = []
for rpm in SWEEP_RPM:
    last = None
    for i in range(SETTLE_STEPS):
        last = plant.step(
            dt_s=DT,
            cluster_current_a=560.0,
            pump_speed_rpm=BASELINE_PUMP_RPM,
            compressor_speed_command_rpm=float(rpm),
            fan_speed_rpm=FAN_SPEED_RPM,
            ambient_temperature_k=AMBIENT_TEMPERATURE_K,
            direction="forward",
        )
    qgen_pp = float(last["cluster_q_gen_total_w"]) / 5.0
    rows.append(
        (
            rpm,
            float(last["q_evap_applied_w"]),
            float(last["tank_temperature_after_k"] - 273.15),
            float(last["cluster_supply_temperature_k"] - 273.15),
            float(
                np.mean(last["cluster_result"]["pack_battery_average_temperatures_k"])
                - 273.15
            ),
            qgen_pp,
        )
    )
    print(f"sweep {rpm} rpm done, qgen {qgen_pp:.0f} W/pack", flush=True)

print("cmd   q_evap   tank   supply  bat    qgen_pp")
for r in rows:
    print(
        f"{r[0]:5.0f}  {r[1]:7.0f}  {r[2]:5.2f}  {r[3]:5.2f}  {r[4]:5.2f}  {r[5]:6.0f}"
    )

out = HERE / "_probe_sweep_heat_anchors.csv"
np.savetxt(
    out,
    np.array(rows),
    delimiter=",",
    header="cmd_rpm,q_evap_plant_w,tank_c,supply_c,bat_c,qgen_per_pack_w",
    comments="",
)
print("saved", out)
