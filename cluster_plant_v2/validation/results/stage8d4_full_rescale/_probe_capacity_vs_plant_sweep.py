"""Probe: plant steady cooling capacity vs the 8D4e capacity map.

Sweeps compressor commands on the full plant at fixed pump speed, lets
each point settle, and compares the achieved steady q_evap and supply
temperature against the artifact capacity map at the same inputs. This
isolates whether the residual NMPC offset comes from the capacity map
over-promising cooling at the settled operating point. Kept for
reference, not deleted.
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
from single_pack_plant.predictor.physics_p_casadi import CasadiPhysicsP

HERE = Path(__file__).resolve().parent
DT = 5.0
SETTLE_STEPS = 240  # 20 min settle per point
SWEEP_RPM = (1000, 1300, 1600, 1900, 2200, 2500)

predictor = CasadiPhysicsP(
    Path("single_pack_plant/model_data/physics_p_operational_8d4.json"), dt_s=DT
)

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
    tank_c = float(last["tank_temperature_after_k"] - 273.15)
    pump_virtual = min(
        BASELINE_PUMP_RPM * PUMP_VIRTUAL_SPEED_SCALE, 4800.0
    )
    q_map = float(predictor._capacity(float(rpm), pump_virtual, tank_c, 35.0))
    rows.append(
        (
            rpm,
            float(last["compressor_speed_rpm"]),
            float(last["q_evap_applied_w"]),
            q_map,
            tank_c,
            float(last["cluster_supply_temperature_k"] - 273.15),
            float(
                np.mean(
                    last["cluster_result"]["pack_battery_average_temperatures_k"]
                )
                - 273.15
            ),
        )
    )
    print(f"sweep {rpm} rpm done", flush=True)

print(
    "cmd   act   q_evap_plant  q_map   ratio  tank   supply  bat"
)
for r in rows:
    print(
        f"{r[0]:5.0f}  {r[1]:5.0f}  {r[2]:11.0f}  {r[3]:7.0f}  "
        f"{r[2] / max(r[3], 1):5.2f}  {r[4]:5.2f}  {r[5]:5.2f}  {r[6]:5.2f}"
    )

out = HERE / "_probe_capacity_vs_plant_sweep.csv"
np.savetxt(
    out,
    np.array(rows),
    delimiter=",",
    header="cmd_rpm,actual_rpm,q_evap_plant_w,q_map_w,tank_c,supply_c,bat_c",
    comments="",
)
print("saved", out)
