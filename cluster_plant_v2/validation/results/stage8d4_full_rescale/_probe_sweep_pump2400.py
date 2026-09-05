"""Probe: sweep anchors at the NMPC pump envelope (actual 2400 rpm).

The earlier sweep probes ran the plant at BASELINE_PUMP_RPM = 3600 rpm
(~0.9 kg/s), but the frozen predictor's virtual pump speed caps at 4800
rpm, which maps to actual 2400 rpm (~0.6 kg/s) via the 8D4c x2 scale.
The NMPC never commands more than virtual 4800, so the anchors must be
taken at actual 2400 rpm where the model/plant flows match exactly.
Same sequential settle protocol as before, plus the measured per-pack
heat. Kept for reference.
"""

from pathlib import Path
import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import numpy as np

from cluster_plant_v2.control.physics_p_nmpc_model import PUMP_VIRTUAL_SPEED_SCALE
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
SETTLE_STEPS = 240
SWEEP_RPM = (1000, 1300, 1600, 1900, 2200, 2500)
# Actual 2400 rpm x virtual scale 2 = virtual 4800 rpm: the top of the
# predictor's pump input domain and of the NMPC's pump envelope.
PUMP_ACTUAL_RPM = 4800.0 / PUMP_VIRTUAL_SPEED_SCALE

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
            pump_speed_rpm=PUMP_ACTUAL_RPM,
            compressor_speed_command_rpm=float(rpm),
            fan_speed_rpm=FAN_SPEED_RPM,
            ambient_temperature_k=AMBIENT_TEMPERATURE_K,
            direction="forward",
        )
    tank_c = float(last["tank_temperature_after_k"] - 273.15)
    pump_virtual = PUMP_ACTUAL_RPM * PUMP_VIRTUAL_SPEED_SCALE
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
                np.mean(last["cluster_result"]["pack_battery_average_temperatures_k"])
                - 273.15
            ),
            float(last["cluster_q_gen_total_w"]) / 5.0,
        )
    )
    print(f"sweep {rpm} rpm done", flush=True)

print("cmd   act   q_evap  q_map  ratio  tank   supply  bat    qgen_pp")
for r in rows:
    print(
        f"{r[0]:5.0f}  {r[1]:5.0f}  {r[2]:7.0f}  {r[3]:6.0f}  "
        f"{r[2]/max(r[3],1):5.2f}  {r[4]:5.2f}  {r[5]:5.2f}  {r[6]:5.2f}  {r[7]:6.0f}"
    )

out = HERE / "_probe_sweep_pump2400.csv"
np.savetxt(
    out,
    np.array(rows),
    delimiter=",",
    header=(
        "cmd_rpm,actual_rpm,q_evap_plant_w,q_map_w,tank_c,supply_c,bat_c,"
        "qgen_per_pack_w"
    ),
    comments="",
)
print("saved", out)
