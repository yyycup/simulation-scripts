"""Probe: plant actual heat generation vs the SOC-aware preview.

Runs the full plant open-loop at fixed compressor/pump speeds under the
S0 560 A schedule and logs cluster_q_gen_total_w from the step result
(divided by 5 packs to match the single-pack preview convention), to
quantify the feedforward mismatch behind the slow steady-state drift in
the 8D4e NMPC rerun. Kept for reference, not deleted.
"""

from pathlib import Path
import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import numpy as np

from cluster_plant_v2.control.physics_p_nmpc_model import ClusterHeatGenerationPreview
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
from cluster_plant_v2.validation.validate_refrigerated_cluster_cooling_loop import (
    AMBIENT_TEMPERATURE_K,
)

DT = 5.0
STEPS = 1280
N_PACKS = 5

case = [c for c in build_cases() if c.case_id == "S0_constant"][0]
currents = case_currents(case, STEPS, DT)
preview = ClusterHeatGenerationPreview(currents, DT)

plant = build_final_plant(
    BASELINE_COMPRESSOR_RPM,
    initial_cluster_current_a=float(currents[0]),
    direction="forward",
)

rows = []
for i in range(STEPS):
    result = plant.step(
        dt_s=DT,
        cluster_current_a=float(currents[i]),
        pump_speed_rpm=BASELINE_PUMP_RPM,
        compressor_speed_command_rpm=BASELINE_COMPRESSOR_RPM,
        fan_speed_rpm=FAN_SPEED_RPM,
        ambient_temperature_k=AMBIENT_TEMPERATURE_K,
        direction="forward",
    )
    rows.append(
        (
            (i + 1) * DT,
            float(result["cluster_q_gen_total_w"]) / N_PACKS,
            float(preview(float(i) * DT, 0.0)),
            float(
                np.mean(
                    result["cluster_result"]["pack_battery_average_temperatures_k"]
                )
                - 273.15
            ),
            float(result["q_evap_applied_w"]),
        )
    )
    if (i + 1) % 320 == 0:
        print(f"step {i + 1}/{STEPS}", flush=True)

arr = np.array(rows)
print("time_min  qgen_per_pack_W  preview_W  ratio  temp_C  q_evap_W")
for idx in range(0, STEPS, 160):
    t, qg, qp, temp, qe = arr[idx]
    print(
        f"{t / 60:6.1f}  {qg:14.0f}  {qp:9.0f}  {qg / qp:5.2f}  {temp:5.2f}  {qe:7.0f}"
    )
late = arr[arr[:, 0] > 5400]
print(
    "late segment (>90 min): per-pack q_gen mean %.0f W, preview %.0f W, ratio %.3f"
    % (late[:, 1].mean(), late[:, 2].mean(), late[:, 1].mean() / late[:, 2].mean())
)

out = Path(__file__).resolve().parent / "_probe_plant_qgen_vs_preview.csv"
np.savetxt(
    out,
    arr,
    delimiter=",",
    header="time_s,qgen_per_pack_w,qgen_preview_w,battery_avg_temp_c,q_evap_w",
    comments="",
)
print("saved", out)
