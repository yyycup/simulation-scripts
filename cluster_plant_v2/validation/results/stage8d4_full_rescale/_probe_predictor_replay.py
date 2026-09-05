"""Probe: open-loop replay of the frozen predictor on the 8D4e run.

Feeds the measured command/disturbance sequences of the 8D4e NMPC rerun
into the frozen Physics-P predictor and compares the predicted battery
temperature against the plant measurement. If the predictor reproduces
the plant trajectory the residual drift is an optimization-side issue;
if the prediction lags low, the frozen model has steady-state gain
mismatch on the temperature channel. Kept for reference, not deleted.
"""

from pathlib import Path
import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import numpy as np
import pandas as pd

from cluster_plant_v2.control.physics_p_nmpc_model import (
    PUMP_VIRTUAL_SPEED_SCALE,
    ClusterHeatGenerationPreview,
    load_frozen_predictor,
)
from cluster_plant_v2.validation.validate_domp_mpc import (
    BASELINE_COMPRESSOR_RPM,
    build_cases,
    case_currents,
)
from cluster_plant_v2.validation.validate_final_cluster_plant import build_final_plant
from cluster_plant_v2.validation.validate_physics_p_nmpc import (
    KELVIN_OFFSET,
    initial_physics_state,
)
from cluster_plant_v2.validation.validate_refrigerated_cluster_cooling_loop import (
    AMBIENT_TEMPERATURE_K,
)

HERE = Path(__file__).resolve().parent
DT = 5.0
STEPS = 1280

case = [c for c in build_cases() if c.case_id == "S0_constant"][0]
currents = case_currents(case, STEPS, DT)
preview = ClusterHeatGenerationPreview(currents, DT)

nmpc = pd.read_csv(
    HERE / "case_S0_peak_8d4e_nmpc_soc_aware_8d4d_timeseries.csv"
).sort_values("time_s")

plant = build_final_plant(
    BASELINE_COMPRESSOR_RPM,
    initial_cluster_current_a=float(currents[0]),
    direction="forward",
)
predictor = load_frozen_predictor()

x = np.array(initial_physics_state(plant), dtype=float).reshape(-1, 1)
t_amb_c = AMBIENT_TEMPERATURE_K - KELVIN_OFFSET

rows = []
for k in range(STEPS):
    row = nmpc.iloc[k]
    u = np.array(
        [
            float(row["compressor_command_rpm"]),
            float(row["pump_command_rpm"]) * PUMP_VIRTUAL_SPEED_SCALE,
        ]
    ).reshape(-1, 1)
    d = np.array(
        [float(row["current_a"]), t_amb_c, float(preview(float(k) * DT, 0.0))]
    ).reshape(-1, 1)
    x = np.asarray(predictor.function(x, u, d), dtype=float).reshape(-1, 1)
    rows.append(
        (
            float(row["time_s"]),
            float(x[0, 0]),
            float(row["battery_avg_temp_c"]),
            float(row["compressor_command_rpm"]),
        )
    )
    if (k + 1) % 320 == 0:
        print(f"replayed {k + 1}/{STEPS}", flush=True)

arr = np.array(rows)
pred, meas = arr[:, 1], arr[:, 2]
print("time_min  pred_C  meas_C  gap_K  cmd")
for idx in range(0, STEPS, 160):
    t, p, m, cmd = arr[idx]
    print(f"{t / 60:6.1f}  {p:6.2f}  {m:6.2f}  {p - m:+6.2f}  {cmd:6.0f}")
late = arr[arr[:, 0] > 5400]
print(
    "late segment: pred mean %.2f C, meas mean %.2f C, gap %+.3f K"
    % (late[:, 1].mean(), late[:, 2].mean(), (late[:, 1] - late[:, 2]).mean())
)

out = HERE / "_probe_predictor_replay_vs_plant.csv"
pd.DataFrame(
    arr, columns=["time_s", "pred_temp_c", "meas_temp_c", "comp_cmd_rpm"]
).to_csv(out, index=False)
print("saved", out)
