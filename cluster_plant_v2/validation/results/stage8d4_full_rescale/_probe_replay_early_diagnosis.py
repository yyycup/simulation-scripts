"""Probe: early-segment model-vs-plant cooling-path diagnostics.

Prints the 8D4e rerun's first 20 minutes (command vs actual rpm, q_evap,
tank/supply/plate-relevant temperatures) alongside the predictor replay
temperature, to locate where the open-loop replay collapses. Kept for
reference, not deleted.
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

plant = pd.read_csv(
    HERE / "case_S0_peak_8d4e_nmpc_soc_aware_8d4d_timeseries.csv"
).sort_values("time_s")
replay = pd.read_csv(HERE / "_probe_predictor_replay_vs_plant.csv")

plant = plant[plant["time_s"] <= 1500].reset_index(drop=True)
replay = replay[replay["time_s"] <= 1500].reset_index(drop=True)

print(
    "time_s  cmd   act   q_evap  tank   supply  return  bat_meas  bat_pred"
)
for i in range(0, len(plant), 24):
    p = plant.iloc[i]
    r = replay.iloc[i]
    print(
        f"{p.time_s:6.0f}  {p.compressor_command_rpm:5.0f}  "
        f"{p.compressor_actual_rpm:5.0f}  {p.q_evap_applied_w:6.0f}  "
        f"{p.tank_temp_c:5.2f}  {p.supply_temp_c:5.2f}  "
        f"{p.return_temp_c:5.2f}  {p.battery_avg_temp_c:7.2f}  "
        f"{r.pred_temp_c:7.2f}"
    )

late = pd.read_csv(
    HERE / "case_S0_peak_8d4e_nmpc_soc_aware_8d4d_timeseries.csv"
)
late = late[late["time_s"] > 5400]
print(
    "\nlate steady: act rpm mean %.0f, q_evap %.0f W, tank %.2f, supply %.2f, "
    "return %.2f, bat %.2f"
    % (
        late["compressor_actual_rpm"].mean(),
        late["q_evap_applied_w"].mean(),
        late["tank_temp_c"].mean(),
        late["supply_temp_c"].mean(),
        late["return_temp_c"].mean(),
        late["battery_avg_temp_c"].mean(),
    )
)
