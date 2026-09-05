"""Probe: heat-preview energy balance for the 8D4e S0 rerun.

Compares the standard ohmic preview heat at 560 A against the actual
evaporator duty in the command-at-floor segment of the new run and the
steady state of the old SOC-aware run, to check whether the retired
SOC-aware preview was supplying materially more feedforward heat.
Kept for reference, not deleted.
"""

from pathlib import Path

import pandas as pd

RES = Path("cluster_plant_v2/validation/results/stage8d4_full_rescale")

new = pd.read_csv(RES / "case_S0_peak_8d4e_nmpc_soc_aware_8d4d_timeseries.csv")
old = pd.read_csv(RES / "case_S0_peak_8d4d_nmpc_soc_aware_8d4d_timeseries.csv")

preview_w = ((560.0 / 4.0) ** 2) * 0.001 * 52.0
print(f"standard ohmic preview heat @560A = {preview_w:.0f} W")

mid = new[(new["time_s"] > 1800) & (new["time_s"] < 3600)]
print(
    "new run, cmd-at-floor segment (30-60 min): "
    f"q_evap mean {mid['q_evap_applied_w'].mean():.0f} W, "
    f"temp {mid['battery_avg_temp_c'].mean():.2f} C"
)

late_old = old[old["time_s"] > 3000]
print(
    "old SOC-aware run steady state (>50 min): "
    f"q_evap mean {late_old['q_evap_applied_w'].mean():.0f} W, "
    f"temp {late_old['battery_avg_temp_c'].mean():.2f} C, "
    f"cmd mean {late_old['compressor_command_rpm'].mean():.0f} rpm"
)
