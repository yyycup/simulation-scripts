"""Probe: preview temperature sensitivity vs plant effective heat.

Checks whether the HPPC resistance tables make the SOC-aware preview
heat grow with cell temperature (the drift hypothesis) and compares
preview values at 25/26 C against the plant's effective steady heat
(evaporator duty in the late segment). Kept for reference.
"""

from pathlib import Path
import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import numpy as np
import pandas as pd

from cluster_plant_v2.control.physics_p_nmpc_model import ClusterHeatGenerationPreview
from cluster_plant_v2.validation.validate_domp_mpc import build_cases, case_currents

RES = Path("cluster_plant_v2/validation/results/stage8d4_full_rescale")

case = [c for c in build_cases() if c.case_id == "S0_constant"][0]
currents = case_currents(case, 1280, 5.0)

for temp_k in (298.15, 298.65, 299.15):
    pv = ClusterHeatGenerationPreview(currents, 5.0, temperature_k=temp_k)
    print(
        f"preview q_gen @cell T {temp_k - 273.15:.1f} C: "
        f"start {pv.q_gen_schedule_w[0]:.0f} W, "
        f"end {pv.q_gen_schedule_w[-1]:.0f} W"
    )

new = pd.read_csv(RES / "case_S0_peak_8d4e_nmpc_soc_aware_8d4d_timeseries.csv")
late = new[new["time_s"] > 5400]
print(
    "plant late segment: q_evap mean "
    f"{late['q_evap_applied_w'].mean():.0f} W, temp {late['battery_avg_temp_c'].mean():.2f} C"
)
old = pd.read_csv(RES / "case_S0_peak_8d4d_nmpc_soc_aware_8d4d_timeseries.csv")
late_old = old[old["time_s"] > 5400]
print(
    "pre-fix late segment: q_evap mean "
    f"{late_old['q_evap_applied_w'].mean():.0f} W, temp {late_old['battery_avg_temp_c'].mean():.2f} C"
)
