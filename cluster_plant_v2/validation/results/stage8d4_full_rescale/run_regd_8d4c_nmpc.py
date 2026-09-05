"""Stage 8D4c RegD verification: SOC-aware NMPC with the pump domain scaling.

The 8D4b run showed the rescaled chiller holds the upper band perfectly
(above-band 0 s) but the frozen predictor -- calibrated on the legacy
28 L/min pump -- underestimates the doubled coolant flow, and the loop
drifted 5125 s below the band. Stage 8D4c runs the NMPC pump input in the
virtual legacy-pump domain (virtual = 2x actual, see
PUMP_VIRTUAL_SPEED_SCALE), which reproduces the true flow exactly inside
the predictor training range. The fixed-speed baseline pass is reused from
the 8D4b run (unchanged hardware for the baseline).
"""

from pathlib import Path
import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import numpy as np
import pandas as pd

from cluster_plant_v2.control.physics_p_nmpc_model import ClusterHeatGenerationPreview
from cluster_plant_v2.validation.validate_domp_mpc import (
    build_cases,
    case_currents,
)
from cluster_plant_v2.validation.validate_physics_p_nmpc import run_nmpc

HERE = Path(__file__).resolve().parent
DT = 5.0
STEPS = 1280
DURATION = STEPS * DT
REF_C = 25.0
BAND_HALF = 0.65

case = [c for c in build_cases() if c.case_id == "T2_regd"][0]
currents = case_currents(case, STEPS, DT)
print(
    f"running {case.case_id} for {DURATION:.0f} s; "
    f"min/max/mean current {currents.min():.1f}/{currents.max():.1f}/"
    f"{currents.mean():.1f} A",
    flush=True,
)

print("pass 1: SOC-aware NMPC with pump domain scaling ...", flush=True)
socaware = run_nmpc(
    case,
    duration_s=DURATION,
    dt_s=DT,
    heat_preview=ClusterHeatGenerationPreview(currents, DT),
)
print("SOC-aware NMPC done", flush=True)

rows = []
frames = {
    "baseline_fixed": pd.read_csv(
        HERE / "case_T2_regd_8d4_baseline_fixed_timeseries.csv"
    ).sort_values("time_s"),
    "nmpc_soc_aware_8d4c": socaware.sort_values("time_s"),
}
# Stage 8D4b baseline CSV carries pump power under the legacy reference
# power (27.4 W); the rescaled pump draws exactly 8x at the same speed
# (network dP ~ q^2, flow x2). Correct for the energy comparison.
PUMP_POWER_CORRECTION = 8.0
for name, frame in frames.items():
    temp = frame["battery_avg_temp_c"].to_numpy()
    out_band = float(np.sum(np.abs(temp - REF_C) > BAND_HALF)) * DT
    comp_kwh = float(frame["compressor_power_w"].sum()) * DT / 3.6e6
    pump_kwh = float(frame["pump_power_w"].sum()) * PUMP_POWER_CORRECTION * DT / 3.6e6
    rows.append(
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
    if name != "baseline_fixed":
        frame.to_csv(HERE / f"case_T2_regd_8d4c_{name}_timeseries.csv", index=False)

summary = pd.DataFrame(rows)
summary["scenario"] = "T2_regd_6400s_8d4c"
summary.to_csv(HERE / "regd_8d4c_summary.csv", index=False)
print(summary.to_string(index=False), flush=True)
print("done", flush=True)
