"""Stage 8D3 RegD verification: baseline vs SOC-aware NMPC after chiller rescale.

Trimmed rerun of the RegD closed loop under the Stage 8D3 chiller (displacement
32 cc/rev, condenser air 4.5 kg/s): fixed-speed baseline plus the SOC-aware
NMPC only, to verify the rescaled capacity holds the 25 +/- 0.65 K band.
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
from cluster_plant_v2.validation.validate_physics_p_nmpc import run_baseline, run_nmpc

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

print("pass 0: fixed-speed baseline ...", flush=True)
baseline = run_baseline(case, duration_s=DURATION, dt_s=DT)
print("baseline done", flush=True)

print("pass 1: SOC-aware NMPC ...", flush=True)
socaware = run_nmpc(
    case,
    duration_s=DURATION,
    dt_s=DT,
    heat_preview=ClusterHeatGenerationPreview(currents, DT),
)
print("SOC-aware NMPC done", flush=True)

rows = []
frames = {
    "baseline_fixed": baseline,
    "nmpc_soc_aware": socaware,
}
for name, frame in frames.items():
    frame = frame.sort_values("time_s")
    temp = frame["battery_avg_temp_c"].to_numpy()
    out_band = float(np.sum(np.abs(temp - REF_C) > BAND_HALF)) * DT
    comp_kwh = float(frame["compressor_power_w"].sum()) * DT / 3.6e6
    pump_kwh = float(frame["pump_power_w"].sum()) * DT / 3.6e6
    rows.append(
        {
            "controller": name,
            "final_temp_c": temp[-1],
            "max_temp_c": temp.max(),
            "max_dev_k": np.abs(temp - REF_C).max(),
            "out_of_band_s": out_band,
            "comp_kwh": comp_kwh,
            "pump_kwh": pump_kwh,
            "total_kwh": comp_kwh + pump_kwh,
        }
    )
    frame.to_csv(HERE / f"case_T2_regd_8d3_{name}_timeseries.csv", index=False)

summary = pd.DataFrame(rows)
summary["scenario"] = "T2_regd_6400s_8d3"
summary.to_csv(HERE / "regd_8d3_summary.csv", index=False)
print(summary.to_string(index=False), flush=True)
print("done", flush=True)
