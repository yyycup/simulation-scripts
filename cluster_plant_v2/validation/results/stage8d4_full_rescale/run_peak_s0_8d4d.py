"""Stage 8D4d/8D4e peak-shaving verification: S0 560 A constant discharge.

The 8D3/8D4 hardware rescale was so far only validated on the RegD
frequency-regulation case (T2). Peak shaving (S0_constant, 560 A
sustained, ~5.1 kW cluster heat) is the other duty the hardware must
serve: a sustained load above the 4.1-4.5 kW compressor floor capacity,
so continuous modulation should dominate and the 8D4d thermostat
cycling supervisor should barely trigger. Runs the fixed-speed baseline
and the SOC-aware NMPC (with cycling enabled) on the current hardware.

Stage 8D4e rerun: same passes, but the frozen predictor capacity block
is the re-derived 8D4 artifact (physics_p_operational_8d4.json); outputs
carry the 8d4e suffix to keep the pre-rerun results for comparison.
Per the updated convention (2026-08-30) the SOC-aware heat preview
variant is retired: pass 2 uses run_nmpc's standard current-based
battery heat preview (heat_preview=None).
"""

from pathlib import Path
import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import numpy as np
import pandas as pd

from cluster_plant_v2.validation.validate_domp_mpc import (
    build_cases,
    case_currents,
    run_baseline,
)
from cluster_plant_v2.validation.validate_physics_p_nmpc import run_nmpc

HERE = Path(__file__).resolve().parent
DT = 5.0
STEPS = 1280
DURATION = STEPS * DT
REF_C = 25.0
BAND_HALF = 0.65

case = [c for c in build_cases() if c.case_id == "S0_constant"][0]
currents = case_currents(case, STEPS, DT)
print(
    f"running {case.case_id} for {DURATION:.0f} s; "
    f"current {currents[0]:.1f} A",
    flush=True,
)

print("pass 1: fixed-speed baseline ...", flush=True)
baseline = run_baseline(case, duration_s=DURATION, dt_s=DT)
print("baseline done", flush=True)

print("pass 2: NMPC with compressor thermostat cycling ...", flush=True)
socaware = run_nmpc(
    case,
    duration_s=DURATION,
    dt_s=DT,
    compressor_thermostat_cycling=True,
)
print("NMPC done", flush=True)

rows = []
frames = {
    "baseline_fixed": baseline.sort_values("time_s"),
    "nmpc_soc_aware_8d4d": socaware.sort_values("time_s"),
}
# Same pump power correction as the RegD scripts: the 8D4b rescaled pump
# draws 8x the legacy reference power at equal speed.
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
    frame.to_csv(HERE / f"case_S0_peak_8d4e_{name}_timeseries.csv", index=False)

comp_cmd = frames["nmpc_soc_aware_8d4d"]["compressor_command_rpm"].to_numpy()
cycle_switches = int(np.sum(np.diff((comp_cmd > 0).astype(int)) != 0))
off_steps = int(np.sum(comp_cmd <= 0.0))
print(
    f"thermostat cycling: {cycle_switches} start/stop switches, "
    f"off time {off_steps * DT:.0f} s",
    flush=True,
)

summary = pd.DataFrame(rows)
summary["scenario"] = "S0_peak_6400s_8d4e"
summary.to_csv(HERE / "s0_peak_8d4e_summary.csv", index=False)
print(summary.to_string(index=False), flush=True)
print("done", flush=True)
