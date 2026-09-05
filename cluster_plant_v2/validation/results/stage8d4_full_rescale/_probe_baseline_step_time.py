"""Probe: time 10 baseline plant steps for the S0 peak rerun (8D4e).

Purpose: 2026-08-30 the detached rerun has spent ~50 min inside the
fixed-speed baseline pass (1280 steps). Time 10 steps to get the
per-step wall cost and judge whether the full run is merely slow or
actually stuck. Kept for reference, not deleted.
"""

from pathlib import Path
import sys
import time

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

from cluster_plant_v2.validation.validate_domp_mpc import (
    AMBIENT_TEMPERATURE_K,
    BASELINE_COMPRESSOR_RPM,
    BASELINE_PUMP_RPM,
    FAN_SPEED_RPM,
    build_cases,
    build_final_plant,
    case_currents,
)

DT = 5.0
case = [c for c in build_cases() if c.case_id == "S0_constant"][0]
currents = case_currents(case, 1280, DT)
plant = build_final_plant(
    BASELINE_COMPRESSOR_RPM,
    initial_cluster_current_a=float(currents[0]),
    direction="forward",
)

t0 = time.perf_counter()
for i in range(10):
    plant.step(
        dt_s=DT,
        cluster_current_a=float(currents[i]),
        pump_speed_rpm=BASELINE_PUMP_RPM,
        compressor_speed_command_rpm=BASELINE_COMPRESSOR_RPM,
        fan_speed_rpm=FAN_SPEED_RPM,
        ambient_temperature_k=AMBIENT_TEMPERATURE_K,
        direction="forward",
    )
elapsed = time.perf_counter() - t0
print(f"10 steps took {elapsed:.2f} s -> {elapsed / 10:.3f} s/step")
print(f"1280-step baseline ETA: {elapsed / 10 * 1280 / 60:.1f} min")
