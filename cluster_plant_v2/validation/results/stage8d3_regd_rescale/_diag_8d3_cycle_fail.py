"""Diagnose the Stage 8D3 cycle closure-gate failure in the RegD baseline loop.

Reruns the baseline step loop with an exception hook that records the failing
operating point (coolant inlet temperature, pump/compressor speeds), then
probes ClosedR134aCycle around that point to map the feasibility boundary.
"""

from pathlib import Path
import sys
import traceback

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import numpy as np

from cluster_plant_v2 import refrigeration as refr
from cluster_plant_v2.refrigeration import ClosedR134aCycle
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

case = [c for c in build_cases() if c.case_id == "T2_regd"][0]
currents = case_currents(case, STEPS, DT)

plant = build_final_plant(
    BASELINE_COMPRESSOR_RPM, initial_cluster_current_a=float(currents[0]), direction="forward"
)

fail_info = None
for step_index in range(STEPS):
    try:
        plant.step(
            dt_s=DT,
            cluster_current_a=float(currents[step_index]),
            pump_speed_rpm=BASELINE_PUMP_RPM,
            compressor_speed_command_rpm=BASELINE_COMPRESSOR_RPM,
            fan_speed_rpm=FAN_SPEED_RPM,
            ambient_temperature_k=AMBIENT_TEMPERATURE_K,
            direction="forward",
        )
    except Exception as exc:
        coolant_k = float(plant.tank.temperature_k)
        fail_info = {
            "step": step_index,
            "time_s": (step_index + 1) * DT,
            "error": str(exc),
            "tank_temp_c": coolant_k - 273.15,
            "current_a": float(currents[step_index]),
        }
        break
print("FAILURE POINT:", fail_info, flush=True)

# ---- probe the cycle around the failing point ------------------------------
if fail_info:
    tin_c = fail_info["tank_temp_c"]
    cycle = ClosedR134aCycle()
    print("\nprobe @4000 rpm, fan 1200, ambient 35 C, sweep coolant inlet:")
    for t_in in [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0]:
        try:
            res = cycle.solve(
                compressor_speed_rpm=4000.0,
                fan_speed_rpm=1200.0,
                coolant_inlet_temperature_k=t_in + 273.15,
                coolant_mass_flow_kg_s=0.44982,
                ambient_temperature_k=308.15,
            )
            print(
                f"  Tin {t_in:5.1f} C: qevap {res['q_evaporator_w']:8.1f} W  "
                f"tevap {res['evaporating_saturation_temperature_k']-273.15:6.2f} C  "
                f"tcond {res['condensing_saturation_temperature_k']-273.15:6.2f} C  "
                f"ok {res['solver_success']}"
            )
        except Exception as exc:
            print(f"  Tin {t_in:5.1f} C: FAIL {type(exc).__name__}: {str(exc)[:90]}")
    print("\nbounds: Tevap_min", refr.MINIMUM_EVAPORATING_SATURATION_TEMPERATURE_K - 273.15,
          "C, Tcond_max", refr.MAXIMUM_CONDENSING_SATURATION_TEMPERATURE_K - 273.15, "C")
