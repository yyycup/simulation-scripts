"""Probe which cycle bound binds during RegD peaks (Stage 8D3 chiller).

During the two out-of-band windows the compressor is LP-unloaded 27%/42% of the
time. This probe solves the closed R134a cycle at representative peak operating
points (coolant inlet ~ tank temp, compressor near/at 6000 rpm) and reports the
resulting evaporating and condensing temperatures, so we can tell whether the
evaporating floor (7 C) or the condensing ceiling (55 C) is the active limit.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import cluster_plant_v2.refrigeration as refr
from cluster_plant_v2.refrigeration import ClosedR134aCycle

print(
    "bounds: Tevap_min",
    refr.MINIMUM_EVAPORATING_SATURATION_TEMPERATURE_K - 273.15,
    "C | Tcond_max",
    refr.MAXIMUM_CONDENSING_SATURATION_TEMPERATURE_K - 273.15,
    "C",
)

cycle = ClosedR134aCycle()
# Peak-window representative points: evaporator coolant inlet = tank temp.
for tin_c in (20.0, 21.0, 22.0, 23.0):
    for rpm in (4500.0, 5500.0, 6000.0):
        try:
            res = cycle.solve(
                compressor_speed_rpm=rpm,
                fan_speed_rpm=1200.0,
                coolant_inlet_temperature_k=tin_c + 273.15,
                coolant_mass_flow_kg_s=0.44982,
                ambient_temperature_k=308.15,
            )
            tevap = res["evaporating_saturation_temperature_k"] - 273.15
            tcond = res["condensing_saturation_temperature_k"] - 273.15
            print(
                f"  Tin {tin_c:4.1f} C, {rpm:5.0f} rpm: qevap {res['q_evaporator_w']:7.1f} W | "
                f"Tevap {tevap:5.2f} C (floor+{tevap-7.0:+.2f}) | "
                f"Tcond {tcond:5.2f} C (ceil{tcond-55.0:+.2f})"
            )
        except Exception as exc:
            print(f"  Tin {tin_c:4.1f} C, {rpm:5.0f} rpm: FAIL {type(exc).__name__}: {str(exc)[:70]}")
