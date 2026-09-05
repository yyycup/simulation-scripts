"""Stage 8D4 probe: full-rescale feasibility before the sizing scan.

Checks two things the 8D3 scan (Tin=25 C nominal point) did not cover:
1. Capacity derating with coolant inlet temperature -- the RegD spikes run
   with the tank at 20-23 C, so the requirement must be met at Tin=20 C.
2. Which constraint saturates first when scaling displacement, evaporator,
   condenser, and condenser air together toward ~17.4 kW peak coverage.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import cluster_plant_v2.refrigeration as refr
from cluster_plant_v2.refrigeration import ClosedR134aCycle

CYCLE = ClosedR134aCycle()
COOLANT_FLOW = 0.44982  # kg/s, 8D2/8D3 scan condition
FAN_RPM = 1200.0
AMBIENT_K = 308.15


def solve(disp_cc: float, evap_m2: float, cond_m2: float, air: float, tin_c: float, rpm: float):
    refr.COMPRESSOR_DISPLACEMENT_M3_PER_REV = disp_cc * 1e-6
    refr.EVAPORATOR_AREA_M2 = evap_m2
    refr.CONDENSER_AREA_M2 = cond_m2
    refr.NOMINAL_AIR_MASS_FLOW_KG_S = air
    try:
        res = CYCLE.solve(
            compressor_speed_rpm=rpm,
            fan_speed_rpm=FAN_RPM,
            coolant_inlet_temperature_k=tin_c + 273.15,
            coolant_mass_flow_kg_s=COOLANT_FLOW,
            ambient_temperature_k=AMBIENT_K,
        )
    except Exception as exc:  # noqa: BLE001 - report infeasible points
        return {"ok": False, "err": f"{type(exc).__name__}: {exc}"[:60]}
    return {
        "ok": bool(res["solver_success"]),
        "q": float(res["q_evaporator_w"]),
        "tevap": float(res["evaporating_saturation_temperature_k"]) - 273.15,
        "tcond": float(res["condensing_saturation_temperature_k"]) - 273.15,
    }


CONFIGS = [
    # (label, disp_cc, evap, cond, air)
    ("8D3 current      ", 32.0, 2.5, 4.0, 4.5),
    ("disp only        ", 48.0, 2.5, 4.0, 4.5),
    ("disp+evap        ", 48.0, 4.5, 4.0, 4.5),
    ("disp+evap+cond   ", 48.0, 4.5, 6.0, 4.5),
    ("full upscale     ", 56.0, 5.5, 8.0, 6.5),
    ("full upscale big ", 64.0, 6.5, 8.0, 6.5),
]

for label, disp, evap, cond, air in CONFIGS:
    for tin in (25.0, 20.0):
        for rpm in (6000.0,):
            r = solve(disp, evap, cond, air, tin, rpm)
            if r["ok"]:
                print(
                    f"{label} Tin {tin:.0f}C @6000: q {r['q']:8.0f} W  "
                    f"tevap {r['tevap']:6.2f} C  tcond {r['tcond']:6.2f} C",
                    flush=True,
                )
            else:
                print(
                    f"{label} Tin {tin:.0f}C @6000: FAIL {r.get('err', '')}",
                    flush=True,
                )
