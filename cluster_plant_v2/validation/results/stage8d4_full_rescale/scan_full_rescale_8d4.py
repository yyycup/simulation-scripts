"""Stage 8D4 full-rescale sizing scan: cover the RegD 1C-charge peak.

Requirement derived from Plant evidence: the RegD full-deviation charge
segment produces ~17.4 kW cluster heat (3486 W/pack x 5, measured at
-1120 A), and the spikes run with the tank at 20-23 C, so the scan gates
the capacity at coolant inlet Tin = 20 C, 6000 rpm.

Feasibility gates (more conservative than 8D3 because the compressor now
runs near its ceiling during the whole charge segment):
  q_evap @ Tin=20 C, 6000 rpm >= 17,500 W
  Tcond <= 53 C (2 K margin to the 55 C model ceiling)
Smallest oversizing score wins.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import pandas as pd

import cluster_plant_v2.refrigeration as refr
from cluster_plant_v2.refrigeration import ClosedR134aCycle

HERE = Path(__file__).resolve().parent
CYCLE = ClosedR134aCycle()

DISPLACEMENTS_CC = (48.0, 56.0, 64.0, 72.0)
EVAPORATOR_AREAS_M2 = (4.5, 5.5, 6.5, 7.5)
CONDENSER_AREAS_M2 = (6.0, 8.0)
AIR_FLOWS_KG_S = (5.5, 6.5)

COOLANT_FLOW_KG_S = 0.44982
FAN_RPM = 1200.0
AMBIENT_K = 308.15
TIN_C = 20.0          # spike-time tank temperature
QEVP_MIN_W = 17500.0  # >= 17.4 kW measured peak heat + margin
TCOND_MAX_C = 53.0

rows = []
for disp, evap, cond, air in itertools.product(
    DISPLACEMENTS_CC, EVAPORATOR_AREAS_M2, CONDENSER_AREAS_M2, AIR_FLOWS_KG_S
):
    refr.COMPRESSOR_DISPLACEMENT_M3_PER_REV = disp * 1e-6
    refr.EVAPORATOR_AREA_M2 = evap
    refr.CONDENSER_AREA_M2 = cond
    refr.NOMINAL_AIR_MASS_FLOW_KG_S = air
    try:
        res = CYCLE.solve(
            compressor_speed_rpm=6000.0,
            fan_speed_rpm=FAN_RPM,
            coolant_inlet_temperature_k=TIN_C + 273.15,
            coolant_mass_flow_kg_s=COOLANT_FLOW_KG_S,
            ambient_temperature_k=AMBIENT_K,
        )
        q = float(res["q_evaporator_w"])
        tcond = float(res["condensing_saturation_temperature_k"]) - 273.15
        tevap = float(res["evaporating_saturation_temperature_k"]) - 273.15
        cop = float(res["cop_shaft"]) if "cop_shaft" in res else None
        ok = bool(res["solver_success"])
    except Exception:  # noqa: BLE001 - infeasible corner
        q, tcond, tevap, cop, ok = float("nan"), float("nan"), float("nan"), None, False
    rows.append(
        {
            "disp_cc": disp,
            "evap_m2": evap,
            "cond_m2": cond,
            "air_kg_s": air,
            "qevap_w": q,
            "tevap_c": tevap,
            "tcond_c": tcond,
            "cop_shaft": cop,
            "feasible": ok,
            "passes": ok and q >= QEVP_MIN_W and tcond <= TCOND_MAX_C,
            "score": disp + 4.0 * evap + 2.0 * cond + air,
        }
    )

frame = pd.DataFrame(rows)
frame.to_csv(HERE / "full_rescale_scan_8d4.csv", index=False)
passing = frame[frame["passes"]].sort_values("score")
print(f"{len(frame)} combos, {len(passing)} pass gates", flush=True)
if len(passing):
    print(passing.head(8).to_string(index=False), flush=True)
else:
    best = frame[frame["feasible"]].sort_values("qevap_w", ascending=False)
    print("no combo passes; top capacity:", flush=True)
    print(best.head(8).to_string(index=False), flush=True)
