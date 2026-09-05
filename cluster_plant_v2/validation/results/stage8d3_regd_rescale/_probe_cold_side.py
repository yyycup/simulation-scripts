"""Cold-side feasibility probe: min coolant inlet temp at 4000 rpm per displacement.

The Stage 8D3 failure (tank overcooled to 11.9 C by the fixed-speed baseline,
then the 32 cc cycle hit the 7 C evaporating-temperature floor) motivates a
displacement x Tin feasibility map to pick the largest displacement whose
4000 rpm operating envelope still covers cold coolant.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import pandas as pd

import cluster_plant_v2.refrigeration as refr
from cluster_plant_v2.refrigeration import ClosedR134aCycle

SPEEDS = (4000.0, 6000.0)
DISPLACEMENTS = (22.0, 26.0, 28.0, 30.0, 32.0)
AIR = (3.5, 4.0, 4.5)
TINS_C = (6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)

rows = []
for disp in DISPLACEMENTS:
    for air in AIR:
        refr.COMPRESSOR_DISPLACEMENT_M3_PER_REV = disp * 1e-6
        refr.NOMINAL_AIR_MASS_FLOW_KG_S = air
        cycle = ClosedR134aCycle()
        # 6000 rpm 热侧能力（Tin 25 C 标定点）
        hot = cycle.solve(
            compressor_speed_rpm=6000.0,
            fan_speed_rpm=1200.0,
            coolant_inlet_temperature_k=298.15,
            coolant_mass_flow_kg_s=0.44982,
            ambient_temperature_k=308.15,
        )
        q6000 = float(hot["q_evaporator_w"])
        # 冷侧：4000 rpm 下最低可行 Tin
        min_feasible_tin = None
        for tin in TINS_C:
            try:
                res = cycle.solve(
                    compressor_speed_rpm=4000.0,
                    fan_speed_rpm=1200.0,
                    coolant_inlet_temperature_k=tin + 273.15,
                    coolant_mass_flow_kg_s=0.44982,
                    ambient_temperature_k=308.15,
                )
                ok = bool(res["solver_success"])
                tevap = float(res["evaporating_saturation_temperature_k"]) - 273.15
            except Exception:
                ok, tevap = False, float("nan")
            rows.append(
                {
                    "disp_cc": disp,
                    "air_kg_s": air,
                    "qevap6000_w": q6000,
                    "tin_c": tin,
                    "ok_4000": ok,
                    "tevap_4000_c": tevap,
                }
            )
            if ok:
                min_feasible_tin = tin
        print(
            f"disp {disp:.0f} air {air:.1f}: Qevap6000 {q6000:7.1f} W, "
            f"min feasible Tin @4000rpm >= {min_feasible_tin} C",
            flush=True,
        )

pd.DataFrame(rows).to_csv(
    "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制/"
    "cluster_plant_v2/validation/results/stage8d3_regd_rescale/cold_side_feasibility.csv",
    index=False,
)
print("csv saved", flush=True)
