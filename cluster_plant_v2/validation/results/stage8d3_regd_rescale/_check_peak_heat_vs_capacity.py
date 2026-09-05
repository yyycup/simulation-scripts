"""Compute the true RegD peak heat generation and compare to chiller capacity.

If the SOC-aware heat generation during the out-of-band windows exceeds the
Stage 8D3 capacity (~10.8 kW at 6000 rpm), no control law can hold the band and
the residual error is a capacity deficit, not a tuning problem.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import numpy as np
import pandas as pd

from cluster_plant_v2.control.physics_p_nmpc_model import ClusterHeatGenerationPreview

NEW = (
    "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制/"
    "cluster_plant_v2/validation/results/stage8d3_regd_rescale/"
    "case_T2_regd_8d3_nmpc_soc_aware_timeseries.csv"
)
DT = 5.0
d = pd.read_csv(NEW).sort_values("time_s").reset_index(drop=True)
currents = d["current_a"].to_numpy()

preview = ClusterHeatGenerationPreview(currents, DT)
q = preview.q_gen_schedule_w  # per-cluster heat generation, W

d["q_gen_w"] = q
print("全时段产热: mean %.0f W | max %.0f W" % (q.mean(), q.max()))
for lo, hi in [(3200, 4000), (5890, 6400)]:
    w = d[(d.time_s >= lo) & (d.time_s <= hi)]
    qe = w.q_evap_applied_w.to_numpy()
    qg = w.q_gen_w.to_numpy()
    print(
        f"尖峰 {lo}-{hi}s: 产热 mean {qg.mean():.0f} max {qg.max():.0f} W | "
        f"蒸发器实发 mean {qe.mean():.0f} max {qe.max():.0f} W | "
        f"缺口(产热-冷量) mean {np.mean(qg-qe):+.0f} W"
    )
# 产热超过 10.7kW 容量的时长
over = (q > 10700.0).sum() * DT
print("产热 > 10.7 kW 的累计时长: %.0f s" % over)
print("产热 > 9.5 kW 的累计时长: %.0f s" % ((q > 9500.0).sum() * DT))
