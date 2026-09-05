"""Stage 8D3 vs old chiller: RegD average-temperature comparison.

Old curve = physics_p_nmpc rows of the pre-rescale full run
(results/physics_p_nmpc_regd_full_20260827/case_T2_regd_full_timeseries.csv).
New curve = SOC-aware NMPC run under the 8D3 rescaled chiller
(case_T2_regd_8d3_nmpc_soc_aware_timeseries.csv).
"""

from pathlib import Path
import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OLD_CSV = (
    Path("C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")
    / "cluster_plant_v2/validation/results/physics_p_nmpc_regd_full_20260827"
    / "case_T2_regd_full_timeseries.csv"
)
NEW_CSV = HERE / "case_T2_regd_8d3_nmpc_soc_aware_timeseries.csv"

old = pd.read_csv(OLD_CSV)
# The old CSV stores two identical 6400 s passes of the SOC-aware NMPC
# back-to-back under the same controller label; keep one pass only.
old = (
    old[old["controller"] == "physics_p_nmpc"]
    .reset_index(drop=True)
    .iloc[:1280]
    .sort_values("time_s")
    .reset_index(drop=True)
)
new = pd.read_csv(NEW_CSV).sort_values("time_s").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.axhspan(25.0 - 0.65, 25.0 + 0.65, color="tab:green", alpha=0.12, zorder=0)
ax.axhline(25.0, color="tab:green", lw=1.0, ls="--", alpha=0.8)
ax.plot(old["time_s"], old["battery_avg_temp_c"], color="tab:red", lw=1.6,
        label="旧制冷（8D2 排量 22 cc）")
ax.plot(new["time_s"], new["battery_avg_temp_c"], color="tab:blue", lw=1.6,
        label="扩容后（8D3 排量 32 cc）")
ax.set_xlabel("时间 / s")
ax.set_ylabel("电池平均温度 / °C")
ax.set_title("调频工况：扩容前后 SOC 感知 NMPC 平均温度对比")
ax.legend(loc="upper left")
ax.set_xlim(0, 6400)
ax.set_ylim(24.0, 29.0)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(HERE / "fig_regd_8d3_vs_old_temp.png", dpi=150)
fig.savefig(HERE / "fig_regd_8d3_vs_old_temp.pdf")
print("saved:", HERE / "fig_regd_8d3_vs_old_temp.png")

# 简表：两曲线带外与峰值
for name, frame in (("old", old), ("new", new)):
    temp = frame["battery_avg_temp_c"].to_numpy()
    out = float(np.sum(np.abs(temp - 25.0) > 0.65)) * 5.0
    print(f"{name}: max {temp.max():.2f} C, min {temp.min():.2f} C, out_of_band {out:.0f} s")
