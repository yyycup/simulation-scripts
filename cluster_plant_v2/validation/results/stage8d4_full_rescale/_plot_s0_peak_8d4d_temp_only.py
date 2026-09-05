"""Stage 8D4d peak-shaving plot, temperature panel only (no baseline).

Single-panel figure: battery average temperature of the NMPC +
thermostat-cycling run on S0 (560 A constant), with the control band
shading and the one short off segment. Baseline trace removed per
request. Style per project spec: Microsoft YaHei, Chinese legend.
"""

from pathlib import Path
import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RES = Path("cluster_plant_v2/validation/results/stage8d4_full_rescale")
REF_C, HALF = 25.0, 0.65

nmpc = pd.read_csv(
    RES / "case_S0_peak_8d4d_nmpc_soc_aware_8d4d_timeseries.csv"
).sort_values("time_s")

t = nmpc["time_s"].to_numpy() / 60.0
temp = nmpc["battery_avg_temp_c"].to_numpy()
comp = nmpc["compressor_command_rpm"].to_numpy()
off = comp <= 0.0

fig, ax = plt.subplots(figsize=(11, 4.6))

ax.axhspan(
    REF_C - HALF, REF_C + HALF, color="green", alpha=0.12,
    label="控制带 25±0.65 °C",
)
ax.fill_between(
    t, 24.0, 26.2, where=off, color="gray", alpha=0.18,
    label="压缩机停机段（仅 235 s）",
)
ax.plot(t, temp, color="tab:green", lw=1.4, label="NMPC + 启停（带外 0 s）")
ax.axhline(REF_C, color="k", lw=0.7, ls=":")

ax.set_ylabel("电池簇均温 (°C)")
ax.set_xlabel("时间 (min)")
ax.set_ylim(24.0, 26.2)
ax.set_title(
    "8D4d 调峰工况（S0, 560 A 恒流, 6400 s）：NMPC 全程带内，"
    "均温 24.74~25.39 °C"
)
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(RES / "fig_s0_peak_8d4d_temp_only.png", dpi=150)
fig.savefig(RES / "fig_s0_peak_8d4d_temp_only.pdf")
print("saved to", RES)
