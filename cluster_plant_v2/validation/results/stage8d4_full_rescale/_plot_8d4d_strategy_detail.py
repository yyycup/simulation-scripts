"""Stage 8D4d detail plot: how thermostat cycling holds the band.

Three panels: battery average temperature with band shading and off
segments, compressor command with start/stop sequence, and tank/supply
temperatures showing the off-phase recovery. Style per project spec:
Microsoft YaHei, Chinese legend, band shading, no fine-line clutter.
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

RES = Path("cluster_plant_v2/validation/results")
REF_C, HALF = 25.0, 0.65

f = pd.read_csv(
    RES / "stage8d4_full_rescale/case_T2_regd_8d4d_nmpc_soc_aware_8d4d_timeseries.csv"
).sort_values("time_s")
t = f["time_s"].to_numpy() / 60.0
temp = f["battery_avg_temp_c"].to_numpy()
comp = f["compressor_command_rpm"].to_numpy()
off = comp <= 0.0

fig, (ax0, ax1, ax2) = plt.subplots(
    3,
    1,
    figsize=(11, 10),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1, 1.4]},
)

# Panel 1: battery temperature + band + off segments
ax0.axhspan(REF_C - HALF, REF_C + HALF, color="green", alpha=0.12, label="控制带 25±0.65 °C")
ax0.fill_between(
    t,
    temp.min() - 0.2,
    temp.max() + 0.2,
    where=off,
    color="gray",
    alpha=0.18,
    label="压缩机停机段",
)
ax0.plot(t, temp, color="tab:green", lw=1.3, label="电池簇均温")
above = temp > REF_C + HALF
below = temp < REF_C - HALF
ax0.scatter(t[above], temp[above], s=6, color="tab:red", label="上穿段 (780 s)")
ax0.scatter(t[below], temp[below], s=6, color="tab:blue", label="下穿段 (25 s)")
ax0.set_ylabel("电池簇均温 (°C)")
ax0.set_title(
    "8D4d 压缩机启停策略：RegD 调频 6400 s，带内 87.4%，"
    "均温全程 24.35~26.18 °C"
)
ax0.legend(loc="upper left", fontsize=8, ncol=2)
ax0.grid(alpha=0.3)

# Panel 2: compressor command
ax1.plot(t, comp, color="tab:green", lw=1.0, label="压缩机指令")
ax1.fill_between(
    t, 0, 6600, where=off, color="gray", alpha=0.18, label="停机段"
)
ax1.set_ylabel("压缩机指令 (rpm)")
ax1.set_ylim(-200, 6600)
ax1.legend(loc="upper right", fontsize=8)
ax1.grid(alpha=0.3)

# Panel 3: tank & supply recovery
ax2.plot(t, f["tank_temp_c"], color="tab:purple", lw=1.2, label="水箱温度")
ax2.plot(t, f["supply_temp_c"], color="tab:cyan", lw=1.0, label="供水温度")
ax2.fill_between(
    t,
    f[["tank_temp_c", "supply_temp_c"]].min().min() - 0.2,
    f[["tank_temp_c", "supply_temp_c"]].max().max() + 0.2,
    where=off,
    color="gray",
    alpha=0.18,
    label="停机段（泵满速，回流复温）",
)
ax2.set_ylabel("冷却液温度 (°C)")
ax2.set_xlabel("时间 (min)")
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(alpha=0.3)

fig.tight_layout()
out_dir = RES / "stage8d4_full_rescale"
fig.savefig(out_dir / "fig_regd_8d4d_strategy_detail.png", dpi=150)
fig.savefig(out_dir / "fig_regd_8d4d_strategy_detail.pdf")
print("saved to", out_dir)
