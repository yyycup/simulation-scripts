"""Stage 8D4d peak-shaving plot: S0 560 A constant discharge result.

Layout follows the RegD strategy detail figure (three panels, Chinese
legend, band shading): battery temperature with baseline comparison,
compressor command with the two rare off segments, and tank/supply
temperatures.
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
base = pd.read_csv(
    RES / "case_S0_peak_8d4d_baseline_fixed_timeseries.csv"
).sort_values("time_s")

t = nmpc["time_s"].to_numpy() / 60.0
temp = nmpc["battery_avg_temp_c"].to_numpy()
comp = nmpc["compressor_command_rpm"].to_numpy()
off = comp <= 0.0

fig, (ax0, ax1, ax2) = plt.subplots(
    3,
    1,
    figsize=(11, 10),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1, 1.4]},
)

# Panel 1: battery temperature, NMPC vs baseline
ax0.axhspan(REF_C - HALF, REF_C + HALF, color="green", alpha=0.12, label="控制带 25±0.65 °C")
ax0.plot(
    base["time_s"] / 60.0,
    base["battery_avg_temp_c"],
    color="gray",
    lw=1.0,
    ls="--",
    label="定速基线（带外 6260 s）",
)
ax0.fill_between(
    t,
    13.0,
    26.5,
    where=off,
    color="gray",
    alpha=0.18,
    label="压缩机停机段（仅 235 s）",
)
ax0.plot(t, temp, color="tab:green", lw=1.4, label="NMPC + 启停（带外 0 s）")
ax0.set_ylabel("电池簇均温 (°C)")
ax0.set_title(
    "8D4d 调峰工况（S0, 560 A 恒流, 6400 s）：NMPC 全程带内，"
    "均温 24.74~25.39 °C"
)
ax0.legend(loc="lower left", fontsize=8, ncol=2)
ax0.grid(alpha=0.3)

# Panel 2: compressor command (continuous modulation dominates)
ax1.plot(t, comp, color="tab:green", lw=1.0, label="压缩机指令")
ax1.fill_between(t, 0, 6600, where=off, color="gray", alpha=0.18, label="停机段")
ax1.set_ylabel("压缩机指令 (rpm)")
ax1.set_ylim(-200, 6600)
ax1.legend(loc="upper right", fontsize=8)
ax1.grid(alpha=0.3)

# Panel 3: tank & supply
ax2.plot(t, nmpc["tank_temp_c"], color="tab:purple", lw=1.2, label="水箱温度")
ax2.plot(t, nmpc["supply_temp_c"], color="tab:cyan", lw=1.0, label="供水温度")
ax2.set_ylabel("冷却液温度 (°C)")
ax2.set_xlabel("时间 (min)")
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(RES / "fig_s0_peak_8d4d_detail.png", dpi=150)
fig.savefig(RES / "fig_s0_peak_8d4d_detail.pdf")
print("saved to", RES)
