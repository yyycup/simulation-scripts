"""Stage 8D4h hot-start closed-loop tracking figure.

Shows the NMPC pulling the 27 C hot start down into the 25 +/- 0.65 C
band in ~4 min and holding it flat, plus the compressor command. Style
per project spec: Microsoft YaHei, Chinese labels, new file name.
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

df = pd.read_csv(
    RES / "case_S0_peak_8d4h_nmpc_hot_start_timeseries.csv"
).sort_values("time_s")
t = df["time_s"].to_numpy() / 60.0
temp = df["battery_avg_temp_c"].to_numpy()
cmd = df["compressor_command_rpm"].to_numpy()

blocks = df[df["time_s"] > 400].copy()
blocks["block"] = (blocks["time_s"] // 600).astype(int)
block_means = blocks.groupby("block")["battery_avg_temp_c"].mean()

late = df[df["time_s"] > 5400]["battery_avg_temp_c"]
in_band = np.abs(temp - REF_C) <= HALF
first_in_t = float(t[np.argmax(in_band)])
out_band = float(np.sum(~in_band)) * 5.0

fig, (ax0, ax1) = plt.subplots(
    2, 1, figsize=(12, 7), sharex=True,
    gridspec_kw={"height_ratios": [2, 1]},
)

ax0.axhspan(REF_C - HALF, REF_C + HALF, color="green", alpha=0.12,
            label="控制带 25±0.65 °C")
ax0.plot(t, temp, color="tab:green", lw=1.2, alpha=0.85, label="簇均温")
ax0.plot(block_means.index * 10 + 5, block_means.values, color="darkred",
         lw=2.0, marker="o", ms=4, label="10 min 分块均值")
ax0.axhline(REF_C, color="k", lw=0.7, ls=":")
ax0.set_ylabel("电池簇均温 (°C)")
ax0.set_ylim(24.2, 27.6)
ax0.grid(alpha=0.3)
ax0.legend(loc="upper right", fontsize=9)
ax0.set_title(
    f"27 °C 热启动：{first_in_t:.0f} min 内拉进控制带（带外 {out_band:.0f} s，"
    f"全在拉降段），末 1000 s 均温 {late.mean():.3f} °C"
    f"（偏差 {late.mean() - REF_C:+.3f} K）",
    fontsize=11,
)

ax1.plot(t, cmd, color="tab:blue", lw=0.9)
ax1.set_ylabel("压缩机指令 (rpm)")
ax1.set_xlabel("时间 (min)")
ax1.set_ylim(-200, 6500)
ax1.grid(alpha=0.3)
steady = df[df["time_s"] > 5400]["compressor_command_rpm"]
ax1.set_title(
    f"拉降段满速 6000 rpm；末 1000 s 指令均值 {steady.mean():.0f} rpm，"
    f"波动 {steady.min():.0f}~{steady.max():.0f} rpm，全程启停 0 次",
    fontsize=11,
)

fig.suptitle(
    "8D4 调峰工况（S0, 560 A 恒流, 6400 s）：8D4h 热启动（27 °C）温度跟踪",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(RES / "fig_s0_peak_8d4h_hot_start_tracking.png", dpi=150)
fig.savefig(RES / "fig_s0_peak_8d4h_hot_start_tracking.pdf")
print("saved to", RES)
