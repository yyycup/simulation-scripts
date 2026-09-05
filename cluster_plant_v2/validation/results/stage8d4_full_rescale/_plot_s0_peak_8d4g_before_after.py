"""Stage 8D4g no-cycling before/after comparison figure.

Two rows (8D4f with the 8D4d thermostat cycling overlay vs 8D4g with
the overlay removed and only a band-floor safety net), two columns
(temperature with control band plus 10-min block means, compressor
command). Shows the ~0.5 K start/stop limit cycle disappears once the
NMPC is allowed to modulate continuously. Style per project spec:
Microsoft YaHei, Chinese labels; new file name to avoid caching.
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

runs = {
    "8D4f（带启停监督层，极限环波浪）":
        "case_S0_peak_8d4e_nmpc_soc_aware_8d4d_timeseries.csv",
    "8D4g（去掉启停层，连续调制）":
        "case_S0_peak_8d4g_nmpc_no_cycling_timeseries.csv",
}

fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)

for row, (label, fname) in enumerate(runs.items()):
    df = pd.read_csv(RES / fname).sort_values("time_s")
    t = df["time_s"].to_numpy() / 60.0
    temp = df["battery_avg_temp_c"].to_numpy()
    cmd = df["compressor_command_rpm"].to_numpy()

    blocks = df[df["time_s"] > 400].copy()
    blocks["block"] = (blocks["time_s"] // 600).astype(int)
    block_means = blocks.groupby("block")["battery_avg_temp_c"].mean()

    ax = axes[row, 0]
    ax.axhspan(REF_C - HALF, REF_C + HALF, color="green", alpha=0.12)
    ax.plot(t, temp, color="tab:green", lw=1.2, alpha=0.75)
    ax.plot(
        block_means.index * 10 + 5,
        block_means.values,
        color="darkred",
        lw=2.0,
        marker="o",
        ms=4,
    )
    ax.axhline(REF_C, color="k", lw=0.7, ls=":")
    late = df[df["time_s"] > 5400]["battery_avg_temp_c"]
    out_band = float(np.sum(np.abs(temp - REF_C) > HALF)) * 5.0
    ax.set_ylabel("电池簇均温 (°C)")
    ax.set_ylim(24.0, 26.2)
    ax.grid(alpha=0.3)
    ax.set_title(
        f"{label}：末 1000 s 均温 {late.mean():.2f} °C"
        f"（偏差 {late.mean() - REF_C:+.2f} K），极值 "
        f"{temp.min():.2f}~{temp.max():.2f} °C，带外 {out_band:.0f} s",
        fontsize=10,
    )

    ax = axes[row, 1]
    ax.plot(t, cmd, color="tab:blue", lw=0.9)
    steady = df[df["time_s"] > 5400]["compressor_command_rpm"]
    switches = int(np.sum(np.diff((cmd > 0).astype(int)) != 0))
    ax.set_ylabel("压缩机指令 (rpm)")
    ax.set_ylim(-200, 6500)
    ax.grid(alpha=0.3)
    ax.set_title(
        f"启停 {switches} 次；末 1000 s 指令均值 {steady.mean():.0f} rpm，"
        f"波动 {steady.min():.0f}~{steady.max():.0f}",
        fontsize=10,
    )

for ax in axes[1]:
    ax.set_xlabel("时间 (min)")
axes[0, 0].legend(
    ["控制带 25±0.65 °C", "", "均温", "10 min 分块均值"],
    loc="lower right",
    fontsize=9,
)

fig.suptitle(
    "8D4 调峰工况（S0, 560 A 恒流, 6400 s）：去掉启停层前后对比",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(RES / "fig_s0_peak_8d4g_no_cycling_before_after.png", dpi=150)
fig.savefig(RES / "fig_s0_peak_8d4g_no_cycling_before_after.pdf")
print("saved to", RES)
