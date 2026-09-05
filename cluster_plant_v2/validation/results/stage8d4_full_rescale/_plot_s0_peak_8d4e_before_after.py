"""Stage 8D4e peak-shaving before/after comparison figure.

Two rows (pre-fix 8d4d artifact vs post-fix 8d4e artifact), two columns
(temperature with control band, compressor command). Shows the capacity
mismatch symptom (limit-cycle command oscillation) is gone post-fix
while a slow residual temperature drift remains. Style per project
spec: Microsoft YaHei, Chinese labels; new file name to avoid caching.
"""

from pathlib import Path
import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RES = Path("cluster_plant_v2/validation/results/stage8d4_full_rescale")
REF_C, HALF = 25.0, 0.65

runs = {
    "修复前（旧工件，容量低估 82~88%）": "case_S0_peak_8d4d_nmpc_soc_aware_8d4d_timeseries.csv",
    "修复后（8D4e 新工件，验证误差 ±1%）": "case_S0_peak_8d4e_nmpc_soc_aware_8d4d_timeseries.csv",
}

fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)

for row, (label, fname) in enumerate(runs.items()):
    df = pd.read_csv(RES / fname).sort_values("time_s")
    t = df["time_s"].to_numpy() / 60.0
    temp = df["battery_avg_temp_c"].to_numpy()
    cmd = df["compressor_command_rpm"].to_numpy()

    ax = axes[row, 0]
    ax.axhspan(REF_C - HALF, REF_C + HALF, color="green", alpha=0.12)
    ax.plot(t, temp, color="tab:green", lw=1.4)
    ax.axhline(REF_C, color="k", lw=0.7, ls=":")
    late = df[df["time_s"] > 5400]["battery_avg_temp_c"]
    ax.set_ylabel("电池簇均温 (°C)")
    ax.set_ylim(24.0, 26.2)
    ax.grid(alpha=0.3)
    ax.set_title(
        f"{label}：末 1000 s 均温 {late.mean():.2f} °C（偏差 {late.mean() - REF_C:+.2f} K），带外 0 s",
        fontsize=10,
    )

    ax = axes[row, 1]
    ax.plot(t, cmd, color="tab:blue", lw=0.9)
    steady = df[df["time_s"] > 5400]["compressor_command_rpm"]
    ax.set_ylabel("压缩机指令 (rpm)")
    ax.set_ylim(-200, 6500)
    ax.grid(alpha=0.3)
    ax.set_title(
        f"末 1000 s 指令：均值 {steady.mean():.0f} rpm，波动 {steady.min():.0f}~{steady.max():.0f}",
        fontsize=10,
    )

for ax in axes[1]:
    ax.set_xlabel("时间 (min)")
axes[0, 0].legend(["控制带 25±0.65 °C", "", "NMPC"], loc="lower right", fontsize=9)

fig.suptitle(
    "8D4 调峰工况（S0, 560 A 恒流, 6400 s）：容量重标定前后对比",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(RES / "fig_s0_peak_8d4e_before_after.png", dpi=150)
fig.savefig(RES / "fig_s0_peak_8d4e_before_after.pdf")
print("saved to", RES)
