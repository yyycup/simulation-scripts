"""Stage 8D4d peak-shaving result figure (S0, 560 A constant, 6400 s).

Three-panel result overview from the pre-rerun (8d4d) artifacts;
fixed-speed baseline traces removed per request (kept only as the
energy comparison bar): panel 1 NMPC temperature with control band,
panel 2 compressor command, panel 3 energy bar chart (pump power uses
the 8x rescale correction). Style per project spec: Microsoft YaHei,
Chinese labels. New file names to avoid viewer caching.
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
PUMP_POWER_CORRECTION = 8.0
DT = 5.0

base = pd.read_csv(RES / "case_S0_peak_8d4d_baseline_fixed_timeseries.csv").sort_values("time_s")
nmpc = pd.read_csv(RES / "case_S0_peak_8d4d_nmpc_soc_aware_8d4d_timeseries.csv").sort_values("time_s")

tb = base["time_s"].to_numpy() / 60.0
tn = nmpc["time_s"].to_numpy() / 60.0
temp_b = base["battery_avg_temp_c"].to_numpy()
temp_n = nmpc["battery_avg_temp_c"].to_numpy()
comp_b = base["compressor_command_rpm"].to_numpy()
comp_n = nmpc["compressor_command_rpm"].to_numpy()

def energy_kwh(frame):
    comp = float(frame["compressor_power_w"].sum()) * DT / 3.6e6
    pump = float(frame["pump_power_w"].sum()) * PUMP_POWER_CORRECTION * DT / 3.6e6
    return comp, pump

comp_kb, pump_kb = energy_kwh(base)
comp_kn, pump_kn = energy_kwh(nmpc)

fig, axes = plt.subplots(
    3, 1, figsize=(11, 9.5), gridspec_kw={"height_ratios": [1.25, 0.9, 0.9]}
)

# ---- panel 1: temperature ----
ax = axes[0]
ax.axhspan(REF_C - HALF, REF_C + HALF, color="green", alpha=0.12, label="控制带 25±0.65 °C")
ax.plot(tn, temp_n, color="tab:green", lw=1.4, label="NMPC + 启停（带外 0 s）")
ax.axhline(REF_C, color="k", lw=0.7, ls=":")
ax.set_ylabel("电池簇均温 (°C)")
ax.set_ylim(24.0, 26.2)
ax.set_title("8D4d 调峰工况（S0, 560 A 恒流, 6400 s）：NMPC 全程带内，均温 24.74~25.39 °C")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

# ---- panel 2: compressor command ----
ax = axes[1]
ax.plot(tn, comp_n, color="tab:green", lw=1.0, label="NMPC 指令")
ax.set_ylabel("压缩机指令 (rpm)")
ax.set_ylim(-200, 6500)
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)

# ---- panel 3: energy bars ----
ax = axes[2]
labels = ["定速基线", "NMPC + 启停"]
comp_vals = [comp_kb, comp_kn]
pump_vals = [pump_kb, pump_kn]
x = np.arange(len(labels))
w = 0.32
ax.bar(x - w / 2, comp_vals, width=w, color="tab:blue", label="压缩机")
ax.bar(x + w / 2, pump_vals, width=w, color="tab:orange", label="泵（×8 修正）")
for i in range(len(labels)):
    total = comp_vals[i] + pump_vals[i]
    ax.text(x[i] - w / 2, comp_vals[i] + 0.08, f"{comp_vals[i]:.2f}", ha="center", fontsize=8)
    ax.text(x[i] + w / 2, pump_vals[i] + 0.08, f"{pump_vals[i]:.2f}", ha="center", fontsize=8)
    ax.text(x[i], max(comp_vals[i], pump_vals[i]) + 0.55, f"合计 {total:.2f} kWh", ha="center", fontsize=9)
saving = 1.0 - (comp_kn + pump_kn) / (comp_kb + pump_kb)
ax.set_xticks(x, labels)
ax.set_ylabel("能耗 (kWh)")
ax.set_ylim(0, max(comp_vals) + 1.6)
ax.set_title(f"NMPC 总能耗 {comp_kn + pump_kn:.2f} kWh，较定速基线节省 {saving * 100:.0f}%")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3, axis="y")

axes[0].set_xticklabels([])
axes[1].set_xlabel("时间 (min)")
for axi in axes[:2]:
    axi.set_xlim(0, 107)
    axi.set_xticks(np.arange(0, 107, 15))

fig.tight_layout()
fig.savefig(RES / "fig_s0_peak_8d4d_result_v3.png", dpi=150)
fig.savefig(RES / "fig_s0_peak_8d4d_result_v3.pdf")
print("saved to", RES)
