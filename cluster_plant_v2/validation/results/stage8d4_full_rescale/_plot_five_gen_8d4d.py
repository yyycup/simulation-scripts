"""Stage 8D4d plot: five-generation RegD temperature comparison.

Generations: 8D2 legacy hardware, 8D3 first rescale, 8D4b coolant flow x2,
8D4c pump domain scaling, 8D4d compressor thermostat cycling. The 8D2 CSV
stores both passes under one controller label; the first 1280 rows are the
single-pass NMPC series (established convention). Style per project
spec: Microsoft YaHei, Chinese legend, band shading, no fine-line clutter.
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
DT = 5.0


def load(path: Path, label_filter=None, first_pass=False) -> pd.DataFrame:
    f = pd.read_csv(path).sort_values("time_s")
    if label_filter is not None:
        f = f[f["controller"] == label_filter]
    if first_pass and len(f) > 1280:
        f = f.iloc[:1280]
    return f.reset_index(drop=True)


gens = {
    "8D2 旧硬件 NMPC": load(
        RES / "physics_p_nmpc_regd_full_20260827/case_T2_regd_full_timeseries.csv",
        label_filter="physics_p_nmpc",
        first_pass=True,
    ),
    "8D3 首次扩容": load(
        RES / "stage8d3_regd_rescale/case_T2_regd_8d3_nmpc_soc_aware_timeseries.csv"
    ),
    "8D4b 冷却液流量×2": load(
        RES / "stage8d4_full_rescale/case_T2_regd_8d4_nmpc_soc_aware_timeseries.csv"
    ),
    "8D4c 泵域缩放": load(
        RES
        / "stage8d4_full_rescale/case_T2_regd_8d4c_nmpc_soc_aware_8d4c_timeseries.csv"
    ),
    "8D4d 压缩机启停": load(
        RES
        / "stage8d4_full_rescale/case_T2_regd_8d4d_nmpc_soc_aware_8d4d_timeseries.csv"
    ),
}
baseline = load(
    RES / "stage8d4_full_rescale/case_T2_regd_8d4_baseline_fixed_timeseries.csv"
)

fig, (ax0, ax1) = plt.subplots(
    2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
)

ax0.axhspan(REF_C - HALF, REF_C + HALF, color="green", alpha=0.12, label="控制带 25±0.65 °C")
ax0.plot(
    baseline["time_s"] / 60.0,
    baseline["battery_avg_temp_c"],
    color="gray",
    lw=1.0,
    ls="--",
    label="8D4 定速基线",
)
colors = ["tab:blue", "tab:cyan", "tab:orange", "tab:red", "tab:green"]
for (label, f), c in zip(gens.items(), colors):
    temp = f["battery_avg_temp_c"].to_numpy()
    oob = float(np.sum(np.abs(temp - REF_C) > HALF)) * DT
    ax0.plot(
        f["time_s"] / 60.0, temp, color=c, lw=1.2, label=f"{label}（带外 {oob:.0f} s）"
    )
ax0.set_ylabel("电池簇均温 (°C)")
ax0.set_title("RegD 调频工况：五代硬件/控制演进的温度对比（T2, 6400 s）")
ax0.legend(loc="upper left", fontsize=8, ncol=2)
ax0.grid(alpha=0.3)

d4d = gens["8D4d 压缩机启停"]
off = d4d["compressor_command_rpm"].to_numpy() <= 0.0
ax1.plot(d4d["time_s"] / 60.0, d4d["compressor_command_rpm"], color="tab:green", lw=1.0)
ax1.fill_between(
    d4d["time_s"] / 60.0,
    0,
    np.where(off, 6000, 0),
    color="gray",
    alpha=0.15,
    label="停机段",
)
ax1.set_ylabel("压缩机指令 (rpm)")
ax1.set_xlabel("时间 (min)")
ax1.set_ylim(-200, 6600)
ax1.legend(loc="upper right", fontsize=8)
ax1.grid(alpha=0.3)

fig.tight_layout()
out_dir = RES / "stage8d4_full_rescale"
fig.savefig(out_dir / "fig_regd_five_generation_comparison_8d4d.png", dpi=150)
fig.savefig(out_dir / "fig_regd_five_generation_comparison_8d4d.pdf")
print("saved to", out_dir)
