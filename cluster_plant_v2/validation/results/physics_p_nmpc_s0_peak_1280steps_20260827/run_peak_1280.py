"""1280-step (6400 s) peak-shaving S0 case: Physics-P NMPC vs fixed baseline."""

from pathlib import Path
import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cluster_plant_v2.validation.validate_domp_mpc import build_cases, summarize
from cluster_plant_v2.validation.validate_physics_p_nmpc import (
    plot_case, run_baseline, run_nmpc,
)

HERE = Path(__file__).resolve().parent
DURATION = 6400.0  # 1280 steps x 5 s
DT = 5.0

case = [c for c in build_cases() if c.case_id == "S0_constant"][0]
print(f"running {case.case_id} for {DURATION:.0f} s (1280 steps)", flush=True)

baseline = run_baseline(case, duration_s=DURATION, dt_s=DT)
print("baseline done", flush=True)

nmpc = run_nmpc(case, duration_s=DURATION, dt_s=DT)
print("nmpc done", flush=True)

frame = pd.concat([baseline, nmpc], ignore_index=True)
frame.to_csv(HERE / "case_S0_peak_1280_steps_timeseries.csv", index=False)

summary = pd.DataFrame([
    summarize(case.case_id, baseline, DT),
    summarize(case.case_id, nmpc, DT),
])
summary.to_csv(HERE / "peak_1280_summary.csv", index=False)
with pd.option_context("display.width", 240):
    print(summary.to_string(index=False))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 10, "legend.fontsize": 8.5, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "lines.linewidth": 1.4,
})
B, N = baseline, nmpc
fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.6), sharex=True,
                         gridspec_kw={"height_ratios": [1.3, 1.1, 1.0, 1.0],
                                      "hspace": 0.16})

ax = axes[0]
ax.axhspan(24.35, 25.65, color="#2A9D8F", alpha=0.10, zorder=0)
ax.axhline(25.0, color="#444444", lw=0.8, ls="--", zorder=1)
ax.plot(N.time_s / 60, N.battery_avg_temp_c, color="#D55E00",
        label="Physics-P NMPC")
ax.plot(B.time_s / 60, B.battery_avg_temp_c, color="#0072B2",
        label="Fixed baseline (4000/3600 rpm)")
band_err_b = (B.battery_avg_temp_c - 25.0).abs()
band_err_n = (N.battery_avg_temp_c - 25.0).abs()
ax.text(0.01, 0.03,
        f"|deviation| max: baseline {band_err_b.max():.3f} K, "
        f"NMPC {band_err_n.max():.3f} K\n"
        f"out of \u00b10.65 K band: baseline "
        f"{int((band_err_b > 0.65).sum() * DT)} s, "
        f"NMPC {int((band_err_n > 0.65).sum() * DT)} s",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC", alpha=0.9))
ax.set_ylabel("Battery avg. temp (\u00b0C)")
ax.legend(loc="lower right")
ax.set_title("Peak-shaving S0, 560 A constant discharge over 6400 s (1280 steps)", loc="left")

ax = axes[1]
ax.step(N.time_s / 60, N.compressor_command_rpm, where="post", color="#D55E00")
ax.plot(B.time_s / 60, B.compressor_command_rpm, color="#0072B2")
ax.set_ylabel("Compressor cmd (rpm)")

ax = axes[2]
ax.step(N.time_s / 60, N.pump_command_rpm, where="post", color="#D55E00")
ax.plot(B.time_s / 60, B.pump_command_rpm, color="#0072B2")
ax.set_ylabel("Pump cmd (rpm)")

ax = axes[3]
eb = (B.compressor_power_w + B.pump_power_w).cumsum() * DT / 3.6e6
en = (N.compressor_power_w + N.pump_power_w).cumsum() * DT / 3.6e6
ax.plot(B.time_s / 60, B.compressor_power_w + B.pump_power_w,
        color="#0072B2", alpha=0.5, lw=1.0)
ax.plot(N.time_s / 60, N.compressor_power_w + N.pump_power_w,
        color="#D55E00", alpha=0.5, lw=1.0)
ax.plot(B.time_s / 60, eb, color="#0072B2", lw=1.8,
        label=f"Baseline cumulative {eb.iloc[-1]:.3f} kWh")
ax.plot(N.time_s / 60, en, color="#D55E00", lw=1.8,
        label=f"NMPC cumulative {en.iloc[-1]:.3f} kWh")
ax.set_ylabel("Power (W), cum. kWh")
ax.set_xlabel("Time (min)")
ax.set_xlim(0, 106.7)
ax.legend(loc="upper left")

fig.align_ylabels(axes)
fig.savefig(HERE / "fig_S0_peak_1280_comparison.png", dpi=300)
fig.savefig(HERE / "fig_S0_peak_1280_comparison.pdf")
plot_case(case.case_id + "_peak1280", frame, HERE)
print("saved:", HERE / "fig_S0_peak_1280_comparison.png")
