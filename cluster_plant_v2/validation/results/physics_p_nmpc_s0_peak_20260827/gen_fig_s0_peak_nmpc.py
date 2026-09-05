"""Publication-style comparison figure: Physics-P NMPC vs fixed baseline, S0 peak-shaving case."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "lines.linewidth": 1.6,
})

NMPC_COLOR = "#D55E00"      # Okabe-Ito vermillion — our method
BASE_COLOR = "#0072B2"      # Okabe-Ito blue — baseline
PUMP_ALPHA = 0.85
TARGET_C = 25.0
BAND_K = 0.65

frame = pd.read_csv(HERE / "case_S0_constant_timeseries.csv")
nmpc = frame[frame.controller == "physics_p_nmpc"]
base = frame[frame.controller == "baseline_fixed"]

fig, axes = plt.subplots(
    4, 1, figsize=(6.75, 6.2), sharex=True,
    gridspec_kw={"height_ratios": [1.3, 1.0, 1.0, 1.0], "hspace": 0.16},
)

# --- Panel 1: battery average temperature (zoomed) ---
ax = axes[0]
ax.axhspan(TARGET_C - BAND_K, TARGET_C + BAND_K, color="#2A9D8F", alpha=0.10, zorder=0)
ax.axhline(TARGET_C, color="#444444", lw=0.8, ls="--", zorder=1)
ax.plot(base.time_s, base.battery_avg_temp_c, color=BASE_COLOR, label="Fixed baseline (4000 / 3600 rpm)")
ax.plot(nmpc.time_s, nmpc.battery_avg_temp_c, color=NMPC_COLOR, label="Physics-P NMPC")
ax.set_ylim(24.92, 25.34)
ax.set_ylabel("Battery avg. temp (°C)")
ax.annotate(f"{base.battery_avg_temp_c.iloc[-1]:.2f} °C",
            xy=(base.time_s.iloc[-1], base.battery_avg_temp_c.iloc[-1]),
            xytext=(-4, -14), textcoords="offset points",
            ha="right", fontsize=8, color=BASE_COLOR)
ax.annotate(f"{nmpc.battery_avg_temp_c.iloc[-1]:.2f} °C",
            xy=(nmpc.time_s.iloc[-1], nmpc.battery_avg_temp_c.iloc[-1]),
            xytext=(-4, 8), textcoords="offset points",
            ha="right", fontsize=8, color=NMPC_COLOR)
ax.legend(loc="upper left", ncol=1)
ax.set_title("Peak-shaving case S0: 560 A constant discharge, 600 s (target 25 °C ± 0.65 K)", loc="left")

# --- Panel 2: compressor speed ---
ax = axes[1]
ax.plot(base.time_s, base.compressor_command_rpm, color=BASE_COLOR, lw=1.2)
ax.step(nmpc.time_s, nmpc.compressor_command_rpm, color=NMPC_COLOR, where="post")
ax.set_ylabel("Compressor cmd (rpm)")
ax.set_yticks([3000, 3500, 4000, 4500])
ax.annotate("fixed 4000 rpm", xy=(605, 4000), ha="left", va="center",
            fontsize=8, color=BASE_COLOR)

# --- Panel 3: pump speed ---
ax = axes[2]
ax.plot(base.time_s, base.pump_command_rpm, color=BASE_COLOR, lw=1.2)
ax.step(nmpc.time_s, nmpc.pump_command_rpm, color=NMPC_COLOR, where="post", alpha=PUMP_ALPHA)
ax.set_ylabel("Pump cmd (rpm)")
ax.set_yticks([3600, 4200, 4800])
ax.annotate("fixed 3600 rpm", xy=(605, 3600), ha="left", va="center",
            fontsize=8, color=BASE_COLOR)

# --- Panel 4: total electrical power ---
ax = axes[3]
base_power = base.compressor_power_w + base.pump_power_w
nmpc_power = nmpc.compressor_power_w + nmpc.pump_power_w
ax.plot(base.time_s, base_power, color=BASE_COLOR, label="Fixed baseline")
ax.plot(nmpc.time_s, nmpc_power, color=NMPC_COLOR, alpha=PUMP_ALPHA, label="Physics-P NMPC")
ax.set_ylabel("Total power (W)")
ax.set_xlabel("Time (s)")
ax.set_xlim(0, 600)
ax.set_ylim(500, 1050)
ax.legend(loc="upper left")
e_base = base_power.sum() * 5 / 3.6e6
e_nmpc = nmpc_power.sum() * 5 / 3.6e6
ax.text(
    0.985, 0.04,
    f"Compressor: 0.1483 \u2192 0.1351 kWh (\u22128.9%)\n"
    f"Pump: 0.0033 \u2192 0.0077 kWh\n"
    f"Total: {e_base:.4f} \u2192 {e_nmpc:.4f} kWh (\u22125.8%)",
    transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC", alpha=0.9),
)

fig.align_ylabels(axes)
fig.savefig(HERE / "fig_S0_peak_nmpc_vs_baseline.png", dpi=300)
fig.savefig(HERE / "fig_S0_peak_nmpc_vs_baseline.pdf")
print("saved:", HERE / "fig_S0_peak_nmpc_vs_baseline.png")
