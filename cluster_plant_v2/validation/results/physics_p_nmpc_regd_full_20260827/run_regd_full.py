"""1280-step (6400 s) frequency-regulation T2 RegD: baseline vs static vs SOC-aware.

PJM RegD profile x 1120 A over the full 2 h source window; currents cross zero
(charging intervals included), so the SOC-aware preview exercises the charge
direction of the HPPC resistance tables while the static preview stays
direction-blind. Pass 1 calibrates the two heat-preview conventions against
open-loop Plant truth; passes 2-3 run the two NMPC variants closed loop.
"""

from pathlib import Path
import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cluster_plant_v2.control.physics_p_nmpc_model import (
    ClusterHeatGenerationPreview,
    battery_heat_generation_preview_w,
)
from cluster_plant_v2.validation.validate_domp_mpc import (
    BASELINE_COMPRESSOR_RPM,
    BASELINE_PUMP_RPM,
    build_cases,
    case_currents,
)
from cluster_plant_v2.validation.validate_final_cluster_plant import (
    FAN_SPEED_RPM,
    build_final_plant,
)
from cluster_plant_v2.validation.validate_physics_p_nmpc import run_baseline, run_nmpc
from cluster_plant_v2.validation.validate_refrigerated_cluster_cooling_loop import (
    AMBIENT_TEMPERATURE_K,
)

HERE = Path(__file__).resolve().parent
S0_SUMMARY = HERE.parent / "physics_p_nmpc_s0_socaware_1280steps_20260827" / "three_way_summary.csv"
DT = 5.0
STEPS = 1280
DURATION = STEPS * DT

case = [c for c in build_cases() if c.case_id == "T2_regd"][0]
currents = case_currents(case, STEPS, DT)
print(
    f"running {case.case_id} for {DURATION:.0f} s ({STEPS} steps); "
    f"min/max/mean current {currents.min():.1f}/{currents.max():.1f}/"
    f"{currents.mean():.1f} A",
    flush=True,
)

# ---- Pass 0: fixed-speed closed-loop baseline ------------------------------
print("pass 0: fixed-speed baseline ...", flush=True)
baseline = run_baseline(case, duration_s=DURATION, dt_s=DT)
print("baseline done", flush=True)

# ---- Pass 1: open-loop Plant truth for Qgen / SOC calibration --------------
print("pass 1: open-loop plant under fixed speeds ...", flush=True)
plant = build_final_plant(
    BASELINE_COMPRESSOR_RPM, initial_cluster_current_a=float(currents[0]), direction="forward"
)
battery = plant.cluster.packs[0].battery
checkpoint_step_indices = list(range(159, STEPS, 160))
truth_rows = []
for step_index in range(STEPS):
    result = plant.step(
        dt_s=DT,
        cluster_current_a=float(currents[step_index]),
        pump_speed_rpm=BASELINE_PUMP_RPM,
        compressor_speed_command_rpm=BASELINE_COMPRESSOR_RPM,
        fan_speed_rpm=FAN_SPEED_RPM,
        ambient_temperature_k=AMBIENT_TEMPERATURE_K,
        direction="forward",
    )
    if step_index in checkpoint_step_indices:
        time_s = (step_index + 1) * DT
        truth_rows.append({
            "time_s": time_s,
            "plant_q_gen_w_per_pack": float(battery.q_gen_total),
            "plant_soc_mean": float(np.mean(battery.get_soc_array())),
        })
del plant, battery
print("open-loop truth done", flush=True)

# ---- Pass 2: NMPC with the legacy static heat preview ----------------------
print("pass 2: static-preview NMPC ...", flush=True)
static_frame = run_nmpc(case, duration_s=DURATION, dt_s=DT)
print("static NMPC done", flush=True)

# ---- Pass 3: NMPC with the SOC-aware heat preview ---------------------------
print("pass 3: SOC-aware NMPC ...", flush=True)
socaware_frame = run_nmpc(
    case,
    duration_s=DURATION,
    dt_s=DT,
    heat_preview=ClusterHeatGenerationPreview(currents, DT),
)
print("SOC-aware NMPC done", flush=True)

# ---- Calibration table -------------------------------------------------------
preview = ClusterHeatGenerationPreview(currents, DT)
calibration = pd.DataFrame(truth_rows)
calibration["plant_mean_current_a"] = [
    float(currents[int(t / DT) - 1]) for t in calibration.time_s
]
calibration["soc_aware_preview_w"] = [
    float(preview(t - DT, 560.0)) for t in calibration.time_s
]
calibration["static_preview_w"] = battery_heat_generation_preview_w(560.0)
calibration["preview_soc"] = [preview.soc_at(t - DT) for t in calibration.time_s]
calibration.to_csv(HERE / "qgen_calibration_checkpoints.csv", index=False)
with pd.option_context("display.width", 220):
    print(calibration.to_string(index=False))

# ---- Three-way comparison -----------------------------------------------------
frames = {
    "baseline_fixed": baseline.copy(),
    "nmpc_static_qgen": static_frame.copy(),
    "nmpc_soc_aware": socaware_frame.copy(),
}
timeseries = pd.concat(frames.values(), ignore_index=True)
timeseries.to_csv(HERE / "case_T2_regd_full_timeseries.csv", index=False)


def metrics(frame: pd.DataFrame, name: str) -> dict:
    power = frame.compressor_power_w + frame.pump_power_w
    deviation = (frame.battery_avg_temp_c - 25.0).abs()
    return {
        "controller": name,
        "final_temp_c": round(frame.battery_avg_temp_c.iloc[-1], 3),
        "max_temp_c": round(frame.battery_max_temp_c.max(), 3),
        "max_dev_k": round(float(deviation.max()), 3),
        "out_of_band_s": int((deviation > 0.65).sum() * DT),
        "comp_kwh": round(frame.compressor_power_w.sum() * DT / 3.6e6, 4),
        "pump_kwh": round(frame.pump_power_w.sum() * DT / 3.6e6, 4),
        "total_kwh": round(power.sum() * DT / 3.6e6, 4),
    }


summary = pd.DataFrame([metrics(f, n) for n, f in frames.items()])
summary["scenario"] = "T2_regd_6400s"
s0 = pd.read_csv(S0_SUMMARY)
s0["scenario"] = "S0_peakshave_6400s"
combined = pd.concat([summary, s0], ignore_index=True)
combined.to_csv(HERE / "peak_and_regd_full_summary.csv", index=False)
with pd.option_context("display.width", 240):
    print(summary.to_string(index=False))
    print()
    print(combined.to_string(index=False))

# ---- Figure ---------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 10, "legend.fontsize": 8.5, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "lines.linewidth": 1.4,
})
STYLE = {
    "baseline_fixed": dict(color="#0072B2", ls="-", lw=1.4),
    "nmpc_static_qgen": dict(color="#D55E00", ls="--", lw=1.4),
    "nmpc_soc_aware": dict(color="#009E73", ls="-", lw=1.7),
}
LABEL = {
    "baseline_fixed": "Fixed baseline (4000/3600 rpm)",
    "nmpc_static_qgen": "NMPC, static Qgen preview",
    "nmpc_soc_aware": "NMPC, SOC-aware Qgen preview",
}

fig, axes = plt.subplots(5, 1, figsize=(7.2, 9.2), sharex=True,
                         gridspec_kw={"height_ratios": [1.3, 0.75, 1.05, 1.05, 1.0],
                                      "hspace": 0.16})
ax = axes[0]
ax.plot(timeseries[timeseries.controller == "baseline_fixed"].time_s / 60,
        currents, color="#888888", lw=1.0)
ax.set_ylabel("Current (A)")
ax.set_title("Frequency regulation T2 RegD x1120 A over 6400 s (1280 steps)", loc="left")

ax = axes[1]
ax.axhspan(24.35, 25.65, color="#2A9D8F", alpha=0.10, zorder=0)
ax.axhline(25.0, color="#444444", lw=0.8, ls="--", zorder=1)
for name in frames:
    ax.plot(frames[name].time_s / 60, frames[name].battery_avg_temp_c,
            label=LABEL[name], **STYLE[name])
ax.set_ylabel("Battery avg.\ntemp (\u00b0C)")
ax.legend(loc="upper left")

ax = axes[2]
ax.step(frames["baseline_fixed"].time_s / 60,
        frames["baseline_fixed"].compressor_command_rpm,
        where="post", **STYLE["baseline_fixed"])
ax.step(static_frame.time_s / 60, static_frame.compressor_command_rpm,
        where="post", **STYLE["nmpc_static_qgen"])
ax.step(socaware_frame.time_s / 60, socaware_frame.compressor_command_rpm,
        where="post", **STYLE["nmpc_soc_aware"])
ax.set_ylabel("Compressor cmd (rpm)")

ax = axes[3]
ax.plot(calibration.time_s / 60, calibration.plant_q_gen_w_per_pack * 5,
        "o-", color="#444444", ms=4, label="Plant truth (open loop, 5 packs)")
ax.plot(calibration.time_s / 60, calibration.soc_aware_preview_w * 5,
        "s--", color="#009E73", ms=4, label="SOC-aware preview x5")
ax.plot(calibration.time_s / 60,
        calibration.static_preview_w * 5 * np.ones(len(calibration)),
        "^--", color="#D55E00", ms=4, label="Static preview x5")
ax.set_ylabel("Heat gen (W)")
ax.legend(loc="upper left")

ax = axes[4]
for name, frame in frames.items():
    energy = (frame.compressor_power_w + frame.pump_power_w).cumsum() * DT / 3.6e6
    ax.plot(frame.time_s / 60, energy,
            label=f"{LABEL[name]} ({energy.iloc[-1]:.2f} kWh)", **STYLE[name])
ax.set_ylabel("Cumulative total energy (kWh)")
ax.set_xlabel("Time (min)")
ax.set_xlim(0, DURATION / 60)
ax.legend(loc="upper left")

fig.align_ylabels(axes)
fig.savefig(HERE / "fig_T2_regd_full_three_way.png", dpi=300)
fig.savefig(HERE / "fig_T2_regd_full_three_way.pdf")
print("saved:", HERE / "fig_T2_regd_full_three_way.png")
