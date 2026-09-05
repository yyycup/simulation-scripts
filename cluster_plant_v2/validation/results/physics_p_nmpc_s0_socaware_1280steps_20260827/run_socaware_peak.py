"""1280-step peak-shaving: SOC-aware Qgen preview NMPC vs static preview vs baseline.

Pass 1 calibrates the two heat-preview conventions against open-loop Plant
truth (per-pack ``q_gen_total`` and branch SOC). Pass 2 runs the closed loop.
The NMPC plant in pass 2 starts fresh, so its preview uses the same fresh
initial SOC / temperature as ``build_final_plant`` provides.
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
)
from cluster_plant_v2.validation.validate_final_cluster_plant import (
    FAN_SPEED_RPM,
    build_final_plant,
)
from cluster_plant_v2.validation.validate_physics_p_nmpc import run_nmpc
from cluster_plant_v2.validation.validate_refrigerated_cluster_cooling_loop import (
    AMBIENT_TEMPERATURE_K,
)

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / "physics_p_nmpc_s0_peak_1280steps_20260827"
DT = 5.0
STEPS = 1280
DURATION = STEPS * DT

case = [c for c in build_cases() if c.case_id == "S0_constant"][0]
currents = np.full(STEPS, 560.0)  # mirrors case_currents(S0_constant)

# ---- Pass 1: open-loop Plant truth for Qgen / SOC calibration -------------
print("pass 1: open-loop plant under fixed speeds ...", flush=True)
plant = build_final_plant(
    BASELINE_COMPRESSOR_RPM, initial_cluster_current_a=560.0, direction="forward"
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

# Fresh-state preview matches the closed-loop NMPC plant start.
preview = ClusterHeatGenerationPreview(currents, DT)
calibration = pd.DataFrame(truth_rows)
calibration["soc_aware_preview_w"] = [
    float(preview(t, 560.0)) for t in calibration.time_s
]
calibration["static_preview_w"] = battery_heat_generation_preview_w(560.0)
calibration["preview_soc"] = [preview.soc_at(t) for t in calibration.time_s]
calibration.to_csv(HERE / "qgen_calibration_checkpoints.csv", index=False)
with pd.option_context("display.width", 200):
    print(calibration.to_string(index=False))

# ---- Pass 2: closed-loop SOC-aware NMPC -----------------------------------
print("\npass 2: SOC-aware NMPC closed loop ...", flush=True)
heat_preview = ClusterHeatGenerationPreview(currents, DT)
nmpc = run_nmpc(
    case,
    duration_s=DURATION,
    dt_s=DT,
    heat_preview=heat_preview,
)
print("SOC-aware NMPC done", flush=True)

# ---- Three-way comparison --------------------------------------------------
old = pd.read_csv(PREV / "case_S0_peak_1280_steps_timeseries.csv")
frames = {
    "baseline_fixed": old[old.controller == "baseline_fixed"].copy(),
    "nmpc_static_qgen": old[old.controller == "physics_p_nmpc"].copy(),
    "nmpc_soc_aware": nmpc.copy(),
}
nmpc.to_csv(HERE / "case_S0_peak_1280_socaware_timeseries.csv", index=False)


def metrics(frame: pd.DataFrame) -> dict:
    power = frame.compressor_power_w + frame.pump_power_w
    deviation = (frame.battery_avg_temp_c - 25.0).abs()
    return {
        "controller": frame.controller.iloc[0],
        "final_temp_c": round(frame.battery_avg_temp_c.iloc[-1], 3),
        "max_temp_c": round(frame.battery_max_temp_c.max(), 3),
        "max_dev_k": round(float(deviation.max()), 3),
        "out_of_band_s": int((deviation > 0.65).sum() * DT),
        "comp_kwh": round(frame.compressor_power_w.sum() * DT / 3.6e6, 4),
        "pump_kwh": round(frame.pump_power_w.sum() * DT / 3.6e6, 4),
        "total_kwh": round(power.sum() * DT / 3.6e6, 4),
    }


def metrics_for(frame: pd.DataFrame, name: str) -> dict:
    return metrics(frame.assign(controller=name))


summary = pd.DataFrame([
    metrics_for(frames["baseline_fixed"], "baseline_fixed"),
    metrics_for(frames["nmpc_static_qgen"], "nmpc_static_qgen"),
    metrics_for(frames["nmpc_soc_aware"], "nmpc_soc_aware"),
])
summary.to_csv(HERE / "three_way_summary.csv", index=False)
with pd.option_context("display.width", 240):
    print(summary.to_string(index=False))

# ---- Figure -----------------------------------------------------------------
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
    "nmpc_static_qgen": "NMPC, static Qgen preview (old)",
    "nmpc_soc_aware": "NMPC, SOC-aware Qgen preview (new)",
}

fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.8), sharex=True,
                         gridspec_kw={"height_ratios": [1.3, 1.05, 1.05, 1.0],
                                      "hspace": 0.16})
ax = axes[0]
ax.axhspan(24.35, 25.65, color="#2A9D8F", alpha=0.10, zorder=0)
ax.axhline(25.0, color="#444444", lw=0.8, ls="--", zorder=1)
for name in ("baseline_fixed", "nmpc_static_qgen", "nmpc_soc_aware"):
    ax.plot(frames[name].time_s / 60, frames[name].battery_avg_temp_c,
            label=LABEL[name], **STYLE[name])
ax.set_ylim(24.9, 28.6)
ax.set_ylabel("Battery avg. temp (\u00b0C)")
ax.legend(loc="upper left")
ax.set_title("Peak-shaving S0 1280 steps: SOC-aware vs static Qgen preview", loc="left")

ax = axes[1]
for name in ("baseline_fixed", "nmpc_static_qgen", "nmpc_soc_aware"):
    ax.step(frames[name].time_s / 60, frames[name].compressor_command_rpm,
            where="post", **STYLE[name])
ax.set_ylabel("Compressor cmd (rpm)")

ax = axes[2]
ax.plot(calibration.time_s / 60, calibration.plant_q_gen_w_per_pack * 5,
        "o-", color="#444444", ms=4, label="Plant truth (open loop, 5 packs)")
ax.plot(calibration.time_s / 60, calibration.soc_aware_preview_w * 5,
        "s--", color="#009E73", ms=4, label="SOC-aware preview x5")
ax.plot(calibration.time_s / 60, calibration.static_preview_w * 5 * np.ones(len(calibration)),
        "^--", color="#D55E00", ms=4, label="Static preview x5 (old)")
ax.set_ylabel("Heat gen (W)")
ax.legend(loc="upper left")
ax.set_xlabel("Time (min)")

ax = axes[3]
for name in ("baseline_fixed", "nmpc_static_qgen", "nmpc_soc_aware"):
    f = frames[name]
    energy = (f.compressor_power_w + f.pump_power_w).cumsum() * DT / 3.6e6
    ax.plot(f.time_s / 60, energy, label=f"{LABEL[name]} ({energy.iloc[-1]:.2f} kWh)",
            **STYLE[name])
ax.set_ylabel("Cumulative total energy (kWh)")
ax.set_xlabel("Time (min)")
ax.set_xlim(0, DURATION / 60)
ax.legend(loc="upper left")

fig.align_ylabels(axes)
fig.savefig(HERE / "fig_S0_socaware_three_way.png", dpi=300)
fig.savefig(HERE / "fig_S0_socaware_three_way.pdf")
print("saved:", HERE / "fig_S0_socaware_three_way.png")
