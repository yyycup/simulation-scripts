"""S0 peak-shaving: default NMPC tuning vs smoother move-penalty tunings.

No frozen code is modified; variants only override runtime
``PhysicsPNmpcParameters`` fields.
"""

from pathlib import Path

import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cluster_plant_v2.control.physics_p_nmpc_controller import PhysicsPNmpcParameters
from cluster_plant_v2.validation.validate_domp_mpc import build_cases
from cluster_plant_v2.validation.validate_physics_p_nmpc import run_nmpc

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / "physics_p_nmpc_s0_peak_20260827"
DT = 5.0

VARIANTS = {
    "tuned_mild": PhysicsPNmpcParameters(
        w_dcomp=25.0,
        w_dpump=25.0,
    ),
    "tuned_firm": PhysicsPNmpcParameters(
        w_dcomp=200.0,
        w_dpump=200.0,
        compressor_dmax_rpm=400.0,
    ),
}


def metrics(frame: pd.DataFrame) -> dict:
    power = frame.compressor_power_w + frame.pump_power_w
    return {
        "controller": frame.controller.iloc[0],
        "final_temp_c": round(frame.battery_avg_temp_c.iloc[-1], 3),
        "max_temp_c": round(frame.battery_max_temp_c.max(), 3),
        "out_of_band_s": int(
            ((frame.battery_avg_temp_c - 25.0).abs() > 0.65).sum() * DT
        ),
        "comp_kwh": round(frame.compressor_power_w.sum() * DT / 3.6e6, 4),
        "pump_kwh": round(frame.pump_power_w.sum() * DT / 3.6e6, 4),
        "total_kwh": round(power.sum() * DT / 3.6e6, 4),
        "move_rpm": round(frame.compressor_command_rpm.diff().abs().sum(), 0),
        "max_jump_rpm": round(frame.compressor_command_rpm.diff().abs().max(), 0),
    }


case = [c for c in build_cases() if c.case_id == "S0_constant"][0]
old = pd.read_csv(PREV / "case_S0_constant_timeseries.csv")
frames = {"baseline_fixed": old[old.controller == "baseline_fixed"].copy(),
          "default": old[old.controller == "physics_p_nmpc"].copy()}
for name, params in VARIANTS.items():
    print(f"running {name} ...", flush=True)
    f = run_nmpc(case, duration_s=600.0, dt_s=DT, parameters=params)
    f["controller"] = name
    frames[name] = f

rows = [metrics(f) for f in frames.values()]
table = pd.DataFrame(rows)
table.to_csv(HERE / "tuning_comparison_summary.csv", index=False)
with pd.option_context("display.width", 240):
    print(table.to_string(index=False))

for name, f in frames.items():
    if name not in ("baseline_fixed", "default"):
        out = pd.concat([frames["baseline_fixed"], f], ignore_index=True)
        out.to_csv(HERE / f"case_S0_constant_{name}_timeseries.csv", index=False)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 10, "legend.fontsize": 8.5, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "lines.linewidth": 1.6,
})
STYLE = {
    "baseline_fixed": dict(color="#0072B2", ls="-", lw=1.4),
    "default": dict(color="#D55E00", ls="--", lw=1.4),
    "tuned_mild": dict(color="#009E73", ls="-", lw=1.7),
    "tuned_firm": dict(color="#CC79A7", ls="-", lw=1.7),
}
LABEL = {
    "baseline_fixed": "Fixed baseline (4000/3600 rpm)",
    "default": "NMPC default (w_du=0.01)",
    "tuned_mild": "Tuned mild (w_du=25)",
    "tuned_firm": "Tuned firm (w_du=200, dmax 400)",
}

fig, axes = plt.subplots(3, 1, figsize=(6.75, 5.6), sharex=True,
                         gridspec_kw={"hspace": 0.18})

ax = axes[0]
ax.axhspan(24.35, 25.65, color="#2A9D8F", alpha=0.10, zorder=0)
ax.axhline(25.0, color="#444444", lw=0.8, ls="--", zorder=1)
for name in ("baseline_fixed", "default", "tuned_mild", "tuned_firm"):
    ax.plot(frames[name].time_s, frames[name].battery_avg_temp_c,
            label=LABEL[name], **STYLE[name])
ax.set_ylim(24.96, 25.30)
ax.set_ylabel("Battery avg. temp (\u00b0C)")
ax.legend(loc="lower right")
ax.set_title("S0 peak-shaving tuning comparison: temperature and compressor command", loc="left")

ax = axes[1]
for name in ("baseline_fixed", "default", "tuned_mild", "tuned_firm"):
    ax.step(frames[name].time_s, frames[name].compressor_command_rpm,
            where="post", **STYLE[name])
ax.set_ylabel("Compressor cmd (rpm)")
ax.set_yticks([3000, 3500, 4000, 4500])

ax = axes[2]
for name in ("baseline_fixed", "default", "tuned_mild", "tuned_firm"):
    p = frames[name].compressor_power_w + frames[name].pump_power_w
    ax.plot(frames[name].time_s, p, label=LABEL[name], **STYLE[name])
ax.set_ylabel("Total power (W)")
ax.set_xlabel("Time (s)")
ax.set_xlim(0, 600)

fig.align_ylabels(axes)
fig.savefig(HERE / "fig_S0_tuning_comparison.png", dpi=300)
fig.savefig(HERE / "fig_S0_tuning_comparison.pdf")
print("saved:", HERE / "fig_S0_tuning_comparison.png")
