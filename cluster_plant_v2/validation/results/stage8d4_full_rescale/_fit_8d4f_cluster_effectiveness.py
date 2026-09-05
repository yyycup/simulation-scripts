"""Stage 8D4f: refit the cluster plate-fluid coupling for deployment.

The corrected ClusterCasadiPhysicsP replaces the legacy
``eff x min(flow, ...)`` lumped plate-fluid term (which caps near 371 W/K
under the 1/5 per-pack flow, below the plant-implied ~395 W/K) with a
reference-flow anchor ``plate_fluid_ref_flow_w_k x pump_ratio**0.8`` that
mirrors the plant cold plate's 0.8-power flow scaling. Anchor: the six
8D4e compressor sweep steady points (full plant, 20 min settle each,
_probe_capacity_vs_plant_sweep.csv). For each candidate anchor the
corrected map is rolled to steady state at the sweep commands with the
measured plant states as seeds; the fit minimizes the battery-temperature
error across all points.

Writes single_pack_plant/model_data/physics_p_operational_8d4f.json with a
provenance block and points PHYSICS_P_ARTIFACT at it. Kept for reference.
"""

import json
import sys
from pathlib import Path

ROOT = Path("c:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from cluster_plant_v2.control.physics_p_nmpc_model import (
    ClusterCasadiPhysicsP,
    PHYSICS_P_ARTIFACT,
)
from cluster_plant_v2.control.physics_p_nmpc_model import (
    PUMP_VIRTUAL_SPEED_SCALE as PV,
)
from cluster_plant_v2.control.physics_p_nmpc_model import ClusterHeatGenerationPreview
from cluster_plant_v2.validation.validate_domp_mpc import (
    BASELINE_PUMP_RPM,
    build_cases,
    case_currents,
)
from cluster_plant_v2.validation.validate_refrigerated_cluster_cooling_loop import (
    AMBIENT_TEMPERATURE_K,
)

HERE = Path("cluster_plant_v2/validation/results/stage8d4_full_rescale")
DT = 5.0
# 60 min rolls leave no doubt about steady convergence at every sweep point.
ROLL_STEPS = 720
T_AMB_C = AMBIENT_TEMPERATURE_K - 273.15
SETTLE_TIME_S = 1200.0

# Preferred anchor table: the sweep at actual pump 2400 rpm, where the
# plant flow matches the predictor's virtual-4800 flow exactly (the 3600 rpm
# sweep sits outside the NMPC pump envelope and misreports flow by ~50%).
PUMP2400_CSV = HERE / "_probe_sweep_pump2400.csv"
if PUMP2400_CSV.exists():
    sweep = pd.read_csv(PUMP2400_CSV)
    print("anchor table: _probe_sweep_pump2400.csv (actual pump 2400 rpm)", flush=True)
else:
    sweep = pd.read_csv(HERE / "_probe_capacity_vs_plant_sweep.csv")
    print("anchor table: legacy 3600 rpm sweep (flow mismatch!)", flush=True)
# Preferred heat anchors: measured per-pack plant heat in the anchor table
# itself, else the full-run qgen probe CSV, else the SOC-aware preview.
HEAT_ANCHORS_CSV = HERE / "_probe_sweep_heat_anchors.csv"
HEAT_ANCHORS = (
    pd.read_csv(HEAT_ANCHORS_CSV) if HEAT_ANCHORS_CSV.exists() else None
)
if HEAT_ANCHORS is None:
    print("sweep heat anchor CSV missing; falling back", flush=True)
QGEN_CSV = HERE / "_probe_plant_qgen_vs_preview.csv"
QGEN_PROBE = pd.read_csv(QGEN_CSV) if QGEN_CSV.exists() else None
if HEAT_ANCHORS is None and QGEN_PROBE is None:
    print("qgen probe CSV missing; falling back to the SOC-aware preview", flush=True)
case = [c for c in build_cases() if c.case_id == "S0_constant"][0]
currents = case_currents(case, 1280, DT)
preview = ClusterHeatGenerationPreview(currents, DT)


def heat_at_sweep_end(index: int) -> float:
    """Per-pack plant heat matching the cumulative settle time of point."""
    row = sweep.iloc[index]
    if "qgen_per_pack_w" in sweep.columns:
        return float(row["qgen_per_pack_w"])
    if HEAT_ANCHORS is not None:
        match = HEAT_ANCHORS[
            np.abs(HEAT_ANCHORS["cmd_rpm"] - float(row["cmd_rpm"])) < 1.0
        ]
        if len(match):
            return float(match["qgen_per_pack_w"].iloc[-1])
    t = (index + 1) * SETTLE_TIME_S
    if QGEN_PROBE is not None:
        rows = QGEN_PROBE[QGEN_PROBE["time_s"] <= t]
        if len(rows):
            return float(rows["qgen_per_pack_w"].iloc[-1])
    return float(rows_fallback(t))


def rows_fallback(t: float) -> float:
    return float(preview(min(t, 1279 * DT), 0.0))


def roll_to_steady(predictor, x0, n_comp_cmd, pump_virtual, current_a, q_gen_w):
    x = np.asarray(x0, dtype=float).reshape(-1, 1)
    u = np.array([n_comp_cmd, pump_virtual]).reshape(-1, 1)
    d = np.array([current_a, T_AMB_C, q_gen_w]).reshape(-1, 1)
    for _ in range(ROLL_STEPS):
        x = np.asarray(predictor.function(x, u, d), dtype=float).reshape(-1, 1)
    return x.reshape(-1)


def seed_state(row) -> np.ndarray:
    pump_virtual = min(BASELINE_PUMP_RPM * PV, 4800.0)
    return np.array(
        [
            row["bat_c"],
            row["tank_c"],
            row["bat_c"] - 3.0,
            row["cmd_rpm"],
            pump_virtual,
            row["q_evap_plant_w"],
            pump_virtual,
            row["supply_c"],
            row["supply_c"],
            row["tank_c"],
            row["return_c"] if "return_c" in row else row["tank_c"],
            row["tank_c"],
            row["tank_c"],
            row["tank_c"],
        ],
        dtype=float,
    )


def build_predictor(plate_flow_ref: float, heat_scale: float):
    artifact = json.loads(PHYSICS_P_ARTIFACT.read_text(encoding="utf-8"))
    artifact.setdefault("cluster_thermal", {})["plate_fluid_ref_flow_w_k"] = float(
        plate_flow_ref
    )
    artifact["thermal"]["battery_heat_generation_scale"] = float(heat_scale)
    tmp = HERE / "_tmp_8d4f_artifact.json"
    tmp.write_text(json.dumps(artifact), encoding="utf-8")
    return ClusterCasadiPhysicsP(tmp, dt_s=DT)


def fit_error(plate_flow_ref: float, heat_scale: float) -> float:
    predictor = build_predictor(plate_flow_ref, heat_scale)
    pump_virtual = min(BASELINE_PUMP_RPM * PV, 4800.0)
    errors = []
    for index, row in sweep.iterrows():
        x0 = seed_state(row)
        x_end = roll_to_steady(
            predictor,
            x0,
            float(row["cmd_rpm"]),
            pump_virtual,
            560.0,
            heat_at_sweep_end(int(index)),
        )
        errors.append(x_end[0] - float(row["bat_c"]))
    return float(np.sqrt(np.mean(np.square(errors))))


best = (None, None, np.inf)
# The plant-implied pack-fluid coupling at the S0 settled point is ~395 W/K;
# scan a wide window around it together with the heat-generation scale.
for heat_scale in (0.74, 0.80, 0.90, 1.00):
    for ref_w_k in np.arange(300.0, 470.0, 20.0):
        err = fit_error(ref_w_k, heat_scale)
        print(f"ref {ref_w_k:.0f} scale {heat_scale:.2f}: rms {err:.3f} K", flush=True)
        if err < best[2]:
            best = (ref_w_k, heat_scale, err)

# Refine around the best grid point.
ref0, scale0 = best[0], best[1]
for heat_scale in (scale0 - 0.05, scale0, scale0 + 0.05):
    for ref_w_k in np.arange(ref0 - 20.0, ref0 + 21.0, 5.0):
        if ref_w_k <= 0.0:
            continue
        err = fit_error(ref_w_k, heat_scale)
        print(f"fine ref {ref_w_k:.1f} scale {heat_scale:.3f}: rms {err:.3f} K", flush=True)
        if err < best[2]:
            best = (ref_w_k, heat_scale, err)

ref_fit, heat_scale_fit, rms = best
print(
    f"\nfit: plate_fluid_ref_flow_w_k {ref_fit:.1f}, "
    f"battery_heat_generation_scale {heat_scale_fit:.3f}, rms {rms:.3f} K"
)

# Per-point residuals at the fit.
predictor = build_predictor(ref_fit, heat_scale_fit)
pump_virtual = min(BASELINE_PUMP_RPM * PV, 4800.0)
print("cmd   plant_bat  model_bat  resid")
for index, row in sweep.iterrows():
    x_end = roll_to_steady(
        predictor,
        seed_state(row),
        float(row["cmd_rpm"]),
        pump_virtual,
        560.0,
        heat_at_sweep_end(int(index)),
    )
    print(
        f"{row['cmd_rpm']:5.0f}  {row['bat_c']:8.2f}  {x_end[0]:8.2f}  "
        f"{x_end[0] - row['bat_c']:+6.2f}"
    )

artifact = json.loads(PHYSICS_P_ARTIFACT.read_text(encoding="utf-8"))
artifact.setdefault("cluster_thermal", {})["plate_fluid_ref_flow_w_k"] = float(ref_fit)
artifact["thermal"]["battery_heat_generation_scale"] = float(heat_scale_fit)
artifact["fit"]["stage8d4f_cluster_thermal_refit"] = {
    "source": "cluster plant steady-state sweep anchors (5 packs, 560 A)",
    "changed": [
        "cluster_thermal.plate_fluid_ref_flow_w_k",
        "thermal.battery_heat_generation_scale",
    ],
    "frozen_sections": ["gate", "dynamic", "capacity", "input_domain"],
    "anchor_file": "_probe_capacity_vs_plant_sweep.csv",
    "rms_battery_temp_k": rms,
    "note": (
        "ClusterCasadiPhysicsP gives pack-level states the 1/5 flow share, "
        "sums N packs' plate heat into the return, and anchors the "
        "plate-fluid coupling at the fitted reference-flow W/K with the "
        "plant's 0.8-power flow scaling; the legacy plate_fluid_effectiveness "
        "field is kept for validator compatibility only."
    ),
}
out = ROOT / "single_pack_plant/model_data/physics_p_operational_8d4f.json"
out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
print(f"\nwrote {out}")
