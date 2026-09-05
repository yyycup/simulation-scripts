"""Stage 8D4e: re-derive the frozen Physics-P capacity block on 8D4 hardware.

The frozen predictor ``physics_p_operational_v1.json`` carries a capacity
model whose 20 cubic coefficients and ``q_upper_w = 4800 W`` ceiling were
calibrated on the legacy small chiller. The 8D4 hardware (72 cc
compressor, rescaled condenser/fan) delivers up to ~19.5 kW, so the
NMPC optimizes against a wrong "speed -> capacity" map; the S0 peak run
showed the signature +0.28 K steady offset plus a 1000-3647 rpm command
limit cycle.

This script re-fits ONLY the capacity block (coefficients + q_upper) on
steady-state points of the current ``ClosedR134aCycle``; every other
artifact section (gate, dynamics, thermal, input domain) is kept frozen.
Feature normalization replicates ``CasadiPhysicsP._capacity`` exactly:

    compressor = (n_comp - 4000) / 2000
    pump_ratio = 2000 / n_pump_virtual
    coolant    = (t_tank - 27.5) / 7.5
    ambient    = (t_amb - 30) / 10
    raw        = (sum c_i * feature_i) * n_comp, clipped to [0, q_upper]

The pump axis is the virtual 8D4c domain (virtual = 2 x actual), so the
training grid covers virtual [3200, 4800] = actual [1600, 2400], the
commanded region of the rescaled pump.
"""

import json
import multiprocessing
from pathlib import Path
import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import numpy as np
import pandas as pd

from cluster_plant_v2.hydraulics import CoolantPump
from cluster_plant_v2.parameters import COOLANT_DENSITY_KG_M3
from cluster_plant_v2.refrigeration import ClosedR134aCycle
from cluster_plant_v2.validation.validate_refrigerated_cluster_cooling_loop import (
    build_medium_header_network,
)

HERE = Path(__file__).resolve().parent
ROOT = Path("C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")
LEGACY_ARTIFACT = ROOT / "single_pack_plant/model_data/physics_p_operational_v1.json"
NEW_ARTIFACT = ROOT / "single_pack_plant/model_data/physics_p_operational_8d4.json"

FAN_RPM = 1200.0
RIDGE = 1e-4
WEIGHT_POWER = 0.8
RNG = np.random.default_rng(20260830)
GRID_CACHE = HERE / "_derive_8d4_capacity_grid.csv"

# Training grid inside the frozen input domain; pump in the virtual domain.
COMP_GRID = np.array([1000.0, 1400.0, 1800.0, 2400.0, 3200.0, 4200.0, 5200.0, 6000.0])
PUMP_VIRTUAL_GRID = np.array([3200.0, 3600.0, 4000.0, 4400.0, 4800.0])
TCOOL_GRID = np.array([16.0, 19.0, 22.0, 25.0, 28.0, 31.0, 34.0])
TAMB_GRID = np.array([30.0, 35.0, 40.0])


def pump_flow_by_virtual_rpm() -> dict[float, float]:
    """Actual mass flow for each virtual speed (actual = virtual / 2).

    Pump construction mirrors the 8D4b rescaled selection (reference
    56 L/min, head re-matched to the medium-header network).
    """
    net = build_medium_header_network()
    ref_flow_kg_s = 56.0 / 1000 / 60 * COOLANT_DENSITY_KG_M3
    ref_dp = float(net.solve(ref_flow_kg_s)["network_delta_p"])
    pump = CoolantPump(
        reference_operating_delta_p_pa=ref_dp,
        reference_volume_flow_l_min=56.0,
    )
    flows = {}
    for virtual in PUMP_VIRTUAL_GRID:
        op = pump.solve_operating_point(virtual / 2.0, net)
        flows[float(virtual)] = float(op["total_mass_flow_kg_s"])
    return flows


def features(n_comp, n_pump_virtual, t_cool, t_amb):
    comp = (n_comp - 4000.0) / 2000.0
    pr = 2000.0 / n_pump_virtual
    cool = (t_cool - 27.5) / 7.5
    amb = (t_amb - 30.0) / 10.0
    return np.array(
        [
            1.0, comp, pr, cool, amb,
            comp**2, comp * pr, comp * cool, comp * amb,
            pr**2, pr * cool, pr * amb,
            cool**2, cool * amb,
            comp**2 * pr, comp * pr**2, comp * pr * cool,
            comp * pr * amb, comp * cool * amb, cool**3,
        ]
    )


def _solve_grid_point(args):
    """Worker: one steady-state cycle solve (stateless cycle class)."""
    n_comp, n_pump_v, t_cool, t_amb, mdot = args
    from cluster_plant_v2.refrigeration import ClosedR134aCycle

    try:
        result = ClosedR134aCycle().solve(
            compressor_speed_rpm=n_comp,
            fan_speed_rpm=FAN_RPM,
            coolant_inlet_temperature_k=t_cool + 273.15,
            coolant_mass_flow_kg_s=mdot,
            ambient_temperature_k=t_amb + 273.15,
        )
    except (ValueError, FloatingPointError) as exc:
        return ("error", n_comp, n_pump_v, t_cool, t_amb,
                exc.__class__.__name__)
    if not result["solver_success"]:
        return ("fail", n_comp, n_pump_v, t_cool, t_amb, 0.0)
    return ("ok", n_comp, n_pump_v, t_cool, t_amb,
            float(result["q_evaporator_w"]))


def main() -> None:
    flows = pump_flow_by_virtual_rpm()
    print("pump flow map (virtual rpm -> kg/s):", flush=True)
    for virtual, mdot in flows.items():
        print(f"  {virtual:.0f} -> {mdot:.4f}", flush=True)

    cycle = ClosedR134aCycle()
    total = len(COMP_GRID) * len(PUMP_VIRTUAL_GRID) * len(TCOOL_GRID) * len(TAMB_GRID)
    cached = pd.read_csv(GRID_CACHE) if GRID_CACHE.exists() else None
    cache_keys = (
        {
            tuple(row[["n_comp", "n_pump_v", "t_cool", "t_amb"]].astype(float))
            for _, row in cached.iterrows()
        }
        if cached is not None
        else set()
    )
    rows = (
        [tuple(row[["n_comp", "n_pump_v", "t_cool", "t_amb", "q_evap_w"]].astype(float))
         for _, row in cached.iterrows()]
        if cached is not None
        else []
    )
    print(f"resume: {len(rows)} cached grid points", flush=True)
    tasks = [
        (float(n_comp), float(n_pump_v), float(t_cool), float(t_amb),
         flows[float(n_pump_v)])
        for n_comp in COMP_GRID
        for n_pump_v in PUMP_VIRTUAL_GRID
        for t_cool in TCOOL_GRID
        for t_amb in TAMB_GRID
        if (float(n_comp), float(n_pump_v), float(t_cool), float(t_amb))
        not in cache_keys
    ]
    print(f"grid tasks to solve: {len(tasks)} / {total}", flush=True)
    failures, errors = 0, 0
    with multiprocessing.Pool() as pool:
        for status, n_comp, n_pump_v, t_cool, t_amb, value in pool.imap_unordered(
            _solve_grid_point, tasks, chunksize=4
        ):
            if status == "ok":
                rows.append((n_comp, n_pump_v, t_cool, t_amb, value))
            elif status == "fail":
                failures += 1
            else:
                errors += 1
                print(
                    f"  corner skipped ({value}): comp {n_comp:.0f} / "
                    f"pump_v {n_pump_v:.0f} / Tc {t_cool:.0f} / Tamb {t_amb:.0f}",
                    flush=True,
                )
    rows.sort()
    pd.DataFrame(
        rows, columns=["n_comp", "n_pump_v", "t_cool", "t_amb", "q_evap_w"]
    ).to_csv(GRID_CACHE, index=False)
    print(f"  grid cache written: {GRID_CACHE}", flush=True)
    data = np.array(rows)
    print(
        f"\ngrid solved: {len(data)} points, {failures} solver failures, "
        f"{errors} corner exceptions skipped",
        flush=True,
    )

    q = data[:, 4]
    print(
        f"q_evap range: {q.min():.0f} .. {q.max():.0f} W "
        f"(legacy ceiling was 4800 W)",
        flush=True,
    )

    feat = np.stack(
        [features(*row[:4]) for row in data]
    )  # (N, 20)
    target = q / data[:, 0]  # raw = (c . f) * n_comp
    weights = np.maximum(q, 1.0) ** WEIGHT_POWER

    idx = RNG.permutation(len(data))
    n_val = len(data) // 4
    val, train = idx[:n_val], idx[n_val:]

    def solve_subset(subset):
        f_w = feat[subset] * weights[subset, None]
        y_w = target[subset] * weights[subset]
        gram = f_w.T @ f_w + RIDGE * np.eye(feat.shape[1])
        return np.linalg.solve(gram, f_w.T @ y_w)

    coeffs = solve_subset(train)

    def metrics(subset, c):
        pred = (feat[subset] @ c) * data[subset, 0]
        actual = q[subset]
        err = pred - actual
        active = actual > 100.0
        return {
            "mae_w": float(np.abs(err).mean()),
            "rmse_w": float(np.sqrt((err**2).mean())),
            "active_mape_percent": float(
                np.mean(np.abs(err[active]) / actual[active]) * 100.0
            ),
            "active_max_relative_error_percent": float(
                np.max(np.abs(err[active]) / actual[active]) * 100.0
            ),
        }

    train_metrics, val_metrics = metrics(train, coeffs), metrics(val, coeffs)
    print("\nfit metrics:", flush=True)
    print(f"  train: {train_metrics}", flush=True)
    print(f"  validation: {val_metrics}", flush=True)

    # Refit on the full grid once quality is confirmed.
    coeffs_full = solve_subset(idx)
    q_upper_w = float(np.ceil(q.max() / 100.0) * 100.0)

    artifact = json.loads(LEGACY_ARTIFACT.read_text(encoding="utf-8"))
    capacity = artifact["capacity"]
    capacity["coefficients"] = [float(c) for c in coeffs_full]
    capacity["q_upper_w"] = q_upper_w
    artifact["fit"]["stage8d4_capacity_rederivation"] = {
        "source": "cluster_plant_v2 8D4 hardware steady-state cycle grid",
        "frozen_sections": ["gate", "dynamic", "thermal", "input_domain"],
        "fan_speed_rpm": FAN_RPM,
        "pump_domain": "virtual (2x actual), grid 3200-4800 = actual 1600-2400",
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "solver_failures": failures,
        "validation_metrics": val_metrics,
        "train_metrics": train_metrics,
        "ridge": RIDGE,
        "weight_power": WEIGHT_POWER,
        "legacy_q_upper_w": 4800.0,
        "new_q_upper_w": q_upper_w,
    }
    NEW_ARTIFACT.write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {NEW_ARTIFACT}", flush=True)
    print(f"q_upper_w: 4800 -> {q_upper_w:.0f} W", flush=True)

    # Held-out sanity: legacy vs new artifact predictions vs plant truth.
    sys.path.insert(0, str(ROOT))
    from single_pack_plant.predictor.physics_p_casadi import CasadiPhysicsP

    def capacity_of(art_path, n_comp, n_pump_v, t_cool, t_amb):
        pred = CasadiPhysicsP(art_path, dt_s=5.0)
        # _capacity expects t_cool / t_ambient in degC (model-world units).
        return float(pred._capacity(n_comp, n_pump_v, t_cool, t_amb))

    print("\nheld-out comparison (plant truth vs legacy vs new):", flush=True)
    checks = [
        (1000.0, 4800.0, 25.0, 35.0),
        (1500.0, 4800.0, 25.0, 35.0),
        (2500.0, 4000.0, 22.0, 35.0),
        (4000.0, 3600.0, 25.0, 35.0),
        (6000.0, 4800.0, 20.0, 35.0),
        (6000.0, 3200.0, 30.0, 35.0),
    ]
    for n_comp, n_pump_v, t_cool, t_amb in checks:
        truth = float(
            cycle.solve(
                compressor_speed_rpm=n_comp,
                fan_speed_rpm=FAN_RPM,
                coolant_inlet_temperature_k=t_cool + 273.15,
                coolant_mass_flow_kg_s=flows[float(n_pump_v)],
                ambient_temperature_k=t_amb + 273.15,
            )["q_evaporator_w"]
        )
        legacy = capacity_of(LEGACY_ARTIFACT, n_comp, n_pump_v, t_cool, t_amb)
        new = capacity_of(NEW_ARTIFACT, n_comp, n_pump_v, t_cool, t_amb)
        print(
            f"  comp {n_comp:.0f} / pump_v {n_pump_v:.0f} / "
            f"Tc {t_cool:.0f}: truth {truth:7.0f} | "
            f"legacy {legacy:7.0f} ({100*(legacy/truth-1):+6.1f}%) | "
            f"new {new:7.0f} ({100*(new/truth-1):+6.1f}%)",
            flush=True,
        )
    print("done", flush=True)


if __name__ == "__main__":
    main()
