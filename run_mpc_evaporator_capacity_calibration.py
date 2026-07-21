"""Fit and validate the low-order Candidate-B evaporator capacity model."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import btms_runtime

btms_runtime.ensure_env_library_bin_on_path()

import numpy as np
import pandas as pd

from mpc_evaporator_capacity_model import evaluate_capacity
from thermal_batch_config import (
    AMBIENT_TEMP_K,
    EVAP_CAP_FACTOR,
    EVAP_FLOW_EXP,
    EVAP_UA_FACTOR,
)
from thermal_loop import staged_fan_speed
from thermal_system import (
    REF,
    chiller_model_NTU,
    compressor_model,
    condenser_model_NTU,
    p_sat_from_T,
    pump_model,
    safe_PropsSI,
    solve_saturation_temperatures,
    superheat_desired,
)


OUT = Path("outputs") / "mpc_evaporator_capacity_candidate_b_15c"
N_COMP = (2000.0, 3000.0, 4000.0, 5000.0, 6000.0)
N_PUMP = (1600.0, 2400.0, 3200.0, 4000.0, 4800.0)
T_COOL_TRAIN = (15.0, 20.0, 25.0, 30.0, 35.0)
T_COOL_VALIDATION = (17.5, 22.5, 27.5, 32.5)
T_COOL_ALL = tuple(sorted((*T_COOL_TRAIN, *T_COOL_VALIDATION)))
# Backward-compatible name used by the small grid-export helper.
T_COOL = T_COOL_ALL
MINIMUM_ACTIVE_RPM = 2000.0
MPC_GATE_CENTER_RPM = 1950.0
MPC_GATE_WIDTH_RPM = 10.0
FIT_WEIGHT_POWER = 0.3
VALIDATION_MAPE_TARGET_PERCENT = 10.0
VALIDATION_MAX_RELATIVE_ERROR_TARGET_PERCENT = 20.0


def plant_point(n_comp, n_pump, t_cool_c):
    m_cool, _ = pump_model(n_pump)
    t_cool = t_cool_c + 273.15
    n_fan = staged_fan_speed(n_comp)
    t_evap, t_cond = solve_saturation_temperatures(
        n_comp,
        n_fan,
        t_cool,
        m_cool,
        AMBIENT_TEMP_K,
    )
    p_evap, p_cond = p_sat_from_T(t_evap), p_sat_from_T(t_cond)
    t_suc = safe_PropsSI("T", "P", p_evap, "Q", 1.0, REF) + superheat_desired
    comp = compressor_model(p_evap, t_suc, p_cond, n_comp)
    cond = condenser_model_NTU(
        comp["mdot"], comp["h_out"], p_cond, n_fan, AMBIENT_TEMP_K
    )
    chiller = chiller_model_NTU(
        comp["mdot"], p_evap, cond["h_cond_out"], t_cool, m_cool
    )
    limit = "Q_hx" if chiller["Q_hx_potential"] <= chiller["Q_ref_max"] else "Q_ref_max"
    return chiller["Q_evap"], limit


def build_grid():
    rows = []
    for n_comp in N_COMP:
        for n_pump in N_PUMP:
            for t_cool in T_COOL_ALL:
                q_evap, limit = plant_point(n_comp, n_pump, t_cool)
                rows.append(
                    {
                        "N_comp_rpm": n_comp,
                        "N_pump_rpm": n_pump,
                        "T_cool_in_C": t_cool,
                        "Q_evap_plant_W": q_evap,
                        "limit_type": limit,
                        "split": (
                            "train" if t_cool in T_COOL_TRAIN else "validation"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def features(frame, kind):
    n_comp = frame.N_comp_rpm.to_numpy()
    n_pump = frame.N_pump_rpm.to_numpy()
    t_cool = frame.T_cool_in_C.to_numpy()
    if kind == "candidate_a":
        return np.c_[n_comp]
    if kind == "candidate_b":
        return np.c_[
            n_comp,
            n_comp * 2000.0 / n_pump,
            n_comp * (t_cool - 25.0),
        ]
    if kind == "candidate_c":
        return np.c_[
            np.ones(len(frame)),
            n_comp,
            n_pump,
            t_cool,
            n_comp * n_pump,
            n_comp * t_cool,
            n_comp**2,
        ]
    raise ValueError(f"Unsupported capacity model: {kind}")


def coefficient_names(kind):
    return {
        "candidate_a": ["kq"],
        "candidate_b": ["b0", "b1", "b2"],
        "candidate_c": ["b0", "b1", "b2", "b3", "b4", "b5", "b6"],
    }[kind]


def fit_calibration(frame, kind, q_upper_w):
    actual = frame.Q_evap_plant_W.to_numpy()
    weights = 1.0 / np.maximum(actual, 1.0) ** FIT_WEIGHT_POWER
    beta, *_ = np.linalg.lstsq(
        features(frame, kind) * weights[:, None],
        actual * weights,
        rcond=None,
    )
    return {
        "model_type": kind,
        "coefficients": dict(zip(coefficient_names(kind), map(float, beta))),
        "n_pump_ref_rpm": 2000.0,
        "q_evap_upper_bound_w": float(q_upper_w),
        "minimum_active_rpm": MINIMUM_ACTIVE_RPM,
        "mpc_power_gate": {
            "kind": "algebraic_sqrt",
            "center_rpm": MPC_GATE_CENTER_RPM,
            "width_rpm": MPC_GATE_WIDTH_RPM,
        },
        "optimizer_low_speed_policy": {
            "capacity": "continuous_relaxation_for_nlp",
            "power": "algebraic_sqrt_gate",
            "physical_execution": "hard_off_below_minimum_active_rpm",
        },
    }


def capacity_metrics(calibration, frame):
    prediction = np.array(
        [
            evaluate_capacity(
                calibration,
                row.N_comp_rpm,
                row.N_pump_rpm,
                row.T_cool_in_C,
            )
            for row in frame.itertuples()
        ]
    )
    actual = frame.Q_evap_plant_W.to_numpy()
    error = prediction - actual
    relative = np.abs(error) / np.maximum(actual, 1e-9) * 100.0
    return {
        "MAE_W": float(np.mean(np.abs(error))),
        "RMSE_W": float(np.sqrt(np.mean(error**2))),
        "MAPE_percent": float(np.mean(relative)),
        "max_relative_error_percent": float(np.max(relative)),
    }, prediction


def _prefixed_metrics(prefix, metrics):
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def main():
    frame = build_grid()
    train = frame.loc[frame["split"] == "train"].copy()
    validation = frame.loc[frame["split"] == "validation"].copy()
    q_upper_w = float(frame.Q_evap_plant_W.max())

    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "candidate_b_capacity_grid.csv", index=False, encoding="utf-8-sig")

    metric_rows = []
    train_fits = {}
    for kind in ("candidate_a", "candidate_b", "candidate_c"):
        calibration = fit_calibration(train, kind, q_upper_w)
        train_metrics, _ = capacity_metrics(calibration, train)
        validation_metrics, _ = capacity_metrics(calibration, validation)
        train_fits[kind] = (calibration, train_metrics, validation_metrics)
        metric_rows.append(
            {
                "model_type": kind,
                **_prefixed_metrics("train", train_metrics),
                **_prefixed_metrics("validation", validation_metrics),
            }
        )

    _candidate_b_train, train_metrics, validation_metrics = train_fits["candidate_b"]
    validation_target_met = (
        validation_metrics["MAPE_percent"] <= VALIDATION_MAPE_TARGET_PERCENT
        and validation_metrics["max_relative_error_percent"]
        <= VALIDATION_MAX_RELATIVE_ERROR_TARGET_PERCENT
    )
    if not validation_target_met:
        raise RuntimeError(
            "Candidate B failed low-temperature holdout validation: "
            f"MAPE={validation_metrics['MAPE_percent']:.3f}%, "
            "max_relative_error="
            f"{validation_metrics['max_relative_error_percent']:.3f}%"
        )

    # Refit the accepted simple model on every validated temperature node.
    calibration = fit_calibration(frame, "candidate_b", q_upper_w)
    full_metrics, full_prediction = capacity_metrics(calibration, frame)
    frame["Q_evap_candidate_b_W"] = full_prediction
    frame["error_W"] = full_prediction - frame["Q_evap_plant_W"]
    frame.to_csv(
        OUT / "candidate_b_capacity_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    calibration.update(
        {
            "evaporator_model_version": "candidate B 15C",
            "EVAP_UA_FACTOR": EVAP_UA_FACTOR,
            "EVAP_FLOW_EXP": EVAP_FLOW_EXP,
            "EVAP_CAP_FACTOR": EVAP_CAP_FACTOR,
            "calibration_date": str(date.today()),
            "selection_method": "temperature_holdout_then_refit_all_nodes",
            "fit_weight_power": FIT_WEIGHT_POWER,
            "data_range": {
                "N_comp_rpm": list(N_COMP),
                "N_pump_rpm": list(N_PUMP),
                "T_cool_in_C": list(T_COOL_ALL),
                "T_cool_train_C": list(T_COOL_TRAIN),
                "T_cool_validation_C": list(T_COOL_VALIDATION),
            },
            "validation_targets": {
                "MAPE_percent": VALIDATION_MAPE_TARGET_PERCENT,
                "max_relative_error_percent": VALIDATION_MAX_RELATIVE_ERROR_TARGET_PERCENT,
            },
            "validation_target_met": True,
            "holdout_validation_metrics": validation_metrics,
            "training_metrics": train_metrics,
            "error_metrics": full_metrics,
        }
    )
    artifact_path = OUT / "mpc_evaporator_capacity_candidate_b.json"
    artifact_path.write_text(
        json.dumps(calibration, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(metric_rows).to_csv(
        OUT / "candidate_model_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(json.dumps(calibration, indent=2))
    print(f"ARTIFACT_PATH={artifact_path}")


if __name__ == "__main__":
    main()
