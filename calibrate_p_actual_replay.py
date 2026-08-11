"""Physically constrained correction of predictor P from actual-command replay."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from btms_runtime import ensure_env_library_bin_on_path

ensure_env_library_bin_on_path()

from scipy.optimize import least_squares

from evaluate_p_shadow_actual_replay import evaluate_actual_command_replay
from mpc_physics_predictor import (
    PHYSICAL_COOLANT_CP_J_KG_K,
    load_physics_artifact,
    validate_physics_artifact,
)


HORIZONS_S = (50.0, 100.0, 300.0)

# thermal_loop.py declares these delays but does not apply them in the executed
# plant path.  P must match the plant that is actually simulated.
STRUCTURAL_DELAY_CORRECTIONS = {
    "evap_input_delay_s": 0.0,
    "pump_flow_delay_s": 0.0,
}

EVAP_TAU_COND_CANDIDATES_S = (35.0, 40.0, 45.0, 50.0, 55.0)
EVAP_TAU_EVAP_CANDIDATES_S = (1.0, 5.0, 10.0)

THERMAL_CALIBRATION_KEYS = (
    "coolant_mass_flow_ref_kg_s",
    "coolant_heat_capacity_j_k",
    "battery_plate_conductance_w_k",
    "plate_tau_s",
    "ambient_conductance_w_k",
    "plate_fluid_effectiveness",
)
PHYSICAL_TANK_HEAT_CAPACITY_J_K = 3.0e-3 * 1071.0 * PHYSICAL_COOLANT_CP_J_KG_K
THERMAL_LOWER = np.array(
    [0.10, PHYSICAL_TANK_HEAT_CAPACITY_J_K, 50.0, 1.0, 0.1, 0.1],
    dtype=float,
)
THERMAL_UPPER = np.array([0.22, 40000.0, 600.0, 20.0, 10.0, 0.8], dtype=float)
THERMAL_START = np.array([0.142, 15000.0, 150.0, 1.0, 0.1, 0.25], dtype=float)
THERMAL_ERROR_SCALES = {
    "T_Batt": 0.2,
    "T_Cool": 0.5,
    "T_Plate": 0.7,
    "T_Supply": 0.4,
    "T_Return": 0.7,
}


def apply_structural_evap_correction(
    base_artifact: dict,
    *,
    tau_cond_s: float,
    tau_evap_s: float,
) -> dict:
    candidate = copy.deepcopy(base_artifact)
    candidate["dynamic"].update(STRUCTURAL_DELAY_CORRECTIONS)
    candidate["dynamic"]["tau_cond_s"] = float(tau_cond_s)
    candidate["dynamic"]["tau_evap_s"] = float(tau_evap_s)
    validate_physics_artifact(candidate)
    return candidate


def thermal_candidate(base_artifact: dict, parameters: np.ndarray) -> dict:
    values = np.asarray(parameters, dtype=float)
    if values.shape != (len(THERMAL_CALIBRATION_KEYS),):
        raise ValueError("thermal calibration parameter vector has the wrong length")
    if not np.isfinite(values).all():
        raise ValueError("thermal calibration parameters must be finite")
    candidate = copy.deepcopy(base_artifact)
    for name, value in zip(THERMAL_CALIBRATION_KEYS, values):
        candidate["thermal"][name] = float(value)
    candidate["thermal"]["coolant_cp_j_kg_k"] = PHYSICAL_COOLANT_CP_J_KG_K
    validate_physics_artifact(candidate)
    return candidate


def _details(frame: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    details, _summary = evaluate_actual_command_replay(
        frame,
        artifact,
        dt_s=5.0,
        horizons_s=HORIZONS_S,
    )
    return details


def _q_score(details: pd.DataFrame) -> float:
    q_details = details.loc[details["state"] == "Q_Evap"]
    horizon_mae = q_details.groupby("horizon_s")["abs_error"].mean()
    return float(np.mean(horizon_mae.to_numpy(dtype=float)) / 500.0)


def _thermal_score(details: pd.DataFrame) -> float:
    selected = details.loc[details["state"].isin(THERMAL_ERROR_SCALES)].copy()
    grouped = selected.groupby(["horizon_s", "state"])["abs_error"].mean()
    normalized = [
        float(value) / THERMAL_ERROR_SCALES[state]
        for (_horizon, state), value in grouped.items()
    ]
    return float(np.mean(normalized))


def _thermal_residuals(details: pd.DataFrame) -> np.ndarray:
    selected = details.loc[details["state"].isin(THERMAL_ERROR_SCALES)].copy()
    counts = selected.groupby(["horizon_s", "state"])["error"].transform("size")
    scales = selected["state"].map(THERMAL_ERROR_SCALES).to_numpy(dtype=float)
    return (
        selected["error"].to_numpy(dtype=float)
        / scales
        / np.sqrt(counts.to_numpy(dtype=float))
    )


def _summary_records(frame: pd.DataFrame, artifact: dict) -> list[dict]:
    _details_frame, summary = evaluate_actual_command_replay(
        frame,
        artifact,
        dt_s=5.0,
        horizons_s=HORIZONS_S,
    )
    return summary.to_dict(orient="records")


def calibrate_actual_replay(
    base_artifact: dict,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
) -> tuple[dict, dict]:
    validate_physics_artifact(base_artifact)

    evap_candidates = []
    for tau_cond_s in EVAP_TAU_COND_CANDIDATES_S:
        for tau_evap_s in EVAP_TAU_EVAP_CANDIDATES_S:
            candidate = apply_structural_evap_correction(
                base_artifact,
                tau_cond_s=tau_cond_s,
                tau_evap_s=tau_evap_s,
            )
            score = _q_score(_details(train_frame, candidate))
            evap_candidates.append((score, tau_cond_s, tau_evap_s, candidate))
    evap_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    evap_score, tau_cond_s, tau_evap_s, evap_artifact = evap_candidates[0]

    span = THERMAL_UPPER - THERMAL_LOWER
    start_z = (THERMAL_START - THERMAL_LOWER) / span

    def artifact_from_z(parameters_z: np.ndarray) -> dict:
        return thermal_candidate(
            evap_artifact,
            THERMAL_LOWER + np.asarray(parameters_z, dtype=float) * span,
        )

    def residual(parameters_z: np.ndarray) -> np.ndarray:
        candidate = artifact_from_z(parameters_z)
        dynamic_residuals = _thermal_residuals(_details(train_frame, candidate))
        regularization = 0.02 * (np.asarray(parameters_z) - start_z)
        return np.concatenate((dynamic_residuals, regularization))

    optimizer = least_squares(
        residual,
        start_z,
        bounds=(np.zeros_like(start_z), np.ones_like(start_z)),
        max_nfev=30,
        diff_step=0.02,
        loss="soft_l1",
        x_scale="jac",
    )
    if not optimizer.success or not np.isfinite(optimizer.x).all():
        raise RuntimeError(f"actual-replay thermal calibration failed: {optimizer.message}")
    candidate = artifact_from_z(optimizer.x)

    base_train_details = _details(train_frame, base_artifact)
    candidate_train_details = _details(train_frame, candidate)
    base_thermal_score = _thermal_score(base_train_details)
    candidate_thermal_score = _thermal_score(candidate_train_details)
    if candidate_thermal_score >= base_thermal_score:
        raise RuntimeError("actual-replay thermal calibration did not improve training score")

    fitted_values = {
        name: float(candidate["thermal"][name])
        for name in THERMAL_CALIBRATION_KEYS
    }
    calibration_metadata = {
        "fit_status": "validated",
        "selection_source": "training_scene_only",
        "training_scene": "peak",
        "validation_scene": "freq_holdout",
        "horizons_s": list(HORIZONS_S),
        "structural_delay_corrections": STRUCTURAL_DELAY_CORRECTIONS,
        "selected_evap_parameters": {
            "tau_cond_s": float(tau_cond_s),
            "tau_evap_s": float(tau_evap_s),
        },
        "evap_training_score": float(evap_score),
        "thermal_calibration_keys": list(THERMAL_CALIBRATION_KEYS),
        "thermal_fitted_values": fitted_values,
        "coolant_cp_fixed_j_kg_k": PHYSICAL_COOLANT_CP_J_KG_K,
        "capacity_model_frozen": True,
        "base_training_thermal_score": base_thermal_score,
        "candidate_training_thermal_score": candidate_thermal_score,
        "optimizer": {
            "success": bool(optimizer.success),
            "cost": float(optimizer.cost),
            "nfev": int(optimizer.nfev),
            "message": str(optimizer.message),
        },
    }
    candidate.setdefault("fit", {})["actual_replay_calibration"] = calibration_metadata
    validate_physics_artifact(candidate)

    report = {
        "method": "actual_future_command_replay",
        "selection_uses_validation_targets": False,
        "calibration": calibration_metadata,
        "base_train": _summary_records(train_frame, base_artifact),
        "candidate_train": _summary_records(train_frame, candidate),
        "base_validation": _summary_records(validation_frame, base_artifact),
        "candidate_validation": _summary_records(validation_frame, candidate),
    }
    return candidate, report


def _validate_output_path(path: Path) -> Path:
    resolved = path.resolve()
    model_data = (Path(__file__).resolve().parent / "model_data").resolve()
    if resolved == model_data or model_data in resolved.parents:
        raise ValueError("calibrated artifact must not be written into model_data")
    return resolved


def parse_args():
    parser = argparse.ArgumentParser(
        description="用调峰真实指令回放校准P模型，并用调频数据独立验证"
    )
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--output-artifact", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_physics_artifact(args.base_artifact, require_validated=True)
    train = pd.read_csv(args.train_csv, encoding="utf-8-sig")
    validation = pd.read_csv(args.validation_csv, encoding="utf-8-sig")
    artifact, report = calibrate_actual_replay(base, train, validation)
    output_artifact = _validate_output_path(args.output_artifact)
    output_artifact.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    output_artifact.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata = artifact["fit"]["actual_replay_calibration"]
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
