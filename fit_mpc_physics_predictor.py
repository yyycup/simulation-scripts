import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from btms_runtime import ensure_env_library_bin_on_path

ensure_env_library_bin_on_path()

from scipy.optimize import least_squares

from mpc_physics_predictor import (
    evaluate_physics_capacity,
    validate_physics_artifact,
)
from mpc_predictor_selection import PHYSICS_P
from predictor_identification_data import validate_identification_frame


LOWER = np.array([1900, 10, -2, -2, -2, -2, -2, -2], dtype=float)
UPPER = np.array([2000, 80, 2, 2, 2, 2, 2, 2], dtype=float)
INITIAL = np.array([1950, 25, 0.6, -0.1, 0, 0, 0, 0], dtype=float)
N_PUMP_REF_RPM = 2000.0
Q_UPPER_W = 4800.0
FIT_COLUMNS = (
    "n_comp_eff_rpm",
    "n_pump_eff_rpm",
    "t_cool_c",
    "t_ambient_c",
    "q_evap_ss_w",
)


def _artifact_from_parameters(parameters: np.ndarray) -> dict:
    artifact = {
        "model_type": PHYSICS_P,
        "schema_version": 1,
        "gate": {
            "n_on_rpm": float(parameters[0]),
            "width_rpm": float(parameters[1]),
        },
        "capacity": {
            "coefficients": [float(value) for value in parameters[2:]],
            "n_pump_ref_rpm": N_PUMP_REF_RPM,
            "q_upper_w": Q_UPPER_W,
        },
    }
    validate_physics_artifact(artifact)
    return artifact


def _validate_fit_subset(frame: pd.DataFrame, expected_split: str, allow_empty: bool) -> None:
    missing = [column for column in ("split", *FIT_COLUMNS) if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing fit columns: {missing}")
    if not allow_empty and frame.empty:
        raise ValueError(f"{expected_split} data must not be empty")
    invalid_splits = set(frame["split"].tolist()) - {expected_split}
    if invalid_splits:
        raise ValueError(
            f"{expected_split} data contains other splits: {sorted(map(str, invalid_splits))}"
        )
    if frame.loc[:, FIT_COLUMNS].isna().any().any():
        raise ValueError(f"{expected_split} fit columns contain NaN")
    try:
        values = frame.loc[:, FIT_COLUMNS].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{expected_split} fit columns must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{expected_split} fit columns must be finite")
    if len(values) and np.any(values[:, 1] <= 0.0):
        raise ValueError(f"{expected_split} n_pump_eff_rpm must be greater than zero")


def _predict(parameters: np.ndarray, features: np.ndarray) -> np.ndarray:
    artifact = _artifact_from_parameters(parameters)
    return np.asarray(
        [
            evaluate_physics_capacity(
                n_comp,
                n_pump,
                t_cool,
                t_ambient,
                artifact=artifact,
            )
            for n_comp, n_pump, t_cool, t_ambient in features
        ],
        dtype=float,
    )


def _monotonic_contexts(validation: pd.DataFrame) -> np.ndarray:
    if validation.empty:
        return np.empty((0, 3), dtype=float)
    return validation.loc[
        :, ["n_pump_eff_rpm", "t_cool_c", "t_ambient_c"]
    ].drop_duplicates().to_numpy(dtype=float)


def _monotonic_violations(parameters: np.ndarray, contexts: np.ndarray) -> np.ndarray:
    if len(contexts) == 0:
        return np.empty(0, dtype=float)
    speeds = np.linspace(2000.0, 6000.0, 17)
    violations = []
    for n_pump, t_cool, t_ambient in contexts:
        features = np.column_stack(
            (
                speeds,
                np.full_like(speeds, n_pump),
                np.full_like(speeds, t_cool),
                np.full_like(speeds, t_ambient),
            )
        )
        differences = np.diff(_predict(parameters, features))
        violations.extend(np.maximum(0.0, -differences))
    return np.asarray(violations, dtype=float)


def fit_physics_predictor(
    train_frame: pd.DataFrame, validation_frame: pd.DataFrame
) -> dict:
    _validate_fit_subset(train_frame, "train", allow_empty=False)
    _validate_fit_subset(validation_frame, "validation", allow_empty=True)

    train_features = train_frame.loc[
        :, ["n_comp_eff_rpm", "n_pump_eff_rpm", "t_cool_c", "t_ambient_c"]
    ].to_numpy(dtype=float)
    train_targets = train_frame["q_evap_ss_w"].to_numpy(dtype=float)
    contexts = _monotonic_contexts(validation_frame)

    def residual(parameters: np.ndarray) -> np.ndarray:
        predictions = _predict(parameters, train_features)
        low_speed_excess = np.maximum(
            0.0,
            predictions[train_features[:, 0] < 2000.0] - 25.0,
        )
        monotonic_penalty = _monotonic_violations(parameters, contexts)
        return np.concatenate(
            (
                predictions - train_targets,
                10.0 * low_speed_excess,
                10.0 * monotonic_penalty,
            )
        )

    result = least_squares(
        residual,
        INITIAL.copy(),
        bounds=(LOWER, UPPER),
        tr_solver="lsmr",
    )
    if not result.success:
        raise RuntimeError(f"Physics capacity optimizer failed: {result.message}")
    parameters = np.asarray(result.x, dtype=float)
    if not np.isfinite(parameters).all():
        raise RuntimeError("Physics capacity optimizer returned non-finite parameters")
    if np.any(parameters < LOWER) or np.any(parameters > UPPER):
        raise RuntimeError("Physics capacity optimizer returned parameters outside bounds")

    artifact = _artifact_from_parameters(parameters)
    monotonic_check_contexts = contexts
    if len(monotonic_check_contexts) == 0:
        monotonic_check_contexts = np.array([[2000.0, 27.5, 30.0]], dtype=float)
    if np.any(_monotonic_violations(parameters, monotonic_check_contexts) > 1e-7):
        raise RuntimeError("Fitted physics capacity is not monotonic on the validation grid")

    validation_available = not validation_frame.empty
    validation_metrics = None
    if validation_available:
        validation_features = validation_frame.loc[
            :, ["n_comp_eff_rpm", "n_pump_eff_rpm", "t_cool_c", "t_ambient_c"]
        ].to_numpy(dtype=float)
        validation_targets = validation_frame["q_evap_ss_w"].to_numpy(dtype=float)
        validation_errors = _predict(parameters, validation_features) - validation_targets
        validation_metrics = {
            "mae_w": float(np.mean(np.abs(validation_errors))),
            "rmse_w": float(np.sqrt(np.mean(np.square(validation_errors)))),
        }
        if not all(math.isfinite(value) for value in validation_metrics.values()):
            raise RuntimeError("Validation metrics are not finite")

    artifact["fit"] = {
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "validation_available": validation_available,
        "validation_metrics": validation_metrics,
        "optimizer": {
            "success": bool(result.success),
            "cost": float(result.cost),
        },
    }
    return artifact


def fit_physics_artifact(frame: pd.DataFrame) -> dict:
    validate_identification_frame(frame)
    train_frame = frame.loc[frame["split"] == "train"].copy()
    validation_frame = frame.loc[frame["split"] == "validation"].copy()
    return fit_physics_predictor(train_frame, validation_frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the steady physics-P capacity layer")
    parser.add_argument("--steady-csv", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    frame = pd.read_csv(arguments.steady_csv, encoding="utf-8")
    artifact = fit_physics_artifact(frame)
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"train_rows={artifact['fit']['train_rows']} "
        f"validation_rows={artifact['fit']['validation_rows']}"
    )
    print(f"artifact={output_path}")


if __name__ == "__main__":
    main()
