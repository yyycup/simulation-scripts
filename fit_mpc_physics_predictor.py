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
    DEFAULT_INPUT_DOMAIN,
    evaluate_physics_capacity,
    validate_physics_artifact,
)
from mpc_predictor_selection import PHYSICS_P
from predictor_identification_data import validate_identification_frame


LOWER = np.array([1900, 10, -2, -2, -2, -2, -2, -2], dtype=float)
UPPER = np.array([2000, 80, 2, 2, 2, 2, 2, 2], dtype=float)
STARTS = (
    np.array([1925, 15, 0.6, -0.1, 0, 0, 0, 0], dtype=float),
    np.array([1950, 25, 0.6, -0.1, 0, 0, 0, 0], dtype=float),
    np.array([1975, 50, 0.6, -0.1, 0, 0, 0, 0], dtype=float),
)
FEATURE_CANDIDATES = (
    ("base", (True, True, True, True, False, False)),
    ("base_plus_c4", (True, True, True, True, True, False)),
    ("base_plus_c5", (True, True, True, True, False, True)),
    ("full", (True, True, True, True, True, True)),
)
RIDGE_CANDIDATES = (1e-6, 1e-4, 1e-2)
N_PUMP_REF_RPM = 2000.0
Q_UPPER_W = 4800.0
FIT_COLUMNS = (
    "n_comp_eff_rpm",
    "n_pump_eff_rpm",
    "t_cool_c",
    "t_ambient_c",
    "q_evap_ss_w",
)
FEATURE_COLUMNS = FIT_COLUMNS[:-1]


def _masked_parameters(parameters: np.ndarray, mask: tuple[bool, ...]) -> np.ndarray:
    masked = np.asarray(parameters, dtype=float).copy()
    masked[2:] *= np.asarray(mask, dtype=float)
    return masked


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
        "input_domain": {name: list(limits) for name, limits in DEFAULT_INPUT_DOMAIN.items()},
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
            evaluate_physics_capacity(*feature, artifact=artifact)
            for feature in features
        ],
        dtype=float,
    )


def _constraint_contexts(validation: pd.DataFrame) -> np.ndarray:
    if validation.empty:
        return np.array([[2000.0, 27.5, 30.0]], dtype=float)
    return validation.loc[
        :, ["n_pump_eff_rpm", "t_cool_c", "t_ambient_c"]
    ].drop_duplicates().to_numpy(dtype=float)


def _grid_predictions(
    parameters: np.ndarray, contexts: np.ndarray, speeds: np.ndarray
) -> np.ndarray:
    predictions = []
    for n_pump, t_cool, t_ambient in contexts:
        features = np.column_stack(
            (
                speeds,
                np.full_like(speeds, n_pump),
                np.full_like(speeds, t_cool),
                np.full_like(speeds, t_ambient),
            )
        )
        predictions.append(_predict(parameters, features))
    return np.asarray(predictions, dtype=float)


def _physical_checks(parameters: np.ndarray, contexts: np.ndarray) -> tuple[float, float]:
    low_speed = _grid_predictions(
        parameters, contexts, np.array([1000.0, 1400.0, 1800.0])
    )
    active = _grid_predictions(parameters, contexts, np.linspace(2000.0, 6000.0, 17))
    return float(np.max(low_speed)), float(np.min(np.diff(active, axis=1)))


def _candidate(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    contexts: np.ndarray,
    mask: tuple[bool, ...],
    ridge: float,
    start: np.ndarray,
) -> tuple[object, np.ndarray] | None:
    mask_array = np.asarray(mask, dtype=float)

    def residual(parameters: np.ndarray) -> np.ndarray:
        masked = _masked_parameters(parameters, mask)
        train_errors = _predict(masked, train_features) - train_targets
        low_speed = _grid_predictions(
            masked, contexts, np.array([1000.0, 1400.0, 1800.0])
        )
        active = _grid_predictions(
            masked, contexts, np.linspace(2000.0, 6000.0, 17)
        )
        low_speed_penalty = np.maximum(0.0, low_speed.ravel() - 25.0)
        monotonic_penalty = np.maximum(0.0, -np.diff(active, axis=1).ravel())
        regularization = math.sqrt(ridge) * masked[2:] * mask_array
        return np.concatenate(
            (
                train_errors,
                10.0 * low_speed_penalty,
                10.0 * monotonic_penalty,
                regularization,
            )
        )

    try:
        result = least_squares(
            residual,
            start.copy(),
            bounds=(LOWER, UPPER),
            tr_solver="lsmr",
        )
    except (ValueError, RuntimeError, FloatingPointError):
        return None
    parameters = np.asarray(result.x, dtype=float)
    if (
        not result.success
        or parameters.shape != LOWER.shape
        or not np.isfinite(parameters).all()
        or np.any(parameters < LOWER)
        or np.any(parameters > UPPER)
    ):
        return None
    parameters = _masked_parameters(parameters, mask)
    max_low_speed, min_active_slope = _physical_checks(parameters, contexts)
    if (
        not math.isfinite(max_low_speed)
        or not math.isfinite(min_active_slope)
        or max_low_speed > 25.0 + 1e-7
        or min_active_slope < -1e-7
    ):
        return None
    return result, parameters


def _validation_metrics(parameters: np.ndarray, validation: pd.DataFrame) -> dict:
    features = validation.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    targets = validation["q_evap_ss_w"].to_numpy(dtype=float)
    errors = _predict(parameters, features) - targets
    metrics = {
        "mae_w": float(np.mean(np.abs(errors))),
        "rmse_w": float(np.sqrt(np.mean(np.square(errors)))),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("Validation metrics are not finite")
    return metrics


def fit_physics_predictor(
    train_frame: pd.DataFrame, validation_frame: pd.DataFrame
) -> dict:
    _validate_fit_subset(train_frame, "train", allow_empty=False)
    _validate_fit_subset(validation_frame, "validation", allow_empty=True)
    train_features = train_frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    train_targets = train_frame["q_evap_ss_w"].to_numpy(dtype=float)
    contexts = _constraint_contexts(validation_frame)

    configurations = []
    if validation_frame.empty:
        configurations.append((FEATURE_CANDIDATES[0], RIDGE_CANDIDATES[1], 1))
    else:
        for feature_candidate in FEATURE_CANDIDATES:
            for ridge in RIDGE_CANDIDATES:
                for start_index in range(len(STARTS)):
                    configurations.append((feature_candidate, ridge, start_index))

    candidates = []
    for (feature_name, mask), ridge, start_index in configurations:
        fitted = _candidate(
            train_features,
            train_targets,
            contexts,
            mask,
            ridge,
            STARTS[start_index],
        )
        if fitted is None:
            continue
        result, parameters = fitted
        metrics = None
        if not validation_frame.empty:
            metrics = _validation_metrics(parameters, validation_frame)
        max_low_speed, min_active_slope = _physical_checks(parameters, contexts)
        candidates.append(
            {
                "feature_name": feature_name,
                "mask": mask,
                "ridge": ridge,
                "start_index": start_index,
                "result": result,
                "parameters": parameters,
                "metrics": metrics,
                "max_low_speed_w": max_low_speed,
                "min_active_slope_w_per_step": min_active_slope,
            }
        )
    if not candidates:
        raise RuntimeError("Physics capacity fitting failed: all candidates failed")

    if validation_frame.empty:
        selected = candidates[0]
    else:
        selected = None
        for feature_name, _ in FEATURE_CANDIDATES:
            same_features = [c for c in candidates if c["feature_name"] == feature_name]
            if not same_features:
                continue
            best = min(
                same_features,
                key=lambda c: (c["metrics"]["mae_w"], c["ridge"], c["start_index"]),
            )
            if selected is None or best["metrics"]["mae_w"] < selected["metrics"]["mae_w"] - 1e-9:
                selected = best
        if selected is None:
            raise RuntimeError("Physics capacity fitting failed: all candidates failed")

    parameters = selected["parameters"]
    artifact = _artifact_from_parameters(parameters)
    validation_available = not validation_frame.empty
    artifact["fit"] = {
        "fit_status": "validated" if validation_available else "mechanical_smoke_unvalidated",
        "selection_method": (
            "validation_mae_strict_complexity"
            if validation_available
            else "fixed_mechanical_no_validation"
        ),
        "selection_metric": "validation_mae_w" if validation_available else None,
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "validation_available": validation_available,
        "validation_metrics": selected["metrics"],
        "selected_features": selected["feature_name"],
        "selected_mask": list(selected["mask"]),
        "ridge": selected["ridge"],
        "start_index": selected["start_index"],
        "candidate_count": len(candidates),
        "max_low_speed_w": selected["max_low_speed_w"],
        "min_active_slope_w_per_step": selected["min_active_slope_w_per_step"],
        "optimizer": {
            "success": bool(selected["result"].success),
            "cost": float(selected["result"].cost),
        },
    }
    return artifact


def fit_physics_artifact(frame: pd.DataFrame) -> dict:
    validate_identification_frame(frame)
    train_frame = frame.loc[frame["split"] == "train"].copy()
    validation_frame = frame.loc[frame["split"] == "validation"].copy()
    return fit_physics_predictor(train_frame, validation_frame)


def _validate_output_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    model_data = (Path(__file__).resolve().parent / "model_data").resolve()
    if resolved == model_data or model_data in resolved.parents:
        raise ValueError("Fitted artifacts must not be written directly into model_data")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the steady physics-P capacity layer")
    parser.add_argument("--steady-csv", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    frame = pd.read_csv(arguments.steady_csv, encoding="utf-8")
    artifact = fit_physics_artifact(frame)
    output_path = _validate_output_path(arguments.output)
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
