import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from btms_runtime import ensure_env_library_bin_on_path

ensure_env_library_bin_on_path()

from p_mpc_run_support import PROJECT_ROOT

from scipy.optimize import least_squares

from mpc_physics_predictor import (
    CUBIC_CAPACITY_FEATURE_NAMES,
    CUBIC_CAPACITY_MODEL,
    DEFAULT_DYNAMIC_PARAMETERS,
    DEFAULT_INPUT_DOMAIN,
    DEFAULT_THERMAL_PARAMETERS,
    ENHANCED_CAPACITY_FEATURE_NAMES,
    ENHANCED_CAPACITY_MODEL,
    PHYSICAL_COOLANT_CP_J_KG_K,
    SCHEDULE_PARAMETER_NAMES,
    evaluate_physics_capacity,
    initialize_physics_state,
    load_physics_artifact,
    step_physics_predictor,
    validate_physics_artifact,
)
from mpc_predictor_selection import PHYSICS_P
from .predictor_identification_data import validate_identification_frame


STARTS = (
    np.array([0.6, -0.1, 0.015, -0.002] + [0.0] * 10, dtype=float),
    np.array([0.5, -0.05, 0.05, -0.01] + [0.0] * 10, dtype=float),
)
CUBIC_START = np.array([0.65] + [0.0] * 19, dtype=float)
FEATURE_CANDIDATES = (
    ("base_5", (True,) * 5 + (False,) * 9),
    ("physical_7", (True,) * 7 + (False,) * 7),
    ("physical_10", (True,) * 10 + (False,) * 4),
    ("physical_14", (True,) * 14),
)
RIDGE_CANDIDATES = (1e-6, 1e-4, 1e-2)
WEIGHT_POWER_CANDIDATES = (0.0, 0.5, 1.0)
N_PUMP_REF_RPM = 2000.0
Q_UPPER_W = 4800.0
MINIMUM_ACTIVE_RPM = 1000.0
VALIDATION_TARGETS = {
    "active_mape_percent": 4.25,
    "active_rmse_w": 100.0,
    "active_max_relative_error_percent": 16.5,
}
FIT_COLUMNS = (
    "n_comp_eff_rpm",
    "n_pump_eff_rpm",
    "t_cool_c",
    "t_ambient_c",
    "q_evap_ss_w",
)
FEATURE_COLUMNS = FIT_COLUMNS[:-1]

DYNAMIC_NUMERIC_KEYS = tuple(
    name for name in DEFAULT_DYNAMIC_PARAMETERS if name != "time_constant_model"
)
THERMAL_KEYS = tuple(
    name
    for name in DEFAULT_THERMAL_PARAMETERS
    if name not in {"coolant_cp_j_kg_k", "battery_heat_generation_scale"}
)
SCHEDULE_KEYS = tuple(sorted(SCHEDULE_PARAMETER_NAMES))
DYNAMIC_LOWER = np.array(
    [0.1, 0.1, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    + [0.05, 1600.0, 50000.0, 10000.0, 20.0, 1.0, 0.1, 0.05],
    dtype=float,
)
DYNAMIC_UPPER = np.array(
    [30.0, 30.0, 60.0, 60.0, 200.0, 200.0, 60.0, 60.0]
    + [0.8, 4800.0, 1000000.0, 500000.0, 2000.0, 100.0, 100.0, 1.0],
    dtype=float,
)
DYNAMIC_START = np.array(
    [DEFAULT_DYNAMIC_PARAMETERS[name] for name in DYNAMIC_NUMERIC_KEYS]
    + [DEFAULT_THERMAL_PARAMETERS[name] for name in THERMAL_KEYS],
    dtype=float,
)
PLATE_REFINEMENT_KEYS = (
    "battery_plate_conductance_w_k",
    "plate_tau_s",
    "plate_fluid_effectiveness",
)
PLATE_REFINEMENT_LOWER = np.array([20.0, 1.0, 0.05], dtype=float)
PLATE_REFINEMENT_UPPER = np.array([2000.0, 100.0, 1.0], dtype=float)
DYNAMIC_REQUIRED_COLUMNS = (
    "scenario_id",
    "split",
    "time_s",
    "dt_s",
    "n_comp_cmd_rpm",
    "n_pump_cmd_rpm",
    "n_comp_eff_rpm",
    "n_pump_eff_rpm",
    "q_gen_w",
    "t_ambient_c",
    "q_cond_eff_w",
    "q_evap_eff_w",
    "t_supply_c",
    "t_plate_c",
    "t_return_c",
    "t_batt_c",
    "t_cool_c",
)
DYNAMIC_RESPONSE_FIELDS = (
    ("n_comp_eff_rpm", "n_comp_eff_rpm", 500.0),
    ("n_pump_eff_rpm", "n_pump_eff_rpm", 500.0),
    ("q_evap_w", "q_evap_eff_w", 500.0),
    ("t_supply_c", "t_supply_c", 1.0),
    ("t_plate_c", "t_plate_c", 1.0),
    ("t_return_c", "t_return_c", 1.0),
    ("t_batt_c", "t_batt_c", 0.2),
    ("t_cool_c", "t_cool_c", 0.5),
)
HORIZON_FIELDS = (
    ("q_evap_w", "q_evap_eff_w", 500.0),
    ("t_supply_c", "t_supply_c", 1.0),
    ("t_plate_c", "t_plate_c", 1.0),
    ("t_return_c", "t_return_c", 1.0),
    ("t_batt_c", "t_batt_c", 0.2),
    ("t_cool_c", "t_cool_c", 0.5),
)
TRAINING_HORIZONS = (10, 20, 60)


def _masked_parameters(parameters: np.ndarray, mask: tuple[bool, ...]) -> np.ndarray:
    masked = np.asarray(parameters, dtype=float).copy()
    masked *= np.asarray(mask, dtype=float)
    return masked


def _artifact_from_parameters(
    parameters: np.ndarray,
    capacity_model: str = ENHANCED_CAPACITY_MODEL,
) -> dict:
    feature_names_by_model = {
        ENHANCED_CAPACITY_MODEL: ENHANCED_CAPACITY_FEATURE_NAMES,
        CUBIC_CAPACITY_MODEL: CUBIC_CAPACITY_FEATURE_NAMES,
    }
    if capacity_model not in feature_names_by_model:
        raise ValueError(f"Unsupported capacity model: {capacity_model!r}")
    feature_names = feature_names_by_model[capacity_model]
    artifact = {
        "model_type": PHYSICS_P,
        "schema_version": 1,
        "gate": {
            "mode": "hard",
            "n_on_rpm": MINIMUM_ACTIVE_RPM,
            "width_rpm": 10.0,
        },
        "capacity": {
            "model": capacity_model,
            "coefficients": [float(value) for value in parameters],
            "feature_names": list(feature_names),
            "minimum_active_rpm": MINIMUM_ACTIVE_RPM,
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


def _design_matrix(features: np.ndarray, capacity_model: str) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    pump_ratio = N_PUMP_REF_RPM / values[:, 1]
    coolant = (values[:, 2] - 27.5) / 7.5
    ambient = (values[:, 3] - 30.0) / 10.0
    compressor = (values[:, 0] - 4000.0) / 2000.0
    if capacity_model == ENHANCED_CAPACITY_MODEL:
        return np.column_stack((
            np.ones(len(values)),
            pump_ratio,
            coolant,
            ambient,
            compressor,
            compressor * coolant,
            compressor * ambient,
            pump_ratio * coolant,
            compressor**2,
            coolant * ambient,
            coolant**2,
            ambient**2,
            pump_ratio * ambient,
            pump_ratio * compressor,
        ))
    if capacity_model == CUBIC_CAPACITY_MODEL:
        return np.column_stack((
            np.ones(len(values)),
            compressor,
            pump_ratio,
            coolant,
            ambient,
            compressor**2,
            compressor * pump_ratio,
            compressor * coolant,
            compressor * ambient,
            pump_ratio**2,
            pump_ratio * coolant,
            pump_ratio * ambient,
            coolant**2,
            coolant * ambient,
            compressor**2 * pump_ratio,
            compressor * pump_ratio**2,
            compressor * pump_ratio * coolant,
            compressor * pump_ratio * ambient,
            compressor * coolant * ambient,
            coolant**3,
        ))
    raise ValueError(f"Unsupported capacity model: {capacity_model!r}")


def _predict(
    parameters: np.ndarray,
    features: np.ndarray,
    capacity_model: str = ENHANCED_CAPACITY_MODEL,
) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    n_comp = values[:, 0]
    design = _design_matrix(values, capacity_model)
    raw = n_comp * (design @ np.asarray(parameters, dtype=float))
    active = np.where(n_comp < MINIMUM_ACTIVE_RPM, 0.0, raw)
    return np.clip(active, 0.0, Q_UPPER_W)


def _constraint_contexts(validation: pd.DataFrame) -> np.ndarray:
    if validation.empty:
        return np.array([[2000.0, 27.5, 30.0]], dtype=float)
    return validation.loc[
        :, ["n_pump_eff_rpm", "t_cool_c", "t_ambient_c"]
    ].drop_duplicates().to_numpy(dtype=float)


def _grid_predictions(
    parameters: np.ndarray,
    contexts: np.ndarray,
    speeds: np.ndarray,
    capacity_model: str = ENHANCED_CAPACITY_MODEL,
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
        predictions.append(_predict(parameters, features, capacity_model))
    return np.asarray(predictions, dtype=float)


def _physical_checks(
    parameters: np.ndarray,
    contexts: np.ndarray,
    capacity_model: str = ENHANCED_CAPACITY_MODEL,
) -> tuple[float, float]:
    low_speed = _grid_predictions(
        parameters, contexts, np.array([300.0, 500.0, 999.0]), capacity_model
    )
    active = _grid_predictions(
        parameters, contexts, np.linspace(1000.0, 6000.0, 21), capacity_model
    )
    return float(np.max(low_speed)), float(np.min(np.diff(active, axis=1)))


def _candidate(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    contexts: np.ndarray,
    mask: tuple[bool, ...],
    ridge: float,
    weight_power: float,
    start: np.ndarray,
    capacity_model: str = ENHANCED_CAPACITY_MODEL,
) -> tuple[object, np.ndarray] | None:
    mask_array = np.asarray(mask, dtype=float)
    lower = np.full(len(start), -2.0, dtype=float)
    upper = np.full(len(start), 2.0, dtype=float)

    def residual(parameters: np.ndarray) -> np.ndarray:
        masked = _masked_parameters(parameters, mask)
        train_errors = (
            _predict(masked, train_features, capacity_model) - train_targets
        )
        train_scale = np.maximum(np.abs(train_targets), 1.0) ** weight_power
        weighted_train_errors = train_errors / train_scale
        low_speed = _grid_predictions(
            masked, contexts, np.array([300.0, 500.0, 999.0]), capacity_model
        )
        active = _grid_predictions(
            masked, contexts, np.linspace(1000.0, 6000.0, 21), capacity_model
        )
        low_speed_penalty = np.maximum(0.0, low_speed.ravel() - 25.0)
        monotonic_penalty = np.maximum(0.0, -np.diff(active, axis=1).ravel())
        regularization = math.sqrt(ridge) * masked * mask_array
        return np.concatenate(
            (
                weighted_train_errors,
                low_speed_penalty,
                monotonic_penalty,
                regularization,
            )
        )

    try:
        result = least_squares(
            residual,
            start.copy(),
            bounds=(lower, upper),
            tr_solver="lsmr",
        )
    except (ValueError, RuntimeError, FloatingPointError):
        return None
    parameters = np.asarray(result.x, dtype=float)
    if (
        not result.success
        or parameters.shape != lower.shape
        or not np.isfinite(parameters).all()
        or np.any(parameters < lower)
        or np.any(parameters > upper)
    ):
        return None
    parameters = _masked_parameters(parameters, mask)
    max_low_speed, min_active_slope = _physical_checks(
        parameters, contexts, capacity_model
    )
    if (
        not math.isfinite(max_low_speed)
        or not math.isfinite(min_active_slope)
        or max_low_speed > 25.0 + 1e-7
        or min_active_slope < -1e-7
    ):
        return None
    return result, parameters


def _validation_metrics(
    parameters: np.ndarray,
    validation: pd.DataFrame,
    capacity_model: str = ENHANCED_CAPACITY_MODEL,
) -> dict:
    features = validation.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    targets = validation["q_evap_ss_w"].to_numpy(dtype=float)
    errors = _predict(parameters, features, capacity_model) - targets
    active = features[:, 0] >= MINIMUM_ACTIVE_RPM
    active_targets = targets[active]
    if not len(active_targets) or np.any(np.abs(active_targets) <= 1e-9):
        raise RuntimeError("Validation requires nonzero active capacity targets")
    active_relative_error = np.abs(errors[active]) / np.abs(active_targets)
    metrics = {
        "mae_w": float(np.mean(np.abs(errors))),
        "rmse_w": float(np.sqrt(np.mean(np.square(errors)))),
        "active_rmse_w": float(
            np.sqrt(np.mean(np.square(errors[active])))
        ),
        "active_mape_percent": float(100.0 * np.mean(active_relative_error)),
        "active_max_relative_error_percent": float(
            100.0 * np.max(active_relative_error)
        ),
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
        configurations.append(
            (
                ENHANCED_CAPACITY_MODEL,
                FEATURE_CANDIDATES[0],
                RIDGE_CANDIDATES[1],
                WEIGHT_POWER_CANDIDATES[0],
                0,
                STARTS[0],
            )
        )
    else:
        for feature_candidate in FEATURE_CANDIDATES:
            for ridge in RIDGE_CANDIDATES:
                for weight_power in WEIGHT_POWER_CANDIDATES:
                    for start_index in range(len(STARTS)):
                        configurations.append(
                            (
                                ENHANCED_CAPACITY_MODEL,
                                feature_candidate,
                                ridge,
                                weight_power,
                                start_index,
                                STARTS[start_index],
                            )
                        )
        configurations.append(
            (
                CUBIC_CAPACITY_MODEL,
                ("cubic_20", (True,) * len(CUBIC_CAPACITY_FEATURE_NAMES)),
                1e-4,
                0.8,
                0,
                CUBIC_START,
            )
        )

    candidates = []
    for (
        capacity_model,
        (feature_name, mask),
        ridge,
        weight_power,
        start_index,
        start,
    ) in configurations:
        fitted = _candidate(
            train_features,
            train_targets,
            contexts,
            mask,
            ridge,
            weight_power,
            start,
            capacity_model,
        )
        if fitted is None:
            continue
        result, parameters = fitted
        metrics = None
        if not validation_frame.empty:
            metrics = _validation_metrics(
                parameters, validation_frame, capacity_model
            )
        max_low_speed, min_active_slope = _physical_checks(
            parameters, contexts, capacity_model
        )
        candidates.append(
            {
                "capacity_model": capacity_model,
                "feature_name": feature_name,
                "mask": mask,
                "ridge": ridge,
                "weight_power": weight_power,
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
        accepted = [
            candidate
            for candidate in candidates
            if all(
                candidate["metrics"][name] <= limit
                for name, limit in VALIDATION_TARGETS.items()
            )
        ]
        selection_pool = accepted or candidates
        selected = min(
            selection_pool,
            key=lambda candidate: (
                sum(candidate["mask"]) if accepted else 0,
                candidate["metrics"]["active_mape_percent"],
                candidate["metrics"]["rmse_w"],
                candidate["ridge"],
                candidate["weight_power"],
                candidate["start_index"],
            ),
        )

    parameters = selected["parameters"]
    artifact = _artifact_from_parameters(parameters, selected["capacity_model"])
    validation_available = not validation_frame.empty
    artifact["fit"] = {
        "fit_status": "validated" if validation_available else "mechanical_smoke_unvalidated",
        "selection_method": (
            "validation_targets_then_minimum_complexity"
            if validation_available
            else "fixed_mechanical_no_validation"
        ),
        "selection_metric": (
            "validation_active_mape_percent" if validation_available else None
        ),
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "validation_available": validation_available,
        "validation_metrics": selected["metrics"],
        "selected_features": selected["feature_name"],
        "selected_capacity_model": selected["capacity_model"],
        "selected_mask": list(selected["mask"]),
        "ridge": selected["ridge"],
        "weight_power": selected["weight_power"],
        "start_index": selected["start_index"],
        "candidate_count": len(candidates),
        "validation_targets": VALIDATION_TARGETS if validation_available else None,
        "validation_target_met": (
            all(
                selected["metrics"][name] <= limit
                for name, limit in VALIDATION_TARGETS.items()
            )
            if validation_available
            else None
        ),
        "max_low_speed_w": selected["max_low_speed_w"],
        "min_active_slope_w_per_step": selected["min_active_slope_w_per_step"],
        "optimizer": {
            "success": bool(selected["result"].success),
            "cost": float(selected["result"].cost),
        },
    }
    return artifact


def _dynamic_artifact(
    base_artifact: dict,
    parameters: np.ndarray,
    model: str,
    evap_response_model: str | None = None,
    battery_plate_conductance_model: str | None = None,
) -> dict:
    artifact = json.loads(json.dumps(base_artifact))
    base_count = len(DYNAMIC_NUMERIC_KEYS) + len(THERMAL_KEYS)
    if parameters.shape != (base_count + (len(SCHEDULE_KEYS) if model == "scheduled" else 0),):
        raise ValueError("Dynamic parameter vector has the wrong length")
    artifact["dynamic"] = {"time_constant_model": model}
    for index, name in enumerate(DYNAMIC_NUMERIC_KEYS):
        artifact["dynamic"][name] = float(parameters[index])
    offset = len(DYNAMIC_NUMERIC_KEYS)
    artifact["thermal"] = {
        "coolant_cp_j_kg_k": PHYSICAL_COOLANT_CP_J_KG_K,
        "battery_heat_generation_scale": float(
            base_artifact.get("thermal", {}).get(
                "battery_heat_generation_scale",
                DEFAULT_THERMAL_PARAMETERS["battery_heat_generation_scale"],
            )
        ),
        **{
            name: float(parameters[offset + index])
            for index, name in enumerate(THERMAL_KEYS)
        },
    }
    if model == "scheduled":
        for index, name in enumerate(SCHEDULE_KEYS):
            artifact["dynamic"][name] = float(parameters[base_count + index])
    if evap_response_model is not None:
        artifact["dynamic"]["evap_response_model"] = evap_response_model
    if battery_plate_conductance_model is not None:
        artifact["thermal"][
            "battery_plate_conductance_model"
        ] = battery_plate_conductance_model
    validate_physics_artifact(artifact)
    return artifact


def _validate_dynamic_subset(
    frame: pd.DataFrame, expected_split: str, allow_empty: bool
) -> None:
    missing = [name for name in DYNAMIC_REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing dynamic fit columns: {missing}")
    if not allow_empty and frame.empty:
        raise ValueError(f"{expected_split} dynamic data must not be empty")
    invalid = set(frame["split"].tolist()) - {expected_split}
    if invalid:
        raise ValueError(
            f"{expected_split} dynamic data contains other splits: {sorted(map(str, invalid))}"
        )
    numeric_columns = [name for name in DYNAMIC_REQUIRED_COLUMNS if name not in {"scenario_id", "split"}]
    try:
        values = frame.loc[:, numeric_columns].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dynamic fit columns must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("Dynamic fit columns must be finite")
    if len(frame) and np.any(frame["dt_s"].to_numpy(dtype=float) <= 0.0):
        raise ValueError("Dynamic dt_s must be greater than zero")


def _initial_dynamic_state(row: pd.Series):
    return initialize_physics_state(
        row["n_comp_eff_rpm"],
        row["n_pump_eff_rpm"],
        row["q_cond_eff_w"],
        row["q_evap_eff_w"],
        row["t_supply_c"],
        row["t_plate_c"],
        row["t_return_c"],
        row["t_batt_c"],
        row["t_cool_c"],
    )


def _dynamic_errors(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    errors = []
    for _, scenario in frame.groupby("scenario_id", sort=True):
        ordered = scenario.sort_values("time_s", kind="stable").reset_index(drop=True)
        if len(ordered) < 2:
            continue
        state = _initial_dynamic_state(ordered.iloc[0])
        for step_index in range(1, len(ordered)):
            row = ordered.iloc[step_index]
            state = step_physics_predictor(
                state,
                row["n_comp_cmd_rpm"],
                row["n_pump_cmd_rpm"],
                row["q_gen_w"],
                row["t_ambient_c"],
                row["dt_s"],
                artifact,
            )
            for state_name, column, scale in DYNAMIC_RESPONSE_FIELDS:
                errors.append((getattr(state, state_name) - float(row[column])) / scale)
            if step_index in TRAINING_HORIZONS:
                for state_name, column, scale in HORIZON_FIELDS:
                    errors.append((getattr(state, state_name) - float(row[column])) / scale)
    return np.asarray(errors, dtype=float)


def _fit_dynamic_candidate(
    base_artifact: dict, train: pd.DataFrame, model: str
) -> tuple[dict, object]:
    if model == "scheduled":
        start = np.concatenate((DYNAMIC_START, np.zeros(len(SCHEDULE_KEYS))))
        lower = np.concatenate((DYNAMIC_LOWER, np.full(len(SCHEDULE_KEYS), -100.0)))
        upper = np.concatenate((DYNAMIC_UPPER, np.full(len(SCHEDULE_KEYS), 100.0)))
    else:
        start = DYNAMIC_START.copy()
        lower = DYNAMIC_LOWER.copy()
        upper = DYNAMIC_UPPER.copy()
    delay_index = DYNAMIC_NUMERIC_KEYS.index("evap_input_delay_s")
    tau_evap_index = DYNAMIC_NUMERIC_KEYS.index("tau_evap_s")
    lower[delay_index], upper[delay_index], start[delay_index] = (
        0.0,
        1e-9,
        0.5e-9,
    )
    lower[tau_evap_index], upper[tau_evap_index], start[tau_evap_index] = (
        45.0,
        45.0 + 1e-9,
        45.0 + 0.5e-9,
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        artifact = _dynamic_artifact(
            base_artifact,
            parameters,
            model,
            evap_response_model="direct",
            battery_plate_conductance_model="constant_physical",
        )
        dynamic_errors = _dynamic_errors(artifact, train)
        regularization = 0.05 * (parameters - start) / np.maximum(upper - lower, 1e-9)
        return np.concatenate((dynamic_errors, regularization))

    result = least_squares(
        residual,
        start,
        bounds=(lower, upper),
        max_nfev=80,
        tr_solver="lsmr",
        ftol=1e-5,
        xtol=1e-5,
        gtol=1e-5,
    )
    parameters = np.asarray(result.x, dtype=float)
    parameters[delay_index] = 0.0
    parameters[tau_evap_index] = 45.0
    if (
        not result.success
        or parameters.shape != start.shape
        or not np.isfinite(parameters).all()
        or np.any(parameters < lower)
        or np.any(parameters > upper)
    ):
        raise RuntimeError(f"Dynamic {model} fitting failed")
    return (
        _dynamic_artifact(
            base_artifact,
            parameters,
            model,
            evap_response_model="direct",
            battery_plate_conductance_model="constant_physical",
        ),
        result,
    )


def _dynamic_validation_metric(artifact: dict, validation: pd.DataFrame) -> float | None:
    if validation.empty:
        return None
    errors = _dynamic_errors(artifact, validation)
    if not len(errors):
        raise ValueError("Validation dynamic trajectories need at least two rows")
    metric = float(np.mean(np.abs(errors)))
    if not math.isfinite(metric):
        raise RuntimeError("Dynamic validation metric is not finite")
    return metric


def _short_response_artifact(base_artifact: dict) -> dict:
    artifact = json.loads(json.dumps(base_artifact))
    artifact["dynamic"].update(
        evap_response_model="direct",
        evap_input_delay_s=0.0,
        tau_evap_s=45.0,
    )
    validate_physics_artifact(artifact)
    return artifact


def refine_physics_short_response(
    base_artifact: dict,
    dynamic_frame: pd.DataFrame,
) -> dict:
    validate_physics_artifact(base_artifact)
    if "dynamic" not in base_artifact or "thermal" not in base_artifact:
        raise ValueError("Short-response refinement requires a dynamic physics artifact")
    validate_identification_frame(dynamic_frame)
    validation = dynamic_frame.loc[dynamic_frame["split"] == "validation"].copy()
    _validate_dynamic_subset(validation, "validation", allow_empty=True)

    candidate = _short_response_artifact(base_artifact)
    base_metric = _dynamic_validation_metric(base_artifact, validation)
    candidate_metric = _dynamic_validation_metric(candidate, validation)
    selected = candidate
    if base_metric is not None and candidate_metric is not None:
        if candidate_metric >= base_metric:
            selected = json.loads(json.dumps(base_artifact))
    improvement = None
    if (
        base_metric is not None
        and candidate_metric is not None
        and base_metric > 0.0
    ):
        improvement = (base_metric - candidate_metric) / base_metric

    selected.setdefault("fit", {})["short_response_refinement"] = {
        "fit_status": (
            "validated" if not validation.empty else "mechanical_smoke_unvalidated"
        ),
        "selection_source": (
            "validation_only" if not validation.empty else "plant_structure"
        ),
        "selected": selected is candidate,
        "evap_response_model": "direct",
        "evap_input_delay_s": 0.0,
        "tau_evap_s": 45.0,
        "all_other_dynamic_and_thermal_parameters_frozen": True,
        "base_validation_weighted_mae": base_metric,
        "candidate_validation_weighted_mae": candidate_metric,
        "validation_improvement": improvement,
    }
    validate_physics_artifact(selected)
    return selected


def _plate_structure_artifact(base_artifact: dict) -> dict:
    artifact = json.loads(json.dumps(base_artifact))
    artifact["thermal"][
        "battery_plate_conductance_model"
    ] = "constant_physical"
    validate_physics_artifact(artifact)
    return artifact


def refine_physics_plate_structure(
    base_artifact: dict,
    dynamic_frame: pd.DataFrame,
) -> dict:
    validate_physics_artifact(base_artifact)
    if "dynamic" not in base_artifact or "thermal" not in base_artifact:
        raise ValueError("Plate-structure refinement requires a dynamic physics artifact")
    validate_identification_frame(dynamic_frame)
    validation = dynamic_frame.loc[dynamic_frame["split"] == "validation"].copy()
    _validate_dynamic_subset(validation, "validation", allow_empty=True)

    candidate = _plate_structure_artifact(base_artifact)
    base_metric = _dynamic_validation_metric(base_artifact, validation)
    candidate_metric = _dynamic_validation_metric(candidate, validation)
    selected = candidate
    if base_metric is not None and candidate_metric is not None:
        if candidate_metric >= base_metric:
            selected = json.loads(json.dumps(base_artifact))
    improvement = None
    if (
        base_metric is not None
        and candidate_metric is not None
        and base_metric > 0.0
    ):
        improvement = (base_metric - candidate_metric) / base_metric

    selected.setdefault("fit", {})["plate_structure_refinement"] = {
        "fit_status": (
            "validated" if not validation.empty else "mechanical_smoke_unvalidated"
        ),
        "selection_source": (
            "validation_only" if not validation.empty else "plant_structure"
        ),
        "selected": selected is candidate,
        "battery_plate_conductance_model": "constant_physical",
        "physical_basis": "cell_to_plate_conductance_is_independent_of_pump_speed",
        "all_numeric_parameters_frozen": True,
        "base_validation_weighted_mae": base_metric,
        "candidate_validation_weighted_mae": candidate_metric,
        "validation_improvement": improvement,
    }
    validate_physics_artifact(selected)
    return selected


def _supply_refinement_artifact(base_artifact: dict) -> dict:
    artifact = json.loads(json.dumps(base_artifact))
    artifact["thermal"]["coolant_cp_j_kg_k"] = PHYSICAL_COOLANT_CP_J_KG_K
    validate_physics_artifact(artifact)
    return artifact


def refine_physics_supply(base_artifact: dict, dynamic_frame: pd.DataFrame) -> dict:
    validate_physics_artifact(base_artifact)
    if "dynamic" not in base_artifact or "thermal" not in base_artifact:
        raise ValueError("Supply refinement requires a dynamic physics artifact")
    validate_identification_frame(dynamic_frame)
    validation = dynamic_frame.loc[dynamic_frame["split"] == "validation"].copy()
    _validate_dynamic_subset(validation, "validation", allow_empty=True)

    candidate = _supply_refinement_artifact(base_artifact)
    base_metric = _dynamic_validation_metric(base_artifact, validation)
    candidate_metric = _dynamic_validation_metric(candidate, validation)
    selected = candidate
    if base_metric is not None and candidate_metric is not None:
        if candidate_metric >= base_metric:
            selected = json.loads(json.dumps(base_artifact))

    selected.setdefault("fit", {})["supply_refinement"] = {
        "fit_status": (
            "validated" if not validation.empty else "mechanical_smoke_unvalidated"
        ),
        "selection_source": (
            "validation_only" if not validation.empty else "known_physical_constant"
        ),
        "corrected_parameter": "coolant_cp_j_kg_k",
        "physical_value_j_kg_k": PHYSICAL_COOLANT_CP_J_KG_K,
        "all_other_parameters_frozen": True,
        "selected": selected is candidate,
        "base_validation_weighted_mae": base_metric,
        "candidate_validation_weighted_mae": candidate_metric,
    }
    validate_physics_artifact(selected)
    return selected


def _plate_refinement_artifact(base_artifact: dict, parameters: np.ndarray) -> dict:
    if parameters.shape != (len(PLATE_REFINEMENT_KEYS),):
        raise ValueError("Plate refinement parameter vector has the wrong length")
    artifact = json.loads(json.dumps(base_artifact))
    for name, value in zip(PLATE_REFINEMENT_KEYS, parameters):
        artifact["thermal"][name] = float(value)
    validate_physics_artifact(artifact)
    return artifact


def _fit_plate_refinement_candidate(
    base_artifact: dict, train: pd.DataFrame
) -> tuple[dict, object]:
    thermal = base_artifact["thermal"]
    start = np.array(
        [
            float(thermal.get(name, DEFAULT_THERMAL_PARAMETERS[name]))
            for name in PLATE_REFINEMENT_KEYS
        ],
        dtype=float,
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        artifact = _plate_refinement_artifact(base_artifact, parameters)
        dynamic_errors = _dynamic_errors(artifact, train)
        regularization = 0.05 * (parameters - start) / (
            PLATE_REFINEMENT_UPPER - PLATE_REFINEMENT_LOWER
        )
        return np.concatenate((dynamic_errors, regularization))

    result = least_squares(
        residual,
        start,
        bounds=(PLATE_REFINEMENT_LOWER, PLATE_REFINEMENT_UPPER),
        max_nfev=50,
        tr_solver="lsmr",
    )
    parameters = np.asarray(result.x, dtype=float)
    if (
        not result.success
        or parameters.shape != start.shape
        or not np.isfinite(parameters).all()
        or np.any(parameters < PLATE_REFINEMENT_LOWER)
        or np.any(parameters > PLATE_REFINEMENT_UPPER)
    ):
        raise RuntimeError("Plate refinement fitting failed")
    return _plate_refinement_artifact(base_artifact, parameters), result


def refine_physics_plate(base_artifact: dict, dynamic_frame: pd.DataFrame) -> dict:
    validate_physics_artifact(base_artifact)
    if "dynamic" not in base_artifact or "thermal" not in base_artifact:
        raise ValueError("Plate refinement requires a dynamic physics artifact")
    validate_identification_frame(dynamic_frame)
    train = dynamic_frame.loc[dynamic_frame["split"] == "train"].copy()
    validation = dynamic_frame.loc[dynamic_frame["split"] == "validation"].copy()
    _validate_dynamic_subset(train, "train", allow_empty=False)
    _validate_dynamic_subset(validation, "validation", allow_empty=True)

    candidate, result = _fit_plate_refinement_candidate(base_artifact, train)
    base_metric = _dynamic_validation_metric(base_artifact, validation)
    candidate_metric = _dynamic_validation_metric(candidate, validation)
    selected = candidate
    if base_metric is not None and candidate_metric is not None:
        if candidate_metric >= base_metric:
            selected = json.loads(json.dumps(base_artifact))

    selected.setdefault("fit", {})["plate_refinement"] = {
        "fit_status": (
            "validated" if not validation.empty else "mechanical_smoke_unvalidated"
        ),
        "selection_source": (
            "validation_only" if not validation.empty else "fixed_candidate_no_validation"
        ),
        "refined_parameters": list(PLATE_REFINEMENT_KEYS),
        "all_other_parameters_frozen": True,
        "selected": selected is candidate,
        "base_validation_weighted_mae": base_metric,
        "candidate_validation_weighted_mae": candidate_metric,
        "optimizer": {
            "success": bool(result.success),
            "cost": float(result.cost),
        },
    }
    validate_physics_artifact(selected)
    return selected


def fit_physics_dynamic(base_artifact: dict, dynamic_frame: pd.DataFrame) -> dict:
    validate_physics_artifact(base_artifact)
    validate_identification_frame(dynamic_frame)
    train = dynamic_frame.loc[dynamic_frame["split"] == "train"].copy()
    validation = dynamic_frame.loc[dynamic_frame["split"] == "validation"].copy()
    _validate_dynamic_subset(train, "train", allow_empty=False)
    _validate_dynamic_subset(validation, "validation", allow_empty=True)

    constant, constant_result = _fit_dynamic_candidate(base_artifact, train, "constant")
    constant_metric = _dynamic_validation_metric(constant, validation)
    selected = constant
    selected_result = constant_result
    scheduled_metric = None
    improvement = None
    scheduled_status = "not_attempted"
    if not validation.empty:
        try:
            scheduled, scheduled_result = _fit_dynamic_candidate(
                base_artifact, train, "scheduled"
            )
            scheduled_metric = _dynamic_validation_metric(scheduled, validation)
        except (ValueError, RuntimeError, FloatingPointError):
            scheduled_status = "failed"
        else:
            scheduled_status = "valid"
            if constant_metric == 0.0:
                improvement = 0.0
            else:
                improvement = (constant_metric - scheduled_metric) / constant_metric
            if constant_metric > 0.0 and scheduled_metric <= 0.90 * constant_metric:
                selected = scheduled
                selected_result = scheduled_result

    selected.setdefault("fit", {})["dynamic_fit"] = {
        "fit_status": "validated" if not validation.empty else "mechanical_smoke_unvalidated",
        "selection_source": "validation_only" if not validation.empty else "fixed_constant_no_validation",
        "selected_time_constant_model": selected["dynamic"]["time_constant_model"],
        "scheduled_minimum_improvement": 0.10,
        "scheduled_improvement": improvement,
        "constant_validation_weighted_mae": constant_metric,
        "scheduled_validation_weighted_mae": scheduled_metric,
        "scheduled_candidate_status": scheduled_status,
        "evap_response_model": selected["dynamic"].get(
            "evap_response_model", "cascaded"
        ),
        "evap_input_delay_selection": "fixed_zero_for_direct_response",
        "tau_evap_selection": "fixed_plant_45_s",
        "q_cond_role": "independent_internal_lag_not_evaporator_driver",
        "battery_plate_conductance_model": selected["thermal"].get(
            "battery_plate_conductance_model",
            "legacy_pump_scaled",
        ),
        "excluded_observation_fields": ["q_cond_eff_w"],
        "training_horizons_steps": list(TRAINING_HORIZONS),
        "train_scenarios": int(train["scenario_id"].nunique()),
        "validation_scenarios": int(validation["scenario_id"].nunique()),
        "optimizer": {
            "success": bool(selected_result.success),
            "cost": float(selected_result.cost),
        },
    }
    validate_physics_artifact(selected)
    return selected


def fit_physics_artifact(frame: pd.DataFrame, dynamic_frame: pd.DataFrame | None = None) -> dict:
    validate_identification_frame(frame)
    train_frame = frame.loc[frame["split"] == "train"].copy()
    validation_frame = frame.loc[frame["split"] == "validation"].copy()
    artifact = fit_physics_predictor(train_frame, validation_frame)
    if dynamic_frame is not None:
        artifact = fit_physics_dynamic(artifact, dynamic_frame)
    return artifact


def _validate_output_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    model_data = (PROJECT_ROOT / "model_data").resolve()
    if resolved == model_data or model_data in resolved.parents:
        raise ValueError("Fitted artifacts must not be written directly into model_data")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the layered physics-P predictor")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--steady-csv")
    source.add_argument("--dynamic-base")
    source.add_argument("--short-response-base")
    source.add_argument("--plate-structure-base")
    source.add_argument("--plate-refinement-base")
    source.add_argument("--supply-refinement-base")
    parser.add_argument("--dynamic-csv")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    dynamic_frame = (
        pd.read_csv(arguments.dynamic_csv, encoding="utf-8")
        if arguments.dynamic_csv
        else None
    )
    if arguments.dynamic_base:
        if dynamic_frame is None:
            parser.error("--dynamic-base requires --dynamic-csv")
        base_artifact = load_physics_artifact(
            arguments.dynamic_base,
            require_validated=True,
        )
        artifact = fit_physics_dynamic(base_artifact, dynamic_frame)
    elif arguments.short_response_base:
        if dynamic_frame is None:
            parser.error("--short-response-base requires --dynamic-csv")
        base_artifact = load_physics_artifact(
            arguments.short_response_base,
            require_validated=True,
        )
        artifact = refine_physics_short_response(base_artifact, dynamic_frame)
    elif arguments.plate_structure_base:
        if dynamic_frame is None:
            parser.error("--plate-structure-base requires --dynamic-csv")
        base_artifact = load_physics_artifact(
            arguments.plate_structure_base,
            require_validated=True,
        )
        artifact = refine_physics_plate_structure(base_artifact, dynamic_frame)
    elif arguments.supply_refinement_base:
        if dynamic_frame is None:
            parser.error("--supply-refinement-base requires --dynamic-csv")
        base_artifact = load_physics_artifact(
            arguments.supply_refinement_base,
            require_validated=True,
        )
        artifact = refine_physics_supply(base_artifact, dynamic_frame)
    elif arguments.plate_refinement_base:
        if dynamic_frame is None:
            parser.error("--plate-refinement-base requires --dynamic-csv")
        base_artifact = load_physics_artifact(
            arguments.plate_refinement_base,
            require_validated=True,
        )
        artifact = refine_physics_plate(base_artifact, dynamic_frame)
    else:
        frame = pd.read_csv(arguments.steady_csv, encoding="utf-8")
        artifact = fit_physics_artifact(frame, dynamic_frame=dynamic_frame)
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
    if "dynamic_fit" in artifact["fit"]:
        print(
            "dynamic_model="
            f"{artifact['fit']['dynamic_fit']['selected_time_constant_model']}"
        )
    if "plate_refinement" in artifact["fit"]:
        print(
            "plate_refinement_selected="
            f"{artifact['fit']['plate_refinement']['selected']}"
        )
    if "plate_structure_refinement" in artifact["fit"]:
        print(
            "plate_structure_refinement_selected="
            f"{artifact['fit']['plate_structure_refinement']['selected']}"
        )
    if "supply_refinement" in artifact["fit"]:
        print(
            "supply_refinement_selected="
            f"{artifact['fit']['supply_refinement']['selected']}"
        )
    print(f"artifact={output_path}")


if __name__ == "__main__":
    main()
