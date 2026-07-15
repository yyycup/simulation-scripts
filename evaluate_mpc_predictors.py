"""Reusable error metrics for offline MPC predictor evaluation."""

import numpy as np


def error_metrics(actual, predicted):
    actual_values = np.asarray(actual, dtype=float).reshape(-1)
    predicted_values = np.asarray(predicted, dtype=float).reshape(-1)
    if actual_values.size != predicted_values.size:
        raise ValueError("actual and predicted must have the same length")
    if actual_values.size == 0:
        raise ValueError("actual and predicted must be non-empty")
    if not (np.isfinite(actual_values).all() and np.isfinite(predicted_values).all()):
        raise ValueError("actual and predicted must contain only finite values")

    error = predicted_values - actual_values
    absolute_error = np.abs(error)
    return {
        "mae": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mean_bias": float(np.mean(error)),
        "p95_abs": float(np.quantile(absolute_error, 0.95)),
    }


def capacity_metrics(frame):
    required_columns = {"n_comp_cmd_rpm", "q_evap_ss_w", "q_pred_w"}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"capacity frame is missing required columns: {missing_columns}")

    try:
        values = frame.loc[:, sorted(required_columns)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("capacity metric columns must contain finite numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError("capacity metric columns must contain only finite values")

    low_speed = frame[frame["n_comp_cmd_rpm"] < 2000.0]
    active = frame[frame["n_comp_cmd_rpm"] >= 2000.0]
    if low_speed.empty:
        raise ValueError("capacity metrics require a non-empty low-speed region (<2000 rpm)")
    if active.empty:
        raise ValueError("capacity metrics require a non-empty active region (>=2000 rpm)")

    result = {
        "low_speed_mae_w": error_metrics(
            low_speed["q_evap_ss_w"], low_speed["q_pred_w"]
        )["mae"]
    }
    active_actual = active["q_evap_ss_w"].astype(float)
    active_predicted = active["q_pred_w"].astype(float)
    relative_error = (active_predicted - active_actual).abs() / active_actual.abs().clip(
        lower=1e-9
    )
    result.update(error_metrics(active_actual, active_predicted))
    result["active_mape_percent"] = float(100.0 * relative_error.mean())
    result["active_max_relative_error_percent"] = float(100.0 * relative_error.max())
    return result
