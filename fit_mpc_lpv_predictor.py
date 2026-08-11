import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mpc_lpv_predictor import (
    DISTURBANCE_NAMES,
    INPUT_NAMES,
    STATE_NAMES,
    augmented_state_names,
    spectral_radius_grid,
    validate_lpv_artifact,
)
from mpc_predictor_selection import LPV_L
from predictor_identification_data import validate_identification_frame


ORDERS = (1, 2, 3)
RIDGES = (1e-6, 1e-4, 1e-2, 1.0, 100.0)
REGIMES = ("low", "active")
DIRECTIONS = (-1, 1)
GATE = {"n_on_rpm": 1950.0, "width_rpm": 25.0}
STABILITY_LIMIT = 0.995
TEMPERATURE_WEIGHTS = np.asarray([3.0, 2.0, 1.0, 1.0, 1.0])


def _finite_numeric(frame, columns, name):
    try:
        values = frame.loc[:, columns].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} columns must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{name} columns must be finite")
    return values


def _sample_dt_s(train):
    declared = _finite_numeric(train, ["dt_s"], "dt_s").ravel() if "dt_s" in train else np.array([])
    differences = []
    for _, scenario in train.groupby("scenario_id", sort=True):
        times = np.sort(_finite_numeric(scenario, ["time_s"], "time_s").ravel())
        differences.extend(np.diff(times).tolist())
    if not differences or min(differences) <= 0.0 or not np.allclose(differences, differences[0]):
        raise ValueError("training scenarios must have one positive sampling interval")
    if declared.size:
        if not np.allclose(declared, declared[0]) or declared[0] <= 0.0:
            raise ValueError("training dt_s values must be one positive sampling interval")
        if not np.isclose(declared[0], differences[0]):
            raise ValueError("training dt_s does not match scenario time_s intervals")
        return float(declared[0])
    return float(differences[0])


def _normalization(train, use_pump_schedule):
    result = {}
    for name in ("n_comp_eff_rpm", "t_cool_c") + (("n_pump_eff_rpm",) if use_pump_schedule else ()):
        values = _finite_numeric(train, [name], name).ravel()
        lower, upper = float(np.min(values)), float(np.max(values))
        center = 0.5 * (lower + upper)
        scale = max(0.5 * (upper - lower), 1.0)
        result[name] = {"center": center, "scale": scale}
    return result


def _rho(values, normalization, name):
    item = normalization[name]
    return np.clip((values - item["center"]) / item["scale"], -1.0, 1.0)


def build_grouped_lagged_samples(frame, order, normalization, use_pump_schedule=False):
    """Build one-step ARX samples, independently within each scenario."""
    columns = (*STATE_NAMES, *INPUT_NAMES, *DISTURBANCE_NAMES, "time_s", "flow_direction")
    _finite_numeric(frame, columns, "dynamic")
    samples = []
    for scenario_id, scenario in frame.groupby("scenario_id", sort=True):
        scenario = scenario.sort_values("time_s", kind="mergesort").reset_index(drop=True)
        states = scenario.loc[:, STATE_NAMES].to_numpy(dtype=float)
        for index in range(order - 1, len(scenario) - 1):
            augmented = np.concatenate([states[index - lag] for lag in range(order)])
            row = scenario.iloc[index]
            rho = [
                float(_rho(np.asarray([row["n_comp_eff_rpm"]]), normalization, "n_comp_eff_rpm")[0]),
                float(_rho(np.asarray([row["t_cool_c"]]), normalization, "t_cool_c")[0]),
            ]
            if use_pump_schedule:
                rho.append(float(_rho(
                    np.asarray([row["n_pump_eff_rpm"]]), normalization, "n_pump_eff_rpm"
                )[0]))
            samples.append({
                "scenario_id": str(scenario_id),
                "flow_direction": int(row["flow_direction"]),
                "regime": "low" if float(row["n_comp_eff_rpm"]) < GATE["n_on_rpm"] else "active",
                "z": np.concatenate((
                    augmented,
                    row.loc[list(INPUT_NAMES)].to_numpy(dtype=float),
                    row.loc[list(DISTURBANCE_NAMES)].to_numpy(dtype=float),
                    [1.0],
                )),
                "rho": np.asarray(rho),
                "target": states[index + 1].copy(),
            })
    return samples


def _select_local_samples(samples, direction, regime):
    exact = [sample for sample in samples if sample["flow_direction"] == direction and sample["regime"] == regime]
    if exact:
        return exact, "direction_regime"
    same_direction = [sample for sample in samples if sample["flow_direction"] == direction]
    if same_direction:
        return same_direction, "direction_all_regimes"
    return list(samples), "global_train"


def _solve_local(samples, order, ridge, use_pump_schedule):
    z = np.vstack([sample["z"] for sample in samples])
    rho = np.vstack([sample["rho"] for sample in samples])
    phi = np.hstack([z] + [z * rho[:, [index]] for index in range(rho.shape[1])])
    targets = np.vstack([sample["target"] for sample in samples])
    column_scales = np.maximum(np.sqrt(np.mean(phi * phi, axis=0)), 1.0)
    phi_scaled = phi / column_scales
    penalty = np.eye(phi.shape[1])
    penalty[z.shape[1] - 1, z.shape[1] - 1] = 0.0
    theta_scaled = np.linalg.solve(
        phi_scaled.T @ phi_scaled + ridge * penalty,
        phi_scaled.T @ targets,
    )
    theta = theta_scaled / column_scales[:, None]
    if not np.isfinite(theta).all():
        raise FloatingPointError("ridge solve produced non-finite coefficients")
    n_state = 9 * order
    names = ("M0", "M_comp", "M_temp", "M_pump") if use_pump_schedule else (
        "M0", "M_comp", "M_temp"
    )
    discrete = {name: {} for name in ("A", "B", "E", "c")}
    block_width = z.shape[1]
    for term_index, term in enumerate(names):
        block = theta[term_index * block_width:(term_index + 1) * block_width].T
        a = np.zeros((n_state, n_state))
        b = np.zeros((n_state, 2))
        e = np.zeros((n_state, 2))
        c = np.zeros(n_state)
        a[:9, :] = block[:, :n_state]
        b[:9, :] = block[:, n_state:n_state + 2]
        e[:9, :] = block[:, n_state + 2:n_state + 4]
        c[:9] = block[:, -1]
        if term == "M0" and order > 1:
            a[9:, :-9] = np.eye(n_state - 9)
        discrete["A"][term] = a
        discrete["B"][term] = b
        discrete["E"][term] = e
        discrete["c"][term] = c
    return discrete, column_scales.tolist()


def _rate_matrices(discrete, dt_s):
    n_state = np.asarray(discrete["A"]["M0"]).shape[0]
    result = {name: {} for name in ("A", "B", "E", "c")}
    for matrix_name, terms in discrete.items():
        for term, value in terms.items():
            array = np.asarray(value, dtype=float)
            if matrix_name == "A" and term == "M0":
                array = array - np.eye(n_state)
            result[matrix_name][term] = (array / dt_s).tolist()
    return result


def _json_discrete(discrete):
    return {
        matrix_name: {term: value.tolist() for term, value in terms.items()}
        for matrix_name, terms in discrete.items()
    }


def _candidate_artifact(train, order, ridge, use_pump_schedule, dt_s):
    normalization = _normalization(train, use_pump_schedule)
    samples = build_grouped_lagged_samples(train, order, normalization, use_pump_schedule)
    if not samples:
        raise ValueError("training data do not contain enough grouped lag samples")
    directions = {}
    sources = {}
    for direction in DIRECTIONS:
        directions[str(direction)] = {}
        sources[str(direction)] = {}
        for regime in REGIMES:
            local_samples, source = _select_local_samples(samples, direction, regime)
            discrete, feature_scales = _solve_local(
                local_samples, order, ridge, use_pump_schedule
            )
            directions[str(direction)][regime] = {
                "discrete": _json_discrete(discrete),
                "rate": _rate_matrices(discrete, dt_s),
            }
            sources[str(direction)][regime] = {
                "sample_source": source,
                "sample_count": len(local_samples),
                "independently_fitted": source == "direction_regime",
                "feature_scaling": "train_local_rms_min_1",
                "feature_rms_scales": feature_scales,
            }
    return {
        "model_type": LPV_L,
        "schema_version": 1,
        "dt_s": dt_s,
        "order": order,
        "base_state_count": 9,
        "augmented_state_count": 9 * order,
        "state_names": list(STATE_NAMES),
        "augmented_state_names": list(augmented_state_names(order)),
        "input_names": list(INPUT_NAMES),
        "disturbance_names": list(DISTURBANCE_NAMES),
        "gate": dict(GATE),
        "normalization": normalization,
        "use_pump_schedule": use_pump_schedule,
        "directions": directions,
        "fit": {"sample_sources": sources},
    }


def _rollout_temperature_mae(artifact, validation):
    if validation.empty:
        return float("inf")
    from mpc_lpv_predictor import scheduled_matrices

    order = artifact["order"]
    errors = []
    for _, scenario in validation.groupby("scenario_id", sort=True):
        scenario = scenario.sort_values("time_s", kind="mergesort").reset_index(drop=True)
        if len(scenario) <= order:
            continue
        states = scenario.loc[:, STATE_NAMES].to_numpy(dtype=float)
        current_index = order - 1
        augmented = np.concatenate([states[current_index - lag] for lag in range(order)])
        for index in range(current_index, len(scenario) - 1):
            row = scenario.iloc[index]
            matrices = scheduled_matrices(
                artifact, row["flow_direction"], augmented[7], augmented[1], augmented[8]
            )
            inputs = row.loc[list(INPUT_NAMES)].to_numpy(dtype=float)
            disturbances = row.loc[list(DISTURBANCE_NAMES)].to_numpy(dtype=float)
            augmented = (
                matrices["A"] @ augmented + matrices["B"] @ inputs
                + matrices["E"] @ disturbances + matrices["c"]
            )
            if not np.isfinite(augmented).all():
                return float("inf")
            errors.append(float(np.average(
                np.abs(augmented[:5] - states[index + 1, :5]), weights=TEMPERATURE_WEIGHTS
            )))
    return float(np.mean(errors)) if errors else float("inf")


def pump_schedule_improves(base_mae, pump_mae):
    return bool(np.isfinite(base_mae) and np.isfinite(pump_mae) and pump_mae <= 0.95 * base_mae)


def _stable(artifact):
    return all(record["spectral_radius"] < STABILITY_LIMIT for record in spectral_radius_grid(artifact))


def _best_candidate(train, validation, dt_s, use_pump_schedule):
    accepted = []
    rejected_unstable = 0
    failed = 0
    for order in ORDERS:
        for ridge in RIDGES:
            try:
                artifact = _candidate_artifact(train, order, ridge, use_pump_schedule, dt_s)
                validate_lpv_artifact(artifact)
                if not _stable(artifact):
                    rejected_unstable += 1
                    continue
                metric = _rollout_temperature_mae(artifact, validation)
                if not np.isfinite(metric):
                    metric = _rollout_temperature_mae(artifact, train)
                accepted.append((metric, order, ridge, artifact))
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                failed += 1
    if not accepted:
        raise RuntimeError("all LPV candidates failed or violated the stability margin")
    accepted.sort(key=lambda item: (item[0], item[1], item[2]))
    metric, order, ridge, artifact = accepted[0]
    return artifact, metric, {
        "accepted_candidates": len(accepted),
        "rejected_unstable_candidates": rejected_unstable,
        "failed_candidates": failed,
        "selected_order": order,
        "selected_ridge": ridge,
    }


def fit_lpv_artifact(frame):
    validate_identification_frame(frame)
    train = frame.loc[frame["split"] == "train"].copy()
    validation = frame.loc[frame["split"] == "validation"].copy()
    if train.empty:
        raise ValueError("LPV identification requires training rows")
    _finite_numeric(train, (*STATE_NAMES, *INPUT_NAMES, *DISTURBANCE_NAMES, "time_s", "flow_direction"), "train")
    if not validation.empty:
        _finite_numeric(validation, (*STATE_NAMES, *INPUT_NAMES, *DISTURBANCE_NAMES, "time_s", "flow_direction"), "validation")
    fitted_directions = set(train["flow_direction"].astype(float))
    validation_directions = set(validation["flow_direction"].astype(float))
    if not fitted_directions <= {-1.0, 1.0} or not validation_directions <= {-1.0, 1.0}:
        raise ValueError("flow_direction must contain only -1 or 1")
    dt_s = _sample_dt_s(train)
    base, base_mae, base_search = _best_candidate(train, validation, dt_s, False)
    pump, pump_mae, pump_search = _best_candidate(train, validation, dt_s, True)
    choose_pump = not validation.empty and pump_schedule_improves(base_mae, pump_mae)
    selected = pump if choose_pump else base
    validation_by_direction = {
        str(direction): {
            "scenario_count": int(
                validation.loc[validation["flow_direction"] == direction, "scenario_id"].nunique()
            ),
            "validated": bool(
                validation.loc[validation["flow_direction"] == direction, "scenario_id"].nunique()
            ),
        }
        for direction in DIRECTIONS
    }
    if validation.empty:
        fit_status = "train_only_unvalidated"
    elif all(item["validated"] for item in validation_by_direction.values()):
        fit_status = "validated"
    else:
        fit_status = "partially_validated"
    selected["fit"].update({
        "fit_status": fit_status,
        "selection_method": "validation_multistep_weighted_temperature_mae" if not validation.empty else "training_multistep_weighted_temperature_mae",
        "validation_scenario_count": int(validation["scenario_id"].nunique()),
        "validation_by_direction": validation_by_direction,
        "base_validation_mae_c": base_mae if not validation.empty else None,
        "pump_validation_mae_c": pump_mae if not validation.empty else None,
        "pump_improvement_fraction": (
            (base_mae - pump_mae) / base_mae if not validation.empty and base_mae > 0.0 else None
        ),
        "pump_schedule_selected": choose_pump,
        "base_search": base_search,
        "pump_search": pump_search,
        "stability_limit": STABILITY_LIMIT,
        "max_spectral_radius": max(
            record["spectral_radius"] for record in spectral_radius_grid(selected)
        ),
    })
    validate_lpv_artifact(selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description="Fit the provisional LPV/ARX model L")
    parser.add_argument("--dynamic-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.dynamic_csv)
    artifact = fit_lpv_artifact(frame)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "order": artifact["order"],
        "ridge": artifact["fit"]["base_search"]["selected_ridge"] if not artifact["use_pump_schedule"] else artifact["fit"]["pump_search"]["selected_ridge"],
        "pump_schedule_selected": artifact["use_pump_schedule"],
        "max_spectral_radius": artifact["fit"]["max_spectral_radius"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
