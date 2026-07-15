import math
from numbers import Real
from pathlib import Path

import numpy as np

from mpc_predictor_selection import (
    PHYSICS_P,
    PredictorArtifactError,
    load_predictor_artifact,
)


DEFAULT_PHYSICS_ARTIFACT = {
    "model_type": PHYSICS_P,
    "schema_version": 1,
    "gate": {"n_on_rpm": 1950.0, "width_rpm": 25.0},
    "capacity": {
        "coefficients": [0.6, -0.1, 0.015, -0.002, 0.0, 0.0],
        "n_pump_ref_rpm": 2000.0,
        "q_upper_w": 4800.0,
    },
}


class PhysicsArtifactError(PredictorArtifactError):
    pass


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def smooth_gate(n_comp_rpm: object, n_on_rpm: object, width_rpm: object) -> float:
    n_comp = _finite_number(n_comp_rpm, "n_comp_rpm")
    n_on = _finite_number(n_on_rpm, "n_on_rpm")
    width = _finite_number(width_rpm, "width_rpm")
    if width <= 0.0:
        raise ValueError("width_rpm must be greater than zero")
    result = 0.5 * (1.0 + np.tanh((n_comp - n_on) / width))
    return _finite_number(result, "smooth_gate result")


def active_capacity_w(
    coefficients: object,
    n_comp_rpm: object,
    n_pump_rpm: object,
    t_cool_c: object,
    t_ambient_c: object,
    n_pump_ref_rpm: object,
) -> float:
    if not isinstance(coefficients, (list, tuple, np.ndarray)) or len(coefficients) != 6:
        raise ValueError("coefficients must contain exactly 6 values")
    c0, c1, c2, c3, c4, c5 = [
        _finite_number(value, f"coefficients[{index}]")
        for index, value in enumerate(coefficients)
    ]
    n_comp = _finite_number(n_comp_rpm, "n_comp_rpm")
    n_pump = _finite_number(n_pump_rpm, "n_pump_rpm")
    t_cool = _finite_number(t_cool_c, "t_cool_c")
    t_ambient = _finite_number(t_ambient_c, "t_ambient_c")
    n_pump_ref = _finite_number(n_pump_ref_rpm, "n_pump_ref_rpm")
    if n_pump <= 0.0:
        raise ValueError("n_pump_rpm must be greater than zero")
    if n_pump_ref <= 0.0:
        raise ValueError("n_pump_ref_rpm must be greater than zero")

    n_comp_norm = (n_comp - 4000.0) / 2000.0
    t_cool_norm = (t_cool - 27.5) / 7.5
    t_ambient_norm = (t_ambient - 30.0) / 10.0
    gain = (
        c0
        + c1 * n_pump_ref / max(n_pump, 1e-6)
        + c2 * t_cool_norm
        + c3 * t_ambient_norm
        + c4 * t_cool_norm * t_ambient_norm
        + c5 * n_comp_norm
    )
    return _finite_number(n_comp * gain, "active_capacity_w result")


def validate_physics_artifact(artifact: object) -> dict:
    if not isinstance(artifact, dict):
        raise PhysicsArtifactError("Physics artifact must be a JSON object")
    if artifact.get("model_type") != PHYSICS_P:
        raise PhysicsArtifactError("Physics artifact model_type must be physics_p")
    if type(artifact.get("schema_version")) is not int or artifact["schema_version"] != 1:
        raise PhysicsArtifactError("Physics artifact schema_version must be integer 1")

    gate = artifact.get("gate")
    if not isinstance(gate, dict):
        raise PhysicsArtifactError("Physics artifact gate must be an object")
    capacity = artifact.get("capacity")
    if not isinstance(capacity, dict):
        raise PhysicsArtifactError("Physics artifact capacity must be an object")
    try:
        n_on = _finite_number(gate.get("n_on_rpm"), "gate.n_on_rpm")
        width = _finite_number(gate.get("width_rpm"), "gate.width_rpm")
        if not 1900.0 <= n_on <= 2000.0:
            raise ValueError("gate.n_on_rpm must be within [1900, 2000]")
        if not 10.0 <= width <= 80.0:
            raise ValueError("gate.width_rpm must be within [10, 80]")

        coefficients = capacity.get("coefficients")
        if not isinstance(coefficients, list) or len(coefficients) != 6:
            raise ValueError("capacity.coefficients must be a list of exactly 6 values")
        for index, coefficient in enumerate(coefficients):
            _finite_number(coefficient, f"capacity.coefficients[{index}]")
        n_pump_ref = _finite_number(
            capacity.get("n_pump_ref_rpm"), "capacity.n_pump_ref_rpm"
        )
        q_upper = _finite_number(capacity.get("q_upper_w"), "capacity.q_upper_w")
        if n_pump_ref <= 0.0:
            raise ValueError("capacity.n_pump_ref_rpm must be greater than zero")
        if q_upper <= 0.0:
            raise ValueError("capacity.q_upper_w must be greater than zero")
    except (TypeError, ValueError) as exc:
        raise PhysicsArtifactError(str(exc)) from exc
    return artifact


def load_physics_artifact(path: str | Path) -> dict:
    try:
        artifact = load_predictor_artifact(path, expected_type=PHYSICS_P)
    except PredictorArtifactError as exc:
        raise PhysicsArtifactError(str(exc)) from exc
    return validate_physics_artifact(artifact)


def evaluate_physics_capacity(
    n_comp_rpm: object,
    n_pump_rpm: object,
    t_cool_c: object,
    t_ambient_c: object,
    artifact: object = None,
) -> float:
    selected = DEFAULT_PHYSICS_ARTIFACT if artifact is None else artifact
    validate_physics_artifact(selected)
    gate = selected["gate"]
    capacity = selected["capacity"]
    gate_value = smooth_gate(
        n_comp_rpm,
        gate["n_on_rpm"],
        gate["width_rpm"],
    )
    active = active_capacity_w(
        capacity["coefficients"],
        n_comp_rpm,
        n_pump_rpm,
        t_cool_c,
        t_ambient_c,
        capacity["n_pump_ref_rpm"],
    )
    raw = _finite_number(gate_value * active, "physics capacity result")
    q_upper = _finite_number(capacity["q_upper_w"], "capacity.q_upper_w")
    return max(0.0, min(q_upper, raw))
