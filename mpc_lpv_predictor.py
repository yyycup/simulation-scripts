import math
from pathlib import Path

import numpy as np

from mpc_predictor_selection import LPV_L, PredictorArtifactError, load_predictor_artifact


STATE_NAMES = (
    "t_batt_c", "t_cool_c", "t_plate_c", "t_supply_c", "t_return_c",
    "q_evap_eff_w", "q_cond_eff_w", "n_comp_eff_rpm", "n_pump_eff_rpm",
)
INPUT_NAMES = ("n_comp_cmd_rpm", "n_pump_cmd_rpm")
DISTURBANCE_NAMES = ("q_gen_w", "t_ambient_c")


class LpvArtifactError(PredictorArtifactError):
    pass


def augmented_state_names(order):
    if type(order) is not int or order not in (1, 2, 3):
        raise ValueError("order must be integer 1, 2, or 3")
    return tuple(f"{name}_lag{lag}" for lag in range(order) for name in STATE_NAMES)


def smooth_gate(n_comp_rpm, n_on_rpm, width_rpm):
    values = np.asarray([n_comp_rpm, n_on_rpm, width_rpm], dtype=float)
    if not np.isfinite(values).all() or values[2] <= 0.0:
        raise ValueError("gate inputs must be finite and width_rpm must be positive")
    return float(0.5 * (1.0 + np.tanh((values[0] - values[1]) / values[2])))


def _array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise LpvArtifactError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise LpvArtifactError(f"{name} must contain only finite values")
    return array


def _terms(use_pump_schedule):
    return ("M0", "M_comp", "M_temp", "M_pump") if use_pump_schedule else (
        "M0", "M_comp", "M_temp"
    )


def _validate_matrix_set(matrix_set, n_state, use_pump_schedule, name):
    if not isinstance(matrix_set, dict) or set(matrix_set) != {"A", "B", "E", "c"}:
        raise LpvArtifactError(f"{name} must contain exactly A, B, E, c")
    shapes = {"A": (n_state, n_state), "B": (n_state, 2), "E": (n_state, 2), "c": (n_state,)}
    required_terms = set(_terms(use_pump_schedule))
    result = {}
    for matrix_name, shape in shapes.items():
        value = matrix_set[matrix_name]
        if not isinstance(value, dict) or set(value) != required_terms:
            raise LpvArtifactError(
                f"{name}.{matrix_name} schedule terms must be {sorted(required_terms)}"
            )
        result[matrix_name] = {
            term: _array(value[term], shape, f"{name}.{matrix_name}.{term}")
            for term in _terms(use_pump_schedule)
        }
    return result


def validate_lpv_artifact(artifact):
    if not isinstance(artifact, dict):
        raise LpvArtifactError("LPV artifact must be a JSON object")
    if artifact.get("model_type") != LPV_L or type(artifact.get("schema_version")) is not int or artifact["schema_version"] != 1:
        raise LpvArtifactError("LPV artifact type/version mismatch")
    order = artifact.get("order")
    expected_names = augmented_state_names(order)
    n_state = 9 * order
    if artifact.get("base_state_count") != 9:
        raise LpvArtifactError("base_state_count must be 9")
    if artifact.get("augmented_state_count") != n_state:
        raise LpvArtifactError("augmented_state_count must equal 9 * order")
    if tuple(artifact.get("state_names", ())) != STATE_NAMES:
        raise LpvArtifactError("state_names do not match the canonical contract")
    if tuple(artifact.get("augmented_state_names", ())) != expected_names:
        raise LpvArtifactError("augmented_state_names do not match order")
    if tuple(artifact.get("input_names", ())) != INPUT_NAMES:
        raise LpvArtifactError("input_names do not match the canonical contract")
    if tuple(artifact.get("disturbance_names", ())) != DISTURBANCE_NAMES:
        raise LpvArtifactError("disturbance_names do not match the canonical contract")
    dt_s = artifact.get("dt_s")
    if isinstance(dt_s, bool) or not isinstance(dt_s, (int, float)) or not math.isfinite(dt_s) or dt_s <= 0:
        raise LpvArtifactError("dt_s must be a positive finite number")
    gate = artifact.get("gate")
    if not isinstance(gate, dict) or set(gate) != {"n_on_rpm", "width_rpm"}:
        raise LpvArtifactError("gate must contain n_on_rpm and width_rpm")
    smooth_gate(2000.0, gate["n_on_rpm"], gate["width_rpm"])
    use_pump = artifact.get("use_pump_schedule")
    if type(use_pump) is not bool:
        raise LpvArtifactError("use_pump_schedule must be boolean")
    normalization = artifact.get("normalization")
    expected_norm = {"n_comp_eff_rpm", "t_cool_c"} | ({"n_pump_eff_rpm"} if use_pump else set())
    if not isinstance(normalization, dict) or set(normalization) != expected_norm:
        raise LpvArtifactError(f"normalization keys must be {sorted(expected_norm)}")
    for name, item in normalization.items():
        if not isinstance(item, dict) or set(item) != {"center", "scale"}:
            raise LpvArtifactError(f"normalization.{name} must contain center and scale")
        values = np.asarray([item["center"], item["scale"]], dtype=float)
        if not np.isfinite(values).all() or values[1] <= 0.0:
            raise LpvArtifactError(f"normalization.{name} values are invalid")
    directions = artifact.get("directions")
    if not isinstance(directions, dict) or set(directions) != {"-1", "1"}:
        raise LpvArtifactError("directions must contain exactly -1 and 1")
    identity = np.eye(n_state)
    for direction, regimes in directions.items():
        if not isinstance(regimes, dict) or set(regimes) != {"low", "active"}:
            raise LpvArtifactError(f"directions.{direction} must contain low and active")
        for regime, local in regimes.items():
            if not isinstance(local, dict) or set(local) != {"discrete", "rate"}:
                raise LpvArtifactError(f"directions.{direction}.{regime} must contain discrete and rate")
            discrete = _validate_matrix_set(local["discrete"], n_state, use_pump, "discrete")
            rate = _validate_matrix_set(local["rate"], n_state, use_pump, "rate")
            for matrix_name in ("A", "B", "E", "c"):
                for term in _terms(use_pump):
                    expected = discrete[matrix_name][term] / dt_s
                    if matrix_name == "A" and term == "M0":
                        expected = (discrete[matrix_name][term] - identity) / dt_s
                    if not np.allclose(rate[matrix_name][term], expected, rtol=1e-10, atol=1e-12):
                        raise LpvArtifactError("rate matrices do not match discrete matrices")
    return artifact


def load_lpv_artifact(path: str | Path):
    try:
        artifact = load_predictor_artifact(path, expected_type=LPV_L)
    except PredictorArtifactError as exc:
        raise LpvArtifactError(str(exc)) from exc
    return validate_lpv_artifact(artifact)


def _coordinate(artifact, name, value):
    item = artifact["normalization"][name]
    coordinate = (float(value) - float(item["center"])) / float(item["scale"])
    return float(np.clip(coordinate, -1.0, 1.0))


def _scheduled_local(local, coordinates, use_pump_schedule, form):
    result = {}
    for name, terms in local[form].items():
        value = np.asarray(terms["M0"], dtype=float).copy()
        value += coordinates[0] * np.asarray(terms["M_comp"], dtype=float)
        value += coordinates[1] * np.asarray(terms["M_temp"], dtype=float)
        if use_pump_schedule:
            value += coordinates[2] * np.asarray(terms["M_pump"], dtype=float)
        result[name] = value
    return result


def scheduled_matrices(
    artifact, flow_direction, n_comp_eff_rpm, t_cool_c, n_pump_eff_rpm=None, form="discrete"
):
    validate_lpv_artifact(artifact)
    if form not in {"discrete", "rate"}:
        raise ValueError("form must be discrete or rate")
    direction = str(int(flow_direction))
    if direction not in {"-1", "1"} or float(flow_direction) not in {-1.0, 1.0}:
        raise ValueError("flow_direction must be -1 or 1")
    coordinates = [
        _coordinate(artifact, "n_comp_eff_rpm", n_comp_eff_rpm),
        _coordinate(artifact, "t_cool_c", t_cool_c),
    ]
    if artifact["use_pump_schedule"]:
        if n_pump_eff_rpm is None:
            raise ValueError("n_pump_eff_rpm is required for pump scheduling")
        coordinates.append(_coordinate(artifact, "n_pump_eff_rpm", n_pump_eff_rpm))
    regimes = artifact["directions"][direction]
    low = _scheduled_local(regimes["low"], coordinates, artifact["use_pump_schedule"], form)
    active = _scheduled_local(regimes["active"], coordinates, artifact["use_pump_schedule"], form)
    gate = smooth_gate(n_comp_eff_rpm, artifact["gate"]["n_on_rpm"], artifact["gate"]["width_rpm"])
    return {name: (1.0 - gate) * low[name] + gate * active[name] for name in ("A", "B", "E", "c")}


def spectral_radius_grid(artifact):
    validate_lpv_artifact(artifact)
    records = []
    comp_norm = artifact["normalization"]["n_comp_eff_rpm"]
    temp_norm = artifact["normalization"]["t_cool_c"]
    pump_norm = artifact["normalization"].get("n_pump_eff_rpm", {"center": 0.0, "scale": 1.0})
    pump_coordinates = (-1.0, 0.0, 1.0) if artifact["use_pump_schedule"] else (0.0,)
    for direction in (-1, 1):
        for rho_comp in (-1.0, 0.0, 1.0):
            for rho_temp in (-1.0, 0.0, 1.0):
                for rho_pump in pump_coordinates:
                    n_comp = comp_norm["center"] + rho_comp * comp_norm["scale"]
                    t_cool = temp_norm["center"] + rho_temp * temp_norm["scale"]
                    n_pump = pump_norm["center"] + rho_pump * pump_norm["scale"]
                    matrices = scheduled_matrices(artifact, direction, n_comp, t_cool, n_pump)
                    radius = float(np.max(np.abs(np.linalg.eigvals(matrices["A"]))))
                    records.append({
                        "flow_direction": direction, "rho_comp": rho_comp,
                        "rho_temp": rho_temp, "rho_pump": rho_pump,
                        "spectral_radius": radius,
                    })
    return records


class LpvPredictor:
    def __init__(self, artifact):
        self.artifact = validate_lpv_artifact(artifact)
        self._state = None

    def reset(self, initial_state):
        if not isinstance(initial_state, dict) or set(initial_state) != set(STATE_NAMES):
            raise ValueError("initial_state must contain exactly STATE_NAMES")
        base = np.asarray([initial_state[name] for name in STATE_NAMES], dtype=float)
        if not np.isfinite(base).all():
            raise ValueError("initial_state must be finite")
        self._state = np.tile(base, self.artifact["order"])
        return {name: float(base[index]) for index, name in enumerate(STATE_NAMES)}

    def step(self, inputs):
        if self._state is None:
            raise RuntimeError("reset() must be called before step()")
        expected = set(INPUT_NAMES) | set(DISTURBANCE_NAMES) | {"flow_direction"}
        if not isinstance(inputs, dict) or set(inputs) != expected:
            raise ValueError("inputs must contain inputs, disturbances, and flow_direction")
        values = np.asarray([inputs[name] for name in (*INPUT_NAMES, *DISTURBANCE_NAMES)], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("inputs must be finite")
        current = self._state[:9]
        matrices = scheduled_matrices(
            self.artifact, inputs["flow_direction"], current[7], current[1], current[8]
        )
        next_state = (
            matrices["A"] @ self._state
            + matrices["B"] @ values[:2]
            + matrices["E"] @ values[2:]
            + matrices["c"]
        )
        if next_state.shape != self._state.shape or not np.isfinite(next_state).all():
            raise FloatingPointError("LPV step produced an invalid state")
        self._state = next_state
        return {name: float(next_state[index]) for index, name in enumerate(STATE_NAMES)}
