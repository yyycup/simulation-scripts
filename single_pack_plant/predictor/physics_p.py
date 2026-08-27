"""14 状态离散 Physics-P 预测器的权威 Python 实现。

``initialize_physics_state()`` 初始化预测状态，``step_physics_predictor()``
完成 k 到 k+1 的单步预测。CasADi 优化版本在 ``physics_p_casadi.py``。
"""

import math
from collections import deque
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np
from ..simulation.config import N_COMP_OFF_RPM

from .selection import (
    PHYSICS_P,
    PredictorArtifactError,
    load_predictor_artifact,
)


DEFAULT_INPUT_DOMAIN = {
    "n_comp_rpm": [1000.0, 6000.0],
    "n_pump_rpm": [1600.0, 4800.0],
    "t_cool_c": [15.0, 35.0],
    "t_ambient_c": [20.0, 40.0],
}

DEFAULT_DYNAMIC_PARAMETERS = {
    "time_constant_model": "constant",
    "tau_comp_s": 5.0,
    "tau_pump_s": 5.0,
    "evap_input_delay_s": 20.0,
    "pump_flow_delay_s": 5.0,
    "tau_cond_s": 75.0,
    "tau_evap_s": 45.0,
    "supply_delay_s": 15.0,
    "return_delay_s": 20.0,
}

# Known coolant property; keep synchronized with thermal_system.cp_cool.
PHYSICAL_COOLANT_CP_J_KG_K = 3391.0

DEFAULT_THERMAL_PARAMETERS = {
    "coolant_cp_j_kg_k": PHYSICAL_COOLANT_CP_J_KG_K,
    "coolant_mass_flow_ref_kg_s": 0.25,
    "n_pump_ref_rpm": 2000.0,
    "battery_heat_capacity_j_k": 246844.0,
    "coolant_heat_capacity_j_k": 100000.0,
    "battery_plate_conductance_w_k": 520.0,
    "plate_tau_s": 20.0,
    "ambient_conductance_w_k": 3.0,
    "plate_fluid_effectiveness": 1.0,
    "battery_heat_generation_scale": 1.0,
}

SCHEDULE_PARAMETER_NAMES = {
    "tau_comp_schedule_s",
    "tau_pump_schedule_s",
    "tau_cond_schedule_s",
    "tau_evap_schedule_s",
}
EVAP_RESPONSE_MODELS = {"cascaded", "direct"}
BATTERY_PLATE_CONDUCTANCE_MODELS = {
    "legacy_pump_scaled",
    "constant_physical",
}

LEGACY_CAPACITY_MODEL = "legacy_six_term"
ENHANCED_CAPACITY_MODEL = "single_enhanced_v1"
CUBIC_CAPACITY_MODEL = "single_cubic_v1"
ENHANCED_CAPACITY_FEATURE_NAMES = (
    "base",
    "pump_ratio",
    "coolant_temperature",
    "ambient_temperature",
    "compressor_speed",
    "compressor_x_coolant",
    "compressor_x_ambient",
    "pump_x_coolant",
    "compressor_squared",
    "coolant_x_ambient",
    "coolant_squared",
    "ambient_squared",
    "pump_x_ambient",
    "pump_x_compressor",
)
CUBIC_CAPACITY_FEATURE_NAMES = (
    "base",
    "compressor_speed",
    "pump_ratio",
    "coolant_temperature",
    "ambient_temperature",
    "compressor_squared",
    "compressor_x_pump",
    "compressor_x_coolant",
    "compressor_x_ambient",
    "pump_squared",
    "pump_x_coolant",
    "pump_x_ambient",
    "coolant_squared",
    "coolant_x_ambient",
    "compressor_squared_x_pump",
    "compressor_x_pump_squared",
    "compressor_x_pump_x_coolant",
    "compressor_x_pump_x_ambient",
    "compressor_x_coolant_x_ambient",
    "coolant_cubed",
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
    "input_domain": DEFAULT_INPUT_DOMAIN,
    "dynamic": DEFAULT_DYNAMIC_PARAMETERS,
    "thermal": DEFAULT_THERMAL_PARAMETERS,
}


@dataclass(frozen=True)
class PhysicsPredictorState:
    n_comp_eff_rpm: float
    n_pump_eff_rpm: float
    evap_speed_history_rpm: tuple[float, ...]
    pump_speed_history_rpm: tuple[float, ...]
    q_cond_w: float
    q_evap_w: float
    supply_history_c: tuple[float, ...]
    t_supply_c: float
    t_plate_c: float
    return_history_c: tuple[float, ...]
    t_return_c: float
    t_batt_c: float
    t_cool_c: float


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


def enhanced_active_capacity_w(
    coefficients: object,
    n_comp_rpm: object,
    n_pump_rpm: object,
    t_cool_c: object,
    t_ambient_c: object,
    n_pump_ref_rpm: object,
) -> float:
    if (
        not isinstance(coefficients, (list, tuple, np.ndarray))
        or len(coefficients) != len(ENHANCED_CAPACITY_FEATURE_NAMES)
    ):
        raise ValueError(
            "enhanced coefficients must contain exactly "
            f"{len(ENHANCED_CAPACITY_FEATURE_NAMES)} values"
        )
    values = [
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

    compressor = (n_comp - 4000.0) / 2000.0
    pump_ratio = n_pump_ref / n_pump
    coolant = (t_cool - 27.5) / 7.5
    ambient = (t_ambient - 30.0) / 10.0
    features = (
        1.0,
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
    )
    gain = sum(
        coefficient * feature for coefficient, feature in zip(values, features)
    )
    return _finite_number(n_comp * gain, "enhanced_active_capacity_w result")


def cubic_active_capacity_w(
    coefficients: object,
    n_comp_rpm: object,
    n_pump_rpm: object,
    t_cool_c: object,
    t_ambient_c: object,
    n_pump_ref_rpm: object,
) -> float:
    if (
        not isinstance(coefficients, (list, tuple, np.ndarray))
        or len(coefficients) != len(CUBIC_CAPACITY_FEATURE_NAMES)
    ):
        raise ValueError(
            "cubic coefficients must contain exactly "
            f"{len(CUBIC_CAPACITY_FEATURE_NAMES)} values"
        )
    values = [
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

    compressor = (n_comp - 4000.0) / 2000.0
    pump_ratio = n_pump_ref / n_pump
    coolant = (t_cool - 27.5) / 7.5
    ambient = (t_ambient - 30.0) / 10.0
    features = (
        1.0,
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
    )
    gain = sum(coefficient * feature for coefficient, feature in zip(values, features))
    return _finite_number(n_comp * gain, "cubic_active_capacity_w result")


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
        capacity_model = capacity.get("model", LEGACY_CAPACITY_MODEL)
        if capacity_model not in {
            LEGACY_CAPACITY_MODEL,
            ENHANCED_CAPACITY_MODEL,
            CUBIC_CAPACITY_MODEL,
        }:
            raise ValueError(f"Unsupported capacity.model: {capacity_model!r}")
        if capacity_model == LEGACY_CAPACITY_MODEL:
            if not 1900.0 <= n_on <= 2000.0:
                raise ValueError("legacy gate.n_on_rpm must be within [1900, 2000]")
        elif not 1000.0 <= n_on <= 2000.0:
            raise ValueError("enhanced gate.n_on_rpm must be within [1000, 2000]")
        if not 10.0 <= width <= 80.0:
            raise ValueError("gate.width_rpm must be within [10, 80]")
        coefficients = capacity.get("coefficients")
        expected_coefficient_count = {
            LEGACY_CAPACITY_MODEL: 6,
            ENHANCED_CAPACITY_MODEL: len(ENHANCED_CAPACITY_FEATURE_NAMES),
            CUBIC_CAPACITY_MODEL: len(CUBIC_CAPACITY_FEATURE_NAMES),
        }[capacity_model]
        if (
            not isinstance(coefficients, list)
            or len(coefficients) != expected_coefficient_count
        ):
            raise ValueError(
                "capacity.coefficients must be a list of exactly "
                f"{expected_coefficient_count} values"
            )
        for index, coefficient in enumerate(coefficients):
            _finite_number(coefficient, f"capacity.coefficients[{index}]")
        gate_mode = gate.get("mode", "smooth")
        if gate_mode not in {"smooth", "hard"}:
            raise ValueError("gate.mode must be smooth or hard")
        if capacity_model in {ENHANCED_CAPACITY_MODEL, CUBIC_CAPACITY_MODEL}:
            if gate_mode != "hard":
                raise ValueError(f"{capacity_model} requires gate.mode == 'hard'")
            feature_names = capacity.get("feature_names")
            expected_feature_names = (
                ENHANCED_CAPACITY_FEATURE_NAMES
                if capacity_model == ENHANCED_CAPACITY_MODEL
                else CUBIC_CAPACITY_FEATURE_NAMES
            )
            if feature_names != list(expected_feature_names):
                raise ValueError(
                    "capacity.feature_names must match the declared physical terms"
                )
            minimum_active = _finite_number(
                capacity.get("minimum_active_rpm"),
                "capacity.minimum_active_rpm",
            )
            if not 1000.0 <= minimum_active <= 2000.0:
                raise ValueError(
                    "capacity.minimum_active_rpm must be within [1000, 2000]"
                )
            if minimum_active != n_on:
                raise ValueError(
                    "capacity.minimum_active_rpm must equal gate.n_on_rpm"
                )
        n_pump_ref = _finite_number(
            capacity.get("n_pump_ref_rpm"), "capacity.n_pump_ref_rpm"
        )
        q_upper = _finite_number(capacity.get("q_upper_w"), "capacity.q_upper_w")
        if n_pump_ref <= 0.0:
            raise ValueError("capacity.n_pump_ref_rpm must be greater than zero")
        if q_upper <= 0.0:
            raise ValueError("capacity.q_upper_w must be greater than zero")

        input_domain = artifact.get("input_domain", DEFAULT_INPUT_DOMAIN)
        if not isinstance(input_domain, dict):
            raise ValueError("input_domain must be an object")
        if set(input_domain) != set(DEFAULT_INPUT_DOMAIN):
            raise ValueError(
                "input_domain must contain exactly n_comp_rpm, n_pump_rpm, "
                "t_cool_c, and t_ambient_c"
            )
        for name, limits in input_domain.items():
            if not isinstance(limits, (list, tuple)) or len(limits) != 2:
                raise ValueError(f"input_domain.{name} must be [min, max]")
            lower = _finite_number(limits[0], f"input_domain.{name}[0]")
            upper = _finite_number(limits[1], f"input_domain.{name}[1]")
            if lower >= upper:
                raise ValueError(f"input_domain.{name} must satisfy min < max")

        dynamic = artifact.get("dynamic")
        thermal = artifact.get("thermal")
        if (dynamic is None) != (thermal is None):
            raise ValueError("dynamic and thermal must either both be present or both absent")
        if dynamic is not None:
            if not isinstance(dynamic, dict):
                raise ValueError("dynamic must be an object")
            model = dynamic.get("time_constant_model")
            if model not in {"constant", "scheduled"}:
                raise ValueError("dynamic.time_constant_model must be constant or scheduled")
            required_dynamic = set(DEFAULT_DYNAMIC_PARAMETERS)
            if model == "scheduled":
                required_dynamic |= SCHEDULE_PARAMETER_NAMES
            allowed_dynamic = required_dynamic | {"evap_response_model"}
            dynamic_keys = frozenset(dynamic)
            if dynamic_keys not in {
                frozenset(required_dynamic),
                frozenset(allowed_dynamic),
            }:
                raise ValueError(
                    "dynamic keys must contain the standard fields and may add "
                    "evap_response_model"
                )
            evap_response_model = dynamic.get("evap_response_model", "cascaded")
            if evap_response_model not in EVAP_RESPONSE_MODELS:
                raise ValueError(
                    "dynamic.evap_response_model must be cascaded or direct"
                )
            for name, value in dynamic.items():
                if name in {"time_constant_model", "evap_response_model"}:
                    continue
                number = _finite_number(value, f"dynamic.{name}")
                if name in SCHEDULE_PARAMETER_NAMES:
                    continue
                if number < 0.0:
                    raise ValueError(f"dynamic.{name} must be nonnegative")
            for name in ("tau_comp_s", "tau_pump_s", "tau_cond_s", "tau_evap_s"):
                if float(dynamic[name]) <= 0.0:
                    raise ValueError(f"dynamic.{name} must be greater than zero")

            if not isinstance(thermal, dict):
                raise ValueError("thermal must be an object")
            optional_thermal = {
                "plate_fluid_effectiveness",
                "battery_heat_generation_scale",
                "compressor_displacement_scale",
                "battery_plate_conductance_model"
            }
            required_thermal = set(DEFAULT_THERMAL_PARAMETERS) - optional_thermal
            thermal_keys = set(thermal)
            if not required_thermal.issubset(thermal_keys) or not thermal_keys.issubset(
                required_thermal | optional_thermal
            ):
                raise ValueError(
                    "thermal keys must contain the legacy thermal fields and may "
                    "add plate_fluid_effectiveness, battery_heat_generation_scale, "
                    "compressor_displacement_scale, and battery_plate_conductance_model"
                )
            conductance_model = thermal.get(
                "battery_plate_conductance_model",
                "legacy_pump_scaled",
            )
            if conductance_model not in BATTERY_PLATE_CONDUCTANCE_MODELS:
                raise ValueError(
                    "thermal.battery_plate_conductance_model must be "
                    "legacy_pump_scaled or constant_physical"
                )
            for name, value in thermal.items():
                if name == "battery_plate_conductance_model":
                    continue
                number = _finite_number(value, f"thermal.{name}")
                if number <= 0.0:
                    raise ValueError(f"thermal.{name} must be greater than zero")
            plate_fluid_effectiveness = float(
                thermal.get("plate_fluid_effectiveness", 1.0)
            )
            if plate_fluid_effectiveness > 1.0:
                raise ValueError(
                    "thermal.plate_fluid_effectiveness must not exceed one"
                )
    except (TypeError, ValueError) as exc:
        raise PhysicsArtifactError(str(exc)) from exc
    return artifact


def load_physics_artifact(path: str | Path, require_validated: bool = False) -> dict:
    try:
        artifact = load_predictor_artifact(path, expected_type=PHYSICS_P)
    except PredictorArtifactError as exc:
        raise PhysicsArtifactError(str(exc)) from exc
    validated = validate_physics_artifact(artifact)
    if require_validated:
        fit = validated.get("fit")
        if not isinstance(fit, dict) or fit.get("fit_status") != "validated":
            raise PhysicsArtifactError(
                "Physics artifact must have fit.fit_status == 'validated'"
            )
    return validated


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
    domain = selected.get("input_domain", DEFAULT_INPUT_DOMAIN)
    raw_inputs = {
        "n_comp_rpm": n_comp_rpm,
        "n_pump_rpm": n_pump_rpm,
        "t_cool_c": t_cool_c,
        "t_ambient_c": t_ambient_c,
    }
    finite_inputs = {
        name: _finite_number(value, name) for name, value in raw_inputs.items()
    }
    clipped = {}
    for name, finite in finite_inputs.items():
        lower, upper = domain[name]
        clipped[name] = max(float(lower), min(float(upper), finite))
    capacity_model = capacity.get("model", LEGACY_CAPACITY_MODEL)
    if capacity_model in {ENHANCED_CAPACITY_MODEL, CUBIC_CAPACITY_MODEL}:
        if finite_inputs["n_comp_rpm"] < float(capacity["minimum_active_rpm"]):
            return 0.0
        gate_value = 1.0
        evaluator = (
            enhanced_active_capacity_w
            if capacity_model == ENHANCED_CAPACITY_MODEL
            else cubic_active_capacity_w
        )
        active = evaluator(
            capacity["coefficients"],
            clipped["n_comp_rpm"],
            clipped["n_pump_rpm"],
            clipped["t_cool_c"],
            clipped["t_ambient_c"],
            capacity["n_pump_ref_rpm"],
        )
    else:
        gate_value = smooth_gate(
            clipped["n_comp_rpm"],
            gate["n_on_rpm"],
            gate["width_rpm"],
        )
        active = active_capacity_w(
            capacity["coefficients"],
            clipped["n_comp_rpm"],
            clipped["n_pump_rpm"],
            clipped["t_cool_c"],
            clipped["t_ambient_c"],
            capacity["n_pump_ref_rpm"],
        )
    q_upper = _finite_number(capacity["q_upper_w"], "capacity.q_upper_w")
    clipped_active = max(0.0, min(q_upper, active))
    raw = _finite_number(gate_value * clipped_active, "physics capacity result")
    return max(0.0, min(q_upper, raw))



def physics_p_startup_fraction_value(
    n_comp_rpm: object,
    minimum_active_rpm: object,
) -> float:
    speed = _finite_number(n_comp_rpm, "n_comp_rpm")
    minimum_active = _finite_number(
        minimum_active_rpm, "minimum_active_rpm"
    )
    if minimum_active <= N_COMP_OFF_RPM:
        raise ValueError("minimum_active_rpm must exceed N_COMP_OFF_RPM")
    return float(
        np.clip(
            (speed - N_COMP_OFF_RPM) / (minimum_active - N_COMP_OFF_RPM),
            0.0,
            1.0,
        )
    )


def evaluate_physics_operating_capacity(
    n_comp_rpm: object,
    n_pump_rpm: object,
    t_cool_c: object,
    t_ambient_c: object,
    artifact: object = None,
) -> float:
    """Evaluate P capacity across off, startup, and validated running speeds."""
    selected = DEFAULT_PHYSICS_ARTIFACT if artifact is None else artifact
    validate_physics_artifact(selected)
    speed = _finite_number(n_comp_rpm, "n_comp_rpm")
    capacity = selected["capacity"]
    model = capacity.get("model", LEGACY_CAPACITY_MODEL)
    if model not in {ENHANCED_CAPACITY_MODEL, CUBIC_CAPACITY_MODEL}:
        return evaluate_physics_capacity(
            speed, n_pump_rpm, t_cool_c, t_ambient_c, artifact=selected
        )

    minimum_active = float(capacity["minimum_active_rpm"])
    startup_fraction = physics_p_startup_fraction_value(
        speed, minimum_active
    )
    if startup_fraction <= 0.0:
        return 0.0
    active_speed = max(minimum_active, speed)
    active_capacity = evaluate_physics_capacity(
        active_speed,
        n_pump_rpm,
        t_cool_c,
        t_ambient_c,
        artifact=selected,
    )
    return startup_fraction * active_capacity

def lag_step(previous: object, target: object, dt_s: object, tau_s: object) -> float:
    previous_value = _finite_number(previous, "previous")
    target_value = _finite_number(target, "target")
    dt = _finite_number(dt_s, "dt_s")
    tau = _finite_number(tau_s, "tau_s")
    if dt <= 0.0:
        raise ValueError("dt_s must be greater than zero")
    alpha = 1.0 if tau <= 0.0 else dt / (tau + dt)
    return _finite_number(
        previous_value + alpha * (target_value - previous_value),
        "lag_step result",
    )


def _battery_plate_conductance(thermal: dict, pump_ratio: float) -> float:
    """Return cell-to-plate conductance without conflating coolant flow."""
    base = float(thermal["battery_plate_conductance_w_k"])
    model = thermal.get(
        "battery_plate_conductance_model",
        "legacy_pump_scaled",
    )
    if model == "constant_physical":
        return base
    return base * float(pump_ratio) ** 0.8


def consumed_parameter_names(artifact: object = None) -> set[str]:
    selected = DEFAULT_PHYSICS_ARTIFACT if artifact is None else artifact
    validate_physics_artifact(selected)
    dynamic = selected.get("dynamic", DEFAULT_DYNAMIC_PARAMETERS)
    thermal = selected.get("thermal", DEFAULT_THERMAL_PARAMETERS)
    names = set(dynamic) | set(thermal)
    if dynamic["time_constant_model"] == "scheduled":
        names |= SCHEDULE_PARAMETER_NAMES
    return names


def initialize_physics_state(
    n_comp_eff_rpm: object,
    n_pump_eff_rpm: object,
    q_cond_w: object,
    q_evap_w: object,
    t_supply_c: object,
    t_plate_c: object,
    t_return_c: object,
    t_batt_c: object,
    t_cool_c: object,
) -> PhysicsPredictorState:
    values = {
        name: _finite_number(value, name)
        for name, value in locals().items()
    }
    return PhysicsPredictorState(
        n_comp_eff_rpm=values["n_comp_eff_rpm"],
        n_pump_eff_rpm=values["n_pump_eff_rpm"],
        evap_speed_history_rpm=(),
        pump_speed_history_rpm=(),
        q_cond_w=values["q_cond_w"],
        q_evap_w=values["q_evap_w"],
        supply_history_c=(),
        t_supply_c=values["t_supply_c"],
        t_plate_c=values["t_plate_c"],
        return_history_c=(),
        t_return_c=values["t_return_c"],
        t_batt_c=values["t_batt_c"],
        t_cool_c=values["t_cool_c"],
    )


def _delay_step(
    history: tuple[float, ...],
    new_value: float,
    delay_s: float,
    dt_s: float,
    initial_value: float,
) -> tuple[float, tuple[float, ...]]:
    steps = max(0, int(round(delay_s / dt_s)))
    if steps == 0:
        return new_value, ()
    buffer = deque(history[-steps:], maxlen=steps)
    while len(buffer) < steps:
        buffer.appendleft(initial_value)
    delayed = buffer[0]
    buffer.append(new_value)
    return delayed, tuple(buffer)


def _time_constant(
    dynamic: dict,
    name: str,
    schedule_name: str,
    coordinate: float,
) -> float:
    base = float(dynamic[name])
    if dynamic["time_constant_model"] == "constant":
        return base
    return max(0.1, base + float(dynamic[schedule_name]) * coordinate)


def step_physics_predictor(
    state: PhysicsPredictorState,
    n_comp_cmd_rpm: object,
    n_pump_cmd_rpm: object,
    q_gen_w: object,
    t_ambient_c: object,
    dt_s: object,
    artifact: object = None,
) -> PhysicsPredictorState:
    if not isinstance(state, PhysicsPredictorState):
        raise TypeError("state must be PhysicsPredictorState")
    selected = DEFAULT_PHYSICS_ARTIFACT if artifact is None else artifact
    validate_physics_artifact(selected)
    dynamic = selected.get("dynamic", DEFAULT_DYNAMIC_PARAMETERS)
    thermal = selected.get("thermal", DEFAULT_THERMAL_PARAMETERS)
    dt = _finite_number(dt_s, "dt_s")
    if dt <= 0.0:
        raise ValueError("dt_s must be greater than zero")
    n_comp_cmd = _finite_number(n_comp_cmd_rpm, "n_comp_cmd_rpm")
    n_pump_cmd = _finite_number(n_pump_cmd_rpm, "n_pump_cmd_rpm")
    q_gen = _finite_number(q_gen_w, "q_gen_w")
    ambient = _finite_number(t_ambient_c, "t_ambient_c")

    domain = selected.get("input_domain", DEFAULT_INPUT_DOMAIN)
    comp_mid = 0.5 * sum(domain["n_comp_rpm"])
    comp_half_range = 0.5 * (domain["n_comp_rpm"][1] - domain["n_comp_rpm"][0])
    pump_mid = 0.5 * sum(domain["n_pump_rpm"])
    pump_half_range = 0.5 * (domain["n_pump_rpm"][1] - domain["n_pump_rpm"][0])
    comp_coordinate = max(-1.0, min(1.0, (state.n_comp_eff_rpm - comp_mid) / comp_half_range))
    pump_coordinate = max(-1.0, min(1.0, (state.n_pump_eff_rpm - pump_mid) / pump_half_range))

    # State order 1: effective actuator speeds.
    n_comp_eff = lag_step(
        state.n_comp_eff_rpm,
        n_comp_cmd,
        dt,
        _time_constant(dynamic, "tau_comp_s", "tau_comp_schedule_s", comp_coordinate),
    )
    n_pump_eff = lag_step(
        state.n_pump_eff_rpm,
        n_pump_cmd,
        dt,
        _time_constant(dynamic, "tau_pump_s", "tau_pump_schedule_s", pump_coordinate),
    )

    # State order 2: compressor-input and pump-flow delays.
    n_comp_delayed, evap_history = _delay_step(
        state.evap_speed_history_rpm,
        n_comp_eff,
        float(dynamic["evap_input_delay_s"]),
        dt,
        state.n_comp_eff_rpm,
    )
    n_pump_delayed, pump_history = _delay_step(
        state.pump_speed_history_rpm,
        n_pump_eff,
        float(dynamic["pump_flow_delay_s"]),
        dt,
        state.n_pump_eff_rpm,
    )

    # State order 3-5: P1 steady capacity and cascaded refrigeration dynamics.
    q_steady = evaluate_physics_operating_capacity(
        n_comp_delayed,
        n_pump_delayed,
        state.t_cool_c,
        ambient,
        artifact=selected,
    )
    load_coordinate = max(-1.0, min(1.0, 2.0 * q_steady / selected["capacity"]["q_upper_w"] - 1.0))
    q_cond = lag_step(
        state.q_cond_w,
        q_steady,
        dt,
        _time_constant(dynamic, "tau_cond_s", "tau_cond_schedule_s", load_coordinate),
    )
    q_evap = lag_step(
        state.q_evap_w,
        q_steady
        if dynamic.get("evap_response_model", "cascaded") == "direct"
        else q_cond,
        dt,
        _time_constant(dynamic, "tau_evap_s", "tau_evap_schedule_s", load_coordinate),
    )

    pump_ref = float(thermal["n_pump_ref_rpm"])
    pump_ratio = max(1e-6, n_pump_delayed / pump_ref)
    mass_flow = float(thermal["coolant_mass_flow_ref_kg_s"]) * pump_ratio
    flow_capacity = mass_flow * float(thermal["coolant_cp_j_kg_k"])

    # State order 6-8: supply delay, cold plate, and return delay.
    evaporator_out = state.t_cool_c - q_evap / flow_capacity
    t_supply, supply_history = _delay_step(
        state.supply_history_c,
        evaporator_out,
        float(dynamic["supply_delay_s"]),
        dt,
        state.t_supply_c,
    )
    battery_plate_conductance = _battery_plate_conductance(
        thermal,
        pump_ratio,
    )
    reference_flow_capacity = (
        float(thermal["coolant_mass_flow_ref_kg_s"])
        * float(thermal["coolant_cp_j_kg_k"])
    )
    plate_fluid_conductance = float(
        thermal.get("plate_fluid_effectiveness", 1.0)
    ) * min(
        flow_capacity,
        reference_flow_capacity * pump_ratio**0.8,
    )
    plate_heat_capacity = (
        float(thermal["plate_tau_s"]) * reference_flow_capacity
    )
    total_plate_conductance = (
        battery_plate_conductance + plate_fluid_conductance
    )
    plate_equilibrium = (
        battery_plate_conductance * state.t_batt_c
        + plate_fluid_conductance * t_supply
    ) / total_plate_conductance
    plate_time_constant = plate_heat_capacity / total_plate_conductance
    t_plate = lag_step(
        state.t_plate_c,
        plate_equilibrium,
        dt,
        plate_time_constant,
    )
    q_batt_plate = battery_plate_conductance * (state.t_batt_c - t_plate)
    q_plate_fluid = plate_fluid_conductance * (t_plate - t_supply)
    plate_out = t_supply + q_plate_fluid / flow_capacity
    t_return, return_history = _delay_step(
        state.return_history_c,
        plate_out,
        float(dynamic["return_delay_s"]),
        dt,
        state.t_return_c,
    )

    # State order 9-10: battery and coolant energy balances.
    ambient_loss = float(thermal["ambient_conductance_w_k"]) * (state.t_batt_c - ambient)
    q_gen_effective = q_gen * float(
        thermal.get("battery_heat_generation_scale", 1.0)
    )
    t_batt = state.t_batt_c + dt * (q_gen_effective - q_batt_plate - ambient_loss) / float(
        thermal["battery_heat_capacity_j_k"]
    )
    t_cool = state.t_cool_c + dt * flow_capacity * (t_return - state.t_cool_c) / float(
        thermal["coolant_heat_capacity_j_k"]
    )

    result = PhysicsPredictorState(
        n_comp_eff_rpm=n_comp_eff,
        n_pump_eff_rpm=n_pump_eff,
        evap_speed_history_rpm=evap_history,
        pump_speed_history_rpm=pump_history,
        q_cond_w=q_cond,
        q_evap_w=q_evap,
        supply_history_c=supply_history,
        t_supply_c=t_supply,
        t_plate_c=t_plate,
        return_history_c=return_history,
        t_return_c=t_return,
        t_batt_c=t_batt,
        t_cool_c=t_cool,
    )
    for name, value in vars(result).items():
        values = value if isinstance(value, tuple) else (value,)
        for item in values:
            _finite_number(item, f"state.{name}")
    return result
