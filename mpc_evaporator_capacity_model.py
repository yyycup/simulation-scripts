"""Bounded steady evaporator-capacity surrogates for the MPC reduced model."""

import json
import math
from pathlib import Path


DEFAULT_CALIBRATION_PATH = (
    Path(__file__).resolve().parent
    / "model_data"
    / "mpc_evaporator_capacity_candidate_b.json"
)
DEFAULT_MINIMUM_ACTIVE_RPM = 2000.0
DEFAULT_MPC_POWER_GATE_CENTER_RPM = 1950.0
DEFAULT_MPC_POWER_GATE_WIDTH_RPM = 10.0


def load_capacity_calibration(path=DEFAULT_CALIBRATION_PATH):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def minimum_active_rpm(calibration):
    default = (
        DEFAULT_MINIMUM_ACTIVE_RPM
        if calibration.get("model_type") == "candidate_b"
        else 0.0
    )
    return float(calibration.get("minimum_active_rpm", default))


def mpc_power_gate_parameters(calibration):
    gate = calibration.get("mpc_power_gate", {})
    center = float(gate.get("center_rpm", DEFAULT_MPC_POWER_GATE_CENTER_RPM))
    width = float(gate.get("width_rpm", DEFAULT_MPC_POWER_GATE_WIDTH_RPM))
    if width <= 0.0:
        raise ValueError("mpc_power_gate.width_rpm must be greater than zero")
    return center, width


def smooth_mpc_power_gate(calibration, n_comp_rpm):
    center, width = mpc_power_gate_parameters(calibration)
    offset = float(n_comp_rpm) - center
    return 0.5 * (1.0 + offset / math.sqrt(offset**2 + width**2))


def raw_capacity(calibration, n_comp_rpm, n_pump_rpm, t_cool_c):
    coefficients = calibration["coefficients"]
    kind = calibration["model_type"]
    if kind == "candidate_a":
        return coefficients["kq"] * n_comp_rpm
    if kind == "candidate_b":
        gain = coefficients["b0"] + coefficients["b1"] * calibration["n_pump_ref_rpm"] / n_pump_rpm + coefficients["b2"] * (t_cool_c - 25.0)
        return n_comp_rpm * gain
    if kind == "candidate_c":
        return (coefficients["b0"] + coefficients["b1"] * n_comp_rpm + coefficients["b2"] * n_pump_rpm + coefficients["b3"] * t_cool_c + coefficients["b4"] * n_comp_rpm * n_pump_rpm + coefficients["b5"] * n_comp_rpm * t_cool_c + coefficients["b6"] * n_comp_rpm**2)
    raise ValueError(f"Unsupported capacity model: {kind}")


def evaluate_capacity(calibration, n_comp_rpm, n_pump_rpm, t_cool_c):
    if float(n_comp_rpm) < minimum_active_rpm(calibration):
        return 0.0
    return max(0.0, min(float(calibration["q_evap_upper_bound_w"]), float(raw_capacity(calibration, n_comp_rpm, n_pump_rpm, t_cool_c))))
