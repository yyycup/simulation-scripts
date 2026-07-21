"""Bounded steady evaporator-capacity surrogates for the MPC reduced model."""

import json
from pathlib import Path


DEFAULT_CALIBRATION_PATH = (
    Path(__file__).resolve().parent
    / "model_data"
    / "mpc_evaporator_capacity_candidate_b.json"
)
DEFAULT_COMPRESSOR_OFF_RPM = 300.0
DEFAULT_MINIMUM_STEADY_RPM = 1000.0


def load_capacity_calibration(path=DEFAULT_CALIBRATION_PATH):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compressor_off_rpm(calibration):
    if "compressor_off_rpm" in calibration:
        return float(calibration["compressor_off_rpm"])
    if "minimum_active_rpm" in calibration:
        return float(calibration["minimum_active_rpm"])
    return DEFAULT_COMPRESSOR_OFF_RPM if calibration.get("model_type") == "candidate_b" else 0.0


def minimum_steady_rpm(calibration):
    if "minimum_steady_rpm" in calibration:
        return float(calibration["minimum_steady_rpm"])
    if "minimum_active_rpm" in calibration:
        return float(calibration["minimum_active_rpm"])
    return DEFAULT_MINIMUM_STEADY_RPM if calibration.get("model_type") == "candidate_b" else 0.0


def startup_fraction(calibration, n_comp_rpm):
    off_rpm = compressor_off_rpm(calibration)
    steady_rpm = minimum_steady_rpm(calibration)
    speed = float(n_comp_rpm)
    if speed <= off_rpm:
        return 0.0
    if speed >= steady_rpm or steady_rpm <= off_rpm:
        return 1.0
    return (speed - off_rpm) / (steady_rpm - off_rpm)


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
    fraction = startup_fraction(calibration, n_comp_rpm)
    if fraction <= 0.0:
        return 0.0
    steady_rpm = minimum_steady_rpm(calibration)
    capacity_speed = max(float(n_comp_rpm), steady_rpm)
    capacity = fraction * raw_capacity(
        calibration, capacity_speed, n_pump_rpm, t_cool_c
    )
    return max(0.0, min(float(calibration["q_evap_upper_bound_w"]), float(capacity)))
