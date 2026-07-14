"""Bounded steady evaporator-capacity surrogates for the MPC reduced model."""

import json
from pathlib import Path


DEFAULT_CALIBRATION_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "mpc_evaporator_capacity_candidate_b"
    / "mpc_evaporator_capacity_candidate_b.json"
)


def load_capacity_calibration(path=DEFAULT_CALIBRATION_PATH):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    return max(0.0, min(float(calibration["q_evap_upper_bound_w"]), float(raw_capacity(calibration, n_comp_rpm, n_pump_rpm, t_cool_c))))
