import csv
import json
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from gekko import GEKKO

from ...simulation.config import (
    INITIAL_TEMP_C,
    MPC_CV_BAND_C,
    MPC_CV_TAU,
    MPC_DYNAMIC_TARGET_MAX_C,
    MPC_DYNAMIC_TARGET_MIN_C,
    MPC_COOLANT_RESERVE_ENABLED,
    MPC_HEAT_PREVIEW_ENABLED,
    MPC_HEAT_PREVIEW_NOMINAL_W,
    MPC_HEAT_PREVIEW_QUANTILE,
    MPC_HEAT_PREVIEW_QUANTILE_WEIGHT,
    MPC_HEAT_PREVIEW_WINDOW_S,
    MPC_N_COMP_MIN_RPM,
    MPC_PRECOOL_MAX_C,
    MPC_MIN_HOLD_TIME_S,
    MPC_Q_SPREAD,
    MPC_Q_TRACK,
    MPC_R_COMP,
    MPC_R_PUMP,
    MPC_R_SWITCH,
    MPC_RESERVE_QEVAP_REF_W,
    MPC_RESERVE_QEVAP_SCALE_W,
    MPC_RESERVE_T_PLATE_IN_REF_C,
    MPC_RESERVE_T_SCALE_C,
    MPC_RESERVE_T_TANK_REF_C,
    MPC_WARM_RELIEF_MAX_C,
    MPC_T_BAT_MARGIN_C,
    MPC_T_BAT_MAX_C,
    MPC_T_SPREAD_SCALE_C,
    MPC_T_TRACK_SCALE_C,
    MPC_U_NCOMP_DCOST,
    MPC_U_NCOMP_DMAX,
    MPC_U_NCOMP_INIT,
    MPC_U_NPUMP_DCOST,
    MPC_U_NPUMP_DMAX,
    MPC_U_NPUMP_INIT,
    MPC_REV_DELTA_T_LIM_C,
    MPC_REV_GAMMA_HOT_LIM,
    MPC_REV_H_DOWN_CELL_SCALE_C,
    MPC_REV_J_ON,
    MPC_REV_J_OFF,
    MPC_PRED_DELTA_T_SWITCH_C,
    MPC_PRED_SWITCH_BUFFER_S,
    MPC_REV_W_DELTA_T,
    MPC_REV_W_GAMMA_HOT,
    MPC_REV_W_H_DOWN,
    MPC_W_BAT_SAFETY,
    MPC_W_PLATE_RESERVE,
    MPC_W_QEVAP_RESERVE,
    MPC_W_TANK_RESERVE,
    N_COMP_MAX_RPM,
    N_COMP_MIN_RPM,
    N_COMP_OFF_RPM,
    N_PUMP_MAX_RPM,
    N_PUMP_MIN_RPM,
    SIM_DT,
    TARGET_TEMP_C,
)
from ...plant.thermal_loop import DEFAULT_REFRIGERATION_DYNAMICS, pipe_delay_steps
from ...plant.thermal_system import C_tank, cp_cool, m_dot_nominal
from .evaporator_capacity import load_capacity_calibration
from ...predictor.physics_p import (
    CUBIC_CAPACITY_FEATURE_NAMES,
    CUBIC_CAPACITY_MODEL,
    evaluate_physics_operating_capacity,
    initialize_physics_state,
    step_physics_predictor,
    load_physics_artifact,
    validate_physics_artifact,
    physics_p_startup_fraction_value,
)
from ...predictor.selection import (
    CANDIDATE_B,
    PHYSICS_P,
    normalize_predictor_name,
)
from ...analysis.normalization import (
    OBJECTIVE_KEYS,
    ObjectiveNormalizationScales,
    normalize_raw_terms,
)

PUMP_POWER_SPEED_COEFF = (2.27321928e-09, -1.62756913e-05, 4.41581449e-02, -3.49442214e01)
LEGACY_COMPRESSOR_POWER_SPEED_COEFF = (3.57e-6, 0.442, 34.0)
LEGACY_COMPRESSOR_POWER_MAX_W = (
    LEGACY_COMPRESSOR_POWER_SPEED_COEFF[0] * N_COMP_MAX_RPM**2
    + LEGACY_COMPRESSOR_POWER_SPEED_COEFF[1] * N_COMP_MAX_RPM
    + LEGACY_COMPRESSOR_POWER_SPEED_COEFF[2]
)
# Fit on the current detailed refrigeration-cycle steady grid.  The form keeps
# mass-flow dependence proportional to speed and lets specific work vary with
# coolant/ambient temperature without adding a pump-speed coupling term.
COMPRESSOR_POWER_COEFF = (
    0.03892611257343539,
    2.994839654732717e-06,
    -0.0007386784006792191,
    0.0019398990645880215,
)
COMPRESSOR_POWER_COOLANT_REF_C = 25.0
COMPRESSOR_POWER_AMBIENT_REF_C = 35.0
COMPRESSOR_POWER_DOMAIN = {
    "n_comp_rpm": (1000.0, 6000.0),
    "t_cool_c": (15.0, 35.0),
    "t_ambient_c": (20.0, 40.0),
}
PHYSICS_P_OFF_SNAP_TOLERANCE_RPM = 1.0
PHYSICS_P_COMP_MV_STEP_HOR = 3
PHYSICS_P_PUMP_MV_STEP_HOR = 3
PHYSICS_P_PEAK_SOLVER_TOL = 3e-6
PHYSICS_P_TEMP_BIAS_ALPHA = 0.20
PHYSICS_P_TEMP_BIAS_MAX_STEP_C = 0.012
PHYSICS_P_TEMP_BIAS_MAX_HORIZON_C = 0.16
# Numerical safety guard for the experimental Physics-P controller only.
# This is a wall-clock budget for one MPC control cycle (including recovery),
# not a control weight, prediction horizon, or actuator constraint.
PHYSICS_P_SOLVE_CYCLE_MAX_TIME_S = 60.0
MODEL_DATA_ROOT = Path(__file__).resolve().parent / "model_data"
DEFAULT_REDUCED_MODEL_CALIBRATION_PATH = (
    MODEL_DATA_ROOT / "mpc_reduced_model_best_theta.json"
)


DEFAULT_EVAPORATOR_CAPACITY_TABLE_PATH = (
    MODEL_DATA_ROOT / "candidate_b_capacity_limits_grid.csv"
)


@dataclass(frozen=True)
class EvaporatorCapacityTable:
    n_comp_rpm: np.ndarray
    n_pump_rpm: np.ndarray
    t_cool_in_c: np.ndarray
    q_hx_w: np.ndarray
    q_ref_max_w: np.ndarray


def load_evaporator_capacity_table(path=DEFAULT_EVAPORATOR_CAPACITY_TABLE_PATH):
    """Load and validate the rectangular candidate-B capacity grid."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as fp:
        rows = list(csv.DictReader(fp))
    required = {"N_comp_rpm", "N_pump_rpm", "T_cool_in_C", "Q_hx_potential_W", "Q_ref_max_W"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid evaporator capacity table: {path}")
    n_comp = np.array(sorted({float(row["N_comp_rpm"]) for row in rows}), dtype=float)
    n_pump = np.array(sorted({float(row["N_pump_rpm"]) for row in rows}), dtype=float)
    t_cool = np.array(sorted({float(row["T_cool_in_C"]) for row in rows}), dtype=float)
    shape = (n_comp.size, n_pump.size, t_cool.size)
    q_hx = np.full(shape, np.nan, dtype=float)
    q_ref = np.full(shape, np.nan, dtype=float)
    comp_index = {value: index for index, value in enumerate(n_comp)}
    pump_index = {value: index for index, value in enumerate(n_pump)}
    temp_index = {value: index for index, value in enumerate(t_cool)}
    for row in rows:
        index = (
            comp_index[float(row["N_comp_rpm"])],
            pump_index[float(row["N_pump_rpm"])],
            temp_index[float(row["T_cool_in_C"])],
        )
        q_hx[index] = float(row["Q_hx_potential_W"])
        q_ref[index] = float(row["Q_ref_max_W"])
    if not (np.isfinite(q_hx).all() and np.isfinite(q_ref).all()):
        raise ValueError(f"Evaporator capacity table has missing grid nodes: {path}")
    return EvaporatorCapacityTable(n_comp, n_pump, t_cool, q_hx, q_ref)


def _planned_control_values(values, executed_command):
    """Convert a GEKKO MV trajectory into future applied commands."""
    executed = float(executed_command)
    try:
        raw = np.asarray(list(values), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raw = np.asarray([], dtype=float)
    if raw.size <= 1:
        return [executed]
    plan = raw[1:].copy()
    plan[0] = executed
    previous = executed
    for index, value in enumerate(plan):
        if not np.isfinite(value):
            plan[index] = previous
        previous = float(plan[index])
    return [float(value) for value in plan]


def physics_p_temperature_bias_update(
    *,
    previous_step_bias_c,
    measured_temp_c,
    previous_prediction_1_c,
    horizon_steps,
    alpha=PHYSICS_P_TEMP_BIAS_ALPHA,
    gain=1.0,
    max_step_bias_c=PHYSICS_P_TEMP_BIAS_MAX_STEP_C,
    max_horizon_bias_c=PHYSICS_P_TEMP_BIAS_MAX_HORIZON_C,
):
    """Estimate a one-sided Physics-P temperature innovation over the horizon."""
    horizon_steps = int(horizon_steps)
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    alpha = float(alpha)
    gain = float(gain)
    max_step_bias_c = float(max_step_bias_c)
    max_horizon_bias_c = float(max_horizon_bias_c)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if not np.isfinite(gain) or gain < 0.0:
        raise ValueError("gain must be finite and nonnegative")
    if not np.isfinite(max_step_bias_c) or max_step_bias_c <= 0.0:
        raise ValueError("max_step_bias_c must be positive and finite")
    if not np.isfinite(max_horizon_bias_c) or max_horizon_bias_c <= 0.0:
        raise ValueError("max_horizon_bias_c must be positive and finite")

    previous_step_bias_c = float(previous_step_bias_c)
    if not np.isfinite(previous_step_bias_c):
        previous_step_bias_c = 0.0
    try:
        innovation_c = max(
            0.0,
            float(measured_temp_c) - float(previous_prediction_1_c),
        )
    except (TypeError, ValueError):
        innovation_c = 0.0
    if not np.isfinite(innovation_c):
        innovation_c = 0.0

    if gain == 0.0:
        step_bias_c = 0.0
    else:
        step_bias_c = np.clip(
            (1.0 - alpha) * previous_step_bias_c + alpha * innovation_c,
            0.0,
            max_step_bias_c,
        )
    path_c = np.minimum(
        np.arange(horizon_steps + 1, dtype=float) * gain * step_bias_c,
        max_horizon_bias_c,
    )
    return {
        "innovation_c": float(innovation_c),
        "step_bias_c": float(step_bias_c),
        "path_c": path_c,
    }


def _linear_segment_value(x, x0, x1, y0, y1):
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def _piecewise_linear_value(x, axis, values):
    index = int(np.clip(np.searchsorted(axis, x, side="right") - 1, 0, len(axis) - 2))
    return _linear_segment_value(x, axis[index], axis[index + 1], values[index], values[index + 1])


def trilinear_capacity_value(table, n_comp_rpm, n_pump_rpm, t_cool_in_c, output):
    """Evaluate the capacity grid with trilinear cells and linear endpoint extrapolation."""
    values = getattr(table, output)
    along_temp = np.array([[_piecewise_linear_value(t_cool_in_c, table.t_cool_in_c, values[i, j, :]) for j in range(table.n_pump_rpm.size)] for i in range(table.n_comp_rpm.size)])
    along_pump = np.array([_piecewise_linear_value(n_pump_rpm, table.n_pump_rpm, along_temp[i, :]) for i in range(table.n_comp_rpm.size)])
    return float(_piecewise_linear_value(n_comp_rpm, table.n_comp_rpm, along_pump))


def _gekko_piecewise_linear(m, x, axis, values):
    """Continuous GEKKO if2 representation of a one-dimensional PWL curve."""
    result = _linear_segment_value(x, axis[0], axis[1], values[0], values[1])
    for index in range(1, len(axis) - 1):
        next_segment = _linear_segment_value(x, axis[index], axis[index + 1], values[index], values[index + 1])
        result = m.if2(x - axis[index], result, next_segment)
    return result


def gekko_trilinear_capacity_expr(m, n_comp_rpm, n_pump_rpm, t_cool_in_c, table, output):
    """Build a GEKKO-solvable nested PWL / trilinear capacity expression."""
    values = getattr(table, output)
    along_temp = [[_gekko_piecewise_linear(m, t_cool_in_c, table.t_cool_in_c, values[i, j, :]) for j in range(table.n_pump_rpm.size)] for i in range(table.n_comp_rpm.size)]
    along_pump = [_gekko_piecewise_linear(m, n_pump_rpm, table.n_pump_rpm, along_temp[i]) for i in range(table.n_comp_rpm.size)]
    return _gekko_piecewise_linear(m, n_comp_rpm, table.n_comp_rpm, along_pump)

FINAL_PEAK_CV_WSP = 5e6
FINAL_FREQ_CV_WSP = 5e7
FINAL_PEAK_W_ENERGY_COMP = 600.0
FINAL_FREQ_W_ENERGY_COMP = 300.0
FINAL_PEAK_W_ENERGY_PUMP = 10000.0
FINAL_FREQ_W_ENERGY_PUMP = 15000.0
FINAL_PEAK_DMAX_COMP = 6000.0
FINAL_FREQ_DMAX_COMP = 6000.0
FINAL_PEAK_DMAX_PUMP = 300.0
FINAL_FREQ_DMAX_PUMP = 600.0
FINAL_PEAK_CV_BAND_HALF_WIDTH = 0.30
FINAL_FREQ_CV_BAND_HALF_WIDTH = 0.45



@dataclass(frozen=True)
class MPCParams:
    name: str
    dynamic_target_min: float
    dynamic_target_max: float
    precool_max: float
    warm_relief_max: float
    cv_band_half_width: float
    w_high_temp: float
    w_cold_temp: float
    w_delta_t: float
    w_energy_comp: float
    w_energy_pump: float
    w_dcomp: float
    w_dpump: float
    w_bat_safety: float
    w_temp_obj: float
    w_term_peak: float
    T_term_peak: float
    j_rev_on: float
    j_rev_off: float
    reverse_hold_s: float
    n_comp_min: float
    n_pump_min: float
    mpc_horizon: int
    dmax_comp: float
    dmax_pump: float
    terminal_cost_enabled: bool = False
    terminal_cost_type: str = "empirical_temp"
    w_terminal_temp: float = 0.0
    terminal_temp_scale_c: float = 1.0
    w_dcomp_quadratic: float = 0.0
    comp_command_filter_alpha: float = 1.0
    physics_p_temp_bias_gain: float = 0.0


peak_mpc_params = MPCParams(
    name="peak_mpc_params",
    dynamic_target_min=25.0,
    dynamic_target_max=25.0,
    precool_max=0.0,
    warm_relief_max=0.0,
    cv_band_half_width=FINAL_PEAK_CV_BAND_HALF_WIDTH,
    w_high_temp=FINAL_PEAK_CV_WSP,
    w_cold_temp=FINAL_PEAK_CV_WSP,
    w_delta_t=MPC_REV_W_DELTA_T,
    w_energy_comp=FINAL_PEAK_W_ENERGY_COMP,
    w_energy_pump=FINAL_PEAK_W_ENERGY_PUMP,
    w_dcomp=MPC_U_NCOMP_DCOST,
    w_dpump=MPC_U_NPUMP_DCOST,
    w_bat_safety=0.0,
    w_temp_obj=0.0,
    w_term_peak=0.0,
    T_term_peak=25.0,
    j_rev_on=1.85,
    j_rev_off=1.50,
    reverse_hold_s=200.0,
    n_comp_min=1000.0,
    n_pump_min=N_PUMP_MIN_RPM,
    mpc_horizon=60,
    dmax_comp=FINAL_PEAK_DMAX_COMP,
    dmax_pump=FINAL_PEAK_DMAX_PUMP,
)

freq_mpc_params = MPCParams(
    name="freq_mpc_params",
    dynamic_target_min=25.0,
    dynamic_target_max=25.0,
    precool_max=0.0,
    warm_relief_max=0.0,
    cv_band_half_width=FINAL_FREQ_CV_BAND_HALF_WIDTH,
    w_high_temp=FINAL_FREQ_CV_WSP,
    w_cold_temp=FINAL_FREQ_CV_WSP,
    w_delta_t=peak_mpc_params.w_delta_t * 1.1,
    w_energy_comp=FINAL_FREQ_W_ENERGY_COMP,
    w_energy_pump=FINAL_FREQ_W_ENERGY_PUMP,
    w_dcomp=peak_mpc_params.w_dcomp,
    w_dpump=peak_mpc_params.w_dpump,
    w_bat_safety=peak_mpc_params.w_bat_safety,
    w_temp_obj=0.0,
    w_term_peak=0.0,
    T_term_peak=25.0,
    j_rev_on=1.95,
    j_rev_off=1.60,
    reverse_hold_s=250.0,
    n_comp_min=1000.0,
    n_pump_min=N_PUMP_MIN_RPM,
    mpc_horizon=45,
    dmax_comp=FINAL_FREQ_DMAX_COMP,
    dmax_pump=FINAL_FREQ_DMAX_PUMP,
)


def select_mpc_params_for_scene(case_name=None):
    text = "" if case_name is None else str(case_name).lower()
    if "\u8c03\u9891" in text or "freq" in text or "reg" in text:
        return freq_mpc_params
    if "\u8c03\u5cf0" in text or "peak" in text:
        return peak_mpc_params
    return peak_mpc_params



FINAL_SELECTED_TERMINAL_COST_PARAMS = {
    "peak": {
        "case_label": "peak_empirical_w1e6",
        "terminal_cost_enabled": True,
        "terminal_cost_type": "empirical_temp",
        "w_terminal_temp": 1e6,
        "terminal_temp_scale_c": 1.0,
    },
    "freq": {
        "case_label": "freq_empirical_w5e5",
        "terminal_cost_enabled": True,
        "terminal_cost_type": "empirical_temp",
        "w_terminal_temp": 5e5,
        "terminal_temp_scale_c": 1.0,
    },
}

_FINAL_SELECTED_TERMINAL_COST_FIELDS = (
    "terminal_cost_enabled",
    "terminal_cost_type",
    "w_terminal_temp",
    "terminal_temp_scale_c",
)


def normalize_mpc_scene(scene: object) -> str:
    text = "" if scene is None else str(scene).strip().lower()
    if "调频" in text or "freq" in text or "frequency" in text or "reg" in text:
        return "freq"
    if "调峰" in text or "peak" in text:
        return "peak"
    raise ValueError(f"Unknown MPC scene: {scene!r}")


def apply_final_selected_terminal_cost_params(params, scene: object):
    """Return a copy of params with the final selected empirical terminal-cost overlay applied."""
    scene_key = normalize_mpc_scene(scene)
    profile = FINAL_SELECTED_TERMINAL_COST_PARAMS[scene_key]
    overlay = {field: profile[field] for field in _FINAL_SELECTED_TERMINAL_COST_FIELDS}
    case_label = profile["case_label"]
    if isinstance(params, dict):
        out = deepcopy(params)
        out.update(overlay)
        out["case_label"] = case_label
        return out
    updated = replace(params, **overlay)
    if hasattr(updated, "name"):
        updated = replace(updated, name=f"{params.name}_{case_label}")
    return updated


def final_selected_terminal_cost_params_for_scene(scene: object, base_params: MPCParams | None = None) -> MPCParams:
    """Build the final selected terminal-cost profile without changing the default baseline."""
    scene_key = normalize_mpc_scene(scene)
    baseline = select_mpc_params_for_scene(scene_key) if base_params is None else base_params
    return apply_final_selected_terminal_cost_params(baseline, scene_key)


def runtime_mpc_params_for_scene(scene: object) -> MPCParams:
    """Return normal runtime MPC parameters with the frozen terminal cost enabled."""
    baseline = select_mpc_params_for_scene(scene)
    return apply_final_selected_terminal_cost_params(baseline, scene)


def _pump_power_speed_value(n_rpm):
    a3, a2, a1, a0 = PUMP_POWER_SPEED_COEFF
    return a3 * n_rpm**3 + a2 * n_rpm**2 + a1 * n_rpm + a0


def _pump_power_speed_expr(m, n_rpm):
    a3, a2, a1, a0 = PUMP_POWER_SPEED_COEFF
    return m.Intermediate(a3 * n_rpm**3 + a2 * n_rpm**2 + a1 * n_rpm + a0)


def _legacy_compressor_power_expr(m, n_comp_rpm):
    a2, a1, a0 = LEGACY_COMPRESSOR_POWER_SPEED_COEFF
    return m.Intermediate(a2 * n_comp_rpm**2 + a1 * n_comp_rpm + a0)


def compressor_power_value(n_comp_rpm, t_cool_c, t_ambient_c):
    a0, a1, a2, a3 = COMPRESSOR_POWER_COEFF
    return float(n_comp_rpm) * (
        a0
        + a1 * float(n_comp_rpm)
        + a2 * (float(t_cool_c) - COMPRESSOR_POWER_COOLANT_REF_C)
        + a3 * (float(t_ambient_c) - COMPRESSOR_POWER_AMBIENT_REF_C)
    )



def physics_p_operating_compressor_power_value(
    n_comp_rpm,
    t_cool_c,
    t_ambient_c,
    minimum_active_rpm=N_COMP_MIN_RPM,
    compressor_displacement_scale=1.0,
):
    displacement_scale = float(compressor_displacement_scale)
    if not np.isfinite(displacement_scale) or displacement_scale <= 0.0:
        raise ValueError("compressor_displacement_scale must be positive and finite")
    startup_fraction = physics_p_startup_fraction_value(
        n_comp_rpm, minimum_active_rpm
    )
    active_speed = max(float(n_comp_rpm), float(minimum_active_rpm))
    return displacement_scale * startup_fraction * max(
        0.0,
        compressor_power_value(active_speed, t_cool_c, t_ambient_c),
    )


def normalize_physics_p_compressor_command(n_comp_rpm):
    """Snap a numerically active lower-bound solution to the exact off point."""
    command = float(n_comp_rpm)
    if command <= N_COMP_OFF_RPM + PHYSICS_P_OFF_SNAP_TOLERANCE_RPM:
        return float(N_COMP_OFF_RPM)
    return command

def _compressor_power_expr(m, n_comp_rpm, t_cool_c, t_ambient_c):
    a0, a1, a2, a3 = COMPRESSOR_POWER_COEFF
    return m.Intermediate(
        n_comp_rpm
        * (
            a0
            + a1 * n_comp_rpm
            + a2 * (t_cool_c - COMPRESSOR_POWER_COOLANT_REF_C)
            + a3 * (t_ambient_c - COMPRESSOR_POWER_AMBIENT_REF_C)
        )
    )


def compressor_power_normalization_w():
    return compressor_power_value(
        COMPRESSOR_POWER_DOMAIN["n_comp_rpm"][1],
        COMPRESSOR_POWER_DOMAIN["t_cool_c"][0],
        COMPRESSOR_POWER_DOMAIN["t_ambient_c"][1],
    )


def load_reduced_model_calibration(path=DEFAULT_REDUCED_MODEL_CALIBRATION_PATH):
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def apply_reduced_model_calibration(params, theta):
    calibrated = dict(params)
    theta = theta or {}
    scale_map = {
        "kq_scale": "kq",
        "h1_scale": "h1_ref",
        "C1_scale": "C1",
        "C2_scale": "C2",
        "tau_evap_scale": "tau_evap_s",
        "tau_plate_scale": "tau_plate_s",
    }
    for scale_name, param_name in scale_map.items():
        if scale_name in theta and param_name in calibrated:
            calibrated[param_name] = calibrated[param_name] * float(theta[scale_name])
    if "Cplate_scale" in theta:
        calibrated["Cplate_scale"] = float(theta["Cplate_scale"])
    calibrated["reduced_model_calibration_theta"] = dict(theta)
    return calibrated


def _validated_physics_p_artifact(artifact):
    if artifact is None:
        raise ValueError("physics_p requires an explicit validated predictor artifact")
    if isinstance(artifact, dict):
        validated = validate_physics_artifact(artifact)
        fit = validated.get("fit")
        if not isinstance(fit, dict) or fit.get("fit_status") != "validated":
            raise ValueError("physics_p artifact must have fit.fit_status == 'validated'")
        return validated
    return load_physics_artifact(artifact, require_validated=True)


def _validated_runtime_weight_multiplier(value, name):
    """Return one finite, strictly positive runtime MPC multiplier."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and greater than zero") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return numeric


def physics_p_prediction_domain_info(
    t_cool_pred_c,
    artifact,
    *,
    tolerance_c=1e-3,
):
    """Check the full coolant trajectory against P's fitted capacity domain."""
    selected = _validated_physics_p_artifact(artifact)
    lower_c, upper_c = (
        float(bound) for bound in selected["input_domain"]["t_cool_c"]
    )
    values = np.asarray(t_cool_pred_c, dtype=float).reshape(-1)
    finite = values.size > 0 and bool(np.all(np.isfinite(values)))
    if finite:
        minimum_c = float(np.min(values))
        maximum_c = float(np.max(values))
        violation_c = max(lower_c - minimum_c, maximum_c - upper_c, 0.0)
    else:
        minimum_c = np.nan
        maximum_c = np.nan
        violation_c = np.inf
    return {
        "prediction_domain_valid": bool(
            finite and violation_c <= float(tolerance_c)
        ),
        "t_cool_pred_min_c": minimum_c,
        "t_cool_pred_max_c": maximum_c,
        "t_cool_domain_lower_c": lower_c,
        "t_cool_domain_upper_c": upper_c,
        "t_cool_domain_violation_c": float(violation_c),
    }


def physics_p_params_from_artifact(params, artifact):
    """Map a validated P artifact onto the reduced MPC state parameters."""
    selected = _validated_physics_p_artifact(artifact)
    dynamic = selected["dynamic"]
    thermal = selected["thermal"]
    if dynamic["time_constant_model"] != "constant":
        raise ValueError("MPC physics_p currently supports only constant time constants")

    mapped = dict(params)
    mapped.update(
        {
            "tau_comp_s": float(dynamic["tau_comp_s"]),
            "tau_pump_s": float(dynamic["tau_pump_s"]),
            "tau_evap_s": float(dynamic["tau_evap_s"]),
            "tau_cond_s": float(dynamic["tau_cond_s"]),
            "theta_evap_input_s": float(dynamic["evap_input_delay_s"]),
            "theta_pump_flow_s": float(dynamic["pump_flow_delay_s"]),
            "tau_pipe_supply_s": float(dynamic["supply_delay_s"]),
            "tau_pipe_return_s": float(dynamic["return_delay_s"]),
            "evap_response_model": dynamic.get("evap_response_model", "cascaded"),
            "C1": float(thermal["battery_heat_capacity_j_k"]),
            "C2": float(thermal["coolant_heat_capacity_j_k"]),
            "h1_ref": float(thermal["battery_plate_conductance_w_k"]),
            "N_pump_ref": float(thermal["n_pump_ref_rpm"]),
            "h2": float(thermal["ambient_conductance_w_k"]),
            "cp_cool": float(thermal["coolant_cp_j_kg_k"]),
            "m_dot_ref": float(thermal["coolant_mass_flow_ref_kg_s"]),
            "tau_plate_s": float(thermal["plate_tau_s"]),
            "plate_fluid_effectiveness": float(
                thermal.get("plate_fluid_effectiveness", 1.0)
            ),
            "battery_heat_generation_scale": float(
                thermal.get("battery_heat_generation_scale", 1.0)
            ),
            "compressor_displacement_scale": float(
                thermal.get("compressor_displacement_scale", 1.0)
            ),
            "battery_plate_conductance_model": thermal.get(
                "battery_plate_conductance_model",
                "legacy_pump_scaled",
            ),
        }
    )
    reference_flow_capacity = mapped["m_dot_ref"] * mapped["cp_cool"]
    mapped["C_plate"] = mapped["tau_plate_s"] * reference_flow_capacity
    return mapped


def physics_p_capacity_expr(
    n_comp_rpm,
    n_pump_rpm,
    t_cool_c,
    t_ambient_c,
    artifact,
    sqrt_func=None,
    smooth_clip_names=None,
):
    """Return the P cubic-capacity expression with artifact-domain clipping."""
    selected = _validated_physics_p_artifact(artifact)
    capacity = selected["capacity"]
    if capacity.get("model") != CUBIC_CAPACITY_MODEL:
        raise ValueError("MPC physics_p currently requires single_cubic_v1 capacity")
    coefficients = capacity["coefficients"]
    if len(coefficients) != len(CUBIC_CAPACITY_FEATURE_NAMES):
        raise ValueError("physics_p cubic capacity must contain 20 coefficients")

    domain = selected["input_domain"]

    def clipped(value, name):
        lower, upper = (float(bound) for bound in domain[name])
        if sqrt_func is None:
            return max(lower, min(upper, float(value)))
        if smooth_clip_names is not None and name not in smooth_clip_names:
            return value
        epsilon = 1e-6
        above_lower = 0.5 * (
            value + lower + sqrt_func((value - lower) ** 2 + epsilon)
        )
        return 0.5 * (
            above_lower
            + upper
            - sqrt_func((above_lower - upper) ** 2 + epsilon)
        )

    n_comp_rpm = clipped(n_comp_rpm, "n_comp_rpm")
    n_pump_rpm = clipped(n_pump_rpm, "n_pump_rpm")
    t_cool_c = clipped(t_cool_c, "t_cool_c")
    t_ambient_c = clipped(t_ambient_c, "t_ambient_c")
    compressor = (n_comp_rpm - 4000.0) / 2000.0
    pump_ratio = float(capacity["n_pump_ref_rpm"]) / n_pump_rpm
    coolant = (t_cool_c - 27.5) / 7.5
    ambient = (t_ambient_c - 30.0) / 10.0
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
    gain = sum(float(coefficient) * feature for coefficient, feature in zip(coefficients, features))
    return n_comp_rpm * gain


def smooth_bounded_capacity_expr(raw_capacity_w, upper_w, sqrt_func):
    """Apply a differentiable approximation of clip(raw, 0, upper)."""
    epsilon = 1e-6
    nonnegative = 0.5 * (
        raw_capacity_w + sqrt_func(raw_capacity_w**2 + epsilon)
    )
    return 0.5 * (
        nonnegative
        + float(upper_w)
        - sqrt_func((nonnegative - float(upper_w)) ** 2 + epsilon)
    )



def _physics_p_startup_fraction_expr(n_comp_rpm, minimum_active_rpm, sqrt_func):
    denominator = float(minimum_active_rpm) - float(N_COMP_OFF_RPM)
    if denominator <= 0.0:
        raise ValueError("physics_p minimum active speed must exceed off speed")
    raw_fraction = (n_comp_rpm - float(N_COMP_OFF_RPM)) / denominator
    epsilon = 1e-12
    nonnegative = 0.5 * (
        raw_fraction + sqrt_func(raw_fraction**2 + epsilon)
    )
    return 0.5 * (
        nonnegative
        + 1.0
        - sqrt_func((nonnegative - 1.0) ** 2 + epsilon)
    )


def _physics_p_active_speed_expr(n_comp_rpm, minimum_active_rpm, sqrt_func):
    epsilon = 1e-12
    return 0.5 * (
        n_comp_rpm
        + float(minimum_active_rpm)
        + sqrt_func((n_comp_rpm - float(minimum_active_rpm)) ** 2 + epsilon)
    )


def physics_p_operating_capacity_expr(
    n_comp_rpm,
    n_pump_rpm,
    t_cool_c,
    t_ambient_c,
    artifact,
    sqrt_func,
):
    """Extend the validated active capacity through the plant startup ramp."""
    selected = _validated_physics_p_artifact(artifact)
    minimum_active = float(selected["capacity"]["minimum_active_rpm"])
    startup_fraction = _physics_p_startup_fraction_expr(
        n_comp_rpm, minimum_active, sqrt_func
    )
    active_speed = _physics_p_active_speed_expr(
        n_comp_rpm, minimum_active, sqrt_func
    )
    active_capacity = physics_p_capacity_expr(
        active_speed,
        n_pump_rpm,
        t_cool_c,
        t_ambient_c,
        selected,
        sqrt_func=sqrt_func,
        # active_speed is already bounded by the validated compressor-state
        # limits. Re-smoothing it at 6000 rpm adds a redundant square root and
        # a poorly scaled derivative at the most common saturation point.
        smooth_clip_names=("t_cool_c",),
    )
    return startup_fraction * active_capacity

def _direction_from_reversed(is_reversed):
    return -1 if is_reversed else 1


def _is_reversed_from_direction(direction):
    return direction < 0


class MPCControllerDual:
    """Original two-input reduced MPC model.

    This is the former MPC core: compressor speed and pump speed are still the
    only continuous optimized control variables in this model.
    """

    def __init__(
        self,
        params,
        dt,
        np_horizon=60,
        mode="POINT",
        target_temp_c=TARGET_TEMP_C,
        flow_direction=1,
        mpc_params=None,
        case_name=None,
        predictor=CANDIDATE_B,
        predictor_artifact=None,
        strict_predictor_ablation=False,
    ):
        self.predictor_name = normalize_predictor_name(predictor)
        if self.predictor_name not in {CANDIDATE_B, PHYSICS_P}:
            raise ValueError(
                f"{self.predictor_name} is not available in the closed-loop MPC core"
            )
        self.predictor_artifact = (
            _validated_physics_p_artifact(predictor_artifact)
            if self.predictor_name == PHYSICS_P
            else None
        )
        self.strict_predictor_ablation = bool(strict_predictor_ablation)
        self._construction_params = dict(params)
        self.params = dict(params)
        if self.predictor_name == PHYSICS_P:
            self.params = physics_p_params_from_artifact(
                self.params,
                self.predictor_artifact,
            )
        self.params.setdefault("tau_comp_s", DEFAULT_REFRIGERATION_DYNAMICS["tau_comp_s"])
        self.params.setdefault("tau_pump_s", DEFAULT_REFRIGERATION_DYNAMICS["tau_pump_s"])
        self.params.setdefault("tau_evap_s", DEFAULT_REFRIGERATION_DYNAMICS["tau_evap_s"])
        self.params.setdefault("tau_cond_s", DEFAULT_REFRIGERATION_DYNAMICS["tau_cond_s"])
        self.params.setdefault("theta_evap_input_s", DEFAULT_REFRIGERATION_DYNAMICS["theta_evap_input_s"])
        self.params.setdefault("theta_pump_flow_s", DEFAULT_REFRIGERATION_DYNAMICS["theta_pump_flow_s"])
        self.params.setdefault("tau_pipe_supply_s", DEFAULT_REFRIGERATION_DYNAMICS["tau_pipe_supply_s"])
        self.params.setdefault("tau_pipe_return_s", DEFAULT_REFRIGERATION_DYNAMICS["tau_pipe_return_s"])
        self.params.setdefault("tau_plate_s", 20.0)
        self.params.setdefault("cp_cool", cp_cool)
        self.params.setdefault("m_dot_ref", m_dot_nominal)
        self.dt = dt
        self.np_horizon = np_horizon
        self.mode = mode
        self.target_temp_c = target_temp_c
        self.flow_direction = 1 if flow_direction >= 0 else -1
        self.mpc_params = mpc_params or peak_mpc_params
        self.case_name = case_name
        self.evap_input_delay_steps = pipe_delay_steps(self.params["theta_evap_input_s"], self.dt)
        self.pump_flow_delay_steps = pipe_delay_steps(self.params["theta_pump_flow_s"], self.dt)
        self.pipe_supply_delay_steps = max(1, pipe_delay_steps(self.params["tau_pipe_supply_s"], self.dt))
        self.pipe_return_delay_steps = max(1, pipe_delay_steps(self.params["tau_pipe_return_s"], self.dt))
        self.m = GEKKO(remote=False)
        self.m.time = np.linspace(0, self.np_horizon * self.dt, self.np_horizon + 1)
        self.alpha_temp = 1.0
        self.alpha_comp = 1.0
        self.alpha_pump = 1.0
        self.runtime_weight_mode = "fixed"
        self.runtime_objective_scale_factor = 1.0
        self.objective_normalization_scales = ObjectiveNormalizationScales()
        self.objective_contribution_diagnostics_enabled = False
        self.runtime_WSPLO = float(self.mpc_params.w_cold_temp)
        self.runtime_WSPHI = float(self.mpc_params.w_high_temp)
        self.runtime_w_terminal = float(self.mpc_params.w_terminal_temp)
        self.runtime_w_temp_obj = float(self.mpc_params.w_temp_obj)
        self.runtime_w_energy_comp = float(self.mpc_params.w_energy_comp)
        self.runtime_w_energy_pump = float(self.mpc_params.w_energy_pump)
        self.runtime_w_dcomp = float(self.mpc_params.w_dcomp)
        self.runtime_w_dpump = float(self.mpc_params.w_dpump)
        self.runtime_w_dcomp_quadratic = float(
            self.mpc_params.w_dcomp_quadratic
        )
        self._runtime_w_temp_obj_param = self.m.Param(
            value=self.runtime_w_temp_obj
        )
        self._runtime_w_terminal_param = self.m.Param(
            value=self.runtime_w_terminal
        )
        self._runtime_w_energy_comp_param = self.m.Param(
            value=self.runtime_w_energy_comp
        )
        self._runtime_w_energy_pump_param = self.m.Param(
            value=self.runtime_w_energy_pump
        )
        self._runtime_w_dcomp_quadratic_param = self.m.Param(
            value=self.runtime_w_dcomp_quadratic
        )
        p = self.params
        if self.predictor_name == PHYSICS_P:
            self.evaporator_capacity_calibration = None
            self._capacity_upper_w = float(
                self.predictor_artifact["capacity"]["q_upper_w"]
            )
        else:
            self.evaporator_capacity_calibration = load_capacity_calibration()
            self._capacity_upper_w = float(self.evaporator_capacity_calibration["q_evap_upper_bound_w"])
        # This is only a GEKKO initial guess; measured refrigeration states are
        # applied before every solve. Keeping the established warm start avoids
        # early frequency-case convergence failures.
        if self.predictor_name == PHYSICS_P:
            initial_q_evap_w = float(
                physics_p_capacity_expr(
                    p["N_comp_min"],
                    MPC_U_NPUMP_INIT,
                    p["T_init_cool"] - 273.15,
                    p["T_env"] - 273.15,
                    self.predictor_artifact,
                )
            )
        else:
            initial_q_evap_w = p["kq"] * p["N_comp_min"]

        self.d1_qgen = self.m.Param(value=1000.0)
        self.d2_tamb = self.m.Param(value=p["T_env"])
        self.d3_qgen_risk = self.m.Param(value=0.0)
        self.physics_p_temp_bias_enabled = bool(
            self.predictor_name == PHYSICS_P
            and not self.strict_predictor_ablation
            and float(self.mpc_params.physics_p_temp_bias_gain) > 0.0
        )
        self.physics_p_temp_bias_path_c = self.m.Param(
            value=np.zeros(self.np_horizon + 1, dtype=float)
        )
        self._physics_p_previous_prediction_1_c = np.nan
        self._physics_p_temp_bias_step_c = 0.0
        self._physics_p_temp_bias_innovation_c = 0.0
        self._physics_p_temp_bias_end_c = 0.0
        self.T_batt_K = self.m.SV(value=p["T_init_batt"])
        self.T_cool_K = self.m.SV(value=p["T_init_cool"])
        # P's 15-35 C coolant interval is a fitted-model acceptance domain.
        # Do not map it onto SV bounds: doing so makes the first dynamic cycle
        # appear structurally infeasible before GEKKO can reconcile the measured
        # state with the delayed internal states. The post-solve trajectory check
        # below enforces the domain without changing the physical state equation.
        # Keep solver iterates inside the physical speed domain.  The dynamic
        # equations and command bounds already keep valid trajectories here,
        # but long-horizon IPOPT initialization can otherwise explore negative
        # speeds and make fractional pump-flow terms numerically invalid.
        if self.predictor_name == PHYSICS_P:
            p_domain = self.predictor_artifact["input_domain"]
            n_comp_state_bounds = (
                float(N_COMP_OFF_RPM), float(p_domain["n_comp_rpm"][1])
            )
            n_pump_state_bounds = tuple(
                float(bound) for bound in p_domain["n_pump_rpm"]
            )
        else:
            n_comp_state_bounds = (0.0, N_COMP_MAX_RPM)
            n_pump_state_bounds = (0.0, N_PUMP_MAX_RPM)
        self.N_comp = self.m.SV(
            value=p["N_comp_min"],
            lb=n_comp_state_bounds[0],
            ub=n_comp_state_bounds[1],
        )
        self.N_pump = self.m.SV(
            value=MPC_U_NPUMP_INIT,
            lb=n_pump_state_bounds[0],
            ub=n_pump_state_bounds[1],
        )
        self.Q_evap = self.m.SV(value=initial_q_evap_w)
        self.Q_cond = self.m.SV(value=initial_q_evap_w)
        self.T_plate = self.m.SV(value=p["T_init_cool"] - 273.15)
        if self.predictor_name == PHYSICS_P:
            # GEKKO delay() initializes unavailable pre-horizon history to
            # zero.  Delay temperature deviations from the measured pipe
            # temperatures so an empty history represents the observed
            # thermal state instead of an unphysical 0 degC coolant slug.
            initial_pipe_c = p["T_init_cool"] - 273.15
            self.T_supply_ref_c = self.m.Param(value=initial_pipe_c)
            self.T_return_ref_c = self.m.Param(value=initial_pipe_c)
            self.T_supply_delay_delta_c = self.m.SV(value=0.0)
            self.T_return_delay_delta_c = self.m.SV(value=0.0)
            self.T_supply = self.m.Intermediate(
                self.T_supply_ref_c + self.T_supply_delay_delta_c
            )
            self.T_return = self.m.Intermediate(
                self.T_return_ref_c + self.T_return_delay_delta_c
            )
        else:
            self.T_supply = self.m.Var(value=p["T_init_cool"] - 273.15)
            self.T_return = self.m.Var(value=p["T_init_cool"] - 273.15)
        self.N_comp_delay = self.m.Var(
            value=p["N_comp_min"],
            lb=n_comp_state_bounds[0],
            ub=n_comp_state_bounds[1],
        )
        self.N_pump_delay = self.m.Var(
            value=MPC_U_NPUMP_INIT,
            lb=n_pump_state_bounds[0],
            ub=n_pump_state_bounds[1],
        )
        if self.predictor_name == PHYSICS_P and self.pump_flow_delay_steps > 0:
            self.N_pump_delay_ref_rpm = self.m.Param(
                value=MPC_U_NPUMP_INIT
            )
            self.N_pump_delay_delta_rpm = self.m.SV(value=0.0)

        self.T_batt_K.FSTATUS = 1
        self.T_cool_K.FSTATUS = 1
        self.N_comp.FSTATUS = 1
        self.N_pump.FSTATUS = 1
        self.Q_evap.FSTATUS = 1
        self.Q_cond.FSTATUS = 1
        self.T_plate.FSTATUS = 1
        if self.predictor_name == PHYSICS_P:
            self.T_supply_delay_delta_c.FSTATUS = 1
            self.T_return_delay_delta_c.FSTATUS = 1
            if self.pump_flow_delay_steps > 0:
                self.N_pump_delay_delta_rpm.FSTATUS = 1
            self._physics_p_state_initialized = False
        self._n_comp_command_lower_rpm = float(
            p["N_comp_min"]
            if self.strict_predictor_ablation
            else (
                N_COMP_OFF_RPM
                if self.predictor_name == PHYSICS_P
                else p["N_comp_min"]
            )
        )
        self.u_ncomp = self.m.MV(
            value=MPC_U_NCOMP_INIT,
            lb=self._n_comp_command_lower_rpm,
            ub=N_COMP_MAX_RPM,
        )
        self.u_npump = self.m.MV(value=MPC_U_NPUMP_INIT, lb=p["N_pump_min"], ub=N_PUMP_MAX_RPM)
        self.u_ncomp.STATUS = 1
        self.u_npump.STATUS = 1
        self.u_ncomp.DCOST = self.runtime_w_dcomp
        self.u_npump.DCOST = self.runtime_w_dpump
        self.u_ncomp.DMAX = self.mpc_params.dmax_comp
        self.u_npump.DMAX = self.mpc_params.dmax_pump
        self.quadratic_comp_move_cost_enabled = bool(
            self.predictor_name == PHYSICS_P
            and float(self.mpc_params.w_dcomp_quadratic) > 0.0
        )
        self.previous_n_comp_cmd_rpm = (
            self.m.Param(value=MPC_U_NCOMP_INIT)
            if self.quadratic_comp_move_cost_enabled
            else None
        )
        quadratic_comp_move_mask_values = [0.0] * len(self.m.time)
        if self.quadratic_comp_move_cost_enabled:
            quadratic_comp_move_mask_values[1] = 1.0
        self.quadratic_comp_move_mask = (
            self.m.Param(value=quadratic_comp_move_mask_values)
            if self.quadratic_comp_move_cost_enabled
            else None
        )
        self.j_dcomp_quadratic = (
            self.m.Intermediate(
                self.quadratic_comp_move_mask
                * (
                    (self.u_ncomp - self.previous_n_comp_cmd_rpm)
                    / 1000.0
                )
                ** 2
            )
            if self.quadratic_comp_move_cost_enabled
            else None
        )

        T_batt = self.m.Intermediate(self.T_batt_K - 273.15)
        self.T_batt_control_c = self.m.Intermediate(
            T_batt + self.physics_p_temp_bias_path_c
        )
        T_cool = self.m.Intermediate(self.T_cool_K - 273.15)
        T_plate = self.T_plate
        T_amb = self.m.Intermediate(self.d2_tamb - 273.15)
        if self.evap_input_delay_steps > 0:
            self.m.delay(self.N_comp, self.N_comp_delay, steps=self.evap_input_delay_steps)
            N_comp_delay = self.N_comp_delay
        else:
            N_comp_delay = self.N_comp
        if self.pump_flow_delay_steps > 0:
            if self.predictor_name == PHYSICS_P:
                self.m.delay(
                    self.N_pump - self.N_pump_delay_ref_rpm,
                    self.N_pump_delay_delta_rpm,
                    steps=self.pump_flow_delay_steps,
                )
                delayed_pump_raw_rpm = self.m.Intermediate(
                    self.N_pump_delay_ref_rpm + self.N_pump_delay_delta_rpm
                )
                pump_lower_rpm, pump_upper_rpm = n_pump_state_bounds
                epsilon = 1e-6
                delayed_pump_above_lower_rpm = self.m.Intermediate(
                    0.5
                    * (
                        delayed_pump_raw_rpm
                        + pump_lower_rpm
                        + self.m.sqrt((delayed_pump_raw_rpm - pump_lower_rpm) ** 2 + epsilon)
                    )
                )
                delayed_pump_bounded_rpm = self.m.Intermediate(
                    0.5
                    * (
                        delayed_pump_above_lower_rpm
                        + pump_upper_rpm
                        - self.m.sqrt((delayed_pump_above_lower_rpm - pump_upper_rpm) ** 2 + epsilon)
                    )
                )
                self.m.Equation(self.N_pump_delay == delayed_pump_bounded_rpm)
            else:
                self.m.delay(self.N_pump, self.N_pump_delay, steps=self.pump_flow_delay_steps)
            N_pump_delay = self.N_pump_delay
        else:
            N_pump_delay = self.N_pump
        if self.predictor_name == PHYSICS_P:
            if p["battery_plate_conductance_model"] != "constant_physical":
                raise ValueError(
                    "MPC physics_p requires constant_physical battery-plate conductance"
                )
            H_batt_plate = self.m.Intermediate(p["h1_ref"])
        else:
            H_batt_plate = self.m.Intermediate(p["h1_ref"] * ((N_pump_delay / p["N_pump_ref"]) ** 0.8))
        if self.predictor_name == PHYSICS_P:
            minimum_active = float(
                self.predictor_artifact["capacity"]["minimum_active_rpm"]
            )
            compressor_startup_fraction = _physics_p_startup_fraction_expr(
                self.N_comp, minimum_active, self.m.sqrt
            )
            compressor_active_speed = _physics_p_active_speed_expr(
                self.N_comp, minimum_active, self.m.sqrt
            )
            compressor_active_power = _compressor_power_expr(
                self.m,
                compressor_active_speed,
                T_cool,
                T_amb,
            )
            P_comp = self.m.Intermediate(
                p["compressor_displacement_scale"]
                * compressor_startup_fraction
                * compressor_active_power
            )
        else:
            P_comp = _legacy_compressor_power_expr(self.m, self.N_comp)
        P_pump = _pump_power_speed_expr(self.m, self.N_pump)
        P_comp_max = (
            compressor_power_normalization_w()
            if self.predictor_name == PHYSICS_P
            else LEGACY_COMPRESSOR_POWER_MAX_W
        )
        P_pump_max = _pump_power_speed_value(N_PUMP_MAX_RPM)
        if self.predictor_name == PHYSICS_P:
            q_evap_unbounded_w = self.m.Intermediate(
                physics_p_operating_capacity_expr(
                    N_comp_delay,
                    N_pump_delay,
                    T_cool,
                    T_amb,
                    self.predictor_artifact,
                    sqrt_func=self.m.sqrt,
                )
            )
            q_evap_active_w = self.m.Intermediate(
                smooth_bounded_capacity_expr(
                    q_evap_unbounded_w,
                    self._capacity_upper_w,
                    self.m.sqrt,
                )
            )
        else:
            coeff = self.evaporator_capacity_calibration["coefficients"]
            n_pump_ref = float(self.evaporator_capacity_calibration["n_pump_ref_rpm"])
            n_pump_safe = self.m.Intermediate(N_pump_delay + 1e-6)
            q_evap_active_w = self.m.Intermediate(
                N_comp_delay * (float(coeff["b0"]) + float(coeff["b1"]) * n_pump_ref / n_pump_safe + float(coeff["b2"]) * (T_cool - 25.0))
            )
        q_evap_raw_w = q_evap_active_w
        self.Q_evap_cmd_w = self.m.Var(value=initial_q_evap_w, lb=0.0, ub=self._capacity_upper_w)
        self.m.Equation(self.Q_evap_cmd_w == q_evap_raw_w)
        Q_evap_cmd = self.Q_evap_cmd_w
        C_flow = self.m.Intermediate(p["m_dot_ref"] * p["cp_cool"] * N_pump_delay / p["N_pump_ref"])
        Q_batt_plate = self.m.Intermediate(H_batt_plate * (T_batt - T_plate))
        T_evap_out = self.m.Intermediate(T_cool - self.Q_evap / (C_flow + 1e-6))
        if self.predictor_name == PHYSICS_P:
            pump_ratio = self.m.Intermediate(N_pump_delay / p["N_pump_ref"])
            plate_fluid_flow_limit = C_flow
            plate_fluid_hx_limit = self.m.Intermediate(
                p["m_dot_ref"]
                * p["cp_cool"]
                * (pump_ratio**0.8)
            )
            smooth_min_capacity = self.m.Intermediate(
                0.5
                * (
                    plate_fluid_flow_limit
                    + plate_fluid_hx_limit
                    - self.m.sqrt(
                        (plate_fluid_flow_limit - plate_fluid_hx_limit) ** 2
                        + 1e-6
                    )
                )
            )
            H_plate_fluid = self.m.Intermediate(
                p["plate_fluid_effectiveness"] * smooth_min_capacity
            )
            Q_plate_fluid = self.m.Intermediate(
                H_plate_fluid * (T_plate - self.T_supply)
            )
            T_plate_out = self.m.Intermediate(
                self.T_supply + Q_plate_fluid / (C_flow + 1e-6)
            )
        else:
            Q_plate_fluid = None
            T_plate_out = self.m.Intermediate(self.T_supply + Q_batt_plate / (C_flow + 1e-6))

        self.m.Equation(p["tau_comp_s"] * self.N_comp.dt() == self.u_ncomp - self.N_comp)
        self.m.Equation(p["tau_pump_s"] * self.N_pump.dt() == self.u_npump - self.N_pump)
        self.m.Equation(p["tau_cond_s"] * self.Q_cond.dt() == Q_evap_cmd - self.Q_cond)
        if self.predictor_name == PHYSICS_P and p["evap_response_model"] == "direct":
            self.m.Equation(p["tau_evap_s"] * self.Q_evap.dt() == Q_evap_cmd - self.Q_evap)
        else:
            self.m.Equation(p["tau_evap_s"] * self.Q_evap.dt() == self.Q_cond - self.Q_evap)
        if self.predictor_name == PHYSICS_P:
            self.m.delay(
                T_evap_out - self.T_supply_ref_c,
                self.T_supply_delay_delta_c,
                steps=self.pipe_supply_delay_steps,
            )
        else:
            self.m.delay(T_evap_out, self.T_supply, steps=self.pipe_supply_delay_steps)
        if self.predictor_name == PHYSICS_P:
            self.m.Equation(
                p["C_plate"] * self.T_plate.dt()
                == Q_batt_plate - Q_plate_fluid
            )
        else:
            self.m.Equation(p["tau_plate_s"] * self.T_plate.dt() == self.T_supply - self.T_plate)
        if self.predictor_name == PHYSICS_P:
            self.m.delay(
                T_plate_out - self.T_return_ref_c,
                self.T_return_delay_delta_c,
                steps=self.pipe_return_delay_steps,
            )
        else:
            self.m.delay(T_plate_out, self.T_return, steps=self.pipe_return_delay_steps)
        battery_heat_generation = self.d1_qgen
        if self.predictor_name == PHYSICS_P:
            battery_heat_generation = (
                p["battery_heat_generation_scale"] * self.d1_qgen
            )
        self.m.Equation(
            p["C1"] * self.T_batt_K.dt()
            == battery_heat_generation
            - Q_batt_plate
            - p["h2"] * (T_batt - T_amb)
        )
        self.m.Equation(p["C2"] * self.T_cool_K.dt() == C_flow * (self.T_return - T_cool))
        self.cv_temp = self.m.CV(value=p["T_init_batt"] - 273.15)
        self.cv_temp.STATUS = 1
        self.cv_temp.FSTATUS = 1
        self.m.Equation(self.cv_temp == self.T_batt_control_c)
        self.track_target_c = self.m.Param(value=self.target_temp_c)
        if self.mode == "POINT":
            self.cv_temp.SP = p["T_set"]
            self.cv_temp.SPLO = p["T_set"] - self.mpc_params.cv_band_half_width
            self.cv_temp.SPHI = p["T_set"] + self.mpc_params.cv_band_half_width
            self.cv_temp.WSPLO = self.runtime_WSPLO
            self.cv_temp.WSPHI = self.runtime_WSPHI
            self.cv_temp.TAU = MPC_CV_TAU
            self.cv_temp.TR_INIT = 2
            self.m.options.CV_TYPE = 1

        self.raw_track_error_sq = self.m.Intermediate(
            (self.T_batt_control_c - self.track_target_c) ** 2
        )
        self.raw_sigma_t_sq = self.m.Intermediate(0.0)
        self.raw_w_comp = P_comp
        self.raw_w_pump = P_pump
        self.raw_j_switch = self.m.Intermediate(0.0)
        self.j_track = self.m.Intermediate(self.raw_track_error_sq / (MPC_T_TRACK_SCALE_C**2))
        self.j_spread = self.m.Intermediate(0.0)
        self.j_comp = self.m.Intermediate(self.raw_w_comp / P_comp_max)
        self.j_pump = self.m.Intermediate(self.raw_w_pump / P_pump_max)
        self.j_switch = self.m.Intermediate(0.0)
        terminal_mask_values = [0.0] * len(self.m.time)
        terminal_mask_values[-1] = 1.0
        self.terminal_mask = self.m.Param(value=terminal_mask_values)
        self.terminal_temp_scale_c = max(float(self.mpc_params.terminal_temp_scale_c), 1e-9)
        if self.mpc_params.terminal_cost_type != "empirical_temp":
            raise ValueError(
                f"Unsupported terminal_cost_type: {self.mpc_params.terminal_cost_type!r}. "
                "Only empirical_temp is supported after DARE cleanup."
            )
        self.terminal_cost_enabled = (
            bool(self.mpc_params.terminal_cost_enabled)
            and float(self.mpc_params.w_terminal_temp) > 0.0
        )
        if not bool(self.mpc_params.terminal_cost_enabled):
            if self.mpc_params.terminal_cost_type != "empirical_temp":
                raise AssertionError("terminal_cost_enabled=False requires empirical_temp terminal_cost_type")
            if float(getattr(self.mpc_params, "w_terminal_temp", 0.0)) != 0.0:
                raise AssertionError("terminal_cost_enabled=False requires w_terminal_temp=0")
            print("TerminalBranchActive=False terminal_cost_enabled=False", flush=True)
        # Empirical scalar terminal temperature cost applied only at the prediction horizon end.
        self.e_terminal_path = self.m.Intermediate(
            (self.T_batt_control_c - self.track_target_c)
            / self.terminal_temp_scale_c
        )
        self.j_terminal_path = self.m.Intermediate(self.e_terminal_path ** 2)
        self.weighted_j_terminal_path = self.m.Intermediate(
            self.terminal_mask
            * self._runtime_w_terminal_param
            * self.j_terminal_path
        )
        self.plate_reserve_over_c = self.m.Var(value=0.0, lb=0.0)
        self.tank_reserve_over_c = self.m.Var(value=0.0, lb=0.0)
        self.qevap_reserve_short_w = self.m.Var(value=0.0, lb=0.0)
        self.bat_safety_over_c = self.m.Var(value=0.0, lb=0.0)
        self.m.Equation(self.plate_reserve_over_c >= self.T_supply - MPC_RESERVE_T_PLATE_IN_REF_C)
        self.m.Equation(self.tank_reserve_over_c >= T_cool - MPC_RESERVE_T_TANK_REF_C)
        self.m.Equation(self.qevap_reserve_short_w >= self.d3_qgen_risk * MPC_RESERVE_QEVAP_REF_W - self.Q_evap)
        self.m.Equation(
            self.bat_safety_over_c
            >= self.T_batt_control_c
            - (MPC_T_BAT_MAX_C - MPC_T_BAT_MARGIN_C)
        )
        reserve_enabled = 1.0 if MPC_COOLANT_RESERVE_ENABLED else 0.0
        self.j_plate_reserve = self.m.Intermediate(
            self.d3_qgen_risk * (self.plate_reserve_over_c / MPC_RESERVE_T_SCALE_C) ** 2
        )
        self.j_tank_reserve = self.m.Intermediate(
            self.d3_qgen_risk * (self.tank_reserve_over_c / MPC_RESERVE_T_SCALE_C) ** 2
        )
        self.j_qevap_reserve = self.m.Intermediate(
            self.d3_qgen_risk * (self.qevap_reserve_short_w / MPC_RESERVE_QEVAP_SCALE_W) ** 2
        )
        self.j_bat_safety = self.m.Intermediate((self.bat_safety_over_c / MPC_T_TRACK_SCALE_C) ** 2)
        self.term_mask = self.terminal_mask
        self.term_over = self.m.Var(value=0.0, lb=0.0)
        self.m.Equation(
            self.term_over
            >= self.T_batt_control_c - self.mpc_params.T_term_peak
        )
        self.j_term = self.m.Intermediate((self.term_over / MPC_T_TRACK_SCALE_C) ** 2)
        j_total_expr = (
            self._runtime_w_temp_obj_param * self.j_track
            + self._runtime_w_energy_comp_param * self.j_comp
            + self._runtime_w_energy_pump_param * self.j_pump
        )
        if self.terminal_cost_enabled:
            j_total_expr = j_total_expr + self.weighted_j_terminal_path
        if self.quadratic_comp_move_cost_enabled:
            j_total_expr = (
                j_total_expr
                + self._runtime_w_dcomp_quadratic_param
                * self.j_dcomp_quadratic
            )
        self.j_total = self.m.Intermediate(j_total_expr)
        self.m.Minimize(self.j_total)
        self.m.options.IMODE = 6
        self.m.options.NODES = 2
        self.m.options.SOLVER = 3
        physics_p_comp_mv_step_hor = None
        physics_p_pump_mv_step_hor = None
        if self.predictor_name == PHYSICS_P and not self.strict_predictor_ablation:
            mpc_profile_name = str(self.mpc_params.name).lower()
            if mpc_profile_name.startswith("freq"):
                physics_p_comp_mv_step_hor = PHYSICS_P_COMP_MV_STEP_HOR
                physics_p_pump_mv_step_hor = PHYSICS_P_PUMP_MV_STEP_HOR
            elif mpc_profile_name.startswith("peak"):
                self.m.options.RTOL = PHYSICS_P_PEAK_SOLVER_TOL
                self.m.options.OTOL = PHYSICS_P_PEAK_SOLVER_TOL
        if physics_p_comp_mv_step_hor is not None:
            # The controller still resolves every 5 s and predicts the full
            # horizon. Future moves are blocked only inside the Physics-P NLP;
            # the controller still re-optimizes every 5 s and DMAX is unchanged.
            self.m.options.MV_STEP_HOR = physics_p_comp_mv_step_hor
            self.u_ncomp.MV_STEP_HOR = physics_p_comp_mv_step_hor
            self.u_npump.MV_STEP_HOR = physics_p_pump_mv_step_hor
        if self.strict_predictor_ablation:
            # A strict predictor ablation shares the external optimization
            # contract.  Only the internal prediction equations may differ.
            strict_mv_step_hor = (
                PHYSICS_P_COMP_MV_STEP_HOR
                if str(self.mpc_params.name).lower().startswith("freq")
                else 1
            )
            self.m.options.MV_STEP_HOR = strict_mv_step_hor
            if strict_mv_step_hor > 1:
                self.u_ncomp.MV_STEP_HOR = strict_mv_step_hor
                self.u_npump.MV_STEP_HOR = strict_mv_step_hor
            self.m.options.RTOL = 1.0e-6
            self.m.options.OTOL = 1.0e-6

    def _mean_prediction_value(self, variable, default=np.nan):
        try:
            values = np.array(list(variable.VALUE), dtype=float)
        except Exception:
            return default
        if values.size == 0:
            return default
        if values.size > 1:
            values = values[1:]
        return float(np.mean(values))

    def _update_physics_p_temperature_bias(self, measured_temp_c):
        gain = (
            float(self.mpc_params.physics_p_temp_bias_gain)
            if self.physics_p_temp_bias_enabled
            else 0.0
        )
        update = physics_p_temperature_bias_update(
            previous_step_bias_c=self._physics_p_temp_bias_step_c,
            measured_temp_c=measured_temp_c,
            previous_prediction_1_c=(
                self._physics_p_previous_prediction_1_c
            ),
            horizon_steps=self.np_horizon,
            gain=gain,
        )
        self._physics_p_temp_bias_innovation_c = update["innovation_c"]
        self._physics_p_temp_bias_step_c = update["step_bias_c"]
        self._physics_p_temp_bias_end_c = float(update["path_c"][-1])
        self.physics_p_temp_bias_path_c.VALUE = update["path_c"]
        return update

    def _last_gekko_value(self, obj, name="GEKKO variable"):
        try:
            values = getattr(obj, "value", None)
            if values is None:
                values = getattr(obj, "VALUE", None)
            if values is None:
                raise ValueError("no value/VALUE")
            if hasattr(values, "value"):
                values = values.value
            arr = list(values)
            if len(arr) == 0:
                raise ValueError("empty value")
            value = float(arr[-1])
            if not np.isfinite(value):
                raise ValueError(f"non-finite terminal value {value}")
            return value
        except Exception as exc:
            raise RuntimeError(f"Failed to extract terminal value for {name}: {exc}") from exc


    def _terminal_log_snapshot(self, active_target_temp_c):
        try:
            t_pred_end_c = self._last_gekko_value(
                self.T_batt_control_c,
                name="bias-corrected battery temperature",
            )
        except Exception as exc:
            print(f"WARNING terminal cost log extraction failed: {exc}", flush=True)
            t_pred_end_c = np.nan
        if np.isfinite(t_pred_end_c):
            j_terminal = ((t_pred_end_c - float(active_target_temp_c)) / self.terminal_temp_scale_c) ** 2
        else:
            j_terminal = np.nan
        weighted_j_terminal = (
            self.runtime_w_terminal * j_terminal
            if self.terminal_cost_enabled and np.isfinite(j_terminal)
            else 0.0
        )
        terminal_cost_requested = bool(self.mpc_params.terminal_cost_enabled)
        return {
            "terminal_cost_enabled": terminal_cost_requested,
            "terminal_cost_type": self.mpc_params.terminal_cost_type,
            "w_terminal_temp": self.runtime_w_terminal,
            "terminal_temp_scale_c": float(self.terminal_temp_scale_c),
            "t_pred_end_c": t_pred_end_c,
            "j_terminal": j_terminal,
            "weighted_j_terminal": weighted_j_terminal,
            "Terminal_Cost_Enabled": terminal_cost_requested,
            "Terminal_Cost_Type": self.mpc_params.terminal_cost_type,
            "W_Terminal_Temp": self.runtime_w_terminal,
            "Terminal_Temp_Scale_C": float(self.terminal_temp_scale_c),
            "T_pred_end_C": t_pred_end_c,
            "J_terminal": j_terminal,
            "Weighted_J_terminal": weighted_j_terminal,
        }

    def _objective_snapshot(self, active_target_temp_c=None):
        j_track = self._mean_prediction_value(self.j_track)
        j_spread = self._mean_prediction_value(self.j_spread, default=0.0)
        j_comp = self._mean_prediction_value(self.j_comp)
        j_pump = self._mean_prediction_value(self.j_pump)
        j_switch = self._mean_prediction_value(self.j_switch, default=0.0)
        j_plate_reserve = self._mean_prediction_value(self.j_plate_reserve, default=0.0)
        j_tank_reserve = self._mean_prediction_value(self.j_tank_reserve, default=0.0)
        j_qevap_reserve = self._mean_prediction_value(self.j_qevap_reserve, default=0.0)
        j_bat_safety = self._mean_prediction_value(self.j_bat_safety, default=0.0)
        try:
            j_term = float(np.asarray(self.j_term.VALUE, dtype=float)[-1])
        except Exception:
            j_term = 0.0
        terminal_info = self._terminal_log_snapshot(
            self.target_temp_c if active_target_temp_c is None else active_target_temp_c
        )
        t_pred_end_c = terminal_info["t_pred_end_c"]
        j_terminal = terminal_info["j_terminal"]
        weighted_j_terminal = terminal_info["weighted_j_terminal"]
        reserve_enabled = 1.0 if MPC_COOLANT_RESERVE_ENABLED else 0.0
        j_total = (
            self.runtime_w_temp_obj * j_track
            + self.runtime_w_energy_comp * j_comp
            + self.runtime_w_energy_pump * j_pump
            + weighted_j_terminal
        )
        snapshot = {
            "raw_track_error_sq": self._mean_prediction_value(self.raw_track_error_sq),
            "raw_sigma_t_sq": self._mean_prediction_value(self.raw_sigma_t_sq, default=0.0),
            "raw_w_comp": self._mean_prediction_value(self.raw_w_comp),
            "raw_w_pump": self._mean_prediction_value(self.raw_w_pump),
            "raw_j_switch": self._mean_prediction_value(self.raw_j_switch, default=0.0),
            "j_track": j_track,
            "j_spread": j_spread,
            "j_comp": j_comp,
            "j_pump": j_pump,
            "j_switch": j_switch,
            "j_plate_reserve": j_plate_reserve,
            "j_tank_reserve": j_tank_reserve,
            "j_qevap_reserve": j_qevap_reserve,
            "j_bat_safety": j_bat_safety,
            "j_term": j_term,
            "terminal_cost_enabled": bool(self.mpc_params.terminal_cost_enabled),
            "terminal_cost_type": self.mpc_params.terminal_cost_type,
            "w_terminal_temp": self.runtime_w_terminal,
            "terminal_temp_scale_c": float(self.terminal_temp_scale_c),
            "t_pred_end_c": t_pred_end_c,
            "j_terminal": j_terminal,
            "weighted_j_terminal": weighted_j_terminal,
            "Terminal_Cost_Enabled": terminal_info["Terminal_Cost_Enabled"],
            "Terminal_Cost_Type": terminal_info["Terminal_Cost_Type"],
            "W_Terminal_Temp": terminal_info["W_Terminal_Temp"],
            "Terminal_Temp_Scale_C": terminal_info["Terminal_Temp_Scale_C"],
            "T_pred_end_C": terminal_info["T_pred_end_C"],
            "J_terminal": terminal_info["J_terminal"],
            "Weighted_J_terminal": terminal_info["Weighted_J_terminal"],
            "j_total": j_total,
            "weighted_j_track": self.runtime_w_temp_obj * j_track,
            "weighted_j_spread": self.mpc_params.w_delta_t * j_spread,
            "weighted_j_comp": self.runtime_w_energy_comp * j_comp,
            "weighted_j_pump": self.runtime_w_energy_pump * j_pump,
            "weighted_j_switch": MPC_R_SWITCH * j_switch,
            "weighted_j_plate_reserve": reserve_enabled * MPC_W_PLATE_RESERVE * j_plate_reserve,
            "weighted_j_tank_reserve": reserve_enabled * MPC_W_TANK_RESERVE * j_tank_reserve,
            "weighted_j_qevap_reserve": reserve_enabled * MPC_W_QEVAP_RESERVE * j_qevap_reserve,
            "weighted_j_bat_safety": self.mpc_params.w_bat_safety * j_bat_safety,
            "weighted_j_term": self.mpc_params.w_term_peak * j_term,
            **self._runtime_weight_diagnostics(),
        }
        snapshot.update(self._objective_contribution_snapshot())
        return snapshot

    @staticmethod
    def _prediction_horizon_sum(variable):
        values = np.asarray(list(variable.VALUE), dtype=float)
        if values.size <= 1:
            return 0.0
        return float(np.sum(values[1:]))

    def _objective_contribution_snapshot(self):
        if not self.objective_contribution_diagnostics_enabled:
            return {}

        contribution_keys = (
            "cv",
            "temperature",
            "terminal",
            "comp",
            "pump",
            "dcomp",
            "dpump",
            "other",
        )
        try:
            results_path = Path(self.m.path) / "results.json"
            results = json.loads(results_path.read_text(encoding="utf-8"))
            cv_cost = np.asarray(results[f"{self.cv_temp.name}.cost"], dtype=float)
            cv_error_hi = np.asarray(
                results[f"{self.cv_temp.name}.err_hi"], dtype=float
            )
            cv_error_lo = np.asarray(
                results[f"{self.cv_temp.name}.err_lo"], dtype=float
            )
            cv_raw_hi = float(np.sum(cv_error_hi[1:]))
            cv_raw_lo = float(np.sum(cv_error_lo[1:]))
            compressor_move_values = np.asarray(
                results[self.u_ncomp.name], dtype=float
            )
            pump_move_values = np.asarray(
                results[self.u_npump.name], dtype=float
            )
            terminal_raw = 0.0
            if self.terminal_cost_enabled:
                terminal_values = np.asarray(
                    list(self.j_terminal_path.VALUE), dtype=float
                )
                if terminal_values.size:
                    terminal_raw = float(terminal_values[-1])
            raw_terms = {
                "cv": cv_raw_hi + cv_raw_lo,
                "temperature": self._prediction_horizon_sum(self.j_track),
                "terminal": terminal_raw,
                "compressor": self._prediction_horizon_sum(self.j_comp),
                "pump": self._prediction_horizon_sum(self.j_pump),
                "dcomp": float(
                    np.sum(np.abs(np.diff(compressor_move_values)))
                ),
                "dpump": float(np.sum(np.abs(np.diff(pump_move_values)))),
            }
            normalized_terms = normalize_raw_terms(
                raw_terms, self.objective_normalization_scales
            )
            contributions = {
                "cv": float(
                    self.runtime_WSPHI * cv_raw_hi
                    + self.runtime_WSPLO * cv_raw_lo
                    + np.sum(cv_cost[1:])
                ),
                "temperature": (
                    self.runtime_w_temp_obj * raw_terms["temperature"]
                ),
                "terminal": self._prediction_horizon_sum(
                    self.weighted_j_terminal_path
                ),
                "comp": (
                    self.runtime_w_energy_comp * raw_terms["compressor"]
                ),
                "pump": self.runtime_w_energy_pump * raw_terms["pump"],
                "dcomp": self.runtime_w_dcomp * raw_terms["dcomp"],
                "dpump": self.runtime_w_dpump * raw_terms["dpump"],
            }
            total = float(self.m.options.OBJFCNVAL)
            contributions["other"] = total - sum(contributions.values())
            denominator = total if abs(total) > 1.0e-12 else np.nan
            snapshot = {
                "objective_contributions_valid": True,
                "objective_contributions_error": "",
                "objective_solver_total": total,
                "objective_cv_raw_hi": cv_raw_hi,
                "objective_cv_raw_lo": cv_raw_lo,
            }
            for key in contribution_keys:
                snapshot[f"objective_{key}_contribution"] = contributions[key]
                snapshot[f"objective_{key}_contribution_share"] = (
                    contributions[key] / denominator
                )
            for key in OBJECTIVE_KEYS:
                snapshot[f"objective_{key}_raw"] = raw_terms[key]
                snapshot[f"objective_{key}_scale"] = getattr(
                    self.objective_normalization_scales, key
                )
                snapshot[f"objective_{key}_normalized"] = normalized_terms[key]
            return snapshot
        except Exception as exc:
            snapshot = {
                "objective_contributions_valid": False,
                "objective_contributions_error": str(exc),
                "objective_solver_total": np.nan,
                "objective_cv_raw_hi": np.nan,
                "objective_cv_raw_lo": np.nan,
            }
            for key in contribution_keys:
                snapshot[f"objective_{key}_contribution"] = np.nan
                snapshot[f"objective_{key}_contribution_share"] = np.nan
            for key in OBJECTIVE_KEYS:
                snapshot[f"objective_{key}_raw"] = np.nan
                snapshot[f"objective_{key}_scale"] = getattr(
                    self.objective_normalization_scales, key
                )
                snapshot[f"objective_{key}_normalized"] = np.nan
            return snapshot

    def _runtime_weight_diagnostics(self):
        diagnostics = {
            "alpha_temp": self.alpha_temp,
            "alpha_comp": self.alpha_comp,
            "alpha_pump": self.alpha_pump,
            "runtime_weight_mode": self.runtime_weight_mode,
            "runtime_WSPLO": self.runtime_WSPLO,
            "runtime_WSPHI": self.runtime_WSPHI,
            "runtime_w_terminal": self.runtime_w_terminal,
            "runtime_w_temp_obj": self.runtime_w_temp_obj,
            "runtime_w_energy_comp": self.runtime_w_energy_comp,
            "runtime_w_energy_pump": self.runtime_w_energy_pump,
            "runtime_w_dcomp": self.runtime_w_dcomp,
            "runtime_w_dpump": self.runtime_w_dpump,
            "runtime_w_dcomp_quadratic": self.runtime_w_dcomp_quadratic,
            "runtime_objective_scale_factor": (
                self.runtime_objective_scale_factor
            ),
        }
        for key in OBJECTIVE_KEYS:
            diagnostics[f"objective_{key}_scale"] = getattr(
                self.objective_normalization_scales, key
            )
        return diagnostics

    def set_objective_contribution_diagnostics_enabled(self, enabled=True):
        """Enable read-only extraction of solved objective contributions."""
        self.objective_contribution_diagnostics_enabled = bool(enabled)
        return self

    def _apply_runtime_objective_weights(self):
        scale = self.runtime_objective_scale_factor
        if self.runtime_weight_mode == "legacy_3d":
            wsp_multiplier = self.alpha_temp
            terminal_multiplier = self.alpha_temp
            temp_multiplier = 1.0
            comp_multiplier = self.alpha_comp
            pump_multiplier = self.alpha_pump
        elif self.runtime_weight_mode == "explicit_2d":
            wsp_multiplier = 1.0
            terminal_multiplier = 1.0
            temp_multiplier = self.alpha_temp
            comp_multiplier = self.alpha_comp
            pump_multiplier = 1.0
        else:
            wsp_multiplier = 1.0
            terminal_multiplier = 1.0
            temp_multiplier = 1.0
            comp_multiplier = 1.0
            pump_multiplier = 1.0

        normalization = self.objective_normalization_scales
        self.runtime_WSPLO = (
            float(self.mpc_params.w_cold_temp)
            * wsp_multiplier
            * scale
            / normalization.cv
        )
        self.runtime_WSPHI = (
            float(self.mpc_params.w_high_temp)
            * wsp_multiplier
            * scale
            / normalization.cv
        )
        self.runtime_w_terminal = (
            float(self.mpc_params.w_terminal_temp)
            * terminal_multiplier
            * scale
            / normalization.terminal
        )
        self.runtime_w_temp_obj = (
            float(self.mpc_params.w_temp_obj)
            * temp_multiplier
            * scale
            / normalization.temperature
        )
        self.runtime_w_energy_comp = (
            float(self.mpc_params.w_energy_comp)
            * comp_multiplier
            * scale
            / normalization.compressor
        )
        self.runtime_w_energy_pump = (
            float(self.mpc_params.w_energy_pump)
            * pump_multiplier
            * scale
            / normalization.pump
        )
        self.runtime_w_dcomp = (
            float(self.mpc_params.w_dcomp) * scale / normalization.dcomp
        )
        self.runtime_w_dpump = (
            float(self.mpc_params.w_dpump) * scale / normalization.dpump
        )
        self.runtime_w_dcomp_quadratic = (
            float(self.mpc_params.w_dcomp_quadratic)
            * scale
            / normalization.dcomp
        )

        self._runtime_w_temp_obj_param.VALUE = self.runtime_w_temp_obj
        self._runtime_w_terminal_param.VALUE = self.runtime_w_terminal
        self._runtime_w_energy_comp_param.VALUE = self.runtime_w_energy_comp
        self._runtime_w_energy_pump_param.VALUE = self.runtime_w_energy_pump
        self._runtime_w_dcomp_quadratic_param.VALUE = (
            self.runtime_w_dcomp_quadratic
        )
        self.u_ncomp.DCOST = self.runtime_w_dcomp
        self.u_npump.DCOST = self.runtime_w_dpump
        if self.mode == "POINT":
            self.cv_temp.WSPLO = self.runtime_WSPLO
            self.cv_temp.WSPHI = self.runtime_WSPHI

    def set_runtime_uniform_objective_scale(self, scale_factor=1.0):
        """Multiply every active MPC objective coefficient by one factor."""
        scale_factor = _validated_runtime_weight_multiplier(
            scale_factor, "scale_factor"
        )
        self.runtime_objective_scale_factor = scale_factor
        self._apply_runtime_objective_weights()
        return self

    def set_runtime_objective_normalization(self, scales):
        """Apply fixed positive objective denominators without rebuilding."""
        if not isinstance(scales, ObjectiveNormalizationScales):
            scales = ObjectiveNormalizationScales(**dict(scales))
        self.objective_normalization_scales = scales
        self._apply_runtime_objective_weights()
        return self

    def set_runtime_weight_multipliers(
        self,
        alpha_temp=1.0,
        alpha_comp=1.0,
        alpha_pump=1.0,
    ):
        """Update MPC objective weights in-place without rebuilding the model."""
        alpha_temp = _validated_runtime_weight_multiplier(
            alpha_temp, "alpha_temp"
        )
        alpha_comp = _validated_runtime_weight_multiplier(
            alpha_comp, "alpha_comp"
        )
        alpha_pump = _validated_runtime_weight_multiplier(
            alpha_pump, "alpha_pump"
        )

        self.runtime_weight_mode = "legacy_3d"
        self.alpha_temp = alpha_temp
        self.alpha_comp = alpha_comp
        self.alpha_pump = alpha_pump
        self._apply_runtime_objective_weights()

    def set_runtime_objective_multipliers(
        self,
        alpha_temp=1.0,
        alpha_comp=1.0,
    ):
        """Scale explicit temperature and compressor terms; keep frozen weights fixed."""
        alpha_temp = _validated_runtime_weight_multiplier(
            alpha_temp, "alpha_temp"
        )
        alpha_comp = _validated_runtime_weight_multiplier(
            alpha_comp, "alpha_comp"
        )

        self.runtime_weight_mode = "explicit_2d"
        self.alpha_temp = alpha_temp
        self.alpha_comp = alpha_comp
        self.alpha_pump = 1.0
        self._apply_runtime_objective_weights()
        return self

    def _qgen_risk_window(self, q_window):
        q_values = np.asarray(q_window, dtype=float)
        risk = (q_values - MPC_HEAT_PREVIEW_NOMINAL_W) / max(MPC_HEAT_PREVIEW_NOMINAL_W, 1.0)
        return np.clip(risk, 0.0, 1.0)

    @staticmethod
    def _series_value(values, index, default=np.nan):
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return default
        idx = min(max(0, int(index)), arr.size - 1)
        return float(arr[idx])

    def _seed_physics_p_feasible_trajectory(
        self,
        *,
        t_batt_meas_c,
        t_cool_meas_c,
        n_comp_eff_rpm,
        n_pump_eff_rpm,
        q_cond_w,
        q_evap_w,
        t_supply_c,
        t_plate_c,
        t_return_c,
        n_comp_cmd_rpm,
        n_pump_cmd_rpm,
        q_window_w,
        t_ambient_c,
    ):
        """Seed the first P solve with a physically feasible held-input path.

        GEKKO otherwise expands scalar first-cycle guesses across the complete
        dynamic horizon.  For P's nonlinear coolant and delay equations that
        guess can violate the fitted coolant domain even though the measured
        state and a held-input trajectory are feasible.  This method changes
        only initial values; all MPC equations, bounds, weights, DMAX values,
        and manipulated-variable degrees of freedom remain unchanged.
        """
        if self.predictor_name != PHYSICS_P:
            return None
        q_values = np.asarray(q_window_w, dtype=float).reshape(-1)
        expected_points = self.np_horizon + 1
        if q_values.size < expected_points:
            raise ValueError(
                "physics_p feasible seed requires horizon + 1 heat-load values"
            )

        initial_state = initialize_physics_state(
            n_comp_eff_rpm=n_comp_eff_rpm,
            n_pump_eff_rpm=n_pump_eff_rpm,
            q_cond_w=q_cond_w,
            q_evap_w=q_evap_w,
            t_supply_c=t_supply_c,
            t_plate_c=t_plate_c,
            t_return_c=t_return_c,
            t_batt_c=t_batt_meas_c,
            t_cool_c=t_cool_meas_c,
        )
        states = [initial_state]
        for horizon_index in range(self.np_horizon):
            states.append(
                step_physics_predictor(
                    states[-1],
                    n_comp_cmd_rpm,
                    n_pump_cmd_rpm,
                    q_values[horizon_index],
                    t_ambient_c,
                    self.dt,
                    self.predictor_artifact,
                )
            )

        def state_values(name, offset=0.0):
            return [float(getattr(state, name)) + offset for state in states]

        def delayed_values(name, delay_steps):
            values = state_values(name)
            steps = max(0, int(delay_steps))
            return [values[max(0, index - steps)] for index in range(len(values))]

        t_batt_k = state_values("t_batt_c", 273.15)
        t_cool_k = state_values("t_cool_c", 273.15)
        n_comp = state_values("n_comp_eff_rpm")
        n_pump = state_values("n_pump_eff_rpm")
        q_cond = state_values("q_cond_w")
        q_evap = state_values("q_evap_w")
        t_plate = state_values("t_plate_c")
        t_supply = state_values("t_supply_c")
        t_return = state_values("t_return_c")
        delayed_comp = delayed_values(
            "n_comp_eff_rpm", self.evap_input_delay_steps
        )
        delayed_pump = delayed_values(
            "n_pump_eff_rpm", self.pump_flow_delay_steps
        )
        q_evap_cmd = [
            evaluate_physics_operating_capacity(
                delayed_comp[index],
                delayed_pump[index],
                states[index].t_cool_c,
                t_ambient_c,
                artifact=self.predictor_artifact,
            )
            for index in range(expected_points)
        ]

        self.T_batt_K.VALUE = t_batt_k
        self.T_cool_K.VALUE = t_cool_k
        self.N_comp.VALUE = n_comp
        self.N_pump.VALUE = n_pump
        self.Q_cond.VALUE = q_cond
        self.Q_evap.VALUE = q_evap
        self.T_plate.VALUE = t_plate
        self.N_comp_delay.VALUE = delayed_comp
        self.N_pump_delay.VALUE = delayed_pump
        self.Q_evap_cmd_w.VALUE = q_evap_cmd
        self.T_supply_delay_delta_c.VALUE = [
            value - float(t_supply_c) for value in t_supply
        ]
        self.T_return_delay_delta_c.VALUE = [
            value - float(t_return_c) for value in t_return
        ]
        if self.pump_flow_delay_steps > 0:
            self.N_pump_delay_delta_rpm.VALUE = [
                value - float(n_pump_eff_rpm) for value in delayed_pump
            ]

        return {
            "t_cool_seed_min_c": float(min(state.t_cool_c for state in states)),
            "t_cool_seed_max_c": float(max(state.t_cool_c for state in states)),
            "t_batt_seed_end_c": float(states[-1].t_batt_c),
        }

    def _set_control_initial_guesses(self, n_comp_rpm, n_pump_rpm):
        """Anchor both control guesses to the currently applied commands."""
        self.u_ncomp.VALUE = float(n_comp_rpm)
        self.u_npump.VALUE = float(n_pump_rpm)
        if self.previous_n_comp_cmd_rpm is not None:
            self.previous_n_comp_cmd_rpm.VALUE = float(n_comp_rpm)

    def _solve_with_fixed_control_recovery(
        self,
        *,
        prime_first_physics_p_cycle=False,
        reseed_physics_p_trajectory=None,
    ):
        """Recover a failed cycle while preserving each selected experiment contract."""
        self._last_solve_recovery_used = False
        self._last_solve_recovery_reason = ""
        is_physics_p = getattr(self, "predictor_name", CANDIDATE_B) == PHYSICS_P
        strict_predictor_ablation = bool(
            getattr(self, "strict_predictor_ablation", False)
        )
        use_common_recovery = is_physics_p or strict_predictor_ablation
        disable_time_shift = is_physics_p
        original_solver = getattr(self.m.options, "SOLVER", None)
        original_max_time = getattr(self.m.options, "MAX_TIME", None)
        original_time_shift = getattr(self.m.options, "TIME_SHIFT", None)
        ncomp_status = self.u_ncomp.STATUS
        npump_status = self.u_npump.STATUS
        if disable_time_shift and original_time_shift is not None:
            # Every P cycle is re-anchored to measured plant and delay states.
            # Candidate B instead requires GEKKO's normal horizon roll-forward;
            # forcing TIME_SHIFT=0 changes its closed-loop state estimate.
            self.m.options.TIME_SHIFT = 0
        solve_deadline = (
            time.perf_counter() + PHYSICS_P_SOLVE_CYCLE_MAX_TIME_S
            if use_common_recovery
            else None
        )

        def solve_model():
            if use_common_recovery:
                remaining_s = solve_deadline - time.perf_counter()
                if remaining_s <= 0.0:
                    raise TimeoutError(
                        "MPC cycle exhausted its "
                        f"{PHYSICS_P_SOLVE_CYCLE_MAX_TIME_S:.1f} s solve budget"
                    )
                if original_max_time is not None:
                    remaining_s = min(float(original_max_time), remaining_s)
                self.m.options.MAX_TIME = max(1.0e-3, remaining_s)
            self.m.solve(disp=False)
        def restore_max_time():
            if not use_common_recovery:
                return
            if original_max_time is None:
                try:
                    delattr(self.m.options, "MAX_TIME")
                except AttributeError:
                    pass
            else:
                self.m.options.MAX_TIME = original_max_time

        def restore_time_shift():
            if disable_time_shift and original_time_shift is not None:
                self.m.options.TIME_SHIFT = original_time_shift

        initialization_errors = []
        if is_physics_p and bool(prime_first_physics_p_cycle):
            if original_time_shift is not None:
                self.m.options.TIME_SHIFT = 0
            measured_state_settings = []
            for state_name in (
                "T_batt_K",
                "T_cool_K",
                "N_comp",
                "N_pump",
                "Q_cond",
                "Q_evap",
                "T_plate",
                "T_supply_delay_delta_c",
                "T_return_delay_delta_c",
                "N_pump_delay_delta_rpm",
                "cv_temp",
            ):
                state_variable = getattr(self, state_name, None)
                if state_variable is None or not hasattr(
                    state_variable, "FSTATUS"
                ):
                    continue
                measured_state_settings.append(
                    (
                        state_variable,
                        state_variable.FSTATUS,
                        getattr(state_variable, "MEAS", None),
                    )
                )
                state_variable.FSTATUS = 0
            self.u_ncomp.STATUS = 0
            self.u_npump.STATUS = 0
            try:
                # Build a feasible delay and thermal trajectory before the
                # first nonlinear control optimization. This is one physical
                # cycle, so the following solve must not shift time again.
                solve_model()
            except Exception as initialization_exc:
                initialization_error = (
                    f"{type(initialization_exc).__name__}: {initialization_exc}"
                )
                initialization_errors.append(
                    f"first-cycle trajectory initialization failed ({initialization_error})"
                )
            finally:
                self.u_ncomp.STATUS = ncomp_status
                self.u_npump.STATUS = npump_status
                for state_variable, fstatus, measurement in measured_state_settings:
                    state_variable.FSTATUS = fstatus
                    if measurement is not None:
                        state_variable.MEAS = measurement

            if original_time_shift is not None:
                self.m.options.TIME_SHIFT = 0
        try:
            solve_model()
            restore_max_time()
            restore_time_shift()
            if initialization_errors:
                initialization_reason = "; ".join(initialization_errors)
                self._last_solve_recovery_used = True
                self._last_solve_recovery_reason = initialization_reason
                return True, initialization_reason
            return False, ""
        except Exception as initial_exc:
            initial_error = f"{type(initial_exc).__name__}: {initial_exc}"
            if disable_time_shift and original_time_shift is not None:
                self.m.options.TIME_SHIFT = 0

        recovery_errors = initialization_errors + [f"initial solve failed ({initial_error})"]
        self._last_solve_recovery_used = True
        self._last_solve_recovery_reason = "; ".join(recovery_errors)
        alternate_solver_enabled = False
        ncomp_status = self.u_ncomp.STATUS
        npump_status = self.u_npump.STATUS
        try:
            if use_common_recovery:
                try:
                    # IPOPT can occasionally reject one warm-started P cycle
                    # while an identical immediate retry converges.
                    solve_model()
                    return True, "; ".join(recovery_errors)
                except Exception as retry_exc:
                    retry_error = f"{type(retry_exc).__name__}: {retry_exc}"
                    recovery_errors.append(
                        f"normal retry failed ({retry_error})"
                    )

                if original_solver is not None and int(original_solver) != 1:
                    self.m.options.SOLVER = 1
                    alternate_solver_enabled = True
                    try:
                        # APOPT is slower for this model but is a useful
                        # numerical fallback after repeated IPOPT failures.
                        solve_model()
                        return True, "; ".join(recovery_errors)
                    except Exception as alternate_exc:
                        alternate_error = (
                            f"{type(alternate_exc).__name__}: {alternate_exc}"
                        )
                        recovery_errors.append(
                            f"alternate solver failed ({alternate_error})"
                        )

                    finally:
                        # Fixed-control recovery is formulated for the original
                        # continuous solver. Leaving APOPT active here made one
                        # failed frequency cycle take minutes.
                        self.m.options.SOLVER = original_solver
                        alternate_solver_enabled = False

                if reseed_physics_p_trajectory is not None:
                    measured_state_settings = []
                    try:
                        reseed_physics_p_trajectory()
                        recovery_errors.append(
                            "physics-p observed-state trajectory reseed"
                        )
                        for state_name in (
                            "T_batt_K",
                            "T_cool_K",
                            "N_comp",
                            "N_pump",
                            "Q_cond",
                            "Q_evap",
                            "T_plate",
                            "T_supply_delay_delta_c",
                            "T_return_delay_delta_c",
                            "N_pump_delay_delta_rpm",
                            "cv_temp",
                        ):
                            state_variable = getattr(self, state_name, None)
                            if state_variable is None or not hasattr(
                                state_variable, "FSTATUS"
                            ):
                                continue
                            measured_state_settings.append(
                                (
                                    state_variable,
                                    state_variable.FSTATUS,
                                    getattr(state_variable, "MEAS", None),
                                )
                            )
                            state_variable.FSTATUS = 0
                        self.u_ncomp.STATUS = 0
                        self.u_npump.STATUS = 0
                        solve_model()
                        self.u_ncomp.STATUS = ncomp_status
                        self.u_npump.STATUS = npump_status
                        for state_variable, fstatus, measurement in (
                            measured_state_settings
                        ):
                            state_variable.FSTATUS = fstatus
                            if measurement is not None:
                                state_variable.MEAS = measurement
                        solve_model()
                        return True, "; ".join(recovery_errors)
                    except Exception as reseed_exc:
                        reseed_error = (
                            f"{type(reseed_exc).__name__}: {reseed_exc}"
                        )
                        recovery_errors.append(
                            "physics-p observed-state trajectory reseed failed "
                            f"({reseed_error})"
                        )
                    finally:
                        self.u_ncomp.STATUS = ncomp_status
                        self.u_npump.STATUS = npump_status
                        for state_variable, fstatus, measurement in (
                            measured_state_settings
                        ):
                            state_variable.FSTATUS = fstatus
                            if measurement is not None:
                                state_variable.MEAS = measurement
            # The model has explicit terminal slack variables, so GEKKO's
            # COLDSTART=1 square-model mode is not applicable.  Freeze only
            # the manipulated variables to rebuild a state trajectory, then
            # restore the controller degrees of freedom and optimize again.
            self.u_ncomp.STATUS = 0
            self.u_npump.STATUS = 0
            solve_model()
            self.u_ncomp.STATUS = ncomp_status
            self.u_npump.STATUS = npump_status
            solve_model()
            return True, "; ".join(recovery_errors)
        except Exception as recovery_exc:
            recovery_error = f"{type(recovery_exc).__name__}: {recovery_exc}"
            recovery_errors.append(f"fixed-control recovery failed ({recovery_error})")
            self._last_solve_recovery_used = True
            self._last_solve_recovery_reason = "; ".join(recovery_errors)
            raise RuntimeError(self._last_solve_recovery_reason) from recovery_exc

        finally:
            self.u_ncomp.STATUS = ncomp_status
            self.u_npump.STATUS = npump_status
            if alternate_solver_enabled:
                self.m.options.SOLVER = original_solver
            restore_max_time()
            restore_time_shift()

    def solve_step(
        self,
        i,
        t_batt_meas,
        t_cool_meas,
        qgen_forecast,
        T_amburrent,
        u_ncomp_value=MPC_U_NCOMP_INIT,
        u_npump_value=MPC_U_NPUMP_INIT,
        target_temp_c=None,
        observed_thermal_state=None,
        t_plate_meas=None,
        _allow_physics_p_solver_rebuild=True,
    ):
        active_target_temp_c = self.target_temp_c if target_temp_c is None else float(target_temp_c)
        if self.mode == "POINT":
            self.cv_temp.SP = active_target_temp_c
            self.cv_temp.SPLO = active_target_temp_c - self.mpc_params.cv_band_half_width
            self.cv_temp.SPHI = active_target_temp_c + self.mpc_params.cv_band_half_width
            self.cv_temp.WSPLO = self.runtime_WSPLO
            self.cv_temp.WSPHI = self.runtime_WSPHI
            self.track_target_c.VALUE = active_target_temp_c
        self.cv_temp.MEAS = t_batt_meas - 273.15
        self.T_batt_K.MEAS = t_batt_meas
        self.T_cool_K.MEAS = t_cool_meas
        supply_meas_c = float(t_cool_meas) - 273.15
        return_meas_c = supply_meas_c
        if observed_thermal_state:
            if "N_comp_eff" in observed_thermal_state:
                self.N_comp.MEAS = float(observed_thermal_state["N_comp_eff"])
            if "N_pump_eff" in observed_thermal_state:
                self.N_pump.MEAS = float(observed_thermal_state["N_pump_eff"])
            if "Q_evap_eff" in observed_thermal_state:
                self.Q_evap.MEAS = float(observed_thermal_state["Q_evap_eff"])
            if "Q_cond_eff" in observed_thermal_state:
                self.Q_cond.MEAS = float(observed_thermal_state["Q_cond_eff"])
            if "T_pipe_supply_K" in observed_thermal_state:
                supply_meas_c = (
                    float(observed_thermal_state["T_pipe_supply_K"]) - 273.15
                )
            if "T_pipe_return_K" in observed_thermal_state:
                return_meas_c = (
                    float(observed_thermal_state["T_pipe_return_K"]) - 273.15
                )
        if t_plate_meas is not None:
            self.T_plate.MEAS = float(np.mean(t_plate_meas) - 273.15)
        if self.predictor_name == PHYSICS_P:
            pump_meas_rpm = float(u_npump_value)
            if observed_thermal_state:
                pump_meas_rpm = float(
                    observed_thermal_state.get("N_pump_eff", pump_meas_rpm)
                )
            if self.pump_flow_delay_steps > 0:
                self.N_pump_delay_ref_rpm.VALUE = pump_meas_rpm
                self.N_pump_delay_delta_rpm.MEAS = 0.0
            self.T_supply_ref_c.VALUE = supply_meas_c
            self.T_return_ref_c.VALUE = return_meas_c
            self.T_supply_delay_delta_c.MEAS = 0.0
            self.T_return_delay_delta_c.MEAS = 0.0
            if not self._physics_p_state_initialized:
                # GEKKO applies SV measurements only after the first cycle.
                # Seed the first trajectory explicitly from the observed plant
                # state; subsequent cycles retain the normal warm start.
                self.T_batt_K.VALUE = float(t_batt_meas)
                self.T_cool_K.VALUE = float(t_cool_meas)
                self.cv_temp.VALUE = float(t_batt_meas) - 273.15
                if self.pump_flow_delay_steps > 0:
                    self.N_pump_delay_delta_rpm.VALUE = 0.0
                    self.N_pump_delay.VALUE = pump_meas_rpm
                if observed_thermal_state:
                    self.N_comp.VALUE = float(
                        observed_thermal_state.get("N_comp_eff", u_ncomp_value)
                    )
                    self.N_pump.VALUE = float(
                        observed_thermal_state.get("N_pump_eff", u_npump_value)
                    )
                    self.Q_evap.VALUE = float(
                        observed_thermal_state.get("Q_evap_eff", 0.0)
                    )
                    self.Q_cond.VALUE = float(
                        observed_thermal_state.get("Q_cond_eff", 0.0)
                    )
                if t_plate_meas is not None:
                    self.T_plate.VALUE = float(np.mean(t_plate_meas) - 273.15)
                self.T_supply_delay_delta_c.VALUE = 0.0
                self.T_return_delay_delta_c.VALUE = 0.0
        self._set_control_initial_guesses(u_ncomp_value, u_npump_value)
        self.d2_tamb.VALUE = np.full(self.np_horizon + 1, T_amburrent)
        q_window = [
            qgen_forecast[i + k] if i + k < len(qgen_forecast) else qgen_forecast[-1]
            for k in range(self.np_horizon + 1)
        ]
        self.d1_qgen.VALUE = q_window
        self.d3_qgen_risk.VALUE = self._qgen_risk_window(q_window)
        physics_p_bias_state_before_solve = None
        if self.predictor_name == PHYSICS_P:
            physics_p_bias_state_before_solve = {
                "_physics_p_previous_prediction_1_c": (
                    self._physics_p_previous_prediction_1_c
                ),
                "_physics_p_temp_bias_step_c": self._physics_p_temp_bias_step_c,
                "_physics_p_temp_bias_innovation_c": (
                    self._physics_p_temp_bias_innovation_c
                ),
                "_physics_p_temp_bias_end_c": self._physics_p_temp_bias_end_c,
            }
        temperature_bias_info = self._update_physics_p_temperature_bias(
            float(t_batt_meas) - 273.15
        )
        physics_p_seed_info = None
        reseed_physics_p_trajectory = None
        if self.predictor_name == PHYSICS_P:
            observed = observed_thermal_state or {}
            plate_meas_c = (
                float(np.mean(t_plate_meas) - 273.15)
                if t_plate_meas is not None
                else float(t_cool_meas) - 273.15
            )
            def reseed_physics_p_trajectory():
                self._set_control_initial_guesses(
                    u_ncomp_value,
                    u_npump_value,
                )
                seed_info = self._seed_physics_p_feasible_trajectory(
                    t_batt_meas_c=float(t_batt_meas) - 273.15,
                    t_cool_meas_c=float(t_cool_meas) - 273.15,
                    n_comp_eff_rpm=float(
                        observed.get("N_comp_eff", u_ncomp_value)
                    ),
                    n_pump_eff_rpm=float(
                        observed.get("N_pump_eff", u_npump_value)
                    ),
                    q_cond_w=float(observed.get("Q_cond_eff", 0.0)),
                    q_evap_w=float(observed.get("Q_evap_eff", 0.0)),
                    t_supply_c=supply_meas_c,
                    t_plate_c=plate_meas_c,
                    t_return_c=return_meas_c,
                    n_comp_cmd_rpm=float(u_ncomp_value),
                    n_pump_cmd_rpm=float(u_npump_value),
                    q_window_w=q_window,
                    t_ambient_c=float(T_amburrent) - 273.15,
                )
                # Assigning full SV trajectories resets measurements in GEKKO.
                # Reapply the observed plant state before the recovery solve.
                self.cv_temp.MEAS = float(t_batt_meas) - 273.15
                self.T_batt_K.MEAS = float(t_batt_meas)
                self.T_cool_K.MEAS = float(t_cool_meas)
                self.N_comp.MEAS = float(
                    observed.get("N_comp_eff", u_ncomp_value)
                )
                self.N_pump.MEAS = float(
                    observed.get("N_pump_eff", u_npump_value)
                )
                self.Q_cond.MEAS = float(observed.get("Q_cond_eff", 0.0))
                self.Q_evap.MEAS = float(observed.get("Q_evap_eff", 0.0))
                self.T_plate.MEAS = plate_meas_c
                self.T_supply_delay_delta_c.MEAS = 0.0
                self.T_return_delay_delta_c.MEAS = 0.0
                if self.pump_flow_delay_steps > 0:
                    self.N_pump_delay_delta_rpm.MEAS = 0.0
                return seed_info

            if not self._physics_p_state_initialized:
                physics_p_seed_info = reseed_physics_p_trajectory()
        solve_start = time.perf_counter()
        prediction_domain_info = None
        solve_recovery_used = False
        solve_recovery_reason = ""
        try:
            solve_recovery_used, solve_recovery_reason = (
                self._solve_with_fixed_control_recovery(
                    prime_first_physics_p_cycle=physics_p_seed_info is not None,
                    reseed_physics_p_trajectory=reseed_physics_p_trajectory,
                )
            )
            solve_time_s = time.perf_counter() - solve_start
            n_comp_command = float(self.u_ncomp.NEWVAL)
            if self.predictor_name == PHYSICS_P:
                n_comp_command = normalize_physics_p_compressor_command(
                    n_comp_command
                )
            t_batt_raw_pred_c = (
                np.array(list(self.T_batt_K.VALUE), dtype=float) - 273.15
            )
            temperature_bias_path_c = np.asarray(
                temperature_bias_info["path_c"],
                dtype=float,
            )
            t_batt_control_pred_c = (
                t_batt_raw_pred_c + temperature_bias_path_c
            )
            result = {
                "n_comp": n_comp_command,
                "n_pump": float(self.u_npump.NEWVAL),
                "t_batt_pred_c": t_batt_control_pred_c,
                "t_batt_raw_pred_c": t_batt_raw_pred_c,
                "t_cool_pred_c": np.array(list(self.T_cool_K.VALUE), dtype=float) - 273.15,
                "flow_direction": self.flow_direction,
                "target_temp_c": active_target_temp_c,
                "solve_time_s": float(solve_time_s),
                "solved": True,
                "solve_error": "",
                "solver_model_path": str(self.m.path),
                "solve_recovery_used": bool(solve_recovery_used),
                "solve_recovery_reason": solve_recovery_reason,
                "physics_p_temp_bias_enabled": bool(
                    self.physics_p_temp_bias_enabled
                ),
                "physics_p_temp_bias_innovation_c": float(
                    temperature_bias_info["innovation_c"]
                ),
                "physics_p_temp_bias_step_c": float(
                    temperature_bias_info["step_bias_c"]
                ),
                "physics_p_temp_bias_end_c": float(
                    temperature_bias_path_c[-1]
                ),
            }
            result["n_comp_plan_rpm"] = _planned_control_values(
                self.u_ncomp.VALUE,
                result["n_comp"],
            )
            result["n_pump_plan_rpm"] = _planned_control_values(
                self.u_npump.VALUE,
                result["n_pump"],
            )
            t_batt_pred_c = result["t_batt_pred_c"]
            t_cool_pred_c = result["t_cool_pred_c"]
            if self.predictor_name == PHYSICS_P:
                prediction_domain_info = physics_p_prediction_domain_info(
                    t_cool_pred_c,
                    self.predictor_artifact,
                )
                result.update(prediction_domain_info)
                if not prediction_domain_info["prediction_domain_valid"]:
                    raise RuntimeError(
                        "physics_p coolant prediction leaves validated domain "
                        f"[{prediction_domain_info['t_cool_domain_lower_c']:.3f}, "
                        f"{prediction_domain_info['t_cool_domain_upper_c']:.3f}] C: "
                        f"trajectory range "
                        f"[{prediction_domain_info['t_cool_pred_min_c']:.3f}, "
                        f"{prediction_domain_info['t_cool_pred_max_c']:.3f}] C"
                    )
                self._physics_p_state_initialized = True
                self._physics_p_previous_prediction_1_c = self._series_value(
                    t_batt_pred_c,
                    1,
                )
            t_plate_pred_c = np.array(list(self.T_plate.VALUE), dtype=float)
            qevap_pred_w = np.array(list(self.Q_evap.VALUE), dtype=float)
            qcond_pred_w = np.array(list(self.Q_cond.VALUE), dtype=float)
            qevap_cmd_pred_w = np.array(list(self.Q_evap_cmd_w.VALUE), dtype=float)
            result.update(
                {
                    "t_batt_pred_1_c": self._series_value(t_batt_pred_c, 1),
                    "t_batt_pred_5_c": self._series_value(t_batt_pred_c, 5),
                    "t_batt_pred_10_c": self._series_value(t_batt_pred_c, 10),
                    "t_batt_pred_20_c": self._series_value(t_batt_pred_c, 20),
                    "t_batt_pred_end_c": self._series_value(t_batt_pred_c, len(t_batt_pred_c) - 1),
                    "t_cool_pred_end_c": self._series_value(t_cool_pred_c, len(t_cool_pred_c) - 1),
                    "t_plate_pred_end_c": self._series_value(t_plate_pred_c, len(t_plate_pred_c) - 1),
                    "qevap_eff_pred_mean_w": float(np.mean(qevap_pred_w)) if qevap_pred_w.size else np.nan,
                    "qevap_cmd_pred_1_w": self._series_value(qevap_cmd_pred_w, 1),
                    "qcond_pred_1_w": self._series_value(qcond_pred_w, 1),
                    "qevap_pred_1_w": self._series_value(qevap_pred_w, 1),
                }
            )
            result.update(self._objective_snapshot(active_target_temp_c))
            return result
        except Exception as exc:
            if (
                self.predictor_name == PHYSICS_P
                and not self.strict_predictor_ablation
                and bool(_allow_physics_p_solver_rebuild)
                and prediction_domain_info is None
            ):
                exhausted_reason = str(
                    solve_recovery_reason
                    or getattr(self, "_last_solve_recovery_reason", "")
                    or f"{type(exc).__name__}: {exc}"
                )
                rebuild_config = {
                    "params": dict(
                        getattr(self, "_construction_params", self.params)
                    ),
                    "dt": self.dt,
                    "np_horizon": self.np_horizon,
                    "mode": self.mode,
                    "target_temp_c": self.target_temp_c,
                    "flow_direction": self.flow_direction,
                    "mpc_params": self.mpc_params,
                    "case_name": self.case_name,
                    "predictor": self.predictor_name,
                    "predictor_artifact": self.predictor_artifact,
                    "strict_predictor_ablation": self.strict_predictor_ablation,
                }
                runtime_weight_multipliers = (
                    self.alpha_temp,
                    self.alpha_comp,
                    self.alpha_pump,
                )
                runtime_weight_mode = self.runtime_weight_mode
                runtime_objective_scale_factor = (
                    self.runtime_objective_scale_factor
                )
                objective_normalization_scales = (
                    self.objective_normalization_scales
                )
                objective_diagnostics_enabled = (
                    self.objective_contribution_diagnostics_enabled
                )
                self.__init__(**rebuild_config)
                if runtime_weight_mode == "explicit_2d":
                    self.set_runtime_objective_multipliers(
                        *runtime_weight_multipliers[:2]
                    )
                elif runtime_weight_mode == "legacy_3d":
                    self.set_runtime_weight_multipliers(
                        *runtime_weight_multipliers
                    )
                self.set_runtime_uniform_objective_scale(
                    runtime_objective_scale_factor
                )
                self.set_runtime_objective_normalization(
                    objective_normalization_scales
                )
                self.set_objective_contribution_diagnostics_enabled(
                    objective_diagnostics_enabled
                )
                for state_name, state_value in (
                    physics_p_bias_state_before_solve or {}
                ).items():
                    setattr(self, state_name, state_value)
                rebuilt_result = self.solve_step(
                    i,
                    t_batt_meas,
                    t_cool_meas,
                    qgen_forecast,
                    T_amburrent,
                    u_ncomp_value=u_ncomp_value,
                    u_npump_value=u_npump_value,
                    target_temp_c=target_temp_c,
                    observed_thermal_state=observed_thermal_state,
                    t_plate_meas=t_plate_meas,
                    _allow_physics_p_solver_rebuild=False,
                )
                rebuild_reason = (
                    "physics-p solver model rebuilt from observed state"
                )
                rebuilt_reason = str(
                    rebuilt_result.get("solve_recovery_reason", "")
                )
                rebuilt_result["solve_recovery_used"] = True
                rebuilt_result["solve_recovery_reason"] = "; ".join(
                    reason
                    for reason in (
                        exhausted_reason,
                        rebuild_reason,
                        rebuilt_reason,
                    )
                    if reason
                )
                rebuilt_result["solve_time_s"] = float(
                    time.perf_counter() - solve_start
                )
                if not rebuilt_result.get("solved", False):
                    rebuilt_result["solve_error"] = (
                        f"original solver instance failed ({type(exc).__name__}: "
                        f"{exc}); rebuilt solver instance failed "
                        f"({rebuilt_result.get('solve_error', '')})"
                    )
                return rebuilt_result
            if not solve_recovery_used:
                solve_recovery_used = bool(
                    getattr(self, "_last_solve_recovery_used", False)
                )
            if not solve_recovery_reason:
                solve_recovery_reason = str(
                    getattr(self, "_last_solve_recovery_reason", "")
                )
            solve_time_s = time.perf_counter() - solve_start
            fallback_n_comp = float(self.mpc_params.n_comp_min)
            fallback_n_pump = float(self.mpc_params.n_pump_min)
            if (
                self.predictor_name == PHYSICS_P
                and not self.strict_predictor_ablation
            ):
                fallback_n_comp = float(
                    np.clip(u_ncomp_value, N_COMP_OFF_RPM, N_COMP_MAX_RPM)
                )
                fallback_n_pump = float(
                    np.clip(u_npump_value, N_PUMP_MIN_RPM, N_PUMP_MAX_RPM)
                )
                predicted_domain_under = bool(
                    prediction_domain_info is not None
                    and prediction_domain_info["t_cool_pred_min_c"]
                    < prediction_domain_info["t_cool_domain_lower_c"]
                )
                measured_coolant_floor_c = min(
                    float(t_cool_meas) - 273.15,
                    float(supply_meas_c),
                    float(return_meas_c),
                )
                coolant_domain_lower_c = float(
                    self.predictor_artifact["input_domain"]["t_cool_c"][0]
                )
                measured_coolant_at_domain_floor = bool(
                    measured_coolant_floor_c <= coolant_domain_lower_c
                )
                measured_battery_not_warm = bool(
                    t_batt_meas - 273.15 <= active_target_temp_c
                )
                if (
                    predicted_domain_under
                    or measured_coolant_at_domain_floor
                    or measured_battery_not_warm
                ):
                    fallback_n_comp = float(N_COMP_OFF_RPM)
                    fallback_n_pump = float(self.mpc_params.n_pump_min)
            fallback = {
                "n_comp": fallback_n_comp,
                "n_pump": fallback_n_pump,
                "t_batt_pred_c": np.array([t_batt_meas - 273.15], dtype=float),
                "t_batt_raw_pred_c": np.array(
                    [t_batt_meas - 273.15],
                    dtype=float,
                ),
                "t_cool_pred_c": np.array([t_cool_meas - 273.15], dtype=float),
                "flow_direction": self.flow_direction,
                "target_temp_c": active_target_temp_c,
                "solve_time_s": float(solve_time_s),
                "solved": False,
                "solve_error": f"{type(exc).__name__}: {exc}",
                "solver_model_path": str(self.m.path),
                "solve_recovery_used": bool(solve_recovery_used),
                "solve_recovery_reason": solve_recovery_reason,
                "physics_p_temp_bias_enabled": bool(
                    self.physics_p_temp_bias_enabled
                ),
                "physics_p_temp_bias_innovation_c": float(
                    temperature_bias_info["innovation_c"]
                ),
                "physics_p_temp_bias_step_c": float(
                    temperature_bias_info["step_bias_c"]
                ),
                "physics_p_temp_bias_end_c": float(
                    temperature_bias_info["path_c"][-1]
                ),
                "n_comp_plan_rpm": [fallback_n_comp],
                "n_pump_plan_rpm": [fallback_n_pump],
                "raw_track_error_sq": np.nan,
                "raw_sigma_t_sq": np.nan,
                "raw_w_comp": np.nan,
                "raw_w_pump": np.nan,
                "raw_j_switch": 0.0,
                "j_track": np.nan,
                "j_spread": 0.0,
                "j_comp": np.nan,
                "j_pump": np.nan,
                "j_switch": 0.0,
                "j_total": np.nan,
                "weighted_j_track": np.nan,
                "weighted_j_spread": 0.0,
                "weighted_j_comp": np.nan,
                "weighted_j_pump": np.nan,
                "weighted_j_switch": 0.0,
                "t_batt_pred_1_c": t_batt_meas - 273.15,
                "t_batt_pred_5_c": t_batt_meas - 273.15,
                "t_batt_pred_10_c": t_batt_meas - 273.15,
                "t_batt_pred_20_c": t_batt_meas - 273.15,
                "t_batt_pred_end_c": t_batt_meas - 273.15,
                "t_cool_pred_end_c": t_cool_meas - 273.15,
                "t_plate_pred_end_c": np.nan,
                "qevap_eff_pred_mean_w": np.nan,
                "terminal_cost_enabled": bool(self.mpc_params.terminal_cost_enabled),
                "terminal_cost_type": self.mpc_params.terminal_cost_type,
                "w_terminal_temp": self.runtime_w_terminal,
                "terminal_temp_scale_c": float(self.terminal_temp_scale_c),
                "t_pred_end_c": t_batt_meas - 273.15,
                "j_terminal": np.nan,
                "weighted_j_terminal": 0.0,
                "Terminal_Cost_Enabled": bool(self.mpc_params.terminal_cost_enabled),
                "Terminal_Cost_Type": self.mpc_params.terminal_cost_type,
                "W_Terminal_Temp": self.runtime_w_terminal,
                "Terminal_Temp_Scale_C": float(self.terminal_temp_scale_c),
                "T_pred_end_C": t_batt_meas - 273.15,
                "J_terminal": np.nan,
                "Weighted_J_terminal": 0.0,
            }
            if prediction_domain_info is not None:
                fallback.update(prediction_domain_info)
            fallback.update(self._runtime_weight_diagnostics())
            self._physics_p_previous_prediction_1_c = np.nan
            return fallback


class BaseFlowMPCController:
    owns_flow_direction = False
    flow_mode = "switching"

    def __init__(
        self,
        current_profile,
        dt=SIM_DT,
        target_temp_c=TARGET_TEMP_C,
        case_name=None,
        mpc_params=None,
        predictor=CANDIDATE_B,
        predictor_artifact=None,
        strict_predictor_ablation=False,
    ):
        self.dt = dt
        self.target_temp_c = target_temp_c
        self.case_name = case_name
        self.mpc_params = mpc_params or select_mpc_params_for_scene(case_name)
        self.predictor_name = normalize_predictor_name(predictor)
        self.predictor_artifact = (
            _validated_physics_p_artifact(predictor_artifact)
            if self.predictor_name == PHYSICS_P
            else None
        )
        self.strict_predictor_ablation = bool(strict_predictor_ablation)
        self.qgen_forecast = [((current / 4.0) ** 2) * 0.001 * 52 for current in current_profile]
        self.heat_preview_steps = max(1, int(round(MPC_HEAT_PREVIEW_WINDOW_S / self.dt)))
        params = {
            "C1": 52 * 4747.0,
            "C2": C_tank,
            "h1_ref": 52 * 10.0,
            "N_pump_ref": 2000.0,
            "h2": 30 * 0.1,
            "kq": 1.0,
            "cp_cool": cp_cool,
            "m_dot_ref": m_dot_nominal,
            "T_env": 298.15,
            "T_set": target_temp_c,
            "T_init_batt": INITIAL_TEMP_C + 273.15,
            "T_init_cool": INITIAL_TEMP_C + 273.15,
            "N_comp_min": self.mpc_params.n_comp_min,
            "N_pump_min": self.mpc_params.n_pump_min,
        }
        self.reduced_model_calibration_theta = load_reduced_model_calibration()
        params = apply_reduced_model_calibration(params, self.reduced_model_calibration_theta)
        if self.predictor_name == PHYSICS_P:
            params = physics_p_params_from_artifact(
                params,
                self.predictor_artifact,
            )
        self.forward_model = MPCControllerDual(
            params,
            dt,
            np_horizon=self.mpc_params.mpc_horizon,
            mode="POINT",
            target_temp_c=target_temp_c,
            flow_direction=1,
            mpc_params=self.mpc_params,
            case_name=case_name,
            predictor=self.predictor_name,
            predictor_artifact=self.predictor_artifact,
            strict_predictor_ablation=self.strict_predictor_ablation,
        )
        self.reverse_model = MPCControllerDual(
            params,
            dt,
            np_horizon=self.mpc_params.mpc_horizon,
            mode="POINT",
            target_temp_c=target_temp_c,
            flow_direction=-1,
            mpc_params=self.mpc_params,
            case_name=case_name,
            predictor=self.predictor_name,
            predictor_artifact=self.predictor_artifact,
            strict_predictor_ablation=self.strict_predictor_ablation,
        )
        self.alpha_temp = 1.0
        self.alpha_comp = 1.0
        self.alpha_pump = 1.0
        self.runtime_objective_scale_factor = 1.0
        self.objective_normalization_scales = ObjectiveNormalizationScales()
        self.direction = 1
        self.last_n_comp = MPC_U_NCOMP_INIT
        self.last_n_pump = MPC_U_NPUMP_INIT
        self.last_flow_info = {
            "mode": self.flow_mode,
            "direction": self.direction,
            "delta_t_pred_max": np.nan,
            "switched": False,
        }

    def set_runtime_weight_multipliers(
        self,
        alpha_temp=1.0,
        alpha_comp=1.0,
        alpha_pump=1.0,
    ):
        """Apply one validated multiplier triplet to both flow models."""
        multipliers = (
            _validated_runtime_weight_multiplier(alpha_temp, "alpha_temp"),
            _validated_runtime_weight_multiplier(alpha_comp, "alpha_comp"),
            _validated_runtime_weight_multiplier(alpha_pump, "alpha_pump"),
        )
        self.forward_model.set_runtime_weight_multipliers(*multipliers)
        self.reverse_model.set_runtime_weight_multipliers(*multipliers)
        self.alpha_temp, self.alpha_comp, self.alpha_pump = multipliers
        return self

    def set_runtime_objective_multipliers(
        self,
        alpha_temp=1.0,
        alpha_comp=1.0,
    ):
        """Apply the two-dimensional explicit objective scaling to both models."""
        multipliers = (
            _validated_runtime_weight_multiplier(alpha_temp, "alpha_temp"),
            _validated_runtime_weight_multiplier(alpha_comp, "alpha_comp"),
        )
        self.forward_model.set_runtime_objective_multipliers(*multipliers)
        self.reverse_model.set_runtime_objective_multipliers(*multipliers)
        self.alpha_temp, self.alpha_comp = multipliers
        self.alpha_pump = 1.0
        return self

    def set_runtime_uniform_objective_scale(self, scale_factor=1.0):
        """Apply one validated uniform objective factor to both flow models."""
        scale_factor = _validated_runtime_weight_multiplier(
            scale_factor, "scale_factor"
        )
        self.forward_model.set_runtime_uniform_objective_scale(scale_factor)
        self.reverse_model.set_runtime_uniform_objective_scale(scale_factor)
        self.runtime_objective_scale_factor = scale_factor
        return self

    def set_runtime_objective_normalization(self, scales):
        """Apply validated fixed objective denominators to both flow models."""
        if not isinstance(scales, ObjectiveNormalizationScales):
            scales = ObjectiveNormalizationScales(**dict(scales))
        self.forward_model.set_runtime_objective_normalization(scales)
        self.reverse_model.set_runtime_objective_normalization(scales)
        self.objective_normalization_scales = scales
        return self

    def set_objective_contribution_diagnostics_enabled(self, enabled=True):
        """Enable the same read-only diagnostic on both flow models."""
        self.forward_model.set_objective_contribution_diagnostics_enabled(enabled)
        self.reverse_model.set_objective_contribution_diagnostics_enabled(enabled)
        return self

    @property
    def is_reversed(self):
        return _is_reversed_from_direction(self.direction)

    def _model_for_direction(self, direction):
        return self.forward_model if direction >= 0 else self.reverse_model

    def dynamic_target_for_step(self, step_no):
        if not MPC_HEAT_PREVIEW_ENABLED:
            return self.target_temp_c
        if not self.qgen_forecast:
            return self.target_temp_c
        start = min(max(0, int(step_no)), len(self.qgen_forecast) - 1)
        end = min(len(self.qgen_forecast), start + self.heat_preview_steps)
        q_window = self.qgen_forecast[start:end]
        if not q_window:
            q_window = [self.qgen_forecast[-1]]
        q_mean = float(np.mean(q_window))
        q_high = float(np.quantile(q_window, MPC_HEAT_PREVIEW_QUANTILE))
        q_risk = (
            MPC_HEAT_PREVIEW_QUANTILE_WEIGHT * q_high
            + (1.0 - MPC_HEAT_PREVIEW_QUANTILE_WEIGHT) * q_mean
        )
        risk = (q_risk - MPC_HEAT_PREVIEW_NOMINAL_W) / max(MPC_HEAT_PREVIEW_NOMINAL_W, 1.0)
        precool = self.mpc_params.precool_max * float(np.clip(risk, 0.0, 1.0))
        warm_relief = self.mpc_params.warm_relief_max * float(np.clip(-risk, 0.0, 1.0))
        target = self.target_temp_c - precool + warm_relief
        return float(np.clip(target, self.mpc_params.dynamic_target_min, self.mpc_params.dynamic_target_max))

    def _solve_for_direction(
        self,
        direction,
        step_no,
        pack,
        t_tank_k,
        t_cabinet_k,
        observed_thermal_state=None,
        t_plate_meas=None,
    ):
        model = self._model_for_direction(direction)
        t_batt_meas = (
            pack.get_avg_temp() if step_no > 0 else INITIAL_TEMP_C + 273.15
        )
        target_temp_c = self.dynamic_target_for_step(step_no)
        return model.solve_step(
            step_no,
            t_batt_meas,
            t_tank_k,
            self.qgen_forecast,
            t_cabinet_k,
            u_ncomp_value=self.last_n_comp,
            u_npump_value=self.last_n_pump,
            target_temp_c=target_temp_c,
            observed_thermal_state=observed_thermal_state,
            t_plate_meas=t_plate_meas,
        )

    def _remember_control(self, solution):
        self.last_n_comp = float(solution["n_comp"])
        self.last_n_pump = float(solution["n_pump"])

    def _condition_compressor_solution(self, solution):
        """Apply the optional Physics-P execution filter after optimization."""
        if self.predictor_name != PHYSICS_P:
            return solution
        raw_command_rpm = float(solution["n_comp"])
        alpha = float(self.mpc_params.comp_command_filter_alpha)
        applied_command_rpm = raw_command_rpm
        if bool(solution.get("solved", False)) and alpha < 1.0:
            applied_command_rpm = float(
                np.clip(
                    self.last_n_comp
                    + alpha * (raw_command_rpm - self.last_n_comp),
                    N_COMP_OFF_RPM,
                    N_COMP_MAX_RPM,
                )
            )
        solution["n_comp_raw_rpm"] = raw_command_rpm
        solution["n_comp_applied_rpm"] = applied_command_rpm
        solution["comp_command_filter_alpha"] = alpha
        solution["n_comp"] = applied_command_rpm
        return solution

    def _predict_temperature_grids(self, pack, solution, direction):
        temps_grid_c = pack.temps.reshape(pack.rows, pack.cols) - 273.15
        grid_offset = temps_grid_c - np.mean(temps_grid_c)
        flow_profile = np.linspace(-0.5, 0.5, pack.cols)
        if direction < 0:
            flow_profile = flow_profile[::-1]
        flow_profile = flow_profile.reshape(1, pack.cols)

        pump_ratio = np.clip(solution["n_pump"] / N_PUMP_MAX_RPM, 0.0, 1.0)
        predicted_avg = np.asarray(solution["t_batt_pred_c"], dtype=float)
        if predicted_avg.size == 0:
            predicted_avg = np.array([pack.get_avg_temp() - 273.15])

        grids = []
        for k, avg_temp_c in enumerate(predicted_avg):
            decay = np.exp(-0.03 * pump_ratio * k)
            direction_bias = 0.02 * pump_ratio * k * flow_profile
            predicted_grid = avg_temp_c + grid_offset * decay + direction_bias
            grids.append(predicted_grid)
        return np.asarray(grids, dtype=float)

    def _predict_delta_t_max(self, pack, solution, direction):
        predicted_grids = self._predict_temperature_grids(pack, solution, direction)
        if predicted_grids.size == 0:
            temps_grid_c = pack.temps.reshape(pack.rows, pack.cols) - 273.15
            return float(np.max(temps_grid_c) - np.min(temps_grid_c))
        deltas = [float(np.max(grid) - np.min(grid)) for grid in predicted_grids]
        return max(deltas) if deltas else float(np.max(temps_grid_c) - np.min(temps_grid_c))

    def _downstream_columns(self, cols, direction):
        split = cols // 2
        if direction >= 0:
            return np.arange(split, cols, dtype=int)
        return np.arange(0, max(1, cols - split), dtype=int)

    def _h_down_from_grid(self, grid_c, direction):
        downstream_cols = self._downstream_columns(grid_c.shape[1], direction)
        downstream_temps = grid_c[:, downstream_cols]
        avg_temp = float(np.mean(grid_c))
        return float(np.sum(np.maximum(downstream_temps - avg_temp, 0.0)))

    @staticmethod
    def _gamma_hot_from_time(hot_time_s, obs_time_s):
        n_cells = hot_time_s.size
        if n_cells <= 1 or obs_time_s <= 0.0:
            return 0.0
        x = hot_time_s / obs_time_s
        uniform = 1.0 / n_cells
        return float(np.sqrt(np.sum((x - uniform) ** 2) / (n_cells - 1)))

    def _set_flow_info(self, solution, delta_t_pred_max, switched=False, extra_info=None):
        self.last_flow_info = {
            "mode": self.flow_mode,
            "direction": self.direction,
            "flow_direction_d": self.direction,
            "delta_t_pred_max": float(delta_t_pred_max),
            "switched": bool(switched),
            "solved": bool(solution.get("solved", False)),
        }
        for key in (
            "raw_track_error_sq",
            "raw_sigma_t_sq",
            "raw_w_comp",
            "raw_w_pump",
            "raw_j_switch",
            "j_track",
            "j_spread",
            "j_comp",
            "j_pump",
            "j_switch",
            "j_total",
            "weighted_j_track",
            "weighted_j_spread",
            "weighted_j_comp",
            "weighted_j_pump",
            "weighted_j_switch",
            "target_temp_c",
            "solve_time_s",
            "solve_error",
            "solver_model_path",
            "solve_recovery_used",
            "solve_recovery_reason",
            "prediction_domain_valid",
            "t_cool_pred_min_c",
            "t_cool_pred_max_c",
            "t_cool_domain_lower_c",
            "t_cool_domain_upper_c",
            "t_cool_domain_violation_c",
            "t_batt_pred_1_c",
            "t_batt_pred_5_c",
            "t_batt_pred_10_c",
            "t_batt_pred_20_c",
            "t_batt_pred_end_c",
            "t_cool_pred_end_c",
            "t_plate_pred_end_c",
            "qevap_eff_pred_mean_w",
            "qevap_cmd_pred_1_w",
            "qcond_pred_1_w",
            "qevap_pred_1_w",
            "n_comp_plan_rpm",
            "n_pump_plan_rpm",
            "n_comp_raw_rpm",
            "n_comp_applied_rpm",
            "comp_command_filter_alpha",
            "physics_p_temp_bias_enabled",
            "physics_p_temp_bias_innovation_c",
            "physics_p_temp_bias_step_c",
            "physics_p_temp_bias_end_c",
            "terminal_cost_enabled",
            "terminal_cost_type",
            "w_terminal_temp",
            "terminal_temp_scale_c",
            "t_pred_end_c",
            "j_terminal",
            "weighted_j_terminal",
            "Terminal_Cost_Enabled",
            "Terminal_Cost_Type",
            "W_Terminal_Temp",
            "Terminal_Temp_Scale_C",
            "T_pred_end_C",
            "J_terminal",
            "Weighted_J_terminal",
            "alpha_temp",
            "alpha_comp",
            "alpha_pump",
            "runtime_weight_mode",
            "runtime_WSPLO",
            "runtime_WSPHI",
            "runtime_w_terminal",
            "runtime_w_temp_obj",
            "runtime_w_energy_comp",
            "runtime_w_energy_pump",
            "runtime_w_dcomp",
            "runtime_w_dpump",
            "runtime_w_dcomp_quadratic",
            "runtime_objective_scale_factor",
            "objective_contributions_valid",
            "objective_contributions_error",
            "objective_solver_total",
            "objective_cv_contribution",
            "objective_temperature_contribution",
            "objective_terminal_contribution",
            "objective_comp_contribution",
            "objective_pump_contribution",
            "objective_dcomp_contribution",
            "objective_dpump_contribution",
            "objective_other_contribution",
            "objective_cv_contribution_share",
            "objective_temperature_contribution_share",
            "objective_terminal_contribution_share",
            "objective_comp_contribution_share",
            "objective_pump_contribution_share",
            "objective_dcomp_contribution_share",
            "objective_dpump_contribution_share",
            "objective_other_contribution_share",
            "objective_cv_raw_hi",
            "objective_cv_raw_lo",
            "objective_cv_raw",
            "objective_cv_scale",
            "objective_cv_normalized",
            "objective_temperature_raw",
            "objective_temperature_scale",
            "objective_temperature_normalized",
            "objective_terminal_raw",
            "objective_terminal_scale",
            "objective_terminal_normalized",
            "objective_compressor_raw",
            "objective_compressor_scale",
            "objective_compressor_normalized",
            "objective_pump_raw",
            "objective_pump_scale",
            "objective_pump_normalized",
            "objective_dcomp_raw",
            "objective_dcomp_scale",
            "objective_dcomp_normalized",
            "objective_dpump_raw",
            "objective_dpump_scale",
            "objective_dpump_normalized",
        ):
            if key in solution:
                self.last_flow_info[key] = solution[key]
        if extra_info:
            self.last_flow_info.update(extra_info)


class SwitchingSystemMPC(BaseFlowMPCController):
    """MPC uses the current flow direction to choose the prediction model."""

    flow_mode = "switching"

    def command(
        self,
        step_no,
        pack,
        t_tank_k,
        t_cabinet_k,
        is_reversed=False,
        thermal_state=None,
        plate_temps=None,
        **_kwargs,
    ):
        self.direction = _direction_from_reversed(is_reversed)
        solution = self._solve_for_direction(
            self.direction,
            step_no,
            pack,
            t_tank_k,
            t_cabinet_k,
            observed_thermal_state=thermal_state,
            t_plate_meas=plate_temps,
        )
        self._condition_compressor_solution(solution)
        delta_t_pred_max = self._predict_delta_t_max(pack, solution, self.direction)
        self._set_flow_info(solution, delta_t_pred_max, switched=False)
        self._remember_control(solution)
        return solution["n_comp"], solution["n_pump"]

    def update_after_step(self, pack):
        return None


class StandardMPC(SwitchingSystemMPC):
    """Original long-run MPC path used by the 12-case batch simulation."""

    flow_mode = "standard"


class SupervisoryEventMPC(BaseFlowMPCController):
    """MPC is solved once, then a supervisor may switch direction and re-solve."""

    owns_flow_direction = True
    flow_mode = "supervised"

    def __init__(
        self,
        current_profile,
        dt=SIM_DT,
        target_temp_c=TARGET_TEMP_C,
        min_hold_time=None,
        case_name=None,
        mpc_params=None,
        predictor=CANDIDATE_B,
        predictor_artifact=None,
    ):
        super().__init__(
            current_profile,
            dt=dt,
            target_temp_c=target_temp_c,
            case_name=case_name,
            mpc_params=mpc_params,
            predictor=predictor,
            predictor_artifact=predictor_artifact,
        )
        self.min_hold_time = self.mpc_params.reverse_hold_s if min_hold_time is None else min_hold_time
        self.delta_t_lim = MPC_REV_DELTA_T_LIM_C
        self.h_down_cell_scale = MPC_REV_H_DOWN_CELL_SCALE_C
        self.gamma_hot_lim = MPC_REV_GAMMA_HOT_LIM
        self.j_rev_on = self.mpc_params.j_rev_on
        self.j_rev_off = self.mpc_params.j_rev_off
        self.w_delta_t = self.mpc_params.w_delta_t
        self.w_h_down = MPC_REV_W_H_DOWN
        self.w_gamma_hot = MPC_REV_W_GAMMA_HOT
        self.min_actual_delta_t_switch_c = 0.25
        self.pred_delta_t_switch_c = MPC_PRED_DELTA_T_SWITCH_C
        self.pred_switch_buffer_s = MPC_PRED_SWITCH_BUFFER_S
        self.min_delta_t_improvement_c = 0.0
        self.max_delta_t_worsening_c = 0.02
        self.min_j_rev_improvement = 0.10
        self.last_switch_time = -1e12
        self.switch_armed = True
        self.hot_time_s = None
        self.obs_time_s = 0.0

    def _ensure_hot_history(self, pack):
        n_cells = int(pack.rows * pack.cols)
        if self.hot_time_s is None or self.hot_time_s.size != n_cells:
            self.hot_time_s = np.zeros(n_cells, dtype=float)
            self.obs_time_s = 0.0

    def _accumulate_hot_time(self, hot_time_s, grid_c, duration_s):
        flat = np.asarray(grid_c, dtype=float).reshape(-1)
        hottest = np.flatnonzero(np.isclose(flat, np.max(flat)))
        if hottest.size == 0:
            return hot_time_s
        hot_time_s[hottest] += duration_s / hottest.size
        return hot_time_s

    def _predict_reversal_metrics(self, pack, solution, direction):
        self._ensure_hot_history(pack)
        predicted_grids = self._predict_temperature_grids(pack, solution, direction)
        if predicted_grids.size == 0:
            current_grid = pack.temps.reshape(pack.rows, pack.cols) - 273.15
            predicted_grids = np.asarray([current_grid], dtype=float)

        downstream_cols = self._downstream_columns(pack.cols, direction)
        n_downstream_cells = max(1, pack.rows * len(downstream_cols))
        h_down_lim = self.h_down_cell_scale * n_downstream_cells

        hot_time_s = self.hot_time_s.copy()
        obs_time_s = float(self.obs_time_s)
        delta_t_series = []
        h_down_series = []
        gamma_hot_series = []
        j_rev_series = []

        for grid_c in predicted_grids:
            delta_t = float(np.max(grid_c) - np.min(grid_c))
            h_down = self._h_down_from_grid(grid_c, direction)
            hot_time_s = self._accumulate_hot_time(hot_time_s, grid_c, self.dt)
            obs_time_s += self.dt
            gamma_hot = self._gamma_hot_from_time(hot_time_s, obs_time_s)
            j_rev = (
                self.w_delta_t * (delta_t / max(self.delta_t_lim, 1e-9))
                + self.w_h_down * (h_down / max(h_down_lim, 1e-9))
                + self.w_gamma_hot * (gamma_hot / max(self.gamma_hot_lim, 1e-9))
            )
            delta_t_series.append(delta_t)
            h_down_series.append(h_down)
            gamma_hot_series.append(gamma_hot)
            j_rev_series.append(float(j_rev))

        delta_t_arr = np.asarray(delta_t_series, dtype=float)
        time_s = np.arange(delta_t_arr.size, dtype=float) * float(self.dt)
        buffer_mask = time_s <= float(self.pred_switch_buffer_s)
        if not np.any(buffer_mask):
            buffer_mask = np.ones_like(delta_t_arr, dtype=bool)
        cross_indices = np.flatnonzero(delta_t_arr >= float(self.pred_delta_t_switch_c))
        delta_t_pred_cross_s = float(cross_indices[0] * float(self.dt)) if cross_indices.size else float("nan")

        return {
            "delta_t_pred_max": float(np.max(delta_t_arr)),
            "delta_t_pred_buffer_max": float(np.max(delta_t_arr[buffer_mask])),
            "delta_t_pred_cross_s": delta_t_pred_cross_s,
            "h_down_pred_max": float(np.max(h_down_series)),
            "gamma_hot_pred_max": float(np.max(gamma_hot_series)),
            "j_rev_pred_max": float(np.max(j_rev_series)),
            "delta_t_pred_0": float(delta_t_series[0]),
            "h_down_pred_0": float(h_down_series[0]),
            "gamma_hot_pred_0": float(gamma_hot_series[0]),
            "j_rev_pred_0": float(j_rev_series[0]),
        }

    def _reverse_prediction_benefits(self, current_metrics, reverse_metrics):
        delta_t_improvement = (
            float(current_metrics["delta_t_pred_max"]) - float(reverse_metrics["delta_t_pred_max"])
        )
        j_rev_improvement = (
            float(current_metrics["j_rev_pred_max"]) - float(reverse_metrics["j_rev_pred_max"])
        )
        delta_t_not_worse = delta_t_improvement >= -float(self.max_delta_t_worsening_c)
        delta_t_good_enough = delta_t_improvement >= float(self.min_delta_t_improvement_c)
        j_rev_good_enough = j_rev_improvement >= float(self.min_j_rev_improvement)
        return bool(delta_t_not_worse and (delta_t_good_enough or j_rev_good_enough) and j_rev_good_enough)

    def _actual_delta_t_c(self, pack):
        grid_c = np.asarray(pack.temps, dtype=float).reshape(pack.rows, pack.cols) - 273.15
        return float(np.max(grid_c) - np.min(grid_c))

    def _predictive_switch_gate(self, metrics):
        cross_s = float(metrics.get("delta_t_pred_cross_s", np.nan))
        return bool(np.isfinite(cross_s) and cross_s <= float(self.pred_switch_buffer_s))

    def command(
        self,
        step_no,
        pack,
        t_tank_k,
        t_cabinet_k,
        current_time=0.0,
        flow_enabled=True,
        thermal_state=None,
        plate_temps=None,
        **_kwargs,
    ):
        self._ensure_hot_history(pack)
        if not flow_enabled:
            self.direction = 1
            solution = self._solve_for_direction(
                self.direction,
                step_no,
                pack,
                t_tank_k,
                t_cabinet_k,
                observed_thermal_state=thermal_state,
                t_plate_meas=plate_temps,
            )
            metrics = self._predict_reversal_metrics(pack, solution, self.direction)
            self._set_flow_info(solution, metrics["delta_t_pred_max"], switched=False, extra_info=metrics)
            self._remember_control(solution)
            return solution["n_comp"], solution["n_pump"]

        solution = self._solve_for_direction(
            self.direction,
            step_no,
            pack,
            t_tank_k,
            t_cabinet_k,
            observed_thermal_state=thermal_state,
            t_plate_meas=plate_temps,
        )
        metrics = self._predict_reversal_metrics(pack, solution, self.direction)
        pre_switch_metrics = dict(metrics)
        actual_delta_t_c = self._actual_delta_t_c(pack)
        if metrics["j_rev_pred_max"] <= self.j_rev_off:
            self.switch_armed = True

        actual_switch_gate = actual_delta_t_c >= self.min_actual_delta_t_switch_c
        predictive_switch_gate = self._predictive_switch_gate(metrics)
        hold_time_satisfied = (current_time - self.last_switch_time) >= self.min_hold_time
        can_switch = hold_time_satisfied and (actual_switch_gate or predictive_switch_gate)
        switched = False
        candidate_switch_info = {}
        if self.switch_armed and can_switch and metrics["j_rev_pred_max"] > self.j_rev_on:
            candidate_direction = -self.direction
            candidate_solution = self._solve_for_direction(
                candidate_direction,
                step_no,
                pack,
                t_tank_k,
                t_cabinet_k,
                observed_thermal_state=thermal_state,
                t_plate_meas=plate_temps,
            )
            candidate_metrics = self._predict_reversal_metrics(pack, candidate_solution, candidate_direction)
            candidate_switch_info = {
                "candidate_direction": candidate_direction,
                "candidate_delta_t_pred_max": float(candidate_metrics["delta_t_pred_max"]),
                "candidate_h_down_pred_max": float(candidate_metrics["h_down_pred_max"]),
                "candidate_gamma_hot_pred_max": float(candidate_metrics["gamma_hot_pred_max"]),
                "candidate_j_rev_pred_max": float(candidate_metrics["j_rev_pred_max"]),
                "candidate_reverse_benefits": self._reverse_prediction_benefits(metrics, candidate_metrics),
                "candidate_predictive_switch_gate": self._predictive_switch_gate(candidate_metrics),
            }
            if candidate_switch_info["candidate_reverse_benefits"]:
                self.direction = candidate_direction
                self.last_switch_time = current_time
                self.switch_armed = False
                switched = True
                solution = candidate_solution
                metrics = candidate_metrics

        self._set_flow_info(solution, metrics["delta_t_pred_max"], switched=switched, extra_info=metrics)
        self.last_flow_info["pre_switch_delta_t_pred_max"] = float(pre_switch_metrics["delta_t_pred_max"])
        self.last_flow_info["pre_switch_delta_t_pred_buffer_max"] = float(pre_switch_metrics["delta_t_pred_buffer_max"])
        self.last_flow_info["pre_switch_delta_t_pred_cross_s"] = float(pre_switch_metrics["delta_t_pred_cross_s"])
        self.last_flow_info["pre_switch_h_down_pred_max"] = float(pre_switch_metrics["h_down_pred_max"])
        self.last_flow_info["pre_switch_gamma_hot_pred_max"] = float(pre_switch_metrics["gamma_hot_pred_max"])
        self.last_flow_info["pre_switch_j_rev_pred_max"] = float(pre_switch_metrics["j_rev_pred_max"])
        self.last_flow_info.update(candidate_switch_info)
        self.last_flow_info["actual_delta_t_c"] = float(actual_delta_t_c)
        self.last_flow_info["actual_switch_gate"] = bool(actual_switch_gate)
        self.last_flow_info["predictive_switch_gate"] = bool(predictive_switch_gate)
        self.last_flow_info["hold_time_satisfied"] = bool(hold_time_satisfied)
        self.last_flow_info["min_actual_delta_t_switch_c"] = float(self.min_actual_delta_t_switch_c)
        self.last_flow_info["pred_delta_t_switch_c"] = float(self.pred_delta_t_switch_c)
        self.last_flow_info["pred_switch_buffer_s"] = float(self.pred_switch_buffer_s)
        self.last_flow_info["min_delta_t_improvement_c"] = float(self.min_delta_t_improvement_c)
        self.last_flow_info["max_delta_t_worsening_c"] = float(self.max_delta_t_worsening_c)
        self.last_flow_info["min_j_rev_improvement"] = float(self.min_j_rev_improvement)
        self.last_flow_info["j_rev_on"] = float(self.j_rev_on)
        self.last_flow_info["j_rev_off"] = float(self.j_rev_off)
        self.last_flow_info["triggered_by_prediction"] = bool(switched and pre_switch_metrics["j_rev_pred_max"] > self.j_rev_on)
        self._remember_control(solution)
        return solution["n_comp"], solution["n_pump"]

    def update_after_step(self, pack):
        self._ensure_hot_history(pack)
        grid_c = pack.temps.reshape(pack.rows, pack.cols) - 273.15
        self._accumulate_hot_time(self.hot_time_s, grid_c, self.dt)
        self.obs_time_s += self.dt
        return None


class MixedIntegerFlowMPC:
    """GEKKO/APOPT mixed-integer MPC with signed flow direction.

    d_flow=1 means forward flow, d_flow=-1 means reverse flow. Unlike the
    switching and supervisory wrappers, this class places d_flow directly in
    the GEKKO optimization model together with compressor and pump speed.
    """

    owns_flow_direction = True
    flow_mode = "mixed_integer"

    def __init__(
        self,
        current_profile,
        dt=SIM_DT,
        target_temp_c=TARGET_TEMP_C,
        np_horizon=40,
        min_hold_time=MPC_MIN_HOLD_TIME_S,
        switch_penalty=None,
        t_track_scale=MPC_T_TRACK_SCALE_C,
        t_spread_scale=MPC_T_SPREAD_SCALE_C,
        q_track=MPC_Q_TRACK,
        q_spread=MPC_Q_SPREAD,
        r_comp=MPC_R_COMP,
        r_pump=MPC_R_PUMP,
        r_switch=MPC_R_SWITCH,
    ):
        self.dt = dt
        self.target_temp_c = target_temp_c
        self.np_horizon = np_horizon
        self.min_hold_time = min_hold_time
        if switch_penalty is not None:
            r_switch = switch_penalty
        self.switch_penalty = r_switch
        self.t_track_scale = t_track_scale
        self.t_spread_scale = t_spread_scale
        self.q_track = q_track
        self.q_spread = q_spread
        self.r_comp = r_comp
        self.r_pump = r_pump
        self.r_switch = r_switch
        self.qgen_forecast = [((current / 4.0) ** 2) * 0.001 * 52 for current in current_profile]
        self.direction = 1
        self.last_n_comp = MPC_U_NCOMP_INIT
        self.last_n_pump = MPC_U_NPUMP_INIT
        self.last_switch_time = -1e12
        self.n_cols = 13
        self.last_flow_info = {
            "mode": self.flow_mode,
            "direction": self.direction,
            "flow_direction_d": self.direction,
            "delta_t_pred_max": np.nan,
            "switched": False,
        }
        self._build_integer_model()

    @property
    def is_reversed(self):
        return _is_reversed_from_direction(self.direction)

    def _build_integer_model(self):
        m = GEKKO(remote=False)
        m.time = np.linspace(0, self.np_horizon * self.dt, self.np_horizon + 1)
        self.m = m

        self.qgen = m.Param(value=np.ones(self.np_horizon + 1) * 1000.0)
        self.t_amb = m.Param(value=np.ones(self.np_horizon + 1) * 298.15)
        self.d_prev = m.Param(value=1)

        self.u_ncomp = m.MV(value=MPC_U_NCOMP_INIT, lb=N_COMP_MIN_RPM, ub=N_COMP_MAX_RPM)
        self.u_npump = m.MV(value=MPC_U_NPUMP_INIT, lb=N_PUMP_MIN_RPM, ub=N_PUMP_MAX_RPM)
        self.d_flow = m.MV(value=1, lb=-1, ub=1, integer=True)
        self.u_ncomp.STATUS = 1
        self.u_npump.STATUS = 1
        self.d_flow.STATUS = 1
        self.u_ncomp.DCOST = MPC_U_NCOMP_DCOST
        self.u_npump.DCOST = MPC_U_NPUMP_DCOST
        self.u_ncomp.DMAX = MPC_U_NCOMP_DMAX
        self.u_npump.DMAX = MPC_U_NPUMP_DMAX

        self.t_cool = m.SV(value=298.15)
        self.t_cool.FSTATUS = 1
        self.t_cols = [m.SV(value=298.15) for _ in range(self.n_cols)]
        for t_col in self.t_cols:
            t_col.FSTATUS = 1

        c_col = 4 * 4747.0
        c_cool = C_tank
        h_col_ref = 52 * 10.0 / self.n_cols
        h_air_col = 30 * 0.1 / self.n_cols
        k_col = 2.0
        kq = 1.0
        forward_gain = np.linspace(1.15, 0.85, self.n_cols)
        reverse_gain = forward_gain[::-1]

        pump_ratio = m.Intermediate((self.u_npump / 2000.0) ** 0.8)
        avg_temp = m.Intermediate(sum(self.t_cols) / self.n_cols)
        cooling_terms = []
        forward_weight = m.Intermediate((1 + self.d_flow) / 2)
        reverse_weight = m.Intermediate((1 - self.d_flow) / 2)

        for j, t_col in enumerate(self.t_cols):
            flow_gain = m.Intermediate(forward_weight * forward_gain[j] + reverse_weight * reverse_gain[j])
            cooling = m.Intermediate(h_col_ref * pump_ratio * flow_gain * (t_col - self.t_cool))
            left = self.t_cols[j - 1] if j > 0 else t_col
            right = self.t_cols[j + 1] if j < self.n_cols - 1 else t_col
            lateral = m.Intermediate(k_col * (left + right - 2 * t_col))
            air_loss = m.Intermediate(h_air_col * (t_col - self.t_amb))
            m.Equation(c_col * t_col.dt() == self.qgen / self.n_cols - cooling - air_loss + lateral)
            cooling_terms.append(cooling)

        m.Equation(c_cool * self.t_cool.dt() == sum(cooling_terms) - kq * self.u_ncomp)

        avg_temp_c = m.Intermediate(avg_temp - 273.15)
        spread_sum = sum((t_col - avg_temp) ** 2 for t_col in self.t_cols)
        P_comp = _legacy_compressor_power_expr(m, self.u_ncomp)
        P_pump = _pump_power_speed_expr(m, self.u_npump)
        P_comp_max = LEGACY_COMPRESSOR_POWER_MAX_W
        P_pump_max = _pump_power_speed_value(N_PUMP_MAX_RPM)
        self.switch_abs = m.Var(lb=0)
        m.Equation(self.d_flow * self.d_flow == 1)
        m.Equation(self.switch_abs >= (self.d_flow - self.d_prev) / 2)
        m.Equation(self.switch_abs >= (self.d_prev - self.d_flow) / 2)

        self.raw_track_error_sq = m.Intermediate((avg_temp_c - self.target_temp_c) ** 2)
        self.raw_sigma_t_sq = m.Intermediate(spread_sum / self.n_cols)
        self.raw_w_comp = P_comp
        self.raw_w_pump = P_pump
        self.raw_j_switch = self.switch_abs

        self.j_track = m.Intermediate(self.raw_track_error_sq / (self.t_track_scale**2))
        self.j_spread = m.Intermediate(self.raw_sigma_t_sq / (self.t_spread_scale**2))
        self.j_comp = m.Intermediate(self.raw_w_comp / P_comp_max)
        self.j_pump = m.Intermediate(self.raw_w_pump / P_pump_max)
        self.j_switch = self.raw_j_switch
        self.j_total = m.Intermediate(
            self.q_track * self.j_track
            + self.q_spread * self.j_spread
            + self.r_comp * self.j_comp
            + self.r_pump * self.j_pump
            + self.r_switch * self.j_switch
        )

        m.Minimize(self.j_total)
        m.options.IMODE = 6
        m.options.NODES = 3
        m.options.SOLVER = 1
        m.options.MAX_ITER = 200

    def _newval(self, variable, default):
        try:
            return float(variable.NEWVAL)
        except Exception:
            try:
                values = list(variable.VALUE)
                return float(values[1] if len(values) > 1 else values[0])
            except Exception:
                return default

    def _mean_prediction_value(self, variable, default=np.nan):
        try:
            values = np.array(list(variable.VALUE), dtype=float)
        except Exception:
            return default
        if values.size == 0:
            return default
        if values.size > 1:
            values = values[1:]
        return float(np.mean(values))

    def _objective_snapshot(self, current_d, d_new):
        switch_default = 0.0 if current_d == d_new else 1.0
        j_track = self._mean_prediction_value(self.j_track)
        j_spread = self._mean_prediction_value(self.j_spread)
        j_comp = self._mean_prediction_value(self.j_comp)
        j_pump = self._mean_prediction_value(self.j_pump)
        j_switch = self._mean_prediction_value(self.j_switch, default=switch_default)
        j_total = (
            self.q_track * j_track
            + self.q_spread * j_spread
            + self.r_comp * j_comp
            + self.r_pump * j_pump
            + self.r_switch * j_switch
        )

        return {
            "raw_track_error_sq": self._mean_prediction_value(self.raw_track_error_sq),
            "raw_sigma_t_sq": self._mean_prediction_value(self.raw_sigma_t_sq),
            "raw_w_comp": self._mean_prediction_value(self.raw_w_comp),
            "raw_w_pump": self._mean_prediction_value(self.raw_w_pump),
            "raw_j_switch": self._mean_prediction_value(self.raw_j_switch, default=switch_default),
            "j_track": j_track,
            "j_spread": j_spread,
            "j_comp": j_comp,
            "j_pump": j_pump,
            "j_switch": j_switch,
            "j_total": j_total,
            "weighted_j_track": self.q_track * j_track,
            "weighted_j_spread": self.q_spread * j_spread,
            "weighted_j_comp": self.r_comp * j_comp,
            "weighted_j_pump": self.r_pump * j_pump,
            "weighted_j_switch": self.r_switch * j_switch,
        }

    def _prediction_arrays(self):
        col_values = [np.array(list(t_col.VALUE), dtype=float) - 273.15 for t_col in self.t_cols]
        col_matrix = np.vstack(col_values)
        avg_pred = np.mean(col_matrix, axis=0)
        delta_pred = np.max(col_matrix, axis=0) - np.min(col_matrix, axis=0)
        return avg_pred, delta_pred

    def command(self, step_no, pack, t_tank_k, t_cabinet_k, current_time=0.0, flow_enabled=True, **_kwargs):
        q_window = [
            self.qgen_forecast[step_no + k] if step_no + k < len(self.qgen_forecast) else self.qgen_forecast[-1]
            for k in range(self.np_horizon + 1)
        ]
        self.qgen.VALUE = q_window
        self.t_amb.VALUE = np.full(self.np_horizon + 1, t_cabinet_k)
        self.t_cool.MEAS = t_tank_k
        self.u_ncomp.VALUE = self.last_n_comp
        self.u_npump.VALUE = self.last_n_pump

        col_temps = np.mean(pack.temps.reshape(pack.rows, pack.cols), axis=0)
        for j, t_col in enumerate(self.t_cols):
            t_col.MEAS = float(col_temps[j])

        current_d = 1 if self.direction > 0 else -1
        self.d_prev.VALUE = current_d
        if not flow_enabled:
            fixed_d = 1
        elif (current_time - self.last_switch_time) < self.min_hold_time:
            fixed_d = current_d
        else:
            fixed_d = None

        if fixed_d is None:
            self.d_flow.LOWER = -1
            self.d_flow.UPPER = 1
        else:
            self.d_flow.LOWER = fixed_d
            self.d_flow.UPPER = fixed_d

        solved = True
        try:
            self.m.solve(disp=False)
        except Exception:
            solved = False

        d_raw = self._newval(self.d_flow, current_d)
        d_new = 1 if d_raw >= 0 else -1
        n_comp = self._newval(self.u_ncomp, MPC_U_NCOMP_INIT)
        n_pump = self._newval(self.u_npump, MPC_U_NPUMP_INIT)
        old_direction = self.direction
        self.direction = d_new
        switched = self.direction != old_direction
        if switched:
            self.last_switch_time = current_time

        if solved:
            avg_pred, delta_pred = self._prediction_arrays()
            delta_t_pred_max = float(np.max(delta_pred))
            objective_info = self._objective_snapshot(current_d, d_new)
        else:
            avg_pred = np.array([pack.get_avg_temp() - 273.15])
            delta_t_pred_max = float(np.max(pack.temps) - np.min(pack.temps))
            objective_info = {
                "raw_track_error_sq": np.nan,
                "raw_sigma_t_sq": np.nan,
                "raw_w_comp": np.nan,
                "raw_w_pump": np.nan,
                "raw_j_switch": 0.0 if current_d == d_new else 1.0,
                "j_track": np.nan,
                "j_spread": np.nan,
                "j_comp": np.nan,
                "j_pump": np.nan,
                "j_switch": 0.0 if current_d == d_new else 1.0,
                "j_total": np.nan,
                "weighted_j_track": np.nan,
                "weighted_j_spread": np.nan,
                "weighted_j_comp": np.nan,
                "weighted_j_pump": np.nan,
                "weighted_j_switch": np.nan,
            }

        self.last_flow_info = {
            "mode": self.flow_mode,
            "direction": self.direction,
            "flow_direction_d": self.direction,
            "delta_t_pred_max": delta_t_pred_max,
            "switched": bool(switched),
            "solved": bool(solved),
            "avg_temp_pred_final": float(avg_pred[-1]),
        }
        self.last_flow_info.update(objective_info)
        self.last_n_comp = float(n_comp)
        self.last_n_pump = float(n_pump)
        return n_comp, n_pump

    def update_after_step(self, pack):
        return None


def create_mpc_flow_controller(
    current_profile,
    dt=SIM_DT,
    target_temp_c=TARGET_TEMP_C,
    mpc_flow_mode="switching",
    case_name=None,
    predictor=CANDIDATE_B,
    predictor_artifact=None,
    mpc_horizon_override=None,
    physics_p_mpc_overrides=None,
    strict_predictor_ablation=False,
):
    strict_predictor_ablation = bool(strict_predictor_ablation)
    if strict_predictor_ablation and mpc_flow_mode != "standard":
        raise ValueError(
            "strict_predictor_ablation is only supported for standard MPC"
        )
    mpc_params = runtime_mpc_params_for_scene(case_name)
    if physics_p_mpc_overrides:
        if normalize_predictor_name(predictor) != PHYSICS_P:
            raise ValueError("physics_p_mpc_overrides are only valid for physics_p")
        allowed = {
            "w_energy_comp",
            "w_terminal_temp",
            "w_dcomp",
            "w_dcomp_quadratic",
            "comp_command_filter_alpha",
            "dmax_comp",
            "cv_band_half_width",
            "w_temp_obj",
            "physics_p_temp_bias_gain",
        }
        unknown = set(physics_p_mpc_overrides) - allowed
        if unknown:
            raise ValueError(
                "unsupported physics_p MPC override(s): "
                + ", ".join(sorted(unknown))
            )
        normalized_overrides = {}
        for key, value in physics_p_mpc_overrides.items():
            numeric = float(value)
            if key == "comp_command_filter_alpha" and not (
                np.isfinite(numeric) and 0.0 < numeric <= 1.0
            ):
                raise ValueError(
                    "comp_command_filter_alpha override must be finite and in (0, 1]"
                )
            if not np.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"{key} override must be finite and nonnegative")
            normalized_overrides[key] = numeric
        override_tag = "_".join(
            f"{key}_{value:g}" for key, value in sorted(normalized_overrides.items())
        )
        mpc_params = replace(
            mpc_params,
            name=f"{mpc_params.name}_physics_p_{override_tag}",
            **normalized_overrides,
        )
    if mpc_horizon_override is not None:
        horizon = int(mpc_horizon_override)
        if horizon < 2:
            raise ValueError("mpc_horizon_override must be at least 2")
        mpc_params = replace(
            mpc_params,
            name=f"{mpc_params.name}_horizon{horizon}",
            mpc_horizon=horizon,
        )
    common_kwargs = {
        "dt": dt,
        "target_temp_c": target_temp_c,
        "case_name": case_name,
        "mpc_params": mpc_params,
        "predictor": predictor,
        "predictor_artifact": predictor_artifact,
    }
    if mpc_flow_mode == "standard":
        return StandardMPC(
            current_profile,
            strict_predictor_ablation=strict_predictor_ablation,
            **common_kwargs,
        )
    if mpc_flow_mode == "switching":
        return SwitchingSystemMPC(current_profile, **common_kwargs)
    if mpc_flow_mode == "supervised":
        return SupervisoryEventMPC(current_profile, **common_kwargs)
    if mpc_flow_mode == "single_predictive_delta_t":
        from ..flow_supervisor import SinglePredictiveDeltaTMPC

        return SinglePredictiveDeltaTMPC(current_profile, **common_kwargs)
    if mpc_flow_mode == "mixed_integer":
        if normalize_predictor_name(predictor) != CANDIDATE_B:
            raise ValueError("mixed_integer MPC does not support physics_p")
        return MixedIntegerFlowMPC(current_profile, dt=dt, target_temp_c=target_temp_c)
    raise ValueError(f"Unknown MPC flow mode: {mpc_flow_mode}")






























