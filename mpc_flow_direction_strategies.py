import csv
import json
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from gekko import GEKKO

from thermal_batch_config import (
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
from thermal_loop import DEFAULT_REFRIGERATION_DYNAMICS, pipe_delay_steps
from thermal_system import C_tank, cp_cool, m_dot_nominal
from mpc_evaporator_capacity_model import load_capacity_calibration

PUMP_POWER_SPEED_COEFF = (2.27321928e-09, -1.62756913e-05, 4.41581449e-02, -3.49442214e01)
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
    ):
        self.params = dict(params)
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
        p = self.params
        self.evaporator_capacity_calibration = load_capacity_calibration()
        self._capacity_upper_w = float(self.evaporator_capacity_calibration["q_evap_upper_bound_w"])

        self.d1_qgen = self.m.Param(value=1000.0)
        self.d2_tamb = self.m.Param(value=p["T_env"])
        self.d3_qgen_risk = self.m.Param(value=0.0)
        self.T_batt_K = self.m.SV(value=p["T_init_batt"])
        self.T_cool_K = self.m.SV(value=p["T_init_cool"])
        self.N_comp = self.m.SV(value=p["N_comp_min"])
        self.N_pump = self.m.SV(value=MPC_U_NPUMP_INIT)
        self.Q_evap = self.m.SV(value=p["kq"] * p["N_comp_min"])
        self.Q_cond = self.m.SV(value=p["kq"] * p["N_comp_min"])
        self.T_plate = self.m.SV(value=p["T_init_cool"] - 273.15)
        self.T_supply = self.m.Var(value=p["T_init_cool"] - 273.15)
        self.T_return = self.m.Var(value=p["T_init_cool"] - 273.15)
        self.N_comp_delay = self.m.Var(value=p["N_comp_min"])
        self.N_pump_delay = self.m.Var(value=MPC_U_NPUMP_INIT)
        self.T_batt_K.FSTATUS = 1
        self.T_cool_K.FSTATUS = 1
        self.N_comp.FSTATUS = 1
        self.N_pump.FSTATUS = 1
        self.Q_evap.FSTATUS = 1
        self.Q_cond.FSTATUS = 1
        self.T_plate.FSTATUS = 1
        self.u_ncomp = self.m.MV(value=MPC_U_NCOMP_INIT, lb=p["N_comp_min"], ub=N_COMP_MAX_RPM)
        self.u_npump = self.m.MV(value=MPC_U_NPUMP_INIT, lb=p["N_pump_min"], ub=N_PUMP_MAX_RPM)
        self.u_ncomp.STATUS = 1
        self.u_npump.STATUS = 1
        self.u_ncomp.DCOST = self.mpc_params.w_dcomp
        self.u_npump.DCOST = self.mpc_params.w_dpump
        self.u_ncomp.DMAX = self.mpc_params.dmax_comp
        self.u_npump.DMAX = self.mpc_params.dmax_pump

        T_batt = self.m.Intermediate(self.T_batt_K - 273.15)
        T_cool = self.m.Intermediate(self.T_cool_K - 273.15)
        T_plate = self.T_plate
        T_amb = self.m.Intermediate(self.d2_tamb - 273.15)
        if self.evap_input_delay_steps > 0:
            self.m.delay(self.N_comp, self.N_comp_delay, steps=self.evap_input_delay_steps)
            N_comp_delay = self.N_comp_delay
        else:
            N_comp_delay = self.N_comp
        if self.pump_flow_delay_steps > 0:
            self.m.delay(self.N_pump, self.N_pump_delay, steps=self.pump_flow_delay_steps)
            N_pump_delay = self.N_pump_delay
        else:
            N_pump_delay = self.N_pump
        H_batt_plate = self.m.Intermediate(p["h1_ref"] * ((N_pump_delay / p["N_pump_ref"]) ** 0.8))
        P_comp = self.m.Intermediate(3.57e-6 * self.N_comp**2 + 0.442 * self.N_comp + 34.0)
        P_pump = _pump_power_speed_expr(self.m, self.N_pump)
        P_comp_max = 3.57e-6 * 6000.0**2 + 0.442 * 6000.0 + 34.0
        P_pump_max = _pump_power_speed_value(N_PUMP_MAX_RPM)
        coeff = self.evaporator_capacity_calibration["coefficients"]
        n_pump_ref = float(self.evaporator_capacity_calibration["n_pump_ref_rpm"])
        n_pump_safe = self.m.Intermediate(N_pump_delay + 1e-6)
        q_evap_raw_w = self.m.Intermediate(
            N_comp_delay * (float(coeff["b0"]) + float(coeff["b1"]) * n_pump_ref / n_pump_safe + float(coeff["b2"]) * (T_cool - 25.0))
        )
        self.Q_evap_cmd_w = self.m.Var(value=1000.0, lb=0.0, ub=self._capacity_upper_w)
        self.m.Equation(self.Q_evap_cmd_w == q_evap_raw_w)
        Q_evap_cmd = self.Q_evap_cmd_w
        C_flow = self.m.Intermediate(p["m_dot_ref"] * p["cp_cool"] * N_pump_delay / p["N_pump_ref"])
        Q_batt_plate = self.m.Intermediate(H_batt_plate * (T_batt - T_plate))
        T_evap_out = self.m.Intermediate(T_cool - self.Q_evap / (C_flow + 1e-6))
        T_plate_out = self.m.Intermediate(self.T_supply + Q_batt_plate / (C_flow + 1e-6))

        self.m.Equation(p["tau_comp_s"] * self.N_comp.dt() == self.u_ncomp - self.N_comp)
        self.m.Equation(p["tau_pump_s"] * self.N_pump.dt() == self.u_npump - self.N_pump)
        self.m.Equation(p["tau_cond_s"] * self.Q_cond.dt() == Q_evap_cmd - self.Q_cond)
        self.m.Equation(p["tau_evap_s"] * self.Q_evap.dt() == self.Q_cond - self.Q_evap)
        self.m.delay(T_evap_out, self.T_supply, steps=self.pipe_supply_delay_steps)
        self.m.Equation(p["tau_plate_s"] * self.T_plate.dt() == self.T_supply - self.T_plate)
        self.m.delay(T_plate_out, self.T_return, steps=self.pipe_return_delay_steps)
        self.m.Equation(
            p["C1"] * self.T_batt_K.dt()
            == self.d1_qgen - Q_batt_plate - p["h2"] * (T_batt - T_amb)
        )
        self.m.Equation(p["C2"] * self.T_cool_K.dt() == C_flow * (self.T_return - T_cool))
        self.cv_temp = self.m.CV(value=p["T_init_batt"] - 273.15)
        self.cv_temp.STATUS = 1
        self.cv_temp.FSTATUS = 1
        self.m.Equation(self.cv_temp == T_batt)
        self.track_target_c = self.m.Param(value=self.target_temp_c)
        if self.mode == "POINT":
            self.cv_temp.SP = p["T_set"]
            self.cv_temp.SPLO = p["T_set"] - self.mpc_params.cv_band_half_width
            self.cv_temp.SPHI = p["T_set"] + self.mpc_params.cv_band_half_width
            self.cv_temp.WSPLO = self.mpc_params.w_cold_temp
            self.cv_temp.WSPHI = self.mpc_params.w_high_temp
            self.cv_temp.TAU = MPC_CV_TAU
            self.cv_temp.TR_INIT = 2
            self.m.options.CV_TYPE = 1

        self.raw_track_error_sq = self.m.Intermediate((T_batt - self.track_target_c) ** 2)
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
            (T_batt - self.track_target_c) / self.terminal_temp_scale_c
        )
        self.j_terminal_path = self.m.Intermediate(self.e_terminal_path ** 2)
        self.weighted_j_terminal_path = self.m.Intermediate(
            self.terminal_mask * float(self.mpc_params.w_terminal_temp) * self.j_terminal_path
        )
        self.plate_reserve_over_c = self.m.Var(value=0.0, lb=0.0)
        self.tank_reserve_over_c = self.m.Var(value=0.0, lb=0.0)
        self.qevap_reserve_short_w = self.m.Var(value=0.0, lb=0.0)
        self.bat_safety_over_c = self.m.Var(value=0.0, lb=0.0)
        self.m.Equation(self.plate_reserve_over_c >= self.T_supply - MPC_RESERVE_T_PLATE_IN_REF_C)
        self.m.Equation(self.tank_reserve_over_c >= T_cool - MPC_RESERVE_T_TANK_REF_C)
        self.m.Equation(self.qevap_reserve_short_w >= self.d3_qgen_risk * MPC_RESERVE_QEVAP_REF_W - self.Q_evap)
        self.m.Equation(self.bat_safety_over_c >= T_batt - (MPC_T_BAT_MAX_C - MPC_T_BAT_MARGIN_C))
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
        self.m.Equation(self.term_over >= T_batt - self.mpc_params.T_term_peak)
        self.j_term = self.m.Intermediate((self.term_over / MPC_T_TRACK_SCALE_C) ** 2)
        j_total_expr = (
            self.mpc_params.w_energy_comp * self.j_comp
            + self.mpc_params.w_energy_pump * self.j_pump
        )
        if self.terminal_cost_enabled:
            j_total_expr = j_total_expr + self.weighted_j_terminal_path
        self.j_total = self.m.Intermediate(j_total_expr)
        self.m.Minimize(self.j_total)
        self.m.options.IMODE = 6
        self.m.options.NODES = 2
        self.m.options.SOLVER = 3

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
            t_pred_end_c = self._last_gekko_value(self.T_batt_K) - 273.15
        except Exception as exc:
            print(f"WARNING terminal cost log extraction failed: {exc}", flush=True)
            t_pred_end_c = np.nan
        if np.isfinite(t_pred_end_c):
            j_terminal = ((t_pred_end_c - float(active_target_temp_c)) / self.terminal_temp_scale_c) ** 2
        else:
            j_terminal = np.nan
        weighted_j_terminal = (
            float(self.mpc_params.w_terminal_temp) * j_terminal
            if self.terminal_cost_enabled and np.isfinite(j_terminal)
            else 0.0
        )
        terminal_cost_requested = bool(self.mpc_params.terminal_cost_enabled)
        return {
            "terminal_cost_enabled": terminal_cost_requested,
            "terminal_cost_type": self.mpc_params.terminal_cost_type,
            "w_terminal_temp": float(self.mpc_params.w_terminal_temp),
            "terminal_temp_scale_c": float(self.terminal_temp_scale_c),
            "t_pred_end_c": t_pred_end_c,
            "j_terminal": j_terminal,
            "weighted_j_terminal": weighted_j_terminal,
            "Terminal_Cost_Enabled": terminal_cost_requested,
            "Terminal_Cost_Type": self.mpc_params.terminal_cost_type,
            "W_Terminal_Temp": float(self.mpc_params.w_terminal_temp),
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
            self.mpc_params.w_energy_comp * j_comp
            + self.mpc_params.w_energy_pump * j_pump
            + weighted_j_terminal
        )
        return {
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
            "w_terminal_temp": float(self.mpc_params.w_terminal_temp),
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
            "weighted_j_track": self.mpc_params.w_temp_obj * j_track,
            "weighted_j_spread": self.mpc_params.w_delta_t * j_spread,
            "weighted_j_comp": self.mpc_params.w_energy_comp * j_comp,
            "weighted_j_pump": self.mpc_params.w_energy_pump * j_pump,
            "weighted_j_switch": MPC_R_SWITCH * j_switch,
            "weighted_j_plate_reserve": reserve_enabled * MPC_W_PLATE_RESERVE * j_plate_reserve,
            "weighted_j_tank_reserve": reserve_enabled * MPC_W_TANK_RESERVE * j_tank_reserve,
            "weighted_j_qevap_reserve": reserve_enabled * MPC_W_QEVAP_RESERVE * j_qevap_reserve,
            "weighted_j_bat_safety": self.mpc_params.w_bat_safety * j_bat_safety,
            "weighted_j_term": self.mpc_params.w_term_peak * j_term,
        }

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
    ):
        active_target_temp_c = self.target_temp_c if target_temp_c is None else float(target_temp_c)
        if self.mode == "POINT":
            self.cv_temp.SP = active_target_temp_c
            self.cv_temp.SPLO = active_target_temp_c - self.mpc_params.cv_band_half_width
            self.cv_temp.SPHI = active_target_temp_c + self.mpc_params.cv_band_half_width
            self.cv_temp.WSPLO = self.mpc_params.w_cold_temp
            self.cv_temp.WSPHI = self.mpc_params.w_high_temp
            self.track_target_c.VALUE = active_target_temp_c
        self.cv_temp.MEAS = t_batt_meas - 273.15
        self.T_batt_K.MEAS = t_batt_meas
        self.T_cool_K.MEAS = t_cool_meas
        if observed_thermal_state:
            if "N_comp_eff" in observed_thermal_state:
                self.N_comp.MEAS = float(observed_thermal_state["N_comp_eff"])
            if "N_pump_eff" in observed_thermal_state:
                self.N_pump.MEAS = float(observed_thermal_state["N_pump_eff"])
            if "Q_evap_eff" in observed_thermal_state:
                self.Q_evap.MEAS = float(observed_thermal_state["Q_evap_eff"])
            if "Q_cond_eff" in observed_thermal_state:
                self.Q_cond.MEAS = float(observed_thermal_state["Q_cond_eff"])
        if t_plate_meas is not None:
            self.T_plate.MEAS = float(np.mean(t_plate_meas) - 273.15)
        self.u_ncomp.VALUE = float(u_ncomp_value)
        self.u_npump.VALUE = float(u_npump_value)
        self.d2_tamb.VALUE = np.full(self.np_horizon + 1, T_amburrent)
        q_window = [
            qgen_forecast[i + k] if i + k < len(qgen_forecast) else qgen_forecast[-1]
            for k in range(self.np_horizon + 1)
        ]
        self.d1_qgen.VALUE = q_window
        self.d3_qgen_risk.VALUE = self._qgen_risk_window(q_window)
        solve_start = time.perf_counter()
        try:
            self.m.solve(disp=False)
            solve_time_s = time.perf_counter() - solve_start
            result = {
                "n_comp": float(self.u_ncomp.NEWVAL),
                "n_pump": float(self.u_npump.NEWVAL),
                "t_batt_pred_c": np.array(list(self.T_batt_K.VALUE), dtype=float) - 273.15,
                "t_cool_pred_c": np.array(list(self.T_cool_K.VALUE), dtype=float) - 273.15,
                "flow_direction": self.flow_direction,
                "target_temp_c": active_target_temp_c,
                "solve_time_s": float(solve_time_s),
                "solved": True,
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
            solve_time_s = time.perf_counter() - solve_start
            return {
                "n_comp": self.mpc_params.n_comp_min,
                "n_pump": self.mpc_params.n_pump_min,
                "t_batt_pred_c": np.array([t_batt_meas - 273.15], dtype=float),
                "t_cool_pred_c": np.array([t_cool_meas - 273.15], dtype=float),
                "flow_direction": self.flow_direction,
                "target_temp_c": active_target_temp_c,
                "solve_time_s": float(solve_time_s),
                "solved": False,
                "n_comp_plan_rpm": [float(self.mpc_params.n_comp_min)],
                "n_pump_plan_rpm": [float(self.mpc_params.n_pump_min)],
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
                "w_terminal_temp": float(self.mpc_params.w_terminal_temp),
                "terminal_temp_scale_c": float(self.terminal_temp_scale_c),
                "t_pred_end_c": t_batt_meas - 273.15,
                "j_terminal": np.nan,
                "weighted_j_terminal": 0.0,
                "Terminal_Cost_Enabled": bool(self.mpc_params.terminal_cost_enabled),
                "Terminal_Cost_Type": self.mpc_params.terminal_cost_type,
                "W_Terminal_Temp": float(self.mpc_params.w_terminal_temp),
                "Terminal_Temp_Scale_C": float(self.terminal_temp_scale_c),
                "T_pred_end_C": t_batt_meas - 273.15,
                "J_terminal": np.nan,
                "Weighted_J_terminal": 0.0,
            }


class BaseFlowMPCController:
    owns_flow_direction = False
    flow_mode = "switching"

    def __init__(self, current_profile, dt=SIM_DT, target_temp_c=TARGET_TEMP_C, case_name=None, mpc_params=None):
        self.dt = dt
        self.target_temp_c = target_temp_c
        self.case_name = case_name
        self.mpc_params = mpc_params or select_mpc_params_for_scene(case_name)
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
            "T_init_batt": 298.15,
            "T_init_cool": 298.15,
            "N_comp_min": self.mpc_params.n_comp_min,
            "N_pump_min": self.mpc_params.n_pump_min,
        }
        self.reduced_model_calibration_theta = load_reduced_model_calibration()
        params = apply_reduced_model_calibration(params, self.reduced_model_calibration_theta)
        self.forward_model = MPCControllerDual(
            params,
            dt,
            np_horizon=self.mpc_params.mpc_horizon,
            mode="POINT",
            target_temp_c=target_temp_c,
            flow_direction=1,
            mpc_params=self.mpc_params,
            case_name=case_name,
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
        )
        self.direction = 1
        self.last_n_comp = MPC_U_NCOMP_INIT
        self.last_n_pump = MPC_U_NPUMP_INIT
        self.last_flow_info = {
            "mode": self.flow_mode,
            "direction": self.direction,
            "delta_t_pred_max": np.nan,
            "switched": False,
        }

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
        t_batt_meas = pack.get_avg_temp() if step_no > 0 else 298.15
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
    ):
        super().__init__(current_profile, dt=dt, target_temp_c=target_temp_c, case_name=case_name, mpc_params=mpc_params)
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
        P_comp = m.Intermediate(3.57e-6 * self.u_ncomp**2 + 0.442 * self.u_ncomp + 34.0)
        P_pump = _pump_power_speed_expr(m, self.u_npump)
        P_comp_max = 3.57e-6 * 6000.0**2 + 0.442 * 6000.0 + 34.0
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
):
    mpc_params = runtime_mpc_params_for_scene(case_name)
    if mpc_flow_mode == "standard":
        return StandardMPC(current_profile, dt=dt, target_temp_c=target_temp_c, case_name=case_name, mpc_params=mpc_params)
    if mpc_flow_mode == "switching":
        return SwitchingSystemMPC(
            current_profile,
            dt=dt,
            target_temp_c=target_temp_c,
            case_name=case_name,
            mpc_params=mpc_params,
        )
    if mpc_flow_mode == "supervised":
        return SupervisoryEventMPC(
            current_profile,
            dt=dt,
            target_temp_c=target_temp_c,
            case_name=case_name,
            mpc_params=mpc_params,
        )
    if mpc_flow_mode == "single_predictive_delta_t":
        from predictive_delta_t_flow_controller import SinglePredictiveDeltaTMPC

        return SinglePredictiveDeltaTMPC(
            current_profile,
            dt=dt,
            target_temp_c=target_temp_c,
            case_name=case_name,
            mpc_params=mpc_params,
        )
    if mpc_flow_mode == "mixed_integer":
        return MixedIntegerFlowMPC(current_profile, dt=dt, target_temp_c=target_temp_c)
    raise ValueError(f"Unknown MPC flow mode: {mpc_flow_mode}")






























