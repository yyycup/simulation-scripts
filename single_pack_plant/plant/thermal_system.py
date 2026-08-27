import numpy as np
import CoolProp.CoolProp as CP
from scipy.interpolate import RegularGridInterpolator
from ..simulation.config import (
    COMPRESSOR_MAP_MIN_RPM,
    COMPRESSOR_MIN_STEADY_RPM,
    EVAP_CAP_FACTOR,
    EVAP_FLOW_EXP,
    EVAP_UA_FACTOR,
    N_COMP_OFF_RPM,
)

# --- CoolProp 鍜屽埗鍐峰墏瀹氫箟 ---
PropsSI = CP.PropsSI
REF = "R134a"
T0_EXERGY = 298.15
SAT_TABLE_T_MIN_K = 250.0
SAT_TABLE_T_MAX_K = 360.0
SAT_TABLE_T_STEP_K = 0.2


def _build_saturation_property_table():
    T_values = np.arange(SAT_TABLE_T_MIN_K, SAT_TABLE_T_MAX_K + 0.5 * SAT_TABLE_T_STEP_K, SAT_TABLE_T_STEP_K)
    return {
        "T": T_values,
        "p": np.array([PropsSI("P", "T", T, "Q", 0.0, REF) for T in T_values]),
        "h_liq": np.array([PropsSI("Hmass", "T", T, "Q", 0.0, REF) for T in T_values]),
        "h_vap": np.array([PropsSI("Hmass", "T", T, "Q", 1.0, REF) for T in T_values]),
        "s_liq": np.array([PropsSI("Smass", "T", T, "Q", 0.0, REF) for T in T_values]),
        "s_vap": np.array([PropsSI("Smass", "T", T, "Q", 1.0, REF) for T in T_values]),
    }


SATURATION_PROPERTY_TABLE = _build_saturation_property_table()

# --- 鐗╃悊鍙傛暟 ---
rho_cool = 1071.0
cp_cool = 3391.0
V_tank_L = 3.0
V_tank_m3 = V_tank_L / 1000.0
C_tank = V_tank_m3 * rho_cool * cp_cool
T_EVAP_SAT_RATED_K = 280.15
T_COND_SAT_RATED_K = 328.15

# 鍘嬬缉鏈哄弬鏁?(Map鍥炬暟鎹?
speed_axis = np.array([2000, 3000, 4000, 5000, 6000])
pr_axis = np.array([4.0, 5.0, 6.4, 8.0, 10.0])
eta_vol_map = np.array([
    [0.88, 0.83, 0.76, 0.69, 0.54], [0.94, 0.91, 0.87, 0.82, 0.74],
    [0.96, 0.94, 0.91, 0.86, 0.80], [0.98, 0.96, 0.94, 0.89, 0.84],
    [0.98, 0.96, 0.95, 0.90, 0.87]
])
eta_is_map = np.array([
    [0.63, 0.57, 0.50, 0.43, 0.37], [0.70, 0.63, 0.59, 0.54, 0.46],
    [0.70, 0.62, 0.62, 0.57, 0.51], [0.68, 0.64, 0.64, 0.58, 0.54],
    [0.68, 0.65, 0.63, 0.60, 0.56]
])
eta_vol_interpolator = RegularGridInterpolator((speed_axis, pr_axis), eta_vol_map, bounds_error=False, fill_value=None)
eta_is_interpolator = RegularGridInterpolator((speed_axis, pr_axis), eta_is_map, bounds_error=False, fill_value=None)
eta_mech = 0.9134570767518371
# Compressor displacement set directly in physical units: 5.525 cm^3/rev.
V_disp_m3_per_rev = 5.525e-06
ETA_VOL_EXTRAPOLATION_BOUNDS = (0.25, 0.99)
ETA_IS_EXTRAPOLATION_BOUNDS = (0.20, 0.80)

# 鍐锋澘涓庢崲鐑櫒鍙傛暟
N_bp = 13
A_bp_total = 0.5
A_bp_seg = A_bp_total / N_bp
h_bp_nominal = 2000.0
m_dot_nominal = 1.2

N_rad = 6
A_rad_total = 0.4
A_rad_seg = A_rad_total / N_rad
h_rad_local = 2.0

# 钂稿彂鍣?鍐峰嚌鍣?椋庢墖鍙傛暟
superheat_desired = 5.0
subcooling_desired = 5.0
A_chiller = 1
h_chiller_cool_nom = 2000.0
h_chiller_ref_nom = 10000.0
EVAPORATOR_UA_SCALE = 0.70
A_evap = A_chiller
h_evap_cool_nom = h_chiller_cool_nom
h_evap_ref_nom = h_chiller_ref_nom
m_dot_ref_nominal = 0.02
A_cond = 1
cp_air = 1005.0
m_dot_air_nominal = 1.5
h_cond_air_nom = 800.0
h_cond_ref_nom = 5000.0
# Condenser fan reference from Energies 2020, 13, 6012:
# SPAL axial fan, diameter 305 mm, speed 800-4000 rpm. Figure 6 validates
# fan power in a 0-0.5 kW range, so 500 W is used as the rated upper point.
N_fan_max_rpm = 4000.0
W_fan_max_power = 500.0

_REFRIGERATION_CYCLE_CACHE = {}
_REFRIGERATION_CYCLE_CACHE_HITS = 0
_REFRIGERATION_CYCLE_CACHE_MISSES = 0
_REFRIGERATION_CYCLE_CACHE_MAX_SIZE = 20000
_REFRIGERATION_CYCLE_CACHE_STEPS = {
    "speed_rpm": 50.0,
    "temperature_K": 0.2,
    "m_dot_kg_s": 0.002,
}

# Pump data digitized from literature curves; final model input is speed, not duty cycle.
pump_speed_points_rpm = np.array([1600.0, 2300.0, 3000.0, 3500.0, 4000.0, 4500.0, 4800.0])
pump_flow_points_L_min = np.array([11.0, 16.0, 21.0, 24.0, 28.0, 32.0, 37.0])
pump_power_coeff = (0.0651, -1.1513, 8.5959)
pump_speed_min_rpm = 1600.0
pump_speed_max_rpm = 4800.0
pump_flow_min_L_min = 10.0
pump_flow_max_L_min = 37.0

# Backward-compatible defaults used by older simulation scripts.
T_env = 298.15
comp_low = 300.0
comp_high = 6000.0
pump_low = 500.0
pump_high = 6000.0


def get_environment_temperatures(t):
    T_outdoor_C = 25.0 + 5.0 * np.sin(2 * np.pi * t / 86400.0)
    T_cabinet_C = 25.0 + 5.0 * np.sin(2 * np.pi * t / 3600.0)
    return T_outdoor_C + 273.15, T_cabinet_C + 273.15


# --- 宸ュ叿鍑芥暟 ---
def safe_PropsSI(output, name1, val1, name2, val2, fluid=REF):
    table_value = _safe_saturation_table_lookup(output, name1, val1, name2, val2, fluid)
    if table_value is not None:
        return table_value
    try:
        v = PropsSI(output, name1, val1, name2, val2, fluid)
        if np.isnan(v):
            if output == 'Dmass': return 1.0
            if output == 'Hmass': return 200000
            return np.nan
        return v
    except ValueError:
        return np.nan


def saturation_property_table_info():
    T = SATURATION_PROPERTY_TABLE["T"]
    return {
        "fluid": REF,
        "T_min_K": float(T[0]),
        "T_max_K": float(T[-1]),
        "T_step_K": SAT_TABLE_T_STEP_K,
        "points": int(T.size),
    }


def _interp_saturation_by_T(field, T):
    table = SATURATION_PROPERTY_TABLE
    if T < table["T"][0] or T > table["T"][-1]:
        return None
    return float(np.interp(T, table["T"], table[field]))


def _interp_saturation_by_p(field, p):
    table = SATURATION_PROPERTY_TABLE
    if p < table["p"][0] or p > table["p"][-1]:
        return None
    return float(np.interp(p, table["p"], table[field]))


def _safe_saturation_table_lookup(output, name1, val1, name2, val2, fluid):
    if fluid != REF:
        return None
    props = {name1: val1, name2: val2}
    quality = props.get("Q")
    if quality not in (0.0, 1.0, 0, 1):
        return None

    field_by_output = {
        ("Hmass", 0.0): "h_liq",
        ("Hmass", 1.0): "h_vap",
        ("Smass", 0.0): "s_liq",
        ("Smass", 1.0): "s_vap",
    }
    q = float(quality)

    if "T" in props:
        T = float(props["T"])
        if output == "P":
            return _interp_saturation_by_T("p", T)
        field = field_by_output.get((output, q))
        return _interp_saturation_by_T(field, T) if field else None

    if "P" in props:
        p = float(props["P"])
        if output == "T":
            return _interp_saturation_by_p("T", p)
        field = field_by_output.get((output, q))
        return _interp_saturation_by_p(field, p) if field else None

    return None


def log_mean_temperature(T_in, T_out):
    if np.isnan(T_in) or np.isnan(T_out) or T_in <= 0.0 or T_out <= 0.0:
        return np.nan
    if abs(T_out - T_in) < 1e-9:
        return float(T_in)
    ratio = T_out / T_in
    if ratio <= 0.0 or abs(np.log(ratio)) < 1e-12:
        return float(0.5 * (T_in + T_out))
    return float((T_out - T_in) / np.log(ratio))


def _finite_nonnegative(value):
    if np.isnan(value):
        return 0.0
    return max(0.0, float(value))


def calculate_single_stage_exergy_losses(
    *,
    m_dot,
    W_comp,
    Q_cond_out,
    Q_evap_in,
    T0,
    h_1,
    h_2,
    h_3,
    h_4,
    s_1,
    s_2,
    s_3,
    s_4,
    T_b_cond,
    T_b_evap,
    W_pump=0.0,
    W_fan=0.0,
):
    E_D_comp = _finite_nonnegative(W_comp + m_dot * ((h_1 - h_2) - T0 * (s_1 - s_2)))
    E_D_cond = _finite_nonnegative(
        m_dot * ((h_2 - h_3) - T0 * (s_2 - s_3)) - (1.0 - T0 / T_b_cond) * Q_cond_out
    )
    E_D_exp = _finite_nonnegative(m_dot * T0 * (s_4 - s_3))
    E_D_evap = _finite_nonnegative(
        (1.0 - T0 / T_b_evap) * Q_evap_in + m_dot * ((h_4 - h_1) - T0 * (s_4 - s_1))
    )
    E_D_total = E_D_comp + E_D_cond + E_D_exp + E_D_evap

    if E_D_total > 1e-12:
        ratios = {
            "E_D_comp_ratio": E_D_comp / E_D_total,
            "E_D_evap_ratio": E_D_evap / E_D_total,
            "E_D_cond_ratio": E_D_cond / E_D_total,
            "E_D_exp_ratio": E_D_exp / E_D_total,
        }
    else:
        ratios = {
            "E_D_comp_ratio": 0.0,
            "E_D_evap_ratio": 0.0,
            "E_D_cond_ratio": 0.0,
            "E_D_exp_ratio": 0.0,
        }

    W_total = W_comp + W_pump + W_fan
    COP_system = Q_evap_in / W_total if W_total > 1e-12 else 0.0
    cop_carnot = T_b_evap / max(T_b_cond - T_b_evap, 1e-6) if T_b_evap > 0.0 and T_b_cond > T_b_evap else np.nan
    exergy_efficiency = COP_system / cop_carnot if not np.isnan(cop_carnot) and cop_carnot > 1e-12 else 0.0

    return {
        "COP_system": _finite_nonnegative(COP_system),
        "COP_Carnot": _finite_nonnegative(cop_carnot),
        "exergy_efficiency": _finite_nonnegative(exergy_efficiency),
        "E_D_comp": E_D_comp,
        "E_D_evap": E_D_evap,
        "E_D_cond": E_D_cond,
        "E_D_exp": E_D_exp,
        "E_D_total": E_D_total,
        **ratios,
    }


def calculate_refrigeration_exergy_metrics(
    *,
    mdot_ref,
    p_evap,
    p_cond,
    h_comp_in,
    h_comp_out,
    h_cond_out,
    h_evap_out,
    W_comp,
    W_fan,
    Q_evap,
    Q_cond,
    T_cool_in,
    T_cool_out,
    T_air_in,
    T_air_out,
    T0=T0_EXERGY,
):
    s_1 = safe_PropsSI("Smass", "P", p_evap, "Hmass", h_comp_in, REF)
    s_2 = safe_PropsSI("Smass", "P", p_cond, "Hmass", h_comp_out, REF)
    s_3 = safe_PropsSI("Smass", "P", p_cond, "Hmass", h_cond_out, REF)
    s_4 = safe_PropsSI("Smass", "P", p_evap, "Hmass", h_cond_out, REF)
    T_b_cond = log_mean_temperature(T_air_in, T_air_out)
    T_b_evap = log_mean_temperature(T_cool_in, T_cool_out)

    return calculate_single_stage_exergy_losses(
        m_dot=mdot_ref,
        W_comp=W_comp,
        Q_cond_out=Q_cond,
        Q_evap_in=Q_evap,
        T0=T0,
        h_1=h_comp_in,
        h_2=h_comp_out,
        h_3=h_cond_out,
        h_4=h_cond_out,
        s_1=s_1,
        s_2=s_2,
        s_3=s_3,
        s_4=s_4,
        T_b_cond=T_b_cond,
        T_b_evap=T_b_evap,
        W_fan=W_fan,
    )


def p_sat_from_T(T):
    T_min_allowable = 200
    if T < T_min_allowable: T = T_min_allowable
    return safe_PropsSI("P", "T", T, "Q", 0.0, REF)


def solve_saturation_temperatures(
    N_rpm_comp,
    N_rpm_fan,
    T_cool_in,
    m_dot_cool,
    T_outdoor,
    low_speed_efficiency_loss_scale=1.0,
):
    T_evap_low = T_EVAP_SAT_RATED_K
    T_evap_high = max(T_evap_low + 1.0, T_cool_in - 2.0)
    T_evap = np.clip(T_cool_in - 10.0, T_evap_low, T_evap_high)
    T_cond = np.clip(T_outdoor + 15.0, T_outdoor + 5.0, T_COND_SAT_RATED_K)

    for _ in range(3):
        T_cond = max(T_cond, T_evap + 5.0)
        p_evap = p_sat_from_T(T_evap)
        p_cond = p_sat_from_T(T_cond)
        if p_evap >= p_cond:
            p_cond = p_evap * 1.1

        T_suc = safe_PropsSI("T", "P", p_evap, "Q", 1.0, REF) + superheat_desired
        comp = compressor_model(
            p_evap,
            T_suc,
            p_cond,
            N_rpm_comp,
            low_speed_efficiency_loss_scale=low_speed_efficiency_loss_scale,
        )
        cond = condenser_model_NTU(comp["mdot"], comp["h_out"], p_cond, N_rpm_fan, T_outdoor)
        chiller = chiller_model_NTU(comp["mdot"], p_evap, cond["h_cond_out"], T_cool_in, m_dot_cool)

        cond_need = cond.get("Q_ref_need", cond["Q_cond"])
        cond_cap = cond.get("Q_hx_potential", cond["Q_cond"])
        cond_scale = max(500.0, abs(cond_need), abs(cond_cap))
        T_cond += np.clip(8.0 * (cond_need - cond_cap) / cond_scale, -4.0, 4.0)

        evap_need = chiller.get("Q_ref_max", chiller["Q_evap"])
        evap_cap = chiller.get("Q_hx_potential", chiller["Q_evap"])
        evap_scale = max(500.0, abs(evap_need), abs(evap_cap))
        T_evap -= np.clip(5.0 * (evap_need - evap_cap) / evap_scale, -3.0, 3.0)

        T_evap = np.clip(T_evap, T_evap_low, T_evap_high)
        T_cond = np.clip(T_cond, max(T_outdoor + 5.0, T_evap + 5.0), T_COND_SAT_RATED_K)

    def cond_residual_at(T_cond_trial):
        p_evap = p_sat_from_T(T_evap)
        p_cond = p_sat_from_T(max(T_cond_trial, T_evap + 5.0))
        if p_evap >= p_cond:
            p_cond = p_evap * 1.1
        T_suc = safe_PropsSI("T", "P", p_evap, "Q", 1.0, REF) + superheat_desired
        comp = compressor_model(
            p_evap,
            T_suc,
            p_cond,
            N_rpm_comp,
            low_speed_efficiency_loss_scale=low_speed_efficiency_loss_scale,
        )
        cond = condenser_model_NTU(comp["mdot"], comp["h_out"], p_cond, N_rpm_fan, T_outdoor)
        return cond.get("Q_hx_potential", cond["Q_cond"]) - cond.get("Q_ref_need", cond["Q_cond"])

    if cond_residual_at(T_cond) < 0.0:
        lo = T_cond
        hi = T_COND_SAT_RATED_K
        if cond_residual_at(hi) < 0.0:
            T_cond = hi
        else:
            for _ in range(10):
                mid = 0.5 * (lo + hi)
                if cond_residual_at(mid) >= 0.0:
                    hi = mid
                else:
                    lo = mid
            T_cond = hi

    return T_evap, T_cond


# --- 缁勪欢妯″瀷 ---
def pump_flow_from_speed(N_pump_rpm):
    speed = np.clip(N_pump_rpm, pump_speed_min_rpm, pump_speed_max_rpm)
    flow = np.interp(speed, pump_speed_points_rpm, pump_flow_points_L_min)
    return np.clip(flow, pump_flow_min_L_min, pump_flow_max_L_min)


def pump_power_from_flow(flow_L_min):
    flow = np.clip(flow_L_min, pump_flow_min_L_min, pump_flow_max_L_min)
    a, b, c = pump_power_coeff
    return np.maximum(0.0, a * flow**2 + b * flow + c)


def pump_model(N_pump_rpm):
    flow_L_min = pump_flow_from_speed(N_pump_rpm)
    m_dot = flow_L_min / 1000.0 / 60.0 * rho_cool
    W_pump = pump_power_from_flow(flow_L_min)
    return m_dot, W_pump


def fan_model(N_fan_rpm):
    N_fan_rpm_clamped = max(0.0, min(N_fan_rpm, N_fan_max_rpm))
    if N_fan_max_rpm <= 0: return 0.0
    return max(0.0, W_fan_max_power * (N_fan_rpm_clamped / N_fan_max_rpm) ** 3)


def compressor_efficiencies(
    N_rpm,
    pressure_ratio,
    low_speed_efficiency_loss_scale=1.0,
):
    """Return bounded map efficiencies, with a marked low-speed extrapolation.

    The measured map starts at 2000 rpm. Between 1000 and 2000 rpm the first
    map segment is extended linearly in speed; pressure ratio remains clamped
    to the measured map domain. This is an explicit engineering assumption,
    not a replacement for future low-speed compressor measurements.
    """
    loss_scale = float(low_speed_efficiency_loss_scale)
    if not np.isfinite(loss_scale) or loss_scale < 0.0:
        raise ValueError("low_speed_efficiency_loss_scale must be finite and nonnegative")

    speed_for_efficiency = float(
        np.clip(N_rpm, COMPRESSOR_MIN_STEADY_RPM, speed_axis.max())
    )
    pressure_ratio_for_map = float(
        np.clip(pressure_ratio, pr_axis.min(), pr_axis.max())
    )
    point = np.array([speed_for_efficiency, pressure_ratio_for_map])
    try:
        eta_vol = float(eta_vol_interpolator(point)[0])
        eta_is_actual = float(eta_is_interpolator(point)[0])
        if speed_for_efficiency < COMPRESSOR_MAP_MIN_RPM:
            edge_point = np.array(
                [COMPRESSOR_MAP_MIN_RPM, pressure_ratio_for_map]
            )
            eta_vol_edge = float(eta_vol_interpolator(edge_point)[0])
            eta_is_edge = float(eta_is_interpolator(edge_point)[0])
            eta_vol = eta_vol_edge + loss_scale * (eta_vol - eta_vol_edge)
            eta_is_actual = eta_is_edge + loss_scale * (
                eta_is_actual - eta_is_edge
            )
    except Exception:
        eta_vol, eta_is_actual = 0.8, 0.6
    eta_vol = float(np.clip(eta_vol, *ETA_VOL_EXTRAPOLATION_BOUNDS))
    eta_is_actual = float(np.clip(eta_is_actual, *ETA_IS_EXTRAPOLATION_BOUNDS))
    return {
        "eta_vol": eta_vol,
        "eta_is": eta_is_actual,
        "speed_for_efficiency_rpm": speed_for_efficiency,
        "pressure_ratio_for_map": pressure_ratio_for_map,
        "low_speed_efficiency_loss_scale": loss_scale,
        "low_speed_extrapolated": speed_for_efficiency < COMPRESSOR_MAP_MIN_RPM,
        "efficiency_source": (
            (
                "linear_extrapolation_from_2000_3000_rpm"
                if loss_scale == 1.0
                else "scaled_linear_extrapolation_from_2000_3000_rpm"
            )
            if speed_for_efficiency < COMPRESSOR_MAP_MIN_RPM
            else "measured_map_interpolation"
        ),
    }


def _compressor_flow_speed_rpm(N_rpm):
    """Map the off-to-minimum-speed actuator transition to continuous flow."""
    speed = float(N_rpm)
    if speed <= N_COMP_OFF_RPM:
        return 0.0, 0.0
    if speed >= COMPRESSOR_MIN_STEADY_RPM:
        return speed, 1.0
    startup_fraction = (speed - N_COMP_OFF_RPM) / (
        COMPRESSOR_MIN_STEADY_RPM - N_COMP_OFF_RPM
    )
    return COMPRESSOR_MIN_STEADY_RPM * startup_fraction, startup_fraction


def compressor_model(
    p_suc,
    T_suc,
    p_dis,
    N_rpm,
    low_speed_efficiency_loss_scale=1.0,
):
    rho_suc = safe_PropsSI("Dmass", "P", p_suc, "T", T_suc, REF)
    if np.isnan(rho_suc) or rho_suc <= 0:
        return {
            "mdot": 0.0,
            "h_out": np.nan,
            "W_dot_elec": 0.0,
            "eta_vol": 0.0,
            "eta_is": 0.0,
            "speed_for_efficiency_rpm": 0.0,
            "pressure_ratio_for_map": np.nan,
            "low_speed_efficiency_loss_scale": float(
                low_speed_efficiency_loss_scale
            ),
            "low_speed_extrapolated": False,
            "efficiency_source": "invalid_suction_state",
            "flow_speed_rpm": 0.0,
            "startup_fraction": 0.0,
        }
    if p_suc <= 0: p_suc = 1e3
    pr = max(1.0, p_dis / p_suc)
    efficiency = compressor_efficiencies(
        N_rpm,
        pr,
        low_speed_efficiency_loss_scale=low_speed_efficiency_loss_scale,
    )
    eta_vol = efficiency["eta_vol"]
    eta_is_actual = efficiency["eta_is"]
    flow_speed_rpm, startup_fraction = _compressor_flow_speed_rpm(N_rpm)
    mdot = rho_suc * eta_vol * V_disp_m3_per_rev * (flow_speed_rpm / 60.0)
    h1 = safe_PropsSI("Hmass", "P", p_suc, "T", T_suc, REF)
    s1 = safe_PropsSI("Smass", "P", p_suc, "T", T_suc, REF)
    h2s = safe_PropsSI("Hmass", "P", p_dis, "Smass", s1, REF)

    if np.isnan(h1) or np.isnan(h2s):
        return {
            "mdot": mdot,
            "h_out": np.nan,
            "W_dot_elec": 0.0,
            **efficiency,
            "flow_speed_rpm": flow_speed_rpm,
            "startup_fraction": startup_fraction,
        }

    h2 = h1 + (h2s - h1) / max(eta_is_actual, 1e-3)
    return {
        "mdot": mdot,
        "h_out": h2,
        "W_dot_elec": mdot * (h2 - h1) / eta_mech,
        **efficiency,
        "flow_speed_rpm": flow_speed_rpm,
        "startup_fraction": startup_fraction,
    }


def radiator_along(T_cool_in, T_air_in, m_dot_cool):
    T_cool = T_cool_in
    Q_total = 0.0
    if m_dot_cool <= 1e-9: return {"T_out": T_cool_in, "Q_total": 0.0}
    for _ in range(N_rad):
        Q_seg = h_rad_local * A_rad_seg * (T_cool - T_air_in)
        T_cool -= Q_seg / (m_dot_cool * cp_cool + 1e-12)
        Q_total += Q_seg
    return {"T_out": T_cool, "Q_total": Q_total}


# === 鏍稿績淇敼锛氶瞾妫掔殑鍐锋澘鎹㈢儹妯″瀷 ===
def cold_plate_fluid_exchange(T_plate_wall_array=None, T_cool_in=None, m_dot_cool=None, **kwargs):
    if T_plate_wall_array is None:
        T_plate_wall_array = kwargs.pop("T_plate_wall", None)
    if T_cool_in is None:
        T_cool_in = kwargs.pop("T_cool_in_to_plate", None)
    if T_plate_wall_array is None or T_cool_in is None or m_dot_cool is None:
        raise TypeError("cold_plate_fluid_exchange requires plate temperature, coolant inlet temperature, and coolant flow")

    if np.isscalar(T_plate_wall_array):
        T_walls = np.full(N_bp, T_plate_wall_array)
    else:
        T_walls = T_plate_wall_array

    if m_dot_cool < 1e-6:
        h_dyn = 50.0
    else:
        h_dyn = max(50.0, h_bp_nominal * (m_dot_cool / m_dot_nominal) ** 0.8)

    T_local = T_cool_in
    Q_total = 0.0
    T_fluid_profile = []

    for i in range(N_bp):
        T_wall = T_walls[i]
        DT_in = T_wall - T_local

        # 棰勪及鍑哄彛娓╁害
        Q_guess = h_dyn * A_bp_seg * DT_in
        T_next = T_local + Q_guess / (m_dot_cool * cp_cool + 1e-12)
        DT_out = T_wall - T_next

        # --- 椴佹鐨?LMTD 璁＄畻 ---
        # 濡傛灉娓╁樊鏋佸皬 (渚嬪 < 1e-4) 鎴栬€呰繘鍑哄彛娓╁樊鍑犱箮鐩哥瓑 (ratio ~ 1)
        # 鐩存帴浣跨敤绠楁湳骞冲潎锛岄伩鍏?log(1)=0 鎴栬€?log(璐熸暟)
        if abs(DT_in) < 1e-4 or abs(DT_out) < 1e-4 or abs(DT_in - DT_out) < 1e-4:
            DT_LMTD = (DT_in + DT_out) / 2.0
        else:
            # Ensure the inlet/outlet temperature differences have the same sign.
            if DT_in * DT_out <= 0:
                # 杩欑鎯呭喌鐞嗚涓婁笉搴斿彂鐢熷湪鍗曠浉鎹㈢儹涓紝闄ら潪闇囪崱
                DT_LMTD = (DT_in + DT_out) / 2.0
            else:
                try:
                    DT_LMTD = (DT_in - DT_out) / np.log(DT_in / DT_out)
                except:
                    DT_LMTD = (DT_in + DT_out) / 2.0

        Q_i = h_dyn * A_bp_seg * DT_LMTD
        T_next = T_local + Q_i / (m_dot_cool * cp_cool + 1e-12)

        T_fluid_profile.append((T_local + T_next) / 2.0)
        Q_total += Q_i
        T_local = T_next

    return {
        "T_out": T_local,
        "Q_total": Q_total,
        "T_fluid_profile": np.array(T_fluid_profile)
    }


def chiller_model_NTU(mdot_ref, p_chiller, h_chiller_in, T_cool_in, m_dot_cool):
    T_sat = safe_PropsSI("T", "P", p_chiller, "Q", 0.0, REF)
    h_f = safe_PropsSI("Hmass", "P", p_chiller, "Q", 0.0, REF)
    h_g = safe_PropsSI("Hmass", "P", p_chiller, "Q", 1.0, REF)

    C_cool = max(1e-6, m_dot_cool * cp_cool)
    h_cool = max(
        50.0,
        EVAP_UA_FACTOR
        * EVAPORATOR_UA_SCALE
        * h_chiller_cool_nom
        * (m_dot_cool / m_dot_nominal) ** EVAP_FLOW_EXP,
    )
    h_ref = max(
        500.0,
        EVAP_UA_FACTOR
        * EVAPORATOR_UA_SCALE
        * h_chiller_ref_nom
        * (mdot_ref / m_dot_ref_nominal) ** 0.8,
    )
    UA = 1.0 / (1 / h_cool / A_chiller + 1 / h_ref / A_chiller)
    NTU = UA / C_cool
    eps = 1.0 - np.exp(-NTU)

    Q_hx = eps * C_cool * max(0.0, T_cool_in - T_sat)
    Q_ref_max = 0.0
    if mdot_ref <= 1e-9 or np.isnan(h_chiller_in) or np.isnan(Q_hx):
        Q_chiller = 0.0
        h_out = h_chiller_in
    else:
        T_ref_out_limit = T_sat + superheat_desired
        h_ref_out_limit = safe_PropsSI("Hmass", "P", p_chiller, "T", T_ref_out_limit, REF)
        if np.isnan(h_ref_out_limit):
            h_ref_out_limit = h_g
        Q_ref_max = EVAP_CAP_FACTOR * max(0.0, mdot_ref * (h_ref_out_limit - h_chiller_in))
        Q_chiller = min(Q_hx, Q_ref_max)
        h_out = h_chiller_in + Q_chiller / mdot_ref

    if np.isnan(h_f) or np.isnan(h_g) or h_g <= h_f or np.isnan(h_out):
        x_out = np.nan
    else:
        x_out = (h_out - h_f) / (h_g - h_f)

    T_cool_out = T_cool_in - Q_chiller / C_cool
    return {
        "Q_chiller": Q_chiller,
        "Q_evap": Q_chiller,
        "h_out": h_out,
        "T_cool_out": T_cool_out,
        "x_out": x_out,
        "T_sat": T_sat,
        "Q_hx_potential": Q_hx,
        "Q_ref_max": Q_ref_max,
    }


def evaporator_model_NTU(mdot_ref, p_evap, h_evap_in, T_cool_in, m_dot_cool):
    return chiller_model_NTU(mdot_ref, p_evap, h_evap_in, T_cool_in, m_dot_cool)


def suction_accumulator_model(mdot_ref, p_suc, h_in):
    T_sat = safe_PropsSI("T", "P", p_suc, "Q", 0.0, REF)
    h_f = safe_PropsSI("Hmass", "P", p_suc, "Q", 0.0, REF)
    h_g = safe_PropsSI("Hmass", "P", p_suc, "Q", 1.0, REF)

    if mdot_ref <= 1e-9 or np.isnan(h_in) or np.isnan(T_sat) or np.isnan(h_f) or np.isnan(h_g) or h_g <= h_f:
        return {
            "x_in": np.nan,
            "h_out": h_in,
            "T_out": T_sat,
            "liquid_hold_rate": 0.0,
            "risk_liquid_slugging": False
        }

    x_in = (h_in - h_f) / (h_g - h_f)
    if x_in < 1.0:
        T_out = T_sat + 0.1
        h_out = safe_PropsSI("Hmass", "P", p_suc, "T", T_out, REF)
        if np.isnan(h_out):
            h_out = h_g
        liquid_hold_rate = mdot_ref * max(0.0, 1.0 - x_in)
        risk_liquid_slugging = True
    else:
        h_out = h_in
        T_out = safe_PropsSI("T", "P", p_suc, "Hmass", h_out, REF)
        if np.isnan(T_out) or T_out <= T_sat:
            T_out = T_sat + 0.1
        liquid_hold_rate = 0.0
        risk_liquid_slugging = False

    return {
        "x_in": x_in,
        "h_out": h_out,
        "T_out": T_out,
        "liquid_hold_rate": liquid_hold_rate,
        "risk_liquid_slugging": risk_liquid_slugging
    }


def condenser_model_NTU(mdot_ref, h_in, p_cond, N_fan, T_air):
    T_sat = safe_PropsSI("T", "P", p_cond, "Q", 0.0, REF)
    h_liq = safe_PropsSI("Hmass", "P", p_cond, "Q", 0.0, REF)
    T_liq_target = max(200.0, T_sat - subcooling_desired)
    h_liq_target = safe_PropsSI("Hmass", "P", p_cond, "T", T_liq_target, REF)
    if np.isnan(h_liq_target) or h_liq_target > h_liq:
        h_liq_target = h_liq
    Q_ref_need = max(0.0, mdot_ref * (h_in - h_liq_target))

    N_fan_eff = max(0.0, min(N_fan, N_fan_max_rpm))
    m_air = m_dot_air_nominal * (N_fan_eff / N_fan_max_rpm)
    C_air = max(1e-6, m_air * cp_air)
    h_air = max(5.0, h_cond_air_nom * (m_air / m_dot_air_nominal) ** 0.8)
    h_ref = max(500.0, h_cond_ref_nom * (mdot_ref / m_dot_ref_nominal) ** 0.8)
    UA = 1.0 / (1 / h_air / A_cond + 1 / h_ref / A_cond)
    NTU = UA / C_air
    eps = 1.0 - np.exp(-NTU)

    Q_hx = eps * C_air * max(0.0, T_sat - T_air)
    Q_act = min(Q_hx, Q_ref_need)
    h_out = h_in - Q_act / (mdot_ref + 1e-9)
    h_cond_out = max(h_out, h_liq_target)
    T_cond_out = safe_PropsSI("T", "P", p_cond, "Hmass", h_cond_out, REF)
    actual_subcooling = max(0.0, T_sat - T_cond_out) if not np.isnan(T_cond_out) else 0.0
    T_air_out = T_air + Q_act / C_air
    return {
        "h_cond_out": h_cond_out,
        "Q_cond": Q_act,
        "T_sat": T_sat,
        "T_cond_out": T_cond_out,
        "subcooling": actual_subcooling,
        "T_air_out": T_air_out,
        "Q_hx_potential": Q_hx,
        "Q_ref_need": Q_ref_need,
    }


def _quantize_for_cache(value, step):
    return round(float(value) / step) * step


def _refrigeration_cycle_cache_key(
    N_rpm_comp,
    N_rpm_fan,
    T_cool_in,
    m_dot_cool,
    T_outdoor,
    low_speed_efficiency_loss_scale,
):
    steps = _REFRIGERATION_CYCLE_CACHE_STEPS
    return (
        _quantize_for_cache(N_rpm_comp, steps["speed_rpm"]),
        _quantize_for_cache(N_rpm_fan, steps["speed_rpm"]),
        _quantize_for_cache(T_cool_in, steps["temperature_K"]),
        _quantize_for_cache(m_dot_cool, steps["m_dot_kg_s"]),
        _quantize_for_cache(T_outdoor, steps["temperature_K"]),
        float(low_speed_efficiency_loss_scale),
    )


def clear_refrigeration_cycle_cache():
    global _REFRIGERATION_CYCLE_CACHE_HITS, _REFRIGERATION_CYCLE_CACHE_MISSES
    _REFRIGERATION_CYCLE_CACHE.clear()
    _REFRIGERATION_CYCLE_CACHE_HITS = 0
    _REFRIGERATION_CYCLE_CACHE_MISSES = 0


def refrigeration_cycle_cache_info():
    return {
        "size": len(_REFRIGERATION_CYCLE_CACHE),
        "hits": _REFRIGERATION_CYCLE_CACHE_HITS,
        "misses": _REFRIGERATION_CYCLE_CACHE_MISSES,
        **_REFRIGERATION_CYCLE_CACHE_STEPS,
    }


def _run_refrigeration_cycle_uncached(
    N_rpm_comp=None,
    N_rpm_fan=None,
    T_cool_in=None,
    m_dot_cool=None,
    T_outdoor=None,
    low_speed_efficiency_loss_scale=1.0,
    **kwargs,
):
    if N_rpm_comp is None:
        N_rpm_comp = kwargs.pop("N_comp_rpm", None)
    if N_rpm_fan is None:
        N_rpm_fan = kwargs.pop("N_fan_rpm", N_rpm_comp)
    if T_cool_in is None:
        T_cool_in = kwargs.pop("T_cool_in_to_evap", None)
    if T_outdoor is None:
        T_outdoor = kwargs.pop("T_env", T_env)
    if N_rpm_comp is None or N_rpm_fan is None or T_cool_in is None or m_dot_cool is None:
        raise TypeError("run_refrigeration_cycle requires compressor speed, fan speed, coolant inlet temperature, and coolant flow")

    if N_rpm_comp <= N_COMP_OFF_RPM:
        T_evap_sat = float(np.clip(T_cool_in - 5.0, T_EVAP_SAT_RATED_K, max(T_EVAP_SAT_RATED_K + 1.0, T_cool_in - 0.5)))
        T_cond_sat = float(np.clip(T_outdoor + 5.0, T_evap_sat + 5.0, T_COND_SAT_RATED_K))
        p_evap = p_sat_from_T(T_evap_sat)
        p_cond = max(p_sat_from_T(T_cond_sat), p_evap * 1.1)
        h_vap = safe_PropsSI("Hmass", "P", p_evap, "Q", 1.0, REF)
        h_liq = safe_PropsSI("Hmass", "P", p_cond, "Q", 0.0, REF)
        return {
            "Q_chiller": 0.0, "Q_evap": 0.0, "Q_cond": 0.0,
            "W_comp": 0.0, "W_fan": fan_model(N_rpm_fan), "T_cool_out": T_cool_in,
            "T_evap_sat": T_evap_sat, "T_cond_sat": T_cond_sat,
            "T_cond_out": T_cond_sat, "subcooling": 0.0, "suction_superheat": 0.0,
            "p_evap": p_evap, "p_cond": p_cond,
            "h_1": h_vap, "h_2": h_vap, "h_3": h_liq, "h_4": h_liq,
            "p_1": p_evap, "p_2": p_cond, "p_3": p_cond, "p_4": p_evap,
            "x_chiller_out": 1.0, "x_evap_out": 1.0, "x_acc_in": 1.0,
            "T_suc_after_acc": T_evap_sat, "h_suc_after_acc": h_vap,
            "liquid_hold_rate": 0.0, "risk_liquid_slugging": False,
            "COP_system": 0.0, "COP_Carnot": 0.0, "exergy_efficiency": 0.0,
            "E_D_comp": 0.0, "E_D_evap": 0.0, "E_D_cond": 0.0, "E_D_exp": 0.0,
            "E_D_total": 0.0, "E_D_comp_ratio": 0.0, "E_D_evap_ratio": 0.0,
            "E_D_cond_ratio": 0.0, "E_D_exp_ratio": 0.0,
            "compressor_eta_vol": 0.0, "compressor_eta_is": 0.0,
            "compressor_low_speed_efficiency_loss_scale": float(
                low_speed_efficiency_loss_scale
            ),
            "compressor_flow_speed_rpm": 0.0,
            "compressor_startup_fraction": 0.0,
            "compressor_low_speed_extrapolated": False,
            "compressor_efficiency_source": "off",
        }

    W_fan = fan_model(N_rpm_fan)
    T_evap_sat, T_cond_sat = solve_saturation_temperatures(
        N_rpm_comp,
        N_rpm_fan,
        T_cool_in,
        m_dot_cool,
        T_outdoor,
        low_speed_efficiency_loss_scale=low_speed_efficiency_loss_scale,
    )
    p_evap = p_sat_from_T(T_evap_sat)
    p_cond = p_sat_from_T(T_cond_sat)

    if p_evap >= p_cond: p_cond = p_evap * 1.1

    T_suc = safe_PropsSI("T", "P", p_evap, "Q", 1.0, REF) + superheat_desired
    comp_for_hx = compressor_model(
        p_evap,
        T_suc,
        p_cond,
        N_rpm_comp,
        low_speed_efficiency_loss_scale=low_speed_efficiency_loss_scale,
    )

    cond = condenser_model_NTU(comp_for_hx["mdot"], comp_for_hx["h_out"], p_cond, N_rpm_fan, T_outdoor)
    chiller = chiller_model_NTU(comp_for_hx["mdot"], p_evap, cond["h_cond_out"], T_cool_in, m_dot_cool)
    h_suc_target = safe_PropsSI("Hmass", "P", p_evap, "T", T_suc, REF)
    mdot_comp_raw = comp_for_hx["mdot"]
    mdot_eff = mdot_comp_raw
    for _ in range(2):
        if mdot_eff <= 1e-9 or h_suc_target <= cond["h_cond_out"]:
            break
        mdot_txv = chiller.get("Q_hx_potential", chiller["Q_evap"]) / (h_suc_target - cond["h_cond_out"])
        mdot_next = float(np.clip(mdot_txv, 0.0, mdot_comp_raw))
        if abs(mdot_next - mdot_eff) <= max(1e-9, 1e-3 * mdot_comp_raw):
            mdot_eff = mdot_next
            break
        mdot_eff = mdot_next
        comp_for_hx = dict(comp_for_hx)
        comp_for_hx["mdot"] = mdot_eff
        cond = condenser_model_NTU(mdot_eff, comp_for_hx["h_out"], p_cond, N_rpm_fan, T_outdoor)
        chiller = chiller_model_NTU(mdot_eff, p_evap, cond["h_cond_out"], T_cool_in, m_dot_cool)

    acc = suction_accumulator_model(comp_for_hx["mdot"], p_evap, chiller["h_out"])
    suction_superheat = max(0.0, acc["T_out"] - safe_PropsSI("T", "P", p_evap, "Q", 1.0, REF))
    comp = compressor_model(
        p_evap,
        T_suc,
        p_cond,
        N_rpm_comp,
        low_speed_efficiency_loss_scale=low_speed_efficiency_loss_scale,
    )
    if comp["mdot"] > 1e-9:
        comp_mdot_raw = comp["mdot"]
        comp = dict(comp)
        comp["mdot"] = comp_for_hx["mdot"]
        comp["W_dot_elec"] *= comp_for_hx["mdot"] / comp_mdot_raw
    exergy_metrics = calculate_refrigeration_exergy_metrics(
        mdot_ref=comp_for_hx["mdot"],
        p_evap=p_evap,
        p_cond=p_cond,
        h_comp_in=acc["h_out"],
        h_comp_out=comp["h_out"],
        h_cond_out=cond["h_cond_out"],
        h_evap_out=chiller["h_out"],
        W_comp=comp["W_dot_elec"],
        W_fan=W_fan,
        Q_evap=chiller["Q_evap"],
        Q_cond=cond["Q_cond"],
        T_cool_in=T_cool_in,
        T_cool_out=chiller["T_cool_out"],
        T_air_in=T_outdoor,
        T_air_out=cond["T_air_out"],
    )

    return {
        "Q_chiller": chiller["Q_chiller"], "Q_evap": chiller["Q_evap"], "Q_hx_potential": chiller["Q_hx_potential"], "Q_ref_max": chiller["Q_ref_max"],
        "Q_cond": cond["Q_cond"],
        "W_comp": comp["W_dot_elec"], "W_fan": W_fan, "T_cool_out": chiller["T_cool_out"],
        "T_evap_sat": T_evap_sat,
        "T_cond_sat": T_cond_sat,
        "T_cond_out": cond["T_cond_out"],
        "subcooling": cond["subcooling"],
        "suction_superheat": suction_superheat,
        "p_evap": p_evap,
        "p_cond": p_cond,
        "h_1": acc["h_out"],
        "h_2": comp["h_out"],
        "h_3": cond["h_cond_out"],
        "h_4": cond["h_cond_out"],
        "p_1": p_evap,
        "p_2": p_cond,
        "p_3": p_cond,
        "p_4": p_evap,
        "x_chiller_out": chiller["x_out"], "x_evap_out": chiller["x_out"], "x_acc_in": acc["x_in"],
        "T_suc_after_acc": acc["T_out"], "h_suc_after_acc": acc["h_out"],
        "liquid_hold_rate": acc["liquid_hold_rate"],
        "risk_liquid_slugging": acc["risk_liquid_slugging"],
        "compressor_eta_vol": comp["eta_vol"],
        "compressor_eta_is": comp["eta_is"],
        "compressor_low_speed_efficiency_loss_scale": comp[
            "low_speed_efficiency_loss_scale"
        ],
        "compressor_flow_speed_rpm": comp["flow_speed_rpm"],
        "compressor_startup_fraction": comp["startup_fraction"],
        "compressor_low_speed_extrapolated": comp["low_speed_extrapolated"],
        "compressor_efficiency_source": comp["efficiency_source"],
        **exergy_metrics,
    }


def run_refrigeration_cycle(
    N_rpm_comp=None,
    N_rpm_fan=None,
    T_cool_in=None,
    m_dot_cool=None,
    T_outdoor=None,
    low_speed_efficiency_loss_scale=1.0,
    **kwargs,
):
    global _REFRIGERATION_CYCLE_CACHE_HITS, _REFRIGERATION_CYCLE_CACHE_MISSES
    if N_rpm_comp is None:
        N_rpm_comp = kwargs.pop("N_comp_rpm", None)
    if N_rpm_fan is None:
        N_rpm_fan = kwargs.pop("N_fan_rpm", N_rpm_comp)
    if T_cool_in is None:
        T_cool_in = kwargs.pop("T_cool_in_to_evap", None)
    if T_outdoor is None:
        T_outdoor = kwargs.pop("T_env", T_env)
    if N_rpm_comp is None or N_rpm_fan is None or T_cool_in is None or m_dot_cool is None:
        raise TypeError("run_refrigeration_cycle requires compressor speed, fan speed, coolant inlet temperature, and coolant flow")

    key = _refrigeration_cycle_cache_key(
        N_rpm_comp,
        N_rpm_fan,
        T_cool_in,
        m_dot_cool,
        T_outdoor,
        low_speed_efficiency_loss_scale,
    )
    cached = _REFRIGERATION_CYCLE_CACHE.get(key)
    if cached is not None:
        _REFRIGERATION_CYCLE_CACHE_HITS += 1
        return dict(cached)

    _REFRIGERATION_CYCLE_CACHE_MISSES += 1
    result = _run_refrigeration_cycle_uncached(
        N_rpm_comp=N_rpm_comp,
        N_rpm_fan=N_rpm_fan,
        T_cool_in=T_cool_in,
        m_dot_cool=m_dot_cool,
        T_outdoor=T_outdoor,
        low_speed_efficiency_loss_scale=low_speed_efficiency_loss_scale,
    )
    if len(_REFRIGERATION_CYCLE_CACHE) >= _REFRIGERATION_CYCLE_CACHE_MAX_SIZE:
        _REFRIGERATION_CYCLE_CACHE.clear()
    _REFRIGERATION_CYCLE_CACHE[key] = dict(result)
    return dict(result)





