import numpy as np

from thermal_batch_config import (
    MPC_MIN_HOLD_TIME_S,
    MPC_REV_DELTA_T_LIM_C,
    MPC_REV_GAMMA_HOT_LIM,
    MPC_REV_H_DOWN_CELL_SCALE_C,
    MPC_REV_J_OFF,
    MPC_REV_J_ON,
    MPC_REV_W_DELTA_T,
    MPC_REV_W_GAMMA_HOT,
    MPC_REV_W_H_DOWN,
    SIM_DT,
)
from thermal_system import (
    A_bp_total,
    C_tank,
    cp_cool,
    cold_plate_fluid_exchange,
    h_bp_nominal,
    m_dot_nominal,
    pump_model,
    radiator_along,
    rho_cool,
    run_refrigeration_cycle,
)


DEFAULT_REFRIGERATION_DYNAMICS = {
    "tau_comp_s": 5.0,
    "tau_pump_s": 5.0,
    "tau_fan_s": 3.0,
    "tau_evap_s": 45.0,
    "tau_cond_s": 75.0,
    "theta_evap_input_s": 20.0,
    "theta_pump_flow_s": 5.0,
    "tau_pipe_supply_s": 15.0,
    "tau_pipe_return_s": 20.0,
}


def first_order_lag(previous, target, dt, tau):
    if tau <= 0.0 or dt <= 0.0:
        return float(target)
    alpha = dt / (tau + dt)
    return float(previous + alpha * (target - previous))


def delay_line_value(previous_values, input_value, initial_value, delay_steps):
    steps = int(max(0, delay_steps))
    if steps <= 0:
        return float(input_value), []
    history = list(previous_values) if previous_values is not None else []
    if not history:
        history = [float(initial_value)] * steps
    while len(history) < steps:
        history.insert(0, float(initial_value))
    history = history[-steps:]
    output = float(history.pop(0))
    history.append(float(input_value))
    return output, history


def pipe_delay_steps(tau_s, dt):
    if tau_s <= 0.0 or dt <= 0.0:
        return 0
    return int(max(1, round(float(tau_s) / float(dt))))


def direction_from_reversed(is_reversed):
    return -1 if is_reversed else 1


def is_reversed_from_direction(direction):
    return direction < 0


def initialize_refrigeration_dynamic_state(N_comp_cmd, N_pump_cmd):
    N_fan_cmd = staged_fan_speed(N_comp_cmd)
    return {
        "N_comp_eff": float(N_comp_cmd),
        "N_pump_eff": float(N_pump_cmd),
        "N_fan_eff": float(N_fan_cmd),
        "Q_evap_eff": 0.0,
        "Q_cond_eff": 0.0,
        "T_pipe_supply_history_K": [],
        "T_pipe_return_history_K": [],
    }


def build_pack_config(
    total_current=560.0,
    initial_soc=0.95,
    initial_temp_c=25.0,
    uniform_air_convection=False,
    dynamic_resistance_update=True,
):
    """构造电池包模型的统一参数配置。

    本函数用于集中管理 `BatteryPack` 的公共建模参数，避免 PID、MPC
    与 on-off 等控制脚本分别维护一套重复配置。函数本身不改变模型功能，
    仅负责生成用于实例化 `BatteryPack` 的参数字典。

    参数说明：
    - total_current：电池包总电流，单位 A
    - initial_soc：初始荷电状态
    - initial_temp_c：初始温度，单位 ℃
    - uniform_air_convection：是否采用均匀空气对流边界
    - dynamic_resistance_update：是否根据实时 SOC 与温度更新内阻

    返回值：
    - config：可直接传入 `BatteryPack(config)` 的配置字典
    """
    return {
        "rows": 4,
        "cols": 13,
        "capacity": 280,
        "initial_soc": initial_soc,
        "initial_temp_c": initial_temp_c,
        "total_current": total_current,
        "cell_thermal_mass": 4747.0,
        "k_intercell": 0.5,
        "h_plate_convection": 10.0,
        "h_air_convection_edge": 0.1,
        "h_air_convection_corner": 0.2,
        "uniform_air_convection": uniform_air_convection,
        "dynamic_resistance_update": dynamic_resistance_update,
    }


def maybe_reverse_flow(
    temps,
    is_reversed,
    current_time,
    last_reverse_time,
    reverse_flow_enabled=False,
    temp_diff_limit=0.5,
    min_reverse_interval=200.0,
):
    """依据温差阈值与最小切换间隔判定是否执行流向反转。

    该函数实现热管理系统中的监督层切换逻辑。判据采用电池包全局温差
    `max(T_cell) - min(T_cell)` 作为触发量，并引入最小翻转时间间隔，
    以避免流向在阈值附近频繁切换。

    参数说明：
    - temps：当前时刻全部电芯温度数组
    - is_reversed：当前流向状态，`False` 表示正向，`True` 表示反向
    - current_time：当前仿真时刻，单位 s
    - last_reverse_time：上一次执行流向翻转的时刻，单位 s
    - reverse_flow_enabled：流向反转总开关
    - temp_diff_limit：触发翻转的全局温差阈值，单位 ℃
    - min_reverse_interval：两次翻转之间的最小时间间隔，单位 s

    返回值：
    - delta_T：当前全局温差，单位 ℃
    - is_reversed：更新后的流向状态
    - last_reverse_time：更新后的上次翻转时刻
    """
    delta_T = float(np.max(temps) - np.min(temps))
    if (
        reverse_flow_enabled
        and delta_T > temp_diff_limit
        and (current_time - last_reverse_time) > min_reverse_interval
    ):
        return delta_T, (not is_reversed), current_time
    return delta_T, is_reversed, last_reverse_time


class SupervisoryFlowController:
    """Shared comprehensive-index flow reversal supervisor."""

    flow_mode = "supervised"

    def __init__(self, dt=SIM_DT, min_hold_time=MPC_MIN_HOLD_TIME_S):
        self.dt = dt
        self.min_hold_time = min_hold_time
        self.delta_t_lim = MPC_REV_DELTA_T_LIM_C
        self.h_down_cell_scale = MPC_REV_H_DOWN_CELL_SCALE_C
        self.gamma_hot_lim = MPC_REV_GAMMA_HOT_LIM
        self.j_rev_on = MPC_REV_J_ON
        self.j_rev_off = MPC_REV_J_OFF
        self.w_delta_t = MPC_REV_W_DELTA_T
        self.w_h_down = MPC_REV_W_H_DOWN
        self.w_gamma_hot = MPC_REV_W_GAMMA_HOT
        self.direction = 1
        self.last_switch_time = -1e12
        self.switch_armed = True
        self.hot_time_s = None
        self.obs_time_s = 0.0
        self.last_flow_info = self._flow_info({}, switched=False)

    @property
    def is_reversed(self):
        return is_reversed_from_direction(self.direction)

    def _ensure_hot_history(self, grid_c):
        n_cells = int(np.asarray(grid_c).size)
        if self.hot_time_s is None or self.hot_time_s.size != n_cells:
            self.hot_time_s = np.zeros(n_cells, dtype=float)
            self.obs_time_s = 0.0

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
    def _accumulate_hot_time(hot_time_s, grid_c, duration_s):
        flat = np.asarray(grid_c, dtype=float).reshape(-1)
        hottest = np.flatnonzero(np.isclose(flat, np.max(flat)))
        if hottest.size == 0:
            return hot_time_s
        hot_time_s[hottest] += duration_s / hottest.size
        return hot_time_s

    @staticmethod
    def _gamma_hot_from_time(hot_time_s, obs_time_s):
        n_cells = hot_time_s.size
        if n_cells <= 1 or obs_time_s <= 0.0:
            return 0.0
        x = hot_time_s / obs_time_s
        uniform = 1.0 / n_cells
        return float(np.sqrt(np.sum((x - uniform) ** 2) / (n_cells - 1)))

    def evaluate(self, predicted_grids_c, direction=None):
        grids = np.asarray(predicted_grids_c, dtype=float)
        if grids.ndim == 2:
            grids = grids.reshape(1, *grids.shape)
        if grids.size == 0:
            raise ValueError("predicted_grids_c must contain at least one grid")

        direction = self.direction if direction is None else direction
        self._ensure_hot_history(grids[0])
        downstream_cols = self._downstream_columns(grids.shape[2], direction)
        n_downstream_cells = max(1, grids.shape[1] * len(downstream_cols))
        h_down_lim = self.h_down_cell_scale * n_downstream_cells

        hot_time_s = self.hot_time_s.copy()
        obs_time_s = float(self.obs_time_s)
        delta_t_series = []
        h_down_series = []
        gamma_hot_series = []
        j_rev_series = []

        for grid_c in grids:
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

        return {
            "delta_t_pred_max": float(np.max(delta_t_series)),
            "h_down_pred_max": float(np.max(h_down_series)),
            "gamma_hot_pred_max": float(np.max(gamma_hot_series)),
            "j_rev_pred_max": float(np.max(j_rev_series)),
            "delta_t_pred_0": float(delta_t_series[0]),
            "h_down_pred_0": float(h_down_series[0]),
            "gamma_hot_pred_0": float(gamma_hot_series[0]),
            "j_rev_pred_0": float(j_rev_series[0]),
        }

    def _flow_info(self, metrics, switched):
        return {
            "mode": self.flow_mode,
            "direction": self.direction,
            "flow_direction_d": self.direction,
            "delta_t_pred_max": metrics.get("delta_t_pred_max", np.nan),
            "h_down_pred_max": metrics.get("h_down_pred_max", np.nan),
            "gamma_hot_pred_max": metrics.get("gamma_hot_pred_max", np.nan),
            "j_rev_pred_max": metrics.get("j_rev_pred_max", np.nan),
            "switched": bool(switched),
            "j_rev_on": float(self.j_rev_on),
            "j_rev_off": float(self.j_rev_off),
        }

    def update(self, temps_c, current_time, flow_enabled=True):
        grid_c = np.asarray(temps_c, dtype=float)
        if grid_c.ndim != 2:
            raise ValueError("temps_c must be a 2D temperature grid")
        self._ensure_hot_history(grid_c)

        if not flow_enabled:
            self.direction = 1
            metrics = self.evaluate(grid_c, direction=self.direction)
            self.last_flow_info = self._flow_info(metrics, switched=False)
            return False

        metrics = self.evaluate(grid_c, direction=self.direction)
        pre_switch_metrics = dict(metrics)
        if metrics["j_rev_pred_max"] <= self.j_rev_off:
            self.switch_armed = True

        can_switch = (current_time - self.last_switch_time) >= self.min_hold_time
        switched = False
        if self.switch_armed and can_switch and metrics["j_rev_pred_max"] > self.j_rev_on:
            self.direction *= -1
            self.last_switch_time = current_time
            self.switch_armed = False
            switched = True
            metrics = self.evaluate(grid_c, direction=self.direction)

        self.last_flow_info = self._flow_info(metrics, switched=switched)
        self.last_flow_info["pre_switch_delta_t_pred_max"] = float(pre_switch_metrics["delta_t_pred_max"])
        self.last_flow_info["pre_switch_h_down_pred_max"] = float(pre_switch_metrics["h_down_pred_max"])
        self.last_flow_info["pre_switch_gamma_hot_pred_max"] = float(pre_switch_metrics["gamma_hot_pred_max"])
        self.last_flow_info["pre_switch_j_rev_pred_max"] = float(pre_switch_metrics["j_rev_pred_max"])
        self.last_flow_info["triggered_by_prediction"] = bool(
            switched and pre_switch_metrics["j_rev_pred_max"] > self.j_rev_on
        )
        return switched

    def update_after_step(self, temps_c):
        grid_c = np.asarray(temps_c, dtype=float)
        self._ensure_hot_history(grid_c)
        self._accumulate_hot_time(self.hot_time_s, grid_c, self.dt)
        self.obs_time_s += self.dt


def staged_fan_speed(N_comp_rpm):
    """Independent four-stage condenser fan command, decoupled from compressor speed."""
    if N_comp_rpm < 2000.0:
        return 0.0
    if N_comp_rpm <= 3500.0:
        return 800.0
    if N_comp_rpm <= 5000.0:
        return 1200.0
    return 1800.0


def simulate_thermal_loop_step(
    pack,
    T_tank_K,
    T_plate_K_array,
    N_comp_cmd,
    N_pump_cmd,
    T_outdoor,
    dt,
    is_reversed=False,
    C_plate_node=None,
    compressor_power_scale=1.0,
    dynamic_state=None,
    dynamics=None,
):
    """推进一步外部热管理回路的热状态与功耗计算。

    本函数描述冷却回路侧的单步离散更新过程，主要包括散热器、冷板、
    冷却液箱以及制冷循环的状态推进。函数输出更新后的冷板温度分布、
    冷却液箱温度和附件功耗，用于与电池包模型进行耦合仿真。

    需要注意的是，本函数不负责电池包内部 electro-thermal 状态的推进；
    电芯温度、SOC 等状态仍由调用方通过 `pack.step()` 更新。

    参数说明：
    - pack：`BatteryPack` 实例，用于读取当前电芯温度场与热参数
    - T_tank_K：当前冷却液箱温度，单位 K
    - T_plate_K_array：当前各列冷板节点温度数组，单位 K
    - N_comp_cmd：压缩机转速指令，单位 rpm
    - N_pump_cmd：水泵转速指令，单位 rpm
    - T_outdoor：室外环境温度，单位 K
    - dt：仿真步长，单位 s
    - is_reversed：当前是否采用反向流动
    - C_plate_node：单列冷板等效热容，单位 J/K
    - compressor_power_scale：压缩机功耗修正系数

    返回值：
    - T_tank_K：更新后的冷却液箱温度，单位 K
    - T_plate_K_array：更新后的各列冷板节点温度，单位 K
    - W_comp_real：压缩机实际功耗，单位 W
    - W_pump_val：水泵功耗，单位 W
    - W_fan_real：风扇功耗，单位 W
    """
    T_plate_next = np.array(T_plate_K_array, dtype=float, copy=True)
    n_cols = T_plate_next.size

    if C_plate_node is None:
        C_plate_node = 6000.0 / max(n_cols, 1)

    dynamics = DEFAULT_REFRIGERATION_DYNAMICS if dynamics is None else dynamics
    if dynamic_state is None:
        dynamic_state = initialize_refrigeration_dynamic_state(N_comp_cmd, N_pump_cmd)

    N_fan_cmd = staged_fan_speed(N_comp_cmd)
    N_comp_eff = first_order_lag(
        dynamic_state.get("N_comp_eff", N_comp_cmd),
        N_comp_cmd,
        dt,
        dynamics.get("tau_comp_s", 0.0),
    )
    N_pump_eff = first_order_lag(
        dynamic_state.get("N_pump_eff", N_pump_cmd),
        N_pump_cmd,
        dt,
        dynamics.get("tau_pump_s", 0.0),
    )
    N_fan_eff = first_order_lag(
        dynamic_state.get("N_fan_eff", N_fan_cmd),
        N_fan_cmd,
        dt,
        dynamics.get("tau_fan_s", 0.0),
    )

    m_dot_cool, W_pump_val = pump_model(N_pump_eff)
    refrig = run_refrigeration_cycle(N_comp_eff, N_fan_eff, T_tank_K, m_dot_cool, T_outdoor)
    W_comp_real = refrig.get("W_comp", 0.0) * compressor_power_scale
    W_fan_real = refrig.get("W_fan", 0.0)
    Q_evap_ss = refrig.get("Q_evap", 0.0)
    Q_cond_ss = refrig.get("Q_cond", 0.0)
    Q_evap_eff = first_order_lag(
        dynamic_state.get("Q_evap_eff", 0.0),
        Q_evap_ss,
        dt,
        dynamics.get("tau_evap_s", 0.0),
    )
    Q_cond_eff = first_order_lag(
        dynamic_state.get("Q_cond_eff", 0.0),
        Q_cond_ss,
        dt,
        dynamics.get("tau_cond_s", 0.0),
    )
    if m_dot_cool > 1e-9:
        T_evap_out = T_tank_K - Q_evap_eff / (m_dot_cool * cp_cool)
    else:
        T_evap_out = T_tank_K
    supply_delay_steps = pipe_delay_steps(dynamics.get("tau_pipe_supply_s", 0.0), dt)
    supply_history = dynamic_state.get("T_pipe_supply_history_K")
    if supply_history is None and "T_pipe_supply_K" in dynamic_state:
        supply_history = [dynamic_state["T_pipe_supply_K"]] * supply_delay_steps
    T_pipe_supply, supply_history_next = delay_line_value(
        supply_history,
        T_evap_out,
        dynamic_state.get("T_pipe_supply_K", T_tank_K),
        supply_delay_steps,
    )

    plate_inputs = T_plate_next[::-1] if is_reversed else T_plate_next
    plate_results = cold_plate_fluid_exchange(plate_inputs, T_pipe_supply, m_dot_cool)
    T_fluid = plate_results["T_fluid_profile"][::-1] if is_reversed else plate_results["T_fluid_profile"]

    h_plate = pack.config.get("h_plate_convection", 10.0)
    temps_grid = pack.temps.reshape(pack.rows, pack.cols)
    Q_cells = np.sum(h_plate * (temps_grid - T_plate_next), axis=0)
    Q_bat_to_plate = max(0.0, float(np.sum(Q_cells)))

    if m_dot_cool > 1e-6:
        h_dyn = max(50.0, h_bp_nominal * (m_dot_cool / m_dot_nominal) ** 0.8)
    else:
        h_dyn = 50.0
    area_per_col = A_bp_total / max(n_cols, 1)
    Q_fluid = h_dyn * area_per_col * (T_plate_next - T_fluid)
    T_plate_next += (Q_cells - Q_fluid) / C_plate_node * dt

    total_power_for_cop = W_comp_real + W_pump_val + W_fan_real
    exergy_metrics = {
        key: refrig.get(key, 0.0)
        for key in (
            "COP_system",
            "COP_Carnot",
            "exergy_efficiency",
            "E_D_comp",
            "E_D_evap",
            "E_D_cond",
            "E_D_exp",
            "E_D_total",
            "E_D_comp_ratio",
            "E_D_evap_ratio",
            "E_D_cond_ratio",
            "E_D_exp_ratio",
        )
    }
    exergy_metrics["COP_system"] = (
        Q_evap_eff / total_power_for_cop if total_power_for_cop > 1e-12 else 0.0
    )
    extra_comp_loss = max(0.0, W_comp_real - refrig.get("W_comp", 0.0))
    if extra_comp_loss > 0.0:
        exergy_metrics["E_D_comp"] += extra_comp_loss
        exergy_metrics["E_D_total"] += extra_comp_loss
        if exergy_metrics["E_D_total"] > 1e-12:
            exergy_metrics["E_D_comp_ratio"] = exergy_metrics["E_D_comp"] / exergy_metrics["E_D_total"]
            exergy_metrics["E_D_evap_ratio"] = exergy_metrics["E_D_evap"] / exergy_metrics["E_D_total"]
            exergy_metrics["E_D_cond_ratio"] = exergy_metrics["E_D_cond"] / exergy_metrics["E_D_total"]
            exergy_metrics["E_D_exp_ratio"] = exergy_metrics["E_D_exp"] / exergy_metrics["E_D_total"]
    if exergy_metrics["COP_Carnot"] > 1e-12:
        exergy_metrics["exergy_efficiency"] = exergy_metrics["COP_system"] / exergy_metrics["COP_Carnot"]
    evap_scale = Q_evap_eff / Q_evap_ss if Q_evap_ss > 1e-12 else 0.0
    cond_scale = Q_cond_eff / Q_cond_ss if Q_cond_ss > 1e-12 else 0.0
    exergy_metrics["E_D_evap"] *= evap_scale
    exergy_metrics["E_D_cond"] *= cond_scale
    exergy_metrics["E_D_exp"] *= evap_scale
    exergy_metrics["E_D_total"] = (
        exergy_metrics["E_D_comp"]
        + exergy_metrics["E_D_evap"]
        + exergy_metrics["E_D_cond"]
        + exergy_metrics["E_D_exp"]
    )
    if exergy_metrics["E_D_total"] > 1e-12:
        exergy_metrics["E_D_comp_ratio"] = exergy_metrics["E_D_comp"] / exergy_metrics["E_D_total"]
        exergy_metrics["E_D_evap_ratio"] = exergy_metrics["E_D_evap"] / exergy_metrics["E_D_total"]
        exergy_metrics["E_D_cond_ratio"] = exergy_metrics["E_D_cond"] / exergy_metrics["E_D_total"]
        exergy_metrics["E_D_exp_ratio"] = exergy_metrics["E_D_exp"] / exergy_metrics["E_D_total"]
    ex_in_cycle = W_comp_real
    ex_dest_aux = 0.0
    ex_useful = max(0.0, ex_in_cycle - exergy_metrics["E_D_total"])
    ex_loss_ambient = max(0.0, ex_in_cycle - ex_useful - exergy_metrics["E_D_total"])

    T_plate_out = plate_results["T_out"]
    return_delay_steps = pipe_delay_steps(dynamics.get("tau_pipe_return_s", 0.0), dt)
    return_history = dynamic_state.get("T_pipe_return_history_K")
    if return_history is None and "T_pipe_return_K" in dynamic_state:
        return_history = [dynamic_state["T_pipe_return_K"]] * return_delay_steps
    T_pipe_return, return_history_next = delay_line_value(
        return_history,
        T_plate_out,
        dynamic_state.get("T_pipe_return_K", T_plate_out),
        return_delay_steps,
    )

    T_tank_next = T_tank_K + (
        m_dot_cool * cp_cool * (T_pipe_return - T_tank_K) / C_tank
    ) * dt
    dynamic_state_next = {
        "N_comp_eff": N_comp_eff,
        "N_pump_eff": N_pump_eff,
        "N_fan_eff": N_fan_eff,
        "Q_evap_eff": Q_evap_eff,
        "Q_cond_eff": Q_cond_eff,
        "T_pipe_supply_K": T_pipe_supply,
        "T_pipe_return_K": T_pipe_return,
        "T_pipe_supply_history_K": supply_history_next,
        "T_pipe_return_history_K": return_history_next,
    }

    return {
        "T_tank_K": T_tank_next,
        "T_plate_K_array": T_plate_next,
        "W_comp_real": W_comp_real,
        "W_pump_val": W_pump_val,
        "W_fan_real": W_fan_real,
        "N_comp_eff": N_comp_eff,
        "N_pump_eff": N_pump_eff,
        "N_fan_cmd": N_fan_cmd,
        "N_fan_eff": N_fan_eff,
        "dynamic_state": dynamic_state_next,
        "m_dot_cool": m_dot_cool,
        "coolant_flow_L_min": m_dot_cool / rho_cool * 1000.0 * 60.0,
        "T_evap_out_K": T_evap_out,
        "T_pipe_supply_K": T_pipe_supply,
        "T_plate_in_K": T_pipe_supply,
        "T_plate_out_K": T_plate_out,
        "T_pipe_return_K": T_pipe_return,
        "T_tank_in_K": T_pipe_return,
        "Q_dot_bat": Q_bat_to_plate,
        "Q_dot_evap": Q_evap_eff,
        "Q_dot_cond": Q_cond_eff,
        "T_evap_sat": refrig.get("T_evap_sat", np.nan),
        "T_cond_sat": refrig.get("T_cond_sat", np.nan),
        "T_cond_out": refrig.get("T_cond_out", np.nan),
        "subcooling": refrig.get("subcooling", np.nan),
        "suction_superheat": refrig.get("suction_superheat", np.nan),
        "p_evap": refrig.get("p_evap", np.nan),
        "p_cond": refrig.get("p_cond", np.nan),
        "h_1": refrig.get("h_1", np.nan),
        "h_2": refrig.get("h_2", np.nan),
        "h_3": refrig.get("h_3", np.nan),
        "h_4": refrig.get("h_4", np.nan),
        "p_1": refrig.get("p_1", np.nan),
        "p_2": refrig.get("p_2", np.nan),
        "p_3": refrig.get("p_3", np.nan),
        "p_4": refrig.get("p_4", np.nan),
        "Ex_dot_in": ex_in_cycle,
        "Ex_dot_useful": ex_useful,
        "Ex_dot_dest_aux": ex_dest_aux,
        "Ex_dot_loss_ambient": ex_loss_ambient,
        **exergy_metrics,
    }




