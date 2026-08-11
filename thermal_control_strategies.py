import numpy as np

from mpc_flow_direction_strategies import create_mpc_flow_controller
from thermal_batch_config import (
    N_COMP_MAX_RPM,
    N_COMP_OFF_RPM,
    PID_N_COMP_MIN_RPM,
    N_PUMP_MAX_RPM,
    N_PUMP_MIN_RPM,
    SIM_DT,
    TARGET_TEMP_C,
)


class SimplePID:
    def __init__(self, kp, ki, kd, dt, output_limits=(0.0, 1.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.min_out, self.max_out = output_limits
        self.integral = 0.0
        self.last_error = 0.0

    def step(self, setpoint, measurement):
        error = measurement - setpoint
        derivative = (error - self.last_error) / self.dt if self.dt > 0 else 0.0
        self.integral += error * self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        clipped = np.clip(output, self.min_out, self.max_out)
        if clipped != output:
            self.integral -= error * self.dt
        self.last_error = error
        return float(clipped)


class OnOffController:
    def __init__(self, boost_on_c=26.0, boost_off_c=24.0):
        self.boost_on_c = boost_on_c
        self.boost_off_c = boost_off_c
        self.state = 1
        self.last_status = {
            "state": self.state,
            "T_avg_C": np.nan,
            "N_comp_cmd_rpm": N_COMP_OFF_RPM,
            "N_pump_cmd_rpm": N_PUMP_MIN_RPM,
            "boost_on_c": self.boost_on_c,
            "boost_off_c": self.boost_off_c,
        }

    def command(self, step_no, pack, t_tank_k, t_cabinet_k, **_kwargs):
        temp_c = pack.get_avg_temp() - 273.15
        if self.state == 1 and temp_c >= self.boost_on_c:
            self.state = 2
        elif self.state == 2 and temp_c <= self.boost_off_c:
            self.state = 1
        if self.state == 2:
            n_comp_cmd, n_pump_cmd = N_COMP_MAX_RPM, N_PUMP_MAX_RPM
        else:
            n_comp_cmd, n_pump_cmd = N_COMP_OFF_RPM, N_PUMP_MIN_RPM
        self.last_status = {
            "state": self.state,
            "T_avg_C": float(temp_c),
            "N_comp_cmd_rpm": float(n_comp_cmd),
            "N_pump_cmd_rpm": float(n_pump_cmd),
            "boost_on_c": self.boost_on_c,
            "boost_off_c": self.boost_off_c,
        }
        return n_comp_cmd, n_pump_cmd

    def update_after_step(self, pack):
        return None


class PIDController:
    def __init__(self, dt=SIM_DT, target_temp_c=TARGET_TEMP_C, pid_params=(0.85, 0.02, 2.0)):
        self.dt = dt
        self.target_temp_k = target_temp_c + 273.15
        self.pid_params = tuple(float(value) for value in pid_params)
        self.pid = SimplePID(*self.pid_params, dt)
        self.n_comp_cmd = PID_N_COMP_MIN_RPM
        self.n_pump_cmd = N_PUMP_MIN_RPM

    def command(self, step_no, pack, t_tank_k, t_cabinet_k, **_kwargs):
        return self.n_comp_cmd, self.n_pump_cmd

    def update_after_step(self, pack):
        pid_output = self.pid.step(self.target_temp_k, pack.get_avg_temp())
        n_comp_target = PID_N_COMP_MIN_RPM + (N_COMP_MAX_RPM - PID_N_COMP_MIN_RPM) * pid_output
        n_pump_target = N_PUMP_MIN_RPM + (N_PUMP_MAX_RPM - N_PUMP_MIN_RPM) * pid_output
        self.n_comp_cmd = float(
            np.clip(
                np.clip(n_comp_target, self.n_comp_cmd - 200 * self.dt, self.n_comp_cmd + 200 * self.dt),
                PID_N_COMP_MIN_RPM,
                N_COMP_MAX_RPM,
            )
        )
        self.n_pump_cmd = float(
            np.clip(
                np.clip(n_pump_target, self.n_pump_cmd - 500 * self.dt, self.n_pump_cmd + 500 * self.dt),
                N_PUMP_MIN_RPM,
                N_PUMP_MAX_RPM,
            )
        )


def create_controller(
    control,
    current_profile,
    dt=SIM_DT,
    target_temp_c=TARGET_TEMP_C,
    mpc_flow_mode="switching",
    case_name=None,
    pid_params=(0.85, 0.02, 2.0),
    mpc_predictor="candidate_b",
    mpc_predictor_artifact=None,
    mpc_horizon_override=None,
    physics_p_mpc_overrides=None,
    strict_predictor_ablation=False,
):
    if control == "on-off":
        return OnOffController()
    if control == "pid":
        return PIDController(dt=dt, target_temp_c=target_temp_c, pid_params=pid_params)
    if control == "mpc":
        return create_mpc_flow_controller(
            current_profile,
            dt=dt,
            target_temp_c=target_temp_c,
            mpc_flow_mode=mpc_flow_mode,
            case_name=case_name,
            predictor=mpc_predictor,
            predictor_artifact=mpc_predictor_artifact,
            mpc_horizon_override=mpc_horizon_override,
            physics_p_mpc_overrides=physics_p_mpc_overrides,
            strict_predictor_ablation=strict_predictor_ablation,
        )
    raise ValueError(f"Unknown control strategy: {control}")
