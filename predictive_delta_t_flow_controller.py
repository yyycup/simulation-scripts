"""MPC flow controller with one predictive temperature-spread trigger only."""

import numpy as np

from mpc_flow_direction_strategies import BaseFlowMPCController, _is_reversed_from_direction
from predictive_delta_t_flow_mpc import PredictiveDeltaTGate
from thermal_batch_config import MPC_PRED_DELTA_T_SWITCH_C, MPC_PRED_SWITCH_BUFFER_S, SIM_DT, TARGET_TEMP_C


class SinglePredictiveDeltaTMPC(BaseFlowMPCController):
    """Reverse only for an imminent predicted maximum-temperature-spread crossing."""

    owns_flow_direction = True
    flow_mode = "single_predictive_delta_t"

    def __init__(
        self,
        current_profile,
        dt=SIM_DT,
        target_temp_c=TARGET_TEMP_C,
        case_name=None,
        mpc_params=None,
        threshold_c=MPC_PRED_DELTA_T_SWITCH_C,
        buffer_s=MPC_PRED_SWITCH_BUFFER_S,
    ):
        super().__init__(current_profile, dt=dt, target_temp_c=target_temp_c, case_name=case_name, mpc_params=mpc_params)
        self.threshold_c = float(threshold_c)
        self.buffer_s = float(buffer_s)
        self.min_hold_s = float(self.mpc_params.reverse_hold_s)
        self.gate = PredictiveDeltaTGate(self.threshold_c, self.buffer_s, self.min_hold_s)
        self.switch_armed = True
        self.last_switch_time = -1e12

    @property
    def is_reversed(self):
        return _is_reversed_from_direction(self.direction)

    def _prediction_delta_t_series(self, pack, solution, direction):
        grids = self._predict_temperature_grids(pack, solution, direction)
        if grids.size == 0:
            grid_c = np.asarray(pack.temps, dtype=float).reshape(pack.rows, pack.cols) - 273.15
            return np.asarray([float(np.max(grid_c) - np.min(grid_c))], dtype=float)
        flattened = np.asarray(grids, dtype=float).reshape(grids.shape[0], -1)
        return np.max(flattened, axis=1) - np.min(flattened, axis=1)

    def _flow_metrics(self, pack, solution, direction, current_time):
        delta_t_series = self._prediction_delta_t_series(pack, solution, direction)
        decision = self.gate.evaluate(delta_t_series, self.dt, current_time, self.last_switch_time)
        crossing_exists = bool(np.isfinite(decision.crossing_time_s))
        buffer_mask = np.arange(delta_t_series.size, dtype=float) * self.dt <= self.buffer_s
        return {
            "delta_t_pred_max": float(np.max(delta_t_series)),
            "delta_t_pred_buffer_max": float(np.max(delta_t_series[buffer_mask])),
            "delta_t_pred_cross_s": float(decision.crossing_time_s),
            "predictive_switch_gate": bool(decision.trigger),
            "hold_time_satisfied": bool(decision.hold_time_satisfied),
            "crossing_exists": crossing_exists,
        }

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
        solution = self._solve_for_direction(
            self.direction,
            step_no,
            pack,
            t_tank_k,
            t_cabinet_k,
            observed_thermal_state=thermal_state,
            t_plate_meas=plate_temps,
        )
        metrics = self._flow_metrics(pack, solution, self.direction, current_time)
        pre_switch_metrics = dict(metrics)
        if not metrics["crossing_exists"]:
            self.switch_armed = True

        switched = False
        if flow_enabled and self.switch_armed and metrics["predictive_switch_gate"]:
            self.direction = -self.direction
            self.last_switch_time = float(current_time)
            self.switch_armed = False
            switched = True
            solution = self._solve_for_direction(
                self.direction,
                step_no,
                pack,
                t_tank_k,
                t_cabinet_k,
                observed_thermal_state=thermal_state,
                t_plate_meas=plate_temps,
            )
            metrics = self._flow_metrics(pack, solution, self.direction, current_time)

        self._set_flow_info(solution, metrics["delta_t_pred_max"], switched=switched, extra_info=metrics)
        self.last_flow_info.update(
            {
                "pre_switch_delta_t_pred_max": pre_switch_metrics["delta_t_pred_max"],
                "pre_switch_delta_t_pred_buffer_max": pre_switch_metrics["delta_t_pred_buffer_max"],
                "pre_switch_delta_t_pred_cross_s": pre_switch_metrics["delta_t_pred_cross_s"],
                "pred_delta_t_switch_c": self.threshold_c,
                "pred_switch_buffer_s": self.buffer_s,
                "min_hold_time_s": self.min_hold_s,
                "switch_armed": bool(self.switch_armed),
                "triggered_by_prediction": bool(switched),
            }
        )
        self._remember_control(solution)
        return solution["n_comp"], solution["n_pump"]

    def update_after_step(self, pack):
        return None
