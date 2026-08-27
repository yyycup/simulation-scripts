"""Closed-loop adapter from the existing Plant interface to State-Space QP-MPC."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ..gekko.mpc import runtime_mpc_params_for_scene
from ...predictor.physics_p import initialize_physics_state
from ...predictor.state_space import PhysicsPStateSpace
from ...predictor.linear_model_bank import OfflinePhysicsPLinearModelBank
from .mpc import QPMPCConfig, QPMPCWeights, StateSpaceQPMPC
from ...simulation.config import (
    MPC_U_NCOMP_INIT,
    MPC_U_NPUMP_INIT,
    N_COMP_MAX_RPM,
    N_COMP_OFF_RPM,
    N_PUMP_MAX_RPM,
    N_PUMP_MIN_RPM,
    SIM_DT,
    TARGET_TEMP_C,
)


class StateSpaceQPPlantController:
    """Expose the new QP through the simulator's established controller API."""

    owns_flow_direction = False
    flow_mode = "state_space_qp"
    predictor_name = "physics_p"

    def __init__(
        self,
        current_profile,
        dt: float = SIM_DT,
        target_temp_c: float = TARGET_TEMP_C,
        case_name=None,
        predictor_artifact: str | Path | None = None,
        horizon_override: int | None = None,
        qp_weights: QPMPCWeights | dict | None = None,
        linear_model_bank_path: str | Path | None = None,
    ) -> None:
        self.dt = float(dt)
        self.target_temp_c = float(target_temp_c)
        self.current_profile = np.asarray(current_profile, dtype=float)
        if self.current_profile.ndim != 1 or self.current_profile.size == 0:
            raise ValueError("current_profile must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(self.current_profile)):
            raise ValueError("current_profile must contain only finite values")
        self.mpc_params = runtime_mpc_params_for_scene(case_name or "peak")
        state_space = (
            PhysicsPStateSpace.from_default_artifact(dt_s=self.dt)
            if predictor_artifact is None
            else PhysicsPStateSpace(predictor_artifact, dt_s=self.dt)
        )
        config = QPMPCConfig.for_scene(case_name or "peak")
        if horizon_override is not None:
            requested = int(horizon_override)
            if requested <= 0:
                raise ValueError("horizon_override must be positive")
            config = replace(config, horizon=requested)
        selected_weights = (
            QPMPCWeights(**qp_weights) if isinstance(qp_weights, dict) else qp_weights
        )
        model_bank = (
            OfflinePhysicsPLinearModelBank.load(linear_model_bank_path, dt_s=self.dt)
            if linear_model_bank_path is not None
            else None
        )
        self.qp = StateSpaceQPMPC(
            state_space, config, weights=selected_weights, model_bank=model_bank
        )
        self.last_n_comp = float(MPC_U_NCOMP_INIT)
        self.last_n_pump = float(MPC_U_NPUMP_INIT)
        self.last_flow_info = {
            "mode": self.flow_mode,
            "direction": 1,
            "solved": False,
            "solve_error": "not solved yet",
        }

    def set_runtime_weight_multipliers(
        self, alpha_q_y=1.0, alpha_r_comp=1.0, alpha_r_pump=1.0
    ):
        self.qp.set_runtime_weight_multipliers(
            alpha_q_y, alpha_r_comp, alpha_r_pump
        )
        return self

    def _disturbance_preview(self, step_no: int, t_ambient_c: float) -> np.ndarray:
        horizon = self.qp.config.horizon
        start = min(max(0, int(step_no)), self.current_profile.size - 1)
        current = self.current_profile[start : start + horizon]
        if current.size < horizon:
            current = np.pad(current, (0, horizon - current.size), mode="edge")
        return np.column_stack(
            [current, np.full(horizon, float(t_ambient_c), dtype=float)]
        )

    def _measured_state(
        self,
        pack,
        t_tank_k: float,
        thermal_state,
        plate_temps,
    ) -> np.ndarray:
        observed = thermal_state or {}
        plate = np.asarray(plate_temps, dtype=float)
        if plate.size == 0:
            t_plate_c = float(t_tank_k) - 273.15
        else:
            t_plate_c = float(np.mean(plate)) - 273.15
        t_supply_c = float(
            observed.get("T_pipe_supply_K", t_tank_k)
        ) - 273.15
        t_return_c = float(
            observed.get("T_pipe_return_K", t_tank_k)
        ) - 273.15
        state = initialize_physics_state(
            n_comp_eff_rpm=observed.get("N_comp_eff", self.last_n_comp),
            n_pump_eff_rpm=observed.get("N_pump_eff", self.last_n_pump),
            q_cond_w=observed.get("Q_cond_eff", 0.0),
            q_evap_w=observed.get("Q_evap_eff", 0.0),
            t_supply_c=t_supply_c,
            t_plate_c=t_plate_c,
            t_return_c=t_return_c,
            t_batt_c=float(pack.get_avg_temp()) - 273.15,
            t_cool_c=float(t_tank_k) - 273.15,
        )
        return self.qp.state_space.pack_state(state)

    @staticmethod
    def _prediction_value(predicted: np.ndarray, index: int, step: int) -> float:
        if predicted.size == 0:
            return float("nan")
        selected = min(max(1, step), predicted.shape[0]) - 1
        return float(predicted[selected, index])

    def command(
        self,
        step_no,
        pack,
        t_tank_k,
        t_cabinet_k,
        thermal_state=None,
        plate_temps=None,
        **_kwargs,
    ):
        measured_state = self._measured_state(
            pack, t_tank_k, thermal_state, plate_temps
        )
        preview = self._disturbance_preview(
            step_no, float(t_cabinet_k) - 273.15
        )
        previous = np.array([self.last_n_comp, self.last_n_pump], dtype=float)
        try:
            result = self.qp.solve(
                measured_state,
                preview,
                self.target_temp_c,
                previous,
            )
            command = result.command_rpm
            predicted = result.predicted_states
            diagnostics = result.diagnostics
            solve_error = diagnostics.fallback_reason or ""
        except Exception as exc:
            command = np.clip(
                previous,
                [N_COMP_OFF_RPM, N_PUMP_MIN_RPM],
                [N_COMP_MAX_RPM, N_PUMP_MAX_RPM],
            )
            predicted = np.empty((0, 14), dtype=float)
            diagnostics = None
            solve_error = f"{type(exc).__name__}: {exc}"

        self.last_n_comp = float(command[0])
        self.last_n_pump = float(command[1])
        solved = bool(diagnostics is not None and diagnostics.solved)
        t_cool_prediction = predicted[:, 1] if predicted.size else np.array([])
        base_weights = self.qp.base_weights
        runtime_weights = self.qp.runtime_weights
        self.last_flow_info = {
            "mode": self.flow_mode,
            "direction": 1,
            "flow_direction_d": 1,
            "target_temp_c": self.target_temp_c,
            "solved": solved,
            "solve_error": solve_error,
            "solver_model_path": "OSQP",
            "linear_model_bank_enabled": self.qp.model_bank is not None,
            "solve_time_s": (
                diagnostics.wall_time_s if diagnostics is not None else np.nan
            ),
            "solve_recovery_used": not solved,
            "solve_recovery_reason": solve_error,
            "n_comp_raw_rpm": self.last_n_comp,
            "n_comp_applied_rpm": self.last_n_comp,
            "comp_command_filter_alpha": 1.0,
            "alpha_temp": runtime_weights.q_y / base_weights.q_y,
            "alpha_comp": runtime_weights.r_comp / base_weights.r_comp,
            "alpha_pump": runtime_weights.r_pump / base_weights.r_pump,
            "runtime_weight_mode": "state_space_qp_multipliers",
            "prediction_domain_valid": bool(
                predicted.size
                and np.all((t_cool_prediction >= 15.0) & (t_cool_prediction <= 35.0))
            ),
            "t_cool_pred_min_c": (
                float(np.min(t_cool_prediction)) if predicted.size else np.nan
            ),
            "t_cool_pred_max_c": (
                float(np.max(t_cool_prediction)) if predicted.size else np.nan
            ),
            "t_cool_domain_lower_c": 15.0,
            "t_cool_domain_upper_c": 35.0,
            "t_cool_domain_violation_c": (
                max(
                    0.0,
                    15.0 - float(np.min(t_cool_prediction)),
                    float(np.max(t_cool_prediction)) - 35.0,
                )
                if predicted.size
                else np.nan
            ),
            "t_batt_pred_1_c": self._prediction_value(predicted, 0, 1),
            "t_batt_pred_5_c": self._prediction_value(predicted, 0, 5),
            "t_batt_pred_10_c": self._prediction_value(predicted, 0, 10),
            "t_batt_pred_20_c": self._prediction_value(predicted, 0, 20),
            "t_batt_pred_end_c": self._prediction_value(
                predicted, 0, predicted.shape[0]
            ),
            "t_cool_pred_end_c": self._prediction_value(
                predicted, 1, predicted.shape[0]
            ),
            "t_plate_pred_end_c": self._prediction_value(
                predicted, 2, predicted.shape[0]
            ),
            "qevap_eff_pred_mean_w": (
                float(np.mean(predicted[:, 5])) if predicted.size else np.nan
            ),
            "n_comp_plan_rpm": (
                result.control_sequence_rpm[:, 0].copy()
                if diagnostics is not None
                else np.array([self.last_n_comp])
            ),
            "n_pump_plan_rpm": (
                result.control_sequence_rpm[:, 1].copy()
                if diagnostics is not None
                else np.array([self.last_n_pump])
            ),
            "qp_status": diagnostics.status if diagnostics is not None else "exception",
            "qp_iterations": diagnostics.iterations if diagnostics is not None else 0,
            "qp_max_constraint_violation": (
                diagnostics.max_constraint_violation
                if diagnostics is not None
                else np.inf
            ),
        }
        return self.last_n_comp, self.last_n_pump

    def update_after_step(self, pack):
        return None
