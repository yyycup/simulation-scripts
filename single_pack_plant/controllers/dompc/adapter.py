"""Plant-facing adapter for the independent do-mpc Physics-P NMPC."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import PhysicsPDoMPCConfig
from .nmpc import PhysicsPDoMPC
from ...predictor.physics_p import initialize_physics_state
from ...predictor.state_space import PhysicsPStateSpace


def forecast_pack_heat_generation_w(pack, current_preview_a, dt_s: float) -> np.ndarray:
    """Forecast total ohmic heat from measured SOC and future pack current.

    The Plant's HPPC resistance maps and its parallel-current distribution are
    reused read-only.  SOC is advanced by Coulomb counting over the MPC
    horizon; cell temperatures remain measured values because the nonlinear
    Physics-P state predicts the thermal response itself.
    """
    currents = np.asarray(current_preview_a, dtype=float).reshape(-1)
    if currents.size == 0:
        raise ValueError("current_preview_a must not be empty")
    if not np.all(np.isfinite(currents)):
        raise ValueError("current_preview_a must be finite")

    soc = np.asarray(pack.socs, dtype=float).copy()
    temp_k = np.asarray(pack.temps, dtype=float)
    rows, cols = int(pack.rows), int(pack.cols)
    if soc.size != rows * cols or temp_k.size != soc.size:
        raise ValueError("pack SOC/temperature shape does not match its cell grid")
    capacity_ah = float(pack.config["capacity"])
    if capacity_ah <= 0.0:
        raise ValueError("pack capacity must be positive")
    hppc = pack._hppc_data
    soc_min, soc_max = float(np.min(hppc["soc"])), float(np.max(hppc["soc"]))
    temp_min, temp_max = float(np.min(hppc["temp"])), float(np.max(hppc["temp"]))

    q_gen = np.empty(currents.size, dtype=float)
    for index, total_current in enumerate(currents):
        points = np.column_stack(
            (
                np.clip(soc, soc_min, soc_max),
                np.clip(temp_k, temp_min, temp_max),
            )
        )
        resistance = (
            pack.interp_R_dis(points)
            if total_current >= 0.0
            else pack.interp_R_chg(points)
        )
        resistance_grid = np.asarray(resistance, dtype=float).reshape(rows, cols)
        branch_resistance = np.sum(resistance_grid, axis=1)
        conductance = 1.0 / (branch_resistance + 1.0e-9)
        branch_current = total_current * conductance / np.sum(conductance)
        q_gen[index] = float(np.sum(branch_current[:, None] ** 2 * resistance_grid))
        soc -= np.repeat(branch_current, cols) * float(dt_s) / (capacity_ah * 3600.0)
    return q_gen


class PhysicsPDoMPCPlantController:
    """Translate observed Plant state into the 14-state direct P NMPC state."""

    owns_flow_direction = False
    flow_mode = "physics_p_dompc"
    predictor_name = "physics_p_casadi"
    uses_dynamic_refrigeration = True

    def __init__(self, current_profile, *, dt=5.0, target_temp_c=25.0, case_name=None,
                 predictor_artifact=None, control_interval_steps=None, control_horizon=None):
        self.current_profile = np.asarray(current_profile, dtype=float)
        self.dt = float(dt)
        self.target_temp_c = float(target_temp_c)
        self.config = PhysicsPDoMPCConfig.for_scene(case_name or "peak")
        if control_interval_steps is not None:
            self.config = replace(self.config, control_interval_steps=int(control_interval_steps)).validated()
        if control_horizon is not None:
            self.config = replace(self.config, control_horizon=int(control_horizon)).validated()
        artifact = Path(predictor_artifact) if predictor_artifact else Path(__file__).resolve().parents[2] / "model_data" / "physics_p_operational_v1.json"
        self.state_space = PhysicsPStateSpace(artifact, dt_s=self.dt)
        self.nmpc = PhysicsPDoMPC(
            artifact,
            self.config,
            dt_s=self.dt,
        )
        self.last_n_comp = 300.0
        self.last_n_pump = 1600.0
        self.solve_calls = 0
        self.last_flow_info: dict[str, object] = {}

    def _measured_state(self, pack, t_tank_k, thermal_state, plate_temps):
        observed = thermal_state or {}
        plate = np.asarray(plate_temps, dtype=float)
        t_plate_c = float(np.mean(plate) if plate.size else t_tank_k) - 273.15
        state = initialize_physics_state(
            n_comp_eff_rpm=observed.get("N_comp_eff", self.last_n_comp),
            n_pump_eff_rpm=observed.get("N_pump_eff", self.last_n_pump),
            q_cond_w=observed.get("Q_cond_eff", 0.0),
            q_evap_w=observed.get("Q_evap_eff", 0.0),
            t_supply_c=float(observed.get("T_pipe_supply_K", t_tank_k)) - 273.15,
            t_plate_c=t_plate_c,
            t_return_c=float(observed.get("T_pipe_return_K", t_tank_k)) - 273.15,
            t_batt_c=float(pack.get_avg_temp()) - 273.15,
            t_cool_c=float(t_tank_k) - 273.15,
        )
        return self.state_space.pack_state(state)

    def command(self, step_no, pack, t_tank_k, t_cabinet_k, thermal_state=None, plate_temps=None, **_kwargs):
        if int(step_no) % self.config.control_interval_steps != 0:
            self.last_flow_info = {
                **self.last_flow_info,
                "solved": True,
                "solve_error": "",
                "solve_time_s": 0.0,
                "optimization_updated": False,
            }
            return self.last_n_comp, self.last_n_pump

        state = self._measured_state(pack, t_tank_k, thermal_state, plate_temps)
        start = min(int(step_no), self.current_profile.size - 1)
        preview = self.current_profile[start : start + self.config.horizon]
        if preview.size < self.config.horizon:
            preview = np.pad(preview, (0, self.config.horizon - preview.size), mode="edge")
        q_gen_preview = forecast_pack_heat_generation_w(pack, preview, self.dt)
        try:
            result = self.nmpc.solve(
                state,
                current_preview_a=preview,
                q_gen_preview_w=q_gen_preview,
                ambient_c=float(t_cabinet_k) - 273.15,
                reference_c=self.target_temp_c,
                previous_input_rpm=(self.last_n_comp, self.last_n_pump),
            )
            self.last_n_comp, self.last_n_pump = map(float, result["command_rpm"])
            self.solve_calls += 1
            solved, error = True, ""
        except Exception as exc:
            solved, error = False, f"{type(exc).__name__}: {exc}"
        self.last_flow_info = {
            "mode": self.flow_mode, "direction": 1, "flow_direction_d": 1,
            "solved": solved, "solve_error": error, "solve_time_s": result["solve_time_s"] if solved else np.nan,
            "solver_model_path": "do-mpc/IPOPT Physics-P", "target_temp_c": self.target_temp_c,
            "n_comp_raw_rpm": float(result["raw_command_rpm"][0]) if solved else np.nan,
            "n_comp_applied_rpm": self.last_n_comp,
            "q_gen_forecast_first_w": float(q_gen_preview[0]),
            "q_gen_forecast_terminal_w": float(q_gen_preview[-1]),
            "dmax_projection_applied": bool(result["dmax_projection_applied"]) if solved else False,
            "optimization_updated": True,
        }
        return self.last_n_comp, self.last_n_pump

    def update_after_step(self, _pack):
        return None
