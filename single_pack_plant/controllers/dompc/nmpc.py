"""do-mpc formulation using the exact symbolic 14-state Physics-P map."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Sequence
import warnings

# Optional do-mpc extras are not used by this controller.
warnings.filterwarnings("ignore", message=r"The (ONNX|opcua|approximateMPC) feature.*", category=UserWarning)
import do_mpc
import casadi as ca
import numpy as np

from ...predictor.physics_p_casadi import CasadiPhysicsP
from ...simulation.runtime import ensure_env_library_bin_on_path
from .config import PhysicsPDoMPCConfig


class PhysicsPDoMPC:
    """Discrete nonlinear MPC with a fully custom, runtime-weighted objective."""

    def __init__(
        self,
        artifact: str | Path,
        config: PhysicsPDoMPCConfig,
        *,
        dt_s: float = 5.0,
    ) -> None:
        ensure_env_library_bin_on_path()
        self.config = config.validated()
        self.horizon = self.config.horizon
        self.control_moves = self.config.control_moves
        self.terminal_hold_steps = self.config.terminal_hold_steps
        self.dt_s = float(dt_s)
        self.weights = self.config.weights
        self._base_weights = self.weights
        self.dmax_comp_rpm = self.config.dmax_comp_rpm
        self.dmax_pump_rpm = self.config.dmax_pump_rpm
        self.physics_p = CasadiPhysicsP(artifact, dt_s=dt_s)
        self._current_preview = np.zeros(self.horizon + 1)
        self._q_gen_preview_w = np.zeros(self.horizon + 1)
        self._ambient = 35.0
        self._reference = 25.0
        self.model, self.mpc = self._build()
        self._initialized = False

    def _build(self):
        model = do_mpc.model.Model("discrete")
        x = model.set_variable("_x", "x", shape=(16, 1))
        n_comp = model.set_variable("_u", "n_comp_cmd")
        n_pump = model.set_variable("_u", "n_pump_cmd")
        current = model.set_variable("_tvp", "current")
        q_gen_w = model.set_variable("_tvp", "q_gen_w")
        tail_current = [model.set_variable("_tvp", f"tail_current_{index}") for index in range(self.terminal_hold_steps)]
        tail_q_gen_w = [model.set_variable("_tvp", f"tail_q_gen_w_{index}") for index in range(self.terminal_hold_steps)]
        ambient = model.set_variable("_tvp", "t_ambient")
        reference = model.set_variable("_tvp", "t_ref")
        w_tavg = model.set_variable("_tvp", "w_tavg")
        w_upper = model.set_variable("_tvp", "w_upper")
        w_comp = model.set_variable("_tvp", "w_comp")
        w_pump = model.set_variable("_tvp", "w_pump")
        p_next = self.physics_p.function(x[:14], ca.vertcat(n_comp, n_pump), ca.vertcat(current, ambient, q_gen_w))
        model.set_rhs("x", ca.vertcat(p_next, n_comp, n_pump))
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_horizon=self.control_moves,
            t_step=self.dt_s,
            state_discretization="discrete",
            store_full_solution=False,
            nlpsol_opts={"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0, "ipopt.max_iter": 100},
        )
        # Use the post-setup symbols here. do-mpc otherwise sees intermediate
        # symbolic Physics-P expressions as free variables in the objective.
        x_objective = model.x["x"]
        n_comp_objective = model.u["n_comp_cmd"]
        n_pump_objective = model.u["n_pump_cmd"]
        reference_objective = model.tvp["t_ref"]
        w_tavg_objective = model.tvp["w_tavg"]
        w_upper_objective = model.tvp["w_upper"]
        w_comp_objective = model.tvp["w_comp"]
        w_pump_objective = model.tvp["w_pump"]
        error = (x_objective[0] - reference_objective) / self.weights.temperature_error_scale_c
        lterm = (
            w_tavg_objective * error**2
            + w_upper_objective * ca.fmax(error, 0.0) ** 2
            + w_comp_objective * (n_comp_objective / 6000.0) ** 3
            + w_pump_objective * (n_pump_objective / 4800.0) ** 3
        )
        # Standard control horizon: the final optimized 5 s command is held
        # over the remaining Np-Nc prediction steps. The exact Physics-P tail
        # is evaluated in the terminal objective, without adding controls.
        tail_state = x_objective[:14]
        tail_control = ca.vertcat(x_objective[14], x_objective[15])
        mterm = 0.0
        for index in range(self.terminal_hold_steps):
            tail_error = (tail_state[0] - reference_objective) / self.weights.temperature_error_scale_c
            mterm += (
                w_tavg_objective * tail_error**2
                + w_upper_objective * ca.fmax(tail_error, 0.0) ** 2
                + w_comp_objective * (tail_control[0] / 6000.0) ** 3
                + w_pump_objective * (tail_control[1] / 4800.0) ** 3
            )
            tail_state = self.physics_p.function(
                tail_state, tail_control,
                ca.vertcat(tail_current[index], model.tvp["t_ambient"], tail_q_gen_w[index]),
            )
        terminal_error = (tail_state[0] - reference_objective) / self.weights.temperature_error_scale_c
        mterm += w_tavg_objective * terminal_error**2
        mpc.set_objective(mterm=mterm, lterm=lterm)
        # do-mpc applies rterm to raw rpm. Convert the documented normalized
        # design w*(delta_n / n_max)^2 into raw-rpm coefficients.
        mpc.set_rterm(
            n_comp_cmd=self.weights.w_delta_comp / self.config.n_comp_max_rpm**2,
            n_pump_cmd=self.weights.w_delta_pump / self.config.n_pump_max_rpm**2,
        )
        mpc.bounds["lower", "_u", "n_comp_cmd"] = self.config.n_comp_min_rpm
        mpc.bounds["upper", "_u", "n_comp_cmd"] = self.config.n_comp_max_rpm
        mpc.bounds["lower", "_u", "n_pump_cmd"] = self.config.n_pump_min_rpm
        mpc.bounds["upper", "_u", "n_pump_cmd"] = self.config.n_pump_max_rpm
        mpc.set_nl_cons("dcomp_up", n_comp - x[14], ub=self.dmax_comp_rpm)
        mpc.set_nl_cons("dcomp_down", x[14] - n_comp, ub=self.dmax_comp_rpm)
        mpc.set_nl_cons("dpump_up", n_pump - x[15], ub=self.dmax_pump_rpm)
        mpc.set_nl_cons("dpump_down", x[15] - n_pump, ub=self.dmax_pump_rpm)
        template = mpc.get_tvp_template()

        def tvp_fun(_time):
            for step in range(self.control_moves + 1):
                preview_step = min(step, self.horizon - 1)
                template["_tvp", step, "current"] = self._current_preview[preview_step]
                template["_tvp", step, "q_gen_w"] = self._q_gen_preview_w[preview_step]
                for index in range(self.terminal_hold_steps):
                    tail_step = min(self.control_moves + index, self.horizon - 1)
                    template["_tvp", step, f"tail_current_{index}"] = self._current_preview[tail_step]
                    template["_tvp", step, f"tail_q_gen_w_{index}"] = self._q_gen_preview_w[tail_step]
                template["_tvp", step, "t_ambient"] = self._ambient
                template["_tvp", step, "t_ref"] = self._reference
                template["_tvp", step, "w_tavg"] = self.weights.w_tavg
                template["_tvp", step, "w_upper"] = self.weights.w_temp_upper
                template["_tvp", step, "w_comp"] = self.weights.w_comp_energy
                template["_tvp", step, "w_pump"] = self.weights.w_pump_energy
            return template

        mpc.set_tvp_fun(tvp_fun)
        mpc.setup()
        return model, mpc

    def set_runtime_weight_multipliers(self, alpha_tavg: float = 1.0, alpha_comp: float = 1.0) -> None:
        if alpha_tavg <= 0.0 or alpha_comp <= 0.0:
            raise ValueError("NMPC runtime multipliers must be positive")
        base = self._base_weights
        self.weights = replace(
            base,
            w_tavg=base.w_tavg * float(alpha_tavg),
            w_temp_upper=base.w_temp_upper * float(alpha_tavg),
            w_comp_energy=base.w_comp_energy * float(alpha_comp),
        )

    def solve(
        self,
        state: Sequence[float],
        *,
        current_preview_a: Sequence[float],
        q_gen_preview_w: Sequence[float],
        ambient_c: float,
        reference_c: float,
        previous_input_rpm: Sequence[float],
    ) -> dict[str, object]:
        physical_state = np.asarray(state, dtype=float).reshape(14)
        previous = np.asarray(previous_input_rpm, dtype=float).reshape(2)
        preview = np.asarray(current_preview_a, dtype=float).reshape(-1)
        if preview.size < self.horizon:
            raise ValueError("current_preview_a must cover the NMPC horizon")
        q_gen_preview = np.asarray(q_gen_preview_w, dtype=float).reshape(-1)
        if q_gen_preview.size < self.horizon:
            raise ValueError("q_gen_preview_w must cover the NMPC horizon")
        self._current_preview = np.concatenate([preview[: self.horizon], preview[self.horizon - 1 : self.horizon]])
        self._q_gen_preview_w = np.concatenate(
            [q_gen_preview[: self.horizon], q_gen_preview[self.horizon - 1 : self.horizon]]
        )
        x = np.concatenate([physical_state, previous]).reshape(16, 1)
        self._ambient = float(ambient_c)
        self._reference = float(reference_c)
        if not self._initialized:
            self.mpc.x0 = x
            self.mpc.set_initial_guess()
            self._initialized = True
        started = perf_counter()
        raw_command = np.asarray(self.mpc.make_step(x), dtype=float).reshape(2)

        # Keep the Plant-facing command safe even if the NLP's first horizon
        # move is initialized before its augmented previous-command state is
        # enforced. Subsequent moves remain constrained in the NLP as well.
        lower = np.asarray((self.config.n_comp_min_rpm, self.config.n_pump_min_rpm))
        upper = np.asarray((self.config.n_comp_max_rpm, self.config.n_pump_max_rpm))
        dmax = np.asarray((self.dmax_comp_rpm, self.dmax_pump_rpm))
        command = np.clip(raw_command, previous - dmax, previous + dmax)
        command = np.clip(command, lower, upper)
        return {
            "command_rpm": command,
            "raw_command_rpm": raw_command,
            "dmax_projection_applied": bool(np.any(np.abs(command - raw_command) > 1.0e-8)),
            "solve_time_s": perf_counter() - started,
        }
