"""Discrete LTV QP-MPC for the 14-state Physics-P adapter.

The controller optimizes compressor and pump commands.  Only the predicted
battery-temperature output is penalized, so the implementation calls its
tracking weight ``Q_y`` rather than implying a full-state ``Q_x`` penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Sequence

import numpy as np
import osqp
from scipy import sparse

from ...predictor.state_space import (
    ND,
    NU,
    NX,
    N_COMP,
    N_PUMP,
    T_BAT,
    T_TANK,
    LinearizedDiscreteModel,
    PhysicsPStateSpace,
)
from ...predictor.linear_model_bank import OfflinePhysicsPLinearModelBank
from ...simulation.config import (
    N_COMP_MAX_RPM,
    N_COMP_OFF_RPM,
    N_PUMP_MAX_RPM,
    N_PUMP_MIN_RPM,
)


@dataclass(frozen=True)
class QPMPCWeights:
    """Base normalized weights selected by the 150 s short-loop validation."""

    q_y: float = 1.0
    r_comp: float = 1.0
    r_pump: float = 1.0e-3
    r_delta_comp: float = 1.0e-2
    r_delta_pump: float = 1.0e-2
    p_f: float = 1.0

    def validated(self) -> "QPMPCWeights":
        values = np.asarray(
            [
                self.q_y,
                self.r_comp,
                self.r_pump,
                self.r_delta_comp,
                self.r_delta_pump,
                self.p_f,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("QP weights must be finite and nonnegative")
        return self


@dataclass(frozen=True)
class QPMPCConfig:
    horizon: int
    dmax_comp_rpm: float
    dmax_pump_rpm: float
    move_block_size: int = 1
    input_min_rpm: tuple[float, float] = (N_COMP_OFF_RPM, N_PUMP_MIN_RPM)
    input_max_rpm: tuple[float, float] = (N_COMP_MAX_RPM, N_PUMP_MAX_RPM)
    enforce_predicted_state_bounds: bool = True
    enforce_reference_upper_soft_constraint: bool = True
    reference_upper_slack_weight: float = 1.0e2
    regularization: float = 1.0e-6
    osqp_eps_abs: float = 1.0e-4
    osqp_eps_rel: float = 1.0e-4
    osqp_max_iter: int = 25000
    osqp_polish: bool = True
    osqp_adaptive_rho: bool = True
    osqp_sigma: float = 1.0e-6

    def validated(self) -> "QPMPCConfig":
        if type(self.horizon) is not int or self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if type(self.move_block_size) is not int or self.move_block_size <= 0:
            raise ValueError("move_block_size must be a positive integer")
        lower = np.asarray(self.input_min_rpm, dtype=float)
        upper = np.asarray(self.input_max_rpm, dtype=float)
        dmax = np.asarray([self.dmax_comp_rpm, self.dmax_pump_rpm], dtype=float)
        if lower.shape != (NU,) or upper.shape != (NU,):
            raise ValueError("input bounds must contain compressor and pump values")
        if not np.all(np.isfinite(np.r_[lower, upper, dmax])):
            raise ValueError("QP configuration must contain only finite values")
        if np.any(lower >= upper) or np.any(dmax <= 0.0):
            raise ValueError("input bounds and DMAX values are invalid")
        if (
            not np.isfinite(self.reference_upper_slack_weight)
            or self.reference_upper_slack_weight < 0.0
        ):
            raise ValueError("reference_upper_slack_weight must be nonnegative")
        return self

    @classmethod
    def for_scene(cls, scene: str) -> "QPMPCConfig":
        from ..gekko.mpc import runtime_mpc_params_for_scene

        params = runtime_mpc_params_for_scene(scene)
        is_frequency = str(params.name).lower().startswith("freq")
        return cls(
            horizon=int(params.mpc_horizon),
            dmax_comp_rpm=float(params.dmax_comp),
            dmax_pump_rpm=float(params.dmax_pump),
            move_block_size=3 if is_frequency else 1,
        )


@dataclass(frozen=True)
class QPProblem:
    H: np.ndarray
    g: np.ndarray
    constraint_matrix: sparse.csc_matrix
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    state_maps: tuple[np.ndarray, ...]
    state_offsets: tuple[np.ndarray, ...]
    nominal_controls: np.ndarray
    n_control_variables: int


@dataclass(frozen=True)
class QPSolveDiagnostics:
    solved: bool
    status: str
    iterations: int
    objective: float
    primal_residual: float
    dual_residual: float
    setup_time_s: float
    solve_time_s: float
    wall_time_s: float
    max_constraint_violation: float
    fallback_reason: str | None


@dataclass(frozen=True)
class QPSolution:
    command_rpm: np.ndarray
    control_sequence_rpm: np.ndarray
    predicted_states: np.ndarray
    diagnostics: QPSolveDiagnostics
    hessian: np.ndarray


def _vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    return result.copy()


def _preview(value: Sequence[Sequence[float]], horizon: int) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape == (ND,):
        result = np.repeat(result[None, :], horizon, axis=0)
    if result.shape != (horizon, ND) or not np.all(np.isfinite(result)):
        raise ValueError(f"disturbance preview must have shape ({horizon}, {ND})")
    return result.copy()


class StateSpaceQPMPC:
    """Constrained QP-MPC with optional offline model-bank scheduling."""

    def __init__(
        self,
        state_space: PhysicsPStateSpace,
        config: QPMPCConfig,
        weights: QPMPCWeights | None = None,
        model_bank: OfflinePhysicsPLinearModelBank | None = None,
    ) -> None:
        if not isinstance(state_space, PhysicsPStateSpace):
            raise TypeError("state_space must be PhysicsPStateSpace")
        self.state_space = state_space
        self.config = config.validated()
        self.model_bank = model_bank
        self.base_weights = (weights or QPMPCWeights()).validated()
        self.runtime_weights = self.base_weights
        self._last_solution_scaled: np.ndarray | None = None
        self._last_solution_physical: np.ndarray | None = None
        self._last_snapshot: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None
        self._last_nominal_controls: np.ndarray | None = None

    def set_runtime_weight_multipliers(
        self,
        alpha_q_y: float = 1.0,
        alpha_r_comp: float = 1.0,
        alpha_r_pump: float = 1.0,
    ) -> None:
        multipliers = np.asarray(
            [alpha_q_y, alpha_r_comp, alpha_r_pump], dtype=float
        )
        if not np.all(np.isfinite(multipliers)) or np.any(multipliers <= 0.0):
            raise ValueError("runtime weight multipliers must be finite and positive")
        self.runtime_weights = replace(
            self.base_weights,
            q_y=self.base_weights.q_y * multipliers[0],
            r_comp=self.base_weights.r_comp * multipliers[1],
            r_pump=self.base_weights.r_pump * multipliers[2],
        )

    def _same_snapshot(
        self,
        state: np.ndarray,
        disturbance: np.ndarray,
        previous_input: np.ndarray,
        reference_c: float,
    ) -> bool:
        if self._last_snapshot is None:
            return False
        old_state, old_disturbance, old_input, old_reference = self._last_snapshot
        return bool(
            np.array_equal(state, old_state)
            and np.array_equal(disturbance, old_disturbance)
            and np.array_equal(previous_input, old_input)
            and reference_c == old_reference
        )

    def _nominal_controls(
        self,
        state: np.ndarray,
        disturbance: np.ndarray,
        previous_input: np.ndarray,
        reference_c: float,
        supplied: Sequence[Sequence[float]] | None,
    ) -> np.ndarray:
        horizon = self.config.horizon
        if supplied is not None:
            controls = np.asarray(supplied, dtype=float)
            if controls.shape != (horizon, NU) or not np.all(np.isfinite(controls)):
                raise ValueError(f"nominal controls must have shape ({horizon}, {NU})")
            return controls.copy()
        if self._same_snapshot(state, disturbance, previous_input, reference_c):
            assert self._last_nominal_controls is not None
            return self._last_nominal_controls.copy()
        if self._last_solution_physical is None:
            return np.repeat(previous_input[None, :], horizon, axis=0)
        return np.vstack(
            [self._last_solution_physical[1:], self._last_solution_physical[-1:]]
        )

    @staticmethod
    def _prediction_maps(
        models: Sequence[LinearizedDiscreteModel],
        scaled_initial_state: np.ndarray,
        scaled_disturbance: np.ndarray,
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        horizon = len(models)
        # Prediction maps operate on controls only. Slack variables belong to
        # the optimization problem and do not affect state propagation.
        n_variables = horizon * NU
        state_map = np.zeros((NX, n_variables), dtype=float)
        state_offset = scaled_initial_state.copy()
        maps: list[np.ndarray] = []
        offsets: list[np.ndarray] = []
        for index, model in enumerate(models):
            state_map = model.A @ state_map
            state_map[:, index * NU : (index + 1) * NU] += model.B
            state_offset = (
                model.A @ state_offset
                + model.E @ scaled_disturbance[index]
                + model.c
            )
            maps.append(state_map.copy())
            offsets.append(state_offset.copy())
        return tuple(maps), tuple(offsets)

    @staticmethod
    def _difference_matrix(horizon: int) -> np.ndarray:
        difference = np.zeros((horizon * NU, horizon * NU), dtype=float)
        for step_index in range(horizon):
            block = slice(step_index * NU, (step_index + 1) * NU)
            difference[block, block] = np.eye(NU)
            if step_index:
                previous = slice((step_index - 1) * NU, step_index * NU)
                difference[block, previous] = -np.eye(NU)
        return difference

    def build_problem(
        self,
        state: Sequence[float],
        disturbance_preview: Sequence[Sequence[float]],
        reference_c: float,
        previous_input_rpm: Sequence[float],
        nominal_controls_rpm: Sequence[Sequence[float]] | None = None,
    ) -> QPProblem:
        x = _vector(state, NX, "state")
        previous_input = _vector(previous_input_rpm, NU, "previous input")
        reference = float(reference_c)
        if not np.isfinite(reference):
            raise ValueError("reference_c must be finite")
        disturbances = _preview(disturbance_preview, self.config.horizon)
        nominal_controls = self._nominal_controls(
            x, disturbances, previous_input, reference, nominal_controls_rpm
        )
        if self.model_bank is None:
            models, _ = self.state_space.linearize_trajectory(
                x, nominal_controls, disturbances
            )
        else:
            models = self.model_bank.select_trajectory(
                x, nominal_controls, disturbances
            )
        scaling = self.state_space.scaling
        scaled_disturbance = np.asarray(
            [scaling.scale_disturbance(row) for row in disturbances]
        )
        state_maps, state_offsets = self._prediction_maps(
            models, scaling.scale_state(x), scaled_disturbance
        )
        horizon = self.config.horizon
        n_control_variables = horizon * NU
        n_slack_variables = (
            horizon if self.config.enforce_reference_upper_soft_constraint else 0
        )
        n_variables = n_control_variables + n_slack_variables

        output_map = np.vstack(
            [scaling.x_scale[T_BAT] * item[T_BAT] for item in state_maps]
        )
        output_offset = np.asarray(
            [
                scaling.x_offset[T_BAT]
                + scaling.x_scale[T_BAT] * item[T_BAT]
                for item in state_offsets
            ]
        )
        output_error = output_offset - reference
        weights = self.runtime_weights.validated()
        q_diagonal = np.full(horizon, weights.q_y, dtype=float)
        q_diagonal[-1] = weights.p_f
        Q = np.diag(q_diagonal)
        R = np.tile([weights.r_comp, weights.r_pump], horizon)
        R_delta = np.tile(
            [weights.r_delta_comp, weights.r_delta_pump], horizon
        )
        difference = self._difference_matrix(horizon)
        previous_scaled = scaling.scale_input(previous_input)
        delta_offset = np.zeros(n_control_variables, dtype=float)
        delta_offset[:NU] = previous_scaled

        H_control = 2.0 * (
            output_map.T @ Q @ output_map
            + np.diag(R)
            + difference.T @ np.diag(R_delta) @ difference
        )
        H = np.zeros((n_variables, n_variables), dtype=float)
        H[:n_control_variables, :n_control_variables] = H_control
        if n_slack_variables:
            H[n_control_variables:, n_control_variables:] = (
                2.0 * self.config.reference_upper_slack_weight * np.eye(horizon)
            )
        H = 0.5 * (H + H.T)
        H += self.config.regularization * np.eye(n_variables)
        g = np.zeros(n_variables, dtype=float)
        g[:n_control_variables] = 2.0 * (
            output_map.T @ Q @ output_error
            - difference.T @ (R_delta * delta_offset)
        )

        constraint_blocks: list[np.ndarray] = []
        lower_blocks: list[np.ndarray] = []
        upper_blocks: list[np.ndarray] = []

        identity = np.hstack(
            [np.eye(n_control_variables), np.zeros((n_control_variables, n_slack_variables))]
        )
        input_min = np.asarray(self.config.input_min_rpm, dtype=float)
        input_max = np.asarray(self.config.input_max_rpm, dtype=float)
        scaled_min = (input_min - scaling.u_offset) / scaling.u_scale
        scaled_max = (input_max - scaling.u_offset) / scaling.u_scale
        constraint_blocks.append(identity)
        lower_blocks.append(np.tile(scaled_min, horizon))
        upper_blocks.append(np.tile(scaled_max, horizon))

        rate_matrix = np.hstack(
            [difference, np.zeros((n_control_variables, n_slack_variables))]
        )
        dmax = np.asarray(
            [self.config.dmax_comp_rpm, self.config.dmax_pump_rpm], dtype=float
        )
        dmax_scaled = dmax / scaling.u_scale
        rate_offset = np.zeros(n_control_variables, dtype=float)
        rate_offset[:NU] = previous_scaled
        constraint_blocks.append(rate_matrix)
        lower_blocks.append(np.tile(-dmax_scaled, horizon) + rate_offset)
        upper_blocks.append(np.tile(dmax_scaled, horizon) + rate_offset)

        if self.config.move_block_size > 1:
            blocking_rows = []
            for step_index in range(1, horizon):
                if step_index % self.config.move_block_size:
                    for input_index in range(NU):
                        row = np.zeros(n_variables, dtype=float)
                        row[step_index * NU + input_index] = 1.0
                        row[(step_index - 1) * NU + input_index] = -1.0
                        blocking_rows.append(row)
            if blocking_rows:
                blocking = np.asarray(blocking_rows)
                constraint_blocks.append(blocking)
                lower_blocks.append(np.zeros(blocking.shape[0]))
                upper_blocks.append(np.zeros(blocking.shape[0]))

        if self.config.enforce_predicted_state_bounds:
            state_rows = []
            state_lower = []
            state_upper = []
            state_limits = (
                (T_TANK, 15.0, 35.0),
                (N_COMP, N_COMP_OFF_RPM, N_COMP_MAX_RPM),
                (N_PUMP, N_PUMP_MIN_RPM, N_PUMP_MAX_RPM),
            )
            for state_map, state_offset in zip(state_maps, state_offsets):
                for state_index, lower, upper in state_limits:
                    scale = scaling.x_scale[state_index]
                    offset = scaling.x_offset[state_index]
                    state_rows.append(
                        np.pad(state_map[state_index], (0, n_slack_variables))
                    )
                    state_lower.append((lower - offset) / scale - state_offset[state_index])
                    state_upper.append((upper - offset) / scale - state_offset[state_index])
            constraint_blocks.append(np.asarray(state_rows))
            lower_blocks.append(np.asarray(state_lower))
            upper_blocks.append(np.asarray(state_upper))

        if n_slack_variables:
            # Exact reference ceiling without a user-specified band:
            # predicted T_bat <= reference + nonnegative slack.  The large
            # slack penalty makes any predicted overshoot higher priority than
            # energy savings while retaining QP feasibility when the plant
            # cannot physically meet 25 degC under the present actuator limits.
            temp_rows = np.hstack([output_map, -np.eye(horizon)])
            constraint_blocks.append(temp_rows)
            lower_blocks.append(np.full(horizon, -np.inf))
            upper_blocks.append(np.full(horizon, reference) - output_offset)
            slack_bounds = np.hstack(
                [np.zeros((horizon, n_control_variables)), np.eye(horizon)]
            )
            constraint_blocks.append(slack_bounds)
            lower_blocks.append(np.zeros(horizon))
            upper_blocks.append(np.full(horizon, np.inf))

        return QPProblem(
            H=H,
            g=g,
            constraint_matrix=sparse.csc_matrix(np.vstack(constraint_blocks)),
            lower_bounds=np.concatenate(lower_blocks),
            upper_bounds=np.concatenate(upper_blocks),
            state_maps=state_maps,
            state_offsets=state_offsets,
            nominal_controls=nominal_controls,
            n_control_variables=n_control_variables,
        )

    @staticmethod
    def _maximum_violation(
        matrix: sparse.csc_matrix,
        value: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> float:
        evaluated = np.asarray(matrix @ value).reshape(-1)
        lower_violation = np.maximum(lower - evaluated, 0.0)
        upper_violation = np.maximum(evaluated - upper, 0.0)
        return float(max(np.max(lower_violation), np.max(upper_violation)))

    def _fallback_command(
        self, state: np.ndarray, previous_input: np.ndarray, reference_c: float
    ) -> np.ndarray:
        lower = np.asarray(self.config.input_min_rpm, dtype=float)
        upper = np.asarray(self.config.input_max_rpm, dtype=float)
        dmax = np.asarray(
            [self.config.dmax_comp_rpm, self.config.dmax_pump_rpm], dtype=float
        )
        if state[T_TANK] <= 15.0 or state[T_BAT] <= reference_c:
            target = lower
        else:
            target = np.clip(previous_input, lower, upper)
        # Always enforce the original DMAX / bounds on the fallback so the
        # safety path itself cannot report a constraint violation.
        limited = np.clip(target, previous_input - dmax, previous_input + dmax)
        return np.clip(limited, lower, upper)

    def _project_actuator_constraints(
        self, control_sequence: np.ndarray, previous_input: np.ndarray
    ) -> np.ndarray:
        lower = np.asarray(self.config.input_min_rpm, dtype=float)
        upper = np.asarray(self.config.input_max_rpm, dtype=float)
        dmax = np.asarray(
            [self.config.dmax_comp_rpm, self.config.dmax_pump_rpm], dtype=float
        )
        projected = np.empty_like(control_sequence)
        prior = previous_input.copy()
        for step_index, control in enumerate(control_sequence):
            candidate = np.clip(control, lower, upper)
            if step_index and step_index % self.config.move_block_size:
                candidate = prior.copy()
            else:
                candidate = np.clip(candidate, prior - dmax, prior + dmax)
                candidate = np.clip(candidate, lower, upper)
            projected[step_index] = candidate
            prior = candidate
        return projected

    def solve(
        self,
        state: Sequence[float],
        disturbance_preview: Sequence[Sequence[float]],
        reference_c: float,
        previous_input_rpm: Sequence[float],
        nominal_controls_rpm: Sequence[Sequence[float]] | None = None,
    ) -> QPSolution:
        wall_start = perf_counter()
        x = _vector(state, NX, "state")
        previous_input = _vector(previous_input_rpm, NU, "previous input")
        disturbances = _preview(disturbance_preview, self.config.horizon)
        problem = self.build_problem(
            x,
            disturbances,
            reference_c,
            previous_input,
            nominal_controls_rpm=nominal_controls_rpm,
        )
        solver = osqp.OSQP()
        solver.setup(
            P=sparse.csc_matrix(np.triu(problem.H)),
            q=problem.g,
            A=problem.constraint_matrix,
            l=problem.lower_bounds,
            u=problem.upper_bounds,
            verbose=False,
            warm_starting=True,
            polishing=self.config.osqp_polish,
            adaptive_rho=self.config.osqp_adaptive_rho,
            sigma=self.config.osqp_sigma,
            eps_abs=self.config.osqp_eps_abs,
            eps_rel=self.config.osqp_eps_rel,
            max_iter=self.config.osqp_max_iter,
        )
        if self._last_solution_scaled is not None:
            # Guard warm-start: must match the complete QP decision vector
            # (controls plus optional temperature slacks) and be finite. A prior
            # fallback clears this entry; otherwise stale scaling can trigger
            # needless active-set thrashing and hit max_iter.
            expected = problem.H.shape[0]
            if (
                self._last_solution_scaled.shape == (expected,)
                and np.all(np.isfinite(self._last_solution_scaled))
            ):
                solver.warm_start(x=self._last_solution_scaled)
            else:
                self._last_solution_scaled = None
                self._last_solution_physical = None
        result = solver.solve(raise_error=False)
        status = str(result.info.status).lower()
        accepted = status in {"solved", "solved inaccurate"}
        fallback_reason: str | None = None
        if accepted and result.x is not None and np.all(np.isfinite(result.x)):
            decision = np.asarray(result.x, dtype=float)
            violation = self._maximum_violation(
                problem.constraint_matrix,
                decision,
                problem.lower_bounds,
                problem.upper_bounds,
            )
            # OSQP eps is 1e-4; keep a small margin above eps so a normally
            # converged solution is not rejected due to floating rounding.
            if violation > 5.0e-4:
                accepted = False
                fallback_reason = "constraint residual exceeded tolerance"
        else:
            decision = np.zeros(problem.H.shape[0], dtype=float)
            violation = float("inf")
            fallback_reason = f"OSQP status: {status}"

        scaling = self.state_space.scaling
        if accepted:
            scaled_controls = decision[:problem.n_control_variables]
            raw_control_sequence = np.asarray(
                [
                    scaling.unscale_input(block)
                    for block in scaled_controls.reshape(self.config.horizon, NU)
                ]
            )
            control_sequence = self._project_actuator_constraints(
                raw_control_sequence, previous_input
            )
            scaled_controls = np.concatenate(
                [scaling.scale_input(block) for block in control_sequence]
            )
            if self.config.enforce_reference_upper_soft_constraint:
                predicted_for_slack = np.asarray(
                    [
                        scaling.unscale_state(state_map @ scaled_controls + state_offset)
                        for state_map, state_offset in zip(
                            problem.state_maps, problem.state_offsets
                        )
                    ]
                )
                slacks = np.maximum(predicted_for_slack[:, T_BAT] - float(reference_c), 0.0)
                decision = np.concatenate([scaled_controls, slacks])
            else:
                decision = scaled_controls
            violation = self._maximum_violation(
                problem.constraint_matrix,
                decision,
                problem.lower_bounds,
                problem.upper_bounds,
            )
            if violation > 5.0e-4:
                accepted = False
                fallback_reason = "projected control violates prediction constraints"

        if accepted:
            predicted_states = np.asarray(
                [
                    scaling.unscale_state(state_map @ scaled_controls + state_offset)
                    for state_map, state_offset in zip(
                        problem.state_maps, problem.state_offsets
                    )
                ]
            )
            self._last_solution_scaled = decision.copy()
            self._last_solution_physical = control_sequence.copy()
        else:
            command = self._fallback_command(x, previous_input, float(reference_c))
            control_sequence = np.repeat(
                command[None, :], self.config.horizon, axis=0
            )
            predicted_states = np.empty((0, NX), dtype=float)
            # Invalidate warm-start so the next cycle does not seed OSQP with a
            # control that was already judged infeasible / max-iter.
            self._last_solution_scaled = None
            self._last_solution_physical = None

        self._last_nominal_controls = problem.nominal_controls.copy()
        self._last_snapshot = (
            x.copy(),
            disturbances.copy(),
            previous_input.copy(),
            float(reference_c),
        )
        diagnostics = QPSolveDiagnostics(
            solved=accepted,
            status=status,
            iterations=int(result.info.iter),
            objective=float(result.info.obj_val),
            primal_residual=float(result.info.prim_res),
            dual_residual=float(result.info.dual_res),
            setup_time_s=float(result.info.setup_time),
            solve_time_s=float(result.info.solve_time),
            wall_time_s=perf_counter() - wall_start,
            max_constraint_violation=violation,
            fallback_reason=fallback_reason,
        )
        return QPSolution(
            command_rpm=control_sequence[0].copy(),
            control_sequence_rpm=control_sequence,
            predicted_states=predicted_states,
            diagnostics=diagnostics,
            hessian=problem.H.copy(),
        )
