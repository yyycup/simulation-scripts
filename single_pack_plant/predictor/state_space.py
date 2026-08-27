"""State-space adapter for the validated 5 s Physics-P predictor.

The authoritative nonlinear equations remain in ``mpc_physics_predictor.py``.
This module only packs the retained 14 states, evaluates one predictor step,
and forms local discrete Jacobians for the QP-MPC.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .physics_p import (
    PhysicsArtifactError,
    PhysicsPredictorState,
    load_physics_artifact,
    step_physics_predictor,
    validate_physics_artifact,
)


STATE_NAMES = (
    "t_bat",
    "t_tank",
    "t_plate",
    "n_comp",
    "n_pump",
    "q_evap",
    "z_pump_1",
    "z_supply_1",
    "z_supply_2",
    "z_supply_3",
    "z_return_1",
    "z_return_2",
    "z_return_3",
    "z_return_4",
)
INPUT_NAMES = ("n_comp_cmd", "n_pump_cmd")
DISTURBANCE_NAMES = ("current", "t_ambient")

NX = len(STATE_NAMES)
NU = len(INPUT_NAMES)
ND = len(DISTURBANCE_NAMES)

T_BAT = 0
T_TANK = 1
T_PLATE = 2
N_COMP = 3
N_PUMP = 4
Q_EVAP = 5
Z_PUMP_1 = 6
Z_SUPPLY = slice(7, 10)
Z_RETURN = slice(10, 14)


@dataclass(frozen=True)
class LinearizedDiscreteModel:
    """Local physical-unit affine model ``x+ = A x + B u + E d + c``."""

    A: np.ndarray
    B: np.ndarray
    E: np.ndarray
    c: np.ndarray
    x_nominal: np.ndarray
    u_nominal: np.ndarray
    d_nominal: np.ndarray


@dataclass(frozen=True)
class StateSpaceScaling:
    """Fixed numerical scaling; it does not change any Physics-P parameter."""

    x_offset: np.ndarray
    x_scale: np.ndarray
    u_offset: np.ndarray
    u_scale: np.ndarray
    d_offset: np.ndarray
    d_scale: np.ndarray

    @classmethod
    def default(cls) -> "StateSpaceScaling":
        return cls(
            x_offset=np.array(
                [25.0, 25.0, 25.0, 300.0, 1600.0, 0.0, 1600.0]
                + [25.0] * 7,
                dtype=float,
            ),
            x_scale=np.array(
                [5.0, 5.0, 5.0, 5700.0, 3200.0, 6000.0, 3200.0]
                + [5.0] * 7,
                dtype=float,
            ),
            u_offset=np.array([300.0, 1600.0], dtype=float),
            u_scale=np.array([5700.0, 3200.0], dtype=float),
            d_offset=np.array([0.0, 30.0], dtype=float),
            d_scale=np.array([1000.0, 10.0], dtype=float),
        )

    def scale_state(self, value: Sequence[float]) -> np.ndarray:
        return (_vector(value, NX, "state") - self.x_offset) / self.x_scale

    def unscale_state(self, value: Sequence[float]) -> np.ndarray:
        return self.x_offset + self.x_scale * _vector(value, NX, "scaled state")

    def scale_input(self, value: Sequence[float]) -> np.ndarray:
        return (_vector(value, NU, "input") - self.u_offset) / self.u_scale

    def unscale_input(self, value: Sequence[float]) -> np.ndarray:
        return self.u_offset + self.u_scale * _vector(value, NU, "scaled input")

    def scale_disturbance(self, value: Sequence[float]) -> np.ndarray:
        return (
            _vector(value, ND, "disturbance") - self.d_offset
        ) / self.d_scale

    def scaled_model(
        self, model: LinearizedDiscreteModel
    ) -> LinearizedDiscreteModel:
        sx_inv = 1.0 / self.x_scale
        A = sx_inv[:, None] * model.A * self.x_scale[None, :]
        B = sx_inv[:, None] * model.B * self.u_scale[None, :]
        E = sx_inv[:, None] * model.E * self.d_scale[None, :]
        c_physical = (
            model.A @ self.x_offset
            + model.B @ self.u_offset
            + model.E @ self.d_offset
            + model.c
            - self.x_offset
        )
        c = sx_inv * c_physical
        return LinearizedDiscreteModel(
            A=A,
            B=B,
            E=E,
            c=c,
            x_nominal=self.scale_state(model.x_nominal),
            u_nominal=self.scale_input(model.u_nominal),
            d_nominal=self.scale_disturbance(model.d_nominal),
        )


def _vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result.copy()


def battery_heat_generation_w(total_current_a: float) -> float:
    """Return the existing 4P13S ohmic heat preview used by the MPC."""

    current = float(total_current_a)
    if not np.isfinite(current):
        raise ValueError("total_current_a must be finite")
    return ((current / 4.0) ** 2) * 0.001 * 52.0


def _canonical_history(
    history: Sequence[float], length: int, initial_value: float
) -> tuple[float, ...]:
    values = [float(item) for item in tuple(history)[-length:]] if length else []
    while len(values) < length:
        values.insert(0, float(initial_value))
    if not np.all(np.isfinite(values)):
        raise ValueError("delay history must contain only finite values")
    return tuple(values)


class PhysicsPStateSpace:
    """Fourteen-state wrapper around the validated direct-response Physics-P."""

    def __init__(
        self,
        artifact: dict | str | Path,
        dt_s: float = 5.0,
        scaling: StateSpaceScaling | None = None,
    ) -> None:
        if isinstance(artifact, (str, Path)):
            selected = load_physics_artifact(artifact, require_validated=True)
        elif isinstance(artifact, dict):
            selected = validate_physics_artifact(artifact)
            if selected.get("fit", {}).get("fit_status") != "validated":
                raise PhysicsArtifactError(
                    "Physics artifact must have fit.fit_status == 'validated'"
                )
        else:
            raise TypeError("artifact must be a validated dict or path")
        self.artifact = selected
        self.dt_s = float(dt_s)
        if not np.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be a finite positive value")
        self.scaling = StateSpaceScaling.default() if scaling is None else scaling
        self._validate_structure()

    @classmethod
    def from_default_artifact(cls, dt_s: float = 5.0) -> "PhysicsPStateSpace":
        path = Path(__file__).resolve().parent / "model_data" / "physics_p_operational_v1.json"
        return cls(path, dt_s=dt_s)

    def _validate_structure(self) -> None:
        dynamic = self.artifact.get("dynamic", {})
        if dynamic.get("evap_response_model") != "direct":
            raise PhysicsArtifactError(
                "14-state model requires dynamic.evap_response_model == 'direct'"
            )
        expected = {
            "evap_input_delay_s": 0,
            "pump_flow_delay_s": 1,
            "supply_delay_s": 3,
            "return_delay_s": 4,
        }
        for name, expected_steps in expected.items():
            actual_steps = int(round(float(dynamic[name]) / self.dt_s))
            if actual_steps != expected_steps:
                raise PhysicsArtifactError(
                    f"14-state model requires {name}={expected_steps} steps, got {actual_steps}"
                )

    def pack_state(self, state: PhysicsPredictorState) -> np.ndarray:
        if not isinstance(state, PhysicsPredictorState):
            raise TypeError("state must be PhysicsPredictorState")
        pump = _canonical_history(
            state.pump_speed_history_rpm, 1, state.n_pump_eff_rpm
        )
        supply = _canonical_history(
            state.supply_history_c, 3, state.t_supply_c
        )
        returned = _canonical_history(
            state.return_history_c, 4, state.t_return_c
        )
        return _vector(
            (
                state.t_batt_c,
                state.t_cool_c,
                state.t_plate_c,
                state.n_comp_eff_rpm,
                state.n_pump_eff_rpm,
                state.q_evap_w,
                pump[0],
                *supply,
                *returned,
            ),
            NX,
            "packed state",
        )

    def unpack_state(
        self, state_vector: Sequence[float], q_cond_w: float | None = None
    ) -> PhysicsPredictorState:
        x = _vector(state_vector, NX, "state")
        q_cond = x[Q_EVAP] if q_cond_w is None else float(q_cond_w)
        if not np.isfinite(q_cond):
            raise ValueError("q_cond_w must be finite")
        return PhysicsPredictorState(
            n_comp_eff_rpm=x[N_COMP],
            n_pump_eff_rpm=x[N_PUMP],
            evap_speed_history_rpm=(),
            pump_speed_history_rpm=(x[Z_PUMP_1],),
            q_cond_w=q_cond,
            q_evap_w=x[Q_EVAP],
            supply_history_c=tuple(x[Z_SUPPLY]),
            t_supply_c=x[Z_SUPPLY.start],
            t_plate_c=x[T_PLATE],
            return_history_c=tuple(x[Z_RETURN]),
            t_return_c=x[Z_RETURN.start],
            t_batt_c=x[T_BAT],
            t_cool_c=x[T_TANK],
        )

    def step(
        self,
        state_vector: Sequence[float],
        control_input: Sequence[float],
        disturbance: Sequence[float],
    ) -> np.ndarray:
        x = _vector(state_vector, NX, "state")
        u = _vector(control_input, NU, "control input")
        d = _vector(disturbance, ND, "disturbance")
        next_state = step_physics_predictor(
            self.unpack_state(x),
            n_comp_cmd_rpm=u[0],
            n_pump_cmd_rpm=u[1],
            q_gen_w=battery_heat_generation_w(d[0]),
            t_ambient_c=d[1],
            dt_s=self.dt_s,
            artifact=self.artifact,
        )
        return self.pack_state(next_state)

    def linearize(
        self,
        state_vector: Sequence[float],
        control_input: Sequence[float],
        disturbance: Sequence[float],
        relative_step: float = 1.0e-5,
    ) -> LinearizedDiscreteModel:
        """Numerically linearize the physical-unit discrete one-step map."""

        x = _vector(state_vector, NX, "state")
        u = _vector(control_input, NU, "control input")
        d = _vector(disturbance, ND, "disturbance")
        relative = float(relative_step)
        if not np.isfinite(relative) or relative <= 0.0:
            raise ValueError("relative_step must be finite and positive")
        f0 = self.step(x, u, d)
        domain = self.artifact["input_domain"]
        minimum_active = float(self.artifact["capacity"]["minimum_active_rpm"])
        state_lower = np.full(NX, -np.inf)
        state_upper = np.full(NX, np.inf)
        state_lower[[T_TANK, N_COMP, N_PUMP, Z_PUMP_1]] = [
            domain["t_cool_c"][0],
            300.0,
            domain["n_pump_rpm"][0],
            domain["n_pump_rpm"][0],
        ]
        state_upper[[T_TANK, N_COMP, N_PUMP, Z_PUMP_1]] = [
            domain["t_cool_c"][1],
            domain["n_comp_rpm"][1],
            domain["n_pump_rpm"][1],
            domain["n_pump_rpm"][1],
        ]
        input_lower = np.array([300.0, domain["n_pump_rpm"][0]])
        input_upper = np.array(
            [domain["n_comp_rpm"][1], domain["n_pump_rpm"][1]]
        )
        disturbance_lower = np.array([-np.inf, domain["t_ambient_c"][0]])
        disturbance_upper = np.array([np.inf, domain["t_ambient_c"][1]])
        A = self._jacobian_argument(
            x,
            u,
            d,
            0,
            self.scaling.x_scale,
            relative,
            f0,
            state_lower,
            state_upper,
            {N_COMP: (300.0, minimum_active)},
        )
        B = self._jacobian_argument(
            x,
            u,
            d,
            1,
            self.scaling.u_scale,
            relative,
            f0,
            input_lower,
            input_upper,
            {0: (300.0, minimum_active)},
        )
        E = self._jacobian_argument(
            x,
            u,
            d,
            2,
            self.scaling.d_scale,
            relative,
            f0,
            disturbance_lower,
            disturbance_upper,
            {},
        )
        c = f0 - A @ x - B @ u - E @ d
        return LinearizedDiscreteModel(A, B, E, c, x, u, d)

    def _jacobian_argument(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        argument: int,
        scales: np.ndarray,
        relative_step: float,
        f0: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        breakpoints: dict[int, tuple[float, ...]],
    ) -> np.ndarray:
        base = (x, u, d)[argument]
        jacobian = np.empty((NX, base.size), dtype=float)
        for column in range(base.size):
            step_size = max(abs(base[column]) * relative_step, scales[column] * relative_step)
            plus = [x.copy(), u.copy(), d.copy()]
            minus = [x.copy(), u.copy(), d.copy()]
            plus[argument][column] += step_size
            minus[argument][column] -= step_size
            crosses_breakpoint = any(
                minus[argument][column] < point < plus[argument][column]
                or base[column] == point
                for point in breakpoints.get(column, ())
            )
            can_step_backward = minus[argument][column] >= lower_bounds[column]
            can_step_forward = plus[argument][column] <= upper_bounds[column]
            if (not can_step_backward or crosses_breakpoint) and can_step_forward:
                jacobian[:, column] = (self.step(*plus) - f0) / step_size
            elif not can_step_forward or crosses_breakpoint:
                if not can_step_backward:
                    raise ValueError("no valid finite-difference direction")
                jacobian[:, column] = (f0 - self.step(*minus)) / step_size
            else:
                jacobian[:, column] = (
                    self.step(*plus) - self.step(*minus)
                ) / (2.0 * step_size)
        return jacobian

    def linearize_scaled(
        self,
        state_vector: Sequence[float],
        control_input: Sequence[float],
        disturbance: Sequence[float],
        relative_step: float = 1.0e-5,
    ) -> LinearizedDiscreteModel:
        return self.scaling.scaled_model(
            self.linearize(
                state_vector,
                control_input,
                disturbance,
                relative_step=relative_step,
            )
        )

    def linearize_trajectory(
        self,
        initial_state: Sequence[float],
        control_sequence: Sequence[Sequence[float]],
        disturbance_sequence: Sequence[Sequence[float]],
    ) -> tuple[list[LinearizedDiscreteModel], np.ndarray]:
        controls = np.asarray(control_sequence, dtype=float)
        disturbances = np.asarray(disturbance_sequence, dtype=float)
        if controls.ndim != 2 or controls.shape[1] != NU:
            raise ValueError("control_sequence must have shape (N, 2)")
        if disturbances.shape != (controls.shape[0], ND):
            raise ValueError("disturbance_sequence must have shape (N, 2)")
        states = [_vector(initial_state, NX, "initial state")]
        models: list[LinearizedDiscreteModel] = []
        for control, disturbance in zip(controls, disturbances):
            models.append(self.linearize_scaled(states[-1], control, disturbance))
            states.append(self.step(states[-1], control, disturbance))
        return models, np.asarray(states)
