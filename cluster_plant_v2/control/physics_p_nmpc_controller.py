"""do-mpc 5.x NMPC configuration on the frozen Physics-P predictor.

Discrete-time nonlinear MPC (no linearization, no QP, no GEKKO): the 16-state
augmented model from :mod:`physics_p_nmpc_model` predicts 5 s steps, the
horizon covers 60 steps (300 s), and the closed loop re-optimizes every
15 s while holding the command for the two intermediate Plant steps.

Move-rate limits are hard constraints expressed against the augmented
previous-command states ``x[14:16]``; at each re-optimization those states
carry the last executed command, so the stage-0 constraint bounds exactly
the applied 15 s move.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import casadi as ca
import numpy as np
from do_mpc.controller import MPC

from cluster_plant_v2.control.physics_p_nmpc_model import (
    N_COMP_CMD_INDEX,
    N_PUMP_CMD_INDEX,
    NOMINAL_RPM,
    TUNABLE_WEIGHT_TVP_NAMES,
)


@dataclass(frozen=True)
class PhysicsPNmpcParameters:
    """Default tuning set from the peak-shaving NMPC specification."""

    horizon_steps: int = 60
    t_step_s: float = 5.0
    reoptimize_every_steps: int = 3
    reference_temperature_c: float = 25.0
    w_track: float = 500.0
    w_upper: float = 5000.0
    w_lower: float = 5000.0
    w_comp: float = 0.1
    w_pump: float = 0.001
    w_dcomp: float = 0.01
    w_dpump: float = 0.01
    compressor_lower_rpm: float = 300.0
    compressor_upper_rpm: float = 6000.0
    # The complete Plant compressor actuator rejects commands below
    # 1000 rpm; commands are clipped to this floor at application time.
    compressor_plant_lower_rpm: float = 1000.0
    # Stage 8D4c: pump bounds are in the virtual legacy-pump domain
    # (actual command = virtual / PUMP_VIRTUAL_SPEED_SCALE). The lower
    # bound 3200 virtual = 1600 actual matches the Plant pump speed floor;
    # the upper 4800 virtual = 2400 actual keeps the predictor input
    # inside its n_pump training domain.
    pump_lower_rpm: float = 3200.0
    pump_upper_rpm: float = 4800.0
    compressor_dmax_rpm: float = 1200.0
    # 600 virtual = 300 actual rpm per reoptimization, preserving the
    # pre-8D4c physical pump move rate.
    pump_dmax_rpm: float = 600.0


def _temperature_cost(
    t_bat, reference_c: float, w_track, w_upper, w_lower
):
    """Asymmetric tracking cost: w_track*e^2 plus one-sided band penalties.

    Accepts either float weights (fixed-cost :func:`build_mpc`) or CasADi
    TVP symbols (tunable :func:`build_tunable_mpc`).
    """
    error = t_bat - reference_c
    overshoot = ca.fmax(error, 0.0)
    undershoot = ca.fmax(-error, 0.0)
    return w_track * error**2 + w_upper * overshoot**2 + w_lower * undershoot**2


def _configure_mpc_core(mpc, model, parameters: PhysicsPNmpcParameters):
    """Settings, bounds and hard move-rate constraints shared by both builds."""
    mpc.settings.t_step = parameters.t_step_s
    mpc.settings.n_horizon = parameters.horizon_steps
    mpc.settings.n_robust = 0
    mpc.settings.store_full_solution = False
    mpc.settings.nlpsol_opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.sb": "yes",
        "ipopt.max_iter": 500,
    }

    n_comp_cmd = model.u["n_comp_cmd"]
    n_pump_cmd = model.u["n_pump_cmd"]
    n_comp_cmd_prev = model.x["x", N_COMP_CMD_INDEX]
    n_pump_cmd_prev = model.x["x", N_PUMP_CMD_INDEX]

    mpc.bounds["lower", "_u", "n_comp_cmd"] = parameters.compressor_lower_rpm
    mpc.bounds["upper", "_u", "n_comp_cmd"] = parameters.compressor_upper_rpm
    mpc.bounds["lower", "_u", "n_pump_cmd"] = parameters.pump_lower_rpm
    mpc.bounds["upper", "_u", "n_pump_cmd"] = parameters.pump_upper_rpm

    # Hard move-rate bounds against the augmented previous-command states.
    # set_nl_cons only supports one-sided ``expr <= ub`` bounds, so each
    # two-sided limit is written as a pair of constraints.
    mpc.set_nl_cons(
        "du_comp_up",
        n_comp_cmd - n_comp_cmd_prev,
        ub=parameters.compressor_dmax_rpm,
    )
    mpc.set_nl_cons(
        "du_comp_dn",
        n_comp_cmd_prev - n_comp_cmd,
        ub=parameters.compressor_dmax_rpm,
    )
    mpc.set_nl_cons(
        "du_pump_up",
        n_pump_cmd - n_pump_cmd_prev,
        ub=parameters.pump_dmax_rpm,
    )
    mpc.set_nl_cons(
        "du_pump_dn",
        n_pump_cmd_prev - n_pump_cmd,
        ub=parameters.pump_dmax_rpm,
    )


def build_mpc(
    model,
    parameters: PhysicsPNmpcParameters | None = None,
    *,
    current_preview: Callable[[float], float],
    ambient_temperature_c: float,
    q_gen_preview: Callable[[float, float], float],
) -> MPC:
    """Configure the discrete NMPC on the frozen Physics-P model."""
    if parameters is None:
        parameters = PhysicsPNmpcParameters()

    mpc = MPC(model)
    _configure_mpc_core(mpc, model, parameters)

    t_bat = model.x["x", 0]
    n_comp_cmd = model.u["n_comp_cmd"]
    n_pump_cmd = model.u["n_pump_cmd"]
    n_comp_cmd_prev = model.x["x", N_COMP_CMD_INDEX]
    n_pump_cmd_prev = model.x["x", N_PUMP_CMD_INDEX]

    comp_nominal = NOMINAL_RPM["n_comp_cmd"]
    pump_nominal = NOMINAL_RPM["n_pump_cmd"]
    d_comp = (n_comp_cmd - n_comp_cmd_prev) / comp_nominal
    d_pump = (n_pump_cmd - n_pump_cmd_prev) / pump_nominal

    stage_cost = _temperature_cost(
        t_bat,
        parameters.reference_temperature_c,
        parameters.w_track,
        parameters.w_upper,
        parameters.w_lower,
    ) + (
        parameters.w_comp * (n_comp_cmd / comp_nominal) ** 3
        + parameters.w_pump * (n_pump_cmd / pump_nominal) ** 3
        + parameters.w_dcomp * d_comp**2
        + parameters.w_dpump * d_pump**2
    )
    # mterm must stay free of control inputs.
    terminal_cost = _temperature_cost(
        t_bat,
        parameters.reference_temperature_c,
        parameters.w_track,
        parameters.w_upper,
        parameters.w_lower,
    )
    mpc.set_objective(mterm=terminal_cost, lterm=stage_cost)

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now: float):
        for k in range(parameters.horizon_steps + 1):
            preview_time = float(t_now) + k * parameters.t_step_s
            current = float(current_preview(preview_time))
            tvp_template["_tvp", k, "current_a"] = current
            tvp_template["_tvp", k, "t_amb_c"] = ambient_temperature_c
            tvp_template["_tvp", k, "q_gen_w"] = float(
                q_gen_preview(preview_time, current)
            )
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)
    mpc.setup()
    return mpc


def default_weight_dict(parameters: PhysicsPNmpcParameters) -> dict[str, float]:
    """The seven tunable weights of ``parameters`` as a mutable dict."""
    return {name: float(getattr(parameters, name)) for name in TUNABLE_WEIGHT_TVP_NAMES}


def build_tunable_mpc(
    model,
    parameters: PhysicsPNmpcParameters | None = None,
    *,
    current_preview: Callable[[float], float],
    ambient_temperature_c: float,
    q_gen_preview: Callable[[float, float], float],
    horizon_steps: int | None = None,
    weight_store: dict[str, float] | None = None,
) -> tuple[MPC, dict[str, float]]:
    """Build the NMPC with the cost weights as TVP symbols.

    ``model`` must have been built with ``include_weight_tvp=True``.
    The weights are filled into the TVP template from the returned mutable
    dict at every call of the internal ``tvp_fun``, so rewriting that dict
    (or :meth:`OnlineTunableNmpc.set_weights`) retunes the controller with
    zero CasADi recompilation. With the default weight values the
    objective is algebraically identical to :func:`build_mpc`.

    Pass ``weight_store`` to make several instances (e.g. a horizon bank)
    read the same mutable weight dict.
    """
    if parameters is None:
        parameters = PhysicsPNmpcParameters()
    if horizon_steps is None:
        horizon_steps = parameters.horizon_steps
    horizon_parameters = replace(parameters, horizon_steps=int(horizon_steps))

    mpc = MPC(model)
    _configure_mpc_core(mpc, model, horizon_parameters)

    t_bat = model.x["x", 0]
    n_comp_cmd = model.u["n_comp_cmd"]
    n_pump_cmd = model.u["n_pump_cmd"]
    n_comp_cmd_prev = model.x["x", N_COMP_CMD_INDEX]
    n_pump_cmd_prev = model.x["x", N_PUMP_CMD_INDEX]

    comp_nominal = NOMINAL_RPM["n_comp_cmd"]
    pump_nominal = NOMINAL_RPM["n_pump_cmd"]
    d_comp = (n_comp_cmd - n_comp_cmd_prev) / comp_nominal
    d_pump = (n_pump_cmd - n_pump_cmd_prev) / pump_nominal

    weight_symbols = {name: model.tvp[name] for name in TUNABLE_WEIGHT_TVP_NAMES}
    reference_c = parameters.reference_temperature_c
    stage_cost = _temperature_cost(
        t_bat,
        reference_c,
        weight_symbols["w_track"],
        weight_symbols["w_upper"],
        weight_symbols["w_lower"],
    ) + (
        weight_symbols["w_comp"] * (n_comp_cmd / comp_nominal) ** 3
        + weight_symbols["w_pump"] * (n_pump_cmd / pump_nominal) ** 3
        + weight_symbols["w_dcomp"] * d_comp**2
        + weight_symbols["w_dpump"] * d_pump**2
    )
    # mterm must stay free of control inputs; TVP symbols are allowed and
    # are evaluated at the terminal stage of the TVP template.
    terminal_cost = _temperature_cost(
        t_bat,
        reference_c,
        weight_symbols["w_track"],
        weight_symbols["w_upper"],
        weight_symbols["w_lower"],
    )
    mpc.set_objective(mterm=terminal_cost, lterm=stage_cost)

    weights = default_weight_dict(parameters) if weight_store is None else weight_store
    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now: float):
        for k in range(horizon_parameters.horizon_steps + 1):
            preview_time = float(t_now) + k * parameters.t_step_s
            current = float(current_preview(preview_time))
            tvp_template["_tvp", k, "current_a"] = current
            tvp_template["_tvp", k, "t_amb_c"] = ambient_temperature_c
            tvp_template["_tvp", k, "q_gen_w"] = float(
                q_gen_preview(preview_time, current)
            )
            for name in TUNABLE_WEIGHT_TVP_NAMES:
                tvp_template["_tvp", k, name] = weights[name]
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)
    mpc.setup()
    return mpc, weights


class OnlineTunableNmpc:
    """TD3-facing facade: zero-recompile weight injection plus horizon switching.

    Weight updates rewrite the TVP template values only (no CasADi
    recompile). Horizon changes switch between MPC instances pre-built at
    construction time for each bank entry, carrying over ``x0``/``u0``
    and the internal clock so the switch is transparent to the driver
    loop.
    """

    def __init__(
        self,
        parameters: PhysicsPNmpcParameters | None = None,
        *,
        horizon_bank: tuple[int, ...] | None = None,
        model=None,
        current_preview: Callable[[float], float],
        ambient_temperature_c: float,
        q_gen_preview: Callable[[float, float], float],
    ) -> None:
        self.parameters = parameters if parameters is not None else PhysicsPNmpcParameters()
        if horizon_bank is None:
            horizon_bank = (self.parameters.horizon_steps,)
        horizon_bank = tuple(int(h) for h in horizon_bank)
        if not horizon_bank:
            raise ValueError("horizon_bank must contain at least one entry")
        if self.parameters.horizon_steps not in horizon_bank:
            raise ValueError(
                "horizon_bank must contain the default horizon_steps "
                f"{self.parameters.horizon_steps}"
            )
        if model is None:
            from cluster_plant_v2.control.physics_p_nmpc_model import (
                build_physics_p_model,
            )

            model = build_physics_p_model(include_weight_tvp=True)

        self._mpc_bank: dict[int, MPC] = {}
        self._shared_weights = default_weight_dict(self.parameters)
        for horizon in horizon_bank:
            mpc_instance, _ = build_tunable_mpc(
                model,
                self.parameters,
                current_preview=current_preview,
                ambient_temperature_c=ambient_temperature_c,
                q_gen_preview=q_gen_preview,
                horizon_steps=horizon,
                # All bank instances read the same mutable weight dict, so
                # a single set_weights call retunes every horizon at once.
                weight_store=self._shared_weights,
            )
            self._mpc_bank[horizon] = mpc_instance
        self._active_horizon = self.parameters.horizon_steps

    @property
    def active_mpc(self) -> MPC:
        return self._mpc_bank[self._active_horizon]

    @property
    def horizon_steps(self) -> int:
        return self._active_horizon

    @property
    def horizon_bank(self) -> tuple[int, ...]:
        return tuple(sorted(self._mpc_bank))

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._shared_weights)

    def set_weights(self, **updates: float) -> None:
        """Update weights in place; takes effect at the next make_step.

        Unknown keys raise; values must be finite and non-negative.
        """
        for name, value in updates.items():
            if name not in TUNABLE_WEIGHT_TVP_NAMES:
                raise ValueError(
                    f"unknown weight {name!r}; expected one of {TUNABLE_WEIGHT_TVP_NAMES}"
                )
            value = float(value)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"weight {name} must be finite and >= 0, got {value}")
        self._shared_weights.update({name: float(v) for name, v in updates.items()})

    def switch_horizon(self, horizon_steps: int) -> bool:
        """Activate a pre-built horizon instance; returns True if it changed.

        Carries over ``x0``, ``u0`` and the internal clock from the
        outgoing instance so the driver loop needs no extra bookkeeping.
        The warm-start trajectory is instance-local and therefore lost;
        the next solve falls back to the initial guess seeding.
        """
        horizon_steps = int(horizon_steps)
        if horizon_steps not in self._mpc_bank:
            raise ValueError(
                f"horizon {horizon_steps} not in bank {self.horizon_bank}"
            )
        if horizon_steps == self._active_horizon:
            return False
        outgoing = self.active_mpc
        incoming = self._mpc_bank[horizon_steps]
        incoming.x0 = outgoing.x0
        incoming.u0 = outgoing.u0
        incoming._t0 = outgoing._t0
        incoming.set_initial_guess()
        self._active_horizon = horizon_steps
        return True

    def align_time(self, time_s: float) -> None:
        """Align the internal MPC clock with the Plant time (TVP axis)."""
        self.active_mpc._t0 = float(time_s)

    def prepare(
        self,
        x0: np.ndarray,
        compressor_command_rpm: float,
        pump_virtual_rpm: float,
    ) -> None:
        """Seed the active instance's initial state and command guess."""
        mpc = self.active_mpc
        mpc.x0["x"] = np.asarray(x0, dtype=float).reshape(-1, 1)
        mpc.u0["n_comp_cmd"] = float(compressor_command_rpm)
        mpc.u0["n_pump_cmd"] = float(pump_virtual_rpm)
        mpc.set_initial_guess()

    def make_step(self, x0) -> np.ndarray:
        return self.active_mpc.make_step(x0)
