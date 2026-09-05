"""do-mpc 5.x MPC configuration for the five-Pack cluster chiller.

The tuning inherits the Stage 9B Phase 2 parameter family: soft temperature
band around 25 degC, compressor/pump energy weights, and compressor/pump move
weights. do-mpc 5.x has no native move-rate bounds, so the DMAX semantics are
enforced by the closed-loop runner as an application-layer rate limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import casadi as ca
from do_mpc.controller import MPC

from cluster_plant_v2.control.cluster_domp_model import (
    ClusterModelCoefficients,
    _surface_expression,
    pump_power_w,
)
from cluster_plant_v2.parameters import (
    MAXIMUM_COMPRESSOR_SPEED_RPM,
    MAX_PUMP_SPEED_RPM,
    MINIMUM_COMPRESSOR_SPEED_RPM,
    MIN_PUMP_SPEED_RPM,
    SIMULATION_TIME_STEP_S,
)


@dataclass(frozen=True)
class ClusterDompcParameters:
    """MPC tuning set inherited from the Stage 9B Phase 2 recommendation."""

    horizon_steps: int = 12
    target_temperature_k: float = 298.15
    temperature_band_half_width_k: float = 0.65
    temperature_weight: float = 1.0
    compressor_energy_weight: float = 0.035
    pump_energy_weight: float = 0.01
    compressor_move_weight: float = 0.50
    pump_move_weight: float = 0.02
    compressor_dmax_rpm: float = 1200.0
    pump_dmax_rpm: float = 300.0


def build_mpc(
    model,
    coefficients: ClusterModelCoefficients,
    parameters: ClusterDompcParameters | None = None,
    *,
    current_preview: Callable[[float], float],
    ambient_temperature_k: float,
) -> MPC:
    """Configure and set up the do-mpc controller with perfect preview."""
    if parameters is None:
        parameters = ClusterDompcParameters()

    mpc = MPC(model)
    mpc.settings.t_step = SIMULATION_TIME_STEP_S
    mpc.settings.n_horizon = parameters.horizon_steps
    mpc.settings.n_robust = 0
    mpc.settings.store_full_solution = False
    mpc.settings.nlpsol_opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.sb": "yes",
        "ipopt.max_iter": 500,
    }

    t_battery = model.x["t_battery"]
    pump_rpm = model.u["pump_rpm"]
    omega_c = model.x["omega_c"]
    mass_flow = coefficients.k_flow_kg_s_per_rpm * pump_rpm

    shaft_power = _surface_expression(
        coefficients.shaft_power_coeffs, omega_c, mass_flow, model.x["t_tank"]
    )
    band_error = ca.fmax(
        ca.fabs(t_battery - parameters.target_temperature_k)
        - parameters.temperature_band_half_width_k,
        0.0,
    )
    stage_cost = (
        parameters.temperature_weight * band_error**2
        + parameters.compressor_energy_weight * shaft_power
        + parameters.pump_energy_weight * pump_power_w(pump_rpm)
    )
    # mterm must be free of control inputs; keep only the band penalty.
    terminal_cost = parameters.temperature_weight * band_error**2
    mpc.set_objective(mterm=terminal_cost, lterm=stage_cost)
    mpc.set_rterm(
        comp_cmd_rpm=parameters.compressor_move_weight,
        pump_rpm=parameters.pump_move_weight,
    )

    mpc.bounds["lower", "_u", "comp_cmd_rpm"] = MINIMUM_COMPRESSOR_SPEED_RPM
    mpc.bounds["upper", "_u", "comp_cmd_rpm"] = MAXIMUM_COMPRESSOR_SPEED_RPM
    mpc.bounds["lower", "_u", "pump_rpm"] = MIN_PUMP_SPEED_RPM
    mpc.bounds["upper", "_u", "pump_rpm"] = MAX_PUMP_SPEED_RPM
    mpc.bounds["lower", "_x", "t_battery"] = 263.15
    mpc.bounds["upper", "_x", "t_battery"] = 343.15

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now: float):
        for k in range(parameters.horizon_steps + 1):
            tvp_template["_tvp", k, "current_a"] = current_preview(
                t_now + k * SIMULATION_TIME_STEP_S
            )
            tvp_template["_tvp", k, "t_amb"] = ambient_temperature_k
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)
    mpc.setup()
    return mpc


def apply_rate_limits(
    command_rpm: float,
    previous_rpm: float,
    dmax_rpm: float,
    lower_rpm: float,
    upper_rpm: float,
) -> float:
    """Clip the command so that the per-step move respects the DMAX bound."""
    clipped = float(previous_rpm) + min(
        max(float(command_rpm) - float(previous_rpm), -float(dmax_rpm)),
        float(dmax_rpm),
    )
    return min(max(clipped, float(lower_rpm)), float(upper_rpm))
