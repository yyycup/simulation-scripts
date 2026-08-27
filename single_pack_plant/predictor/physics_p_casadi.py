"""CasADi expression of the frozen 14-state discrete Physics-P predictor.

This is an optimization adapter only.  The authoritative numerical predictor
remains :mod:`physics_p`; this module must pass step-equivalence checks before
any NMPC controller is allowed to use it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import casadi as ca
import numpy as np

from .physics_p import load_physics_artifact
from .state_space import NU, NX


class CasadiPhysicsP:
    """Build the exact 5 s discrete P map as a CasADi ``Function``."""

    def __init__(self, artifact: str | Path, dt_s: float = 5.0) -> None:
        self.artifact = load_physics_artifact(artifact, require_validated=True)
        self.dt_s = float(dt_s)
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        dynamic = self.artifact["dynamic"]
        if dynamic["time_constant_model"] != "constant":
            raise ValueError("CasadiPhysicsP currently requires constant time constants")
        if dynamic.get("evap_response_model") != "direct":
            raise ValueError("CasadiPhysicsP currently requires direct evaporator response")
        required_delays = {"evap_input_delay_s": 0, "pump_flow_delay_s": 1, "supply_delay_s": 3, "return_delay_s": 4}
        for name, steps in required_delays.items():
            if int(round(float(dynamic[name]) / self.dt_s)) != steps:
                raise ValueError(f"14-state CasADi P requires {name}={steps} steps")
        self.function = self._build_function()

    @staticmethod
    def _clip(value, lower: float, upper: float):
        return ca.fmin(float(upper), ca.fmax(float(lower), value))

    @staticmethod
    def _lag(previous, target, dt: float, tau: float):
        # The frozen artifact has positive actuator/refrigeration time
        # constants; ``tau`` may be a symbolic plate time constant here.
        alpha = dt / (tau + dt)
        return previous + alpha * (target - previous)

    def _capacity(self, n_comp, n_pump, t_cool, t_ambient):
        capacity = self.artifact["capacity"]
        domain = self.artifact["input_domain"]
        comp = self._clip(n_comp, *domain["n_comp_rpm"])
        pump = self._clip(n_pump, *domain["n_pump_rpm"])
        cool = self._clip(t_cool, *domain["t_cool_c"])
        ambient = self._clip(t_ambient, *domain["t_ambient_c"])
        compressor = (comp - 4000.0) / 2000.0
        pump_ratio = float(capacity["n_pump_ref_rpm"]) / pump
        coolant = (cool - 27.5) / 7.5
        amb = (ambient - 30.0) / 10.0
        features = (
            1.0, compressor, pump_ratio, coolant, amb, compressor**2,
            compressor * pump_ratio, compressor * coolant, compressor * amb,
            pump_ratio**2, pump_ratio * coolant, pump_ratio * amb, coolant**2,
            coolant * amb, compressor**2 * pump_ratio,
            compressor * pump_ratio**2, compressor * pump_ratio * coolant,
            compressor * pump_ratio * amb, compressor * coolant * amb, coolant**3,
        )
        raw = sum(float(c) * feature for c, feature in zip(capacity["coefficients"], features)) * comp
        active = self._clip(raw, 0.0, float(capacity["q_upper_w"]))
        startup = self._clip(
            (n_comp - 300.0) / (float(capacity["minimum_active_rpm"]) - 300.0), 0.0, 1.0
        )
        return startup * active

    def _build_function(self) -> ca.Function:
        x = ca.SX.sym("x", NX)
        u = ca.SX.sym("u", NU)
        # ``q_gen_w`` is explicit so the NMPC can forecast SOC-dependent
        # battery heat rather than assuming a fixed resistance from current.
        d = ca.SX.sym("d", 3)  # [current_a, t_ambient_c, q_gen_w]
        dt = self.dt_s
        dynamic = self.artifact["dynamic"]
        thermal = self.artifact["thermal"]

        t_bat, t_tank, t_plate = x[0], x[1], x[2]
        n_comp, n_pump, q_evap = x[3], x[4], x[5]
        z_pump = x[6]
        z_supply = x[7:10]
        z_return = x[10:14]
        n_comp_next = self._lag(n_comp, u[0], dt, float(dynamic["tau_comp_s"]))
        n_pump_next = self._lag(n_pump, u[1], dt, float(dynamic["tau_pump_s"]))
        q_steady = self._capacity(n_comp_next, z_pump, t_tank, d[1])
        q_evap_next = self._lag(q_evap, q_steady, dt, float(dynamic["tau_evap_s"]))

        pump_ref = float(thermal["n_pump_ref_rpm"])
        pump_ratio = ca.fmax(1.0e-6, z_pump / pump_ref)
        flow_capacity = float(thermal["coolant_mass_flow_ref_kg_s"]) * pump_ratio * float(thermal["coolant_cp_j_kg_k"])
        evaporator_out = t_tank - q_evap_next / flow_capacity
        t_supply = z_supply[0]
        conductance = float(thermal["battery_plate_conductance_w_k"])
        ref_flow = float(thermal["coolant_mass_flow_ref_kg_s"]) * float(thermal["coolant_cp_j_kg_k"])
        plate_fluid = float(thermal["plate_fluid_effectiveness"]) * ca.fmin(flow_capacity, ref_flow * pump_ratio**0.8)
        total_g = conductance + plate_fluid
        plate_equilibrium = (conductance * t_bat + plate_fluid * t_supply) / total_g
        plate_tau = float(thermal["plate_tau_s"]) * ref_flow / total_g
        t_plate_next = self._lag(t_plate, plate_equilibrium, dt, plate_tau)
        q_batt_plate = conductance * (t_bat - t_plate_next)
        q_plate_fluid = plate_fluid * (t_plate_next - t_supply)
        plate_out = t_supply + q_plate_fluid / flow_capacity
        q_gen = d[2] * float(thermal.get("battery_heat_generation_scale", 1.0))
        ambient_loss = float(thermal["ambient_conductance_w_k"]) * (t_bat - d[1])
        t_bat_next = t_bat + dt * (q_gen - q_batt_plate - ambient_loss) / float(thermal["battery_heat_capacity_j_k"])
        t_tank_next = t_tank + dt * flow_capacity * (z_return[0] - t_tank) / float(thermal["coolant_heat_capacity_j_k"])
        x_next = ca.vertcat(
            t_bat_next, t_tank_next, t_plate_next, n_comp_next, n_pump_next,
            q_evap_next, n_pump_next, z_supply[1], z_supply[2], evaporator_out,
            z_return[1], z_return[2], z_return[3], plate_out,
        )
        return ca.Function("physics_p_step", [x, u, d], [x_next], ["x", "u", "d"], ["x_next"])

    def step(self, state: Sequence[float], control: Sequence[float], disturbance: Sequence[float]) -> np.ndarray:
        disturbance_array = np.asarray(disturbance, dtype=float).reshape(-1)
        if disturbance_array.size == 2:
            # Compatibility path for existing one-step callers.  The NMPC
            # always supplies the explicit third heat-load disturbance.
            current_a = disturbance_array[0]
            q_gen_w = ((current_a / 4.0) ** 2) * 0.001 * 52.0
            disturbance_array = np.array((current_a, disturbance_array[1], q_gen_w))
        if disturbance_array.size != 3:
            raise ValueError("disturbance must be [current_a, t_ambient_c, q_gen_w]")
        value = self.function(np.asarray(state, dtype=float), np.asarray(control, dtype=float), disturbance_array)
        return np.asarray(value, dtype=float).reshape(NX)
