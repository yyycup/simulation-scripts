"""Harmonized 13-node cold-plate reference for ROM validation (Stage 1.6).

Why this module exists
----------------------
``validation/legacy_cold_plate_reference.py`` normalises its HTC correlation
with ``NOMINAL_COOLANT_MASS_FLOW_KG_S`` = 1.2 kg/s, a *cluster-level* number
that was applied per plate. Both ROMs normalise with
``COLD_PLATE_REFERENCE_MASS_FLOW_KG_S`` = 0.1428 kg/s, the *per-plate design*
flow. The two sides therefore differ by a flow-independent factor

    (1.2 / 0.1428) ** 0.8 = 5.49

in h, which Stage 1.5 showed accounts for essentially all of the apparent
16 % "model error". This module is the harmonized reference: same 13-node
discretization, same frozen constants, but the HTC reference flow is an
**explicit, visible constructor argument** defaulting to the value the ROMs
use.

    !!! The reference flow is a CORRELATION parameter only. It anchors
    !!! h = h_nominal * (m_dot / m_dot_ref) ** 0.8 at h_nominal. It is NOT
    !!! the operating flow -- the operating flow is still passed per step.

``legacy_cold_plate_reference`` is left completely untouched. A bit-exact
equivalence test (``tests/test_harmonized_cold_plate_reference.py``) pins this
module to the legacy path when ``reference_mass_flow`` is set to the legacy
value, so the two cannot silently drift apart.

Two heat-transfer quantities
----------------------------
The legacy 13-node path carries two different Q's and never reconciles them:

* ``q_plate_to_fluid_reported`` -- ``sum(h * A_seg * LMTD)`` from the
  single-guess LMTD sweep. This is the historical output; it is **kept** for
  continuity but is NOT consistent with the wall-state update, and at steady
  state it under-reports by 0.5-1.2 % (Stage 1.5 finding).
* ``q_plate_to_fluid_energy``   -- ``sum(h * A_seg * (T_plate - T_fluid_mean))``,
  i.e. exactly the heat that was removed from the wall states. **All heat
  transfer error metrics in Stage 1.6 and beyond compare against this one.**

Sub-stepping
------------
All three models advance the wall states with explicit Euler, whose stability
limit is ``dt < 2 C / (h A)``. At the harmonized (higher) HTC and 37 L/min that
limit falls to about 3.7 s, below the 5 s plant/MPC interface step. This
adapter therefore sub-divides the incoming ``dt`` into internal steps of at
most ``internal_dt_s`` (default 1 s) and reports outer-step **averages**, so
that both energy paths close exactly across the outer step:

    C * (T_plate_end - T_plate_start) / dt = Q_battery - Q_energy_avg
    Q_energy_avg = G * (T_out_avg - T_in)

Exactly the same convention as ``thermal/cold_plate_heat_current.py``.
"""

from __future__ import annotations

import numpy as np

from cluster_plant_v2.parameters import (
    COLD_PLATE_REFERENCE_MASS_FLOW_KG_S,
    PLATE_NODE_HEAT_CAPACITY_TOTAL,
)
from cluster_plant_v2.validation.legacy_cold_plate_reference import (
    A_bp_seg,
    N_bp,
    cp_cool,
    h_bp_nominal,
    rho_cool,
)

__all__ = [
    "N_bp",
    "A_bp_seg",
    "DEFAULT_REFERENCE_MASS_FLOW_KG_S",
    "DEFAULT_INTERNAL_DT_S",
    "HarmonizedColdPlateAdapter",
    "cold_plate_fluid_exchange",
    "convective_htc",
]

DEFAULT_REFERENCE_MASS_FLOW_KG_S = COLD_PLATE_REFERENCE_MASS_FLOW_KG_S
DEFAULT_INTERNAL_DT_S = 1.0
HTC_FLOOR_W_M2_K = 50.0
HTC_FLOW_EXPONENT = 0.8
#: Balances the floating-point slack in ceil(dt / internal_dt).
_SUBSTEP_TOLERANCE = 1e-9


def convective_htc(mass_flow: float, reference_mass_flow: float) -> float:
    """Flow-dependent HTC. ``reference_mass_flow`` is a correlation anchor."""
    return max(
        HTC_FLOOR_W_M2_K,
        h_bp_nominal * (float(mass_flow) / float(reference_mass_flow)) ** HTC_FLOW_EXPONENT,
    )


def cold_plate_fluid_exchange(
    plate_wall_temperatures,
    coolant_inlet_temperature: float,
    mass_flow: float,
    reference_mass_flow: float,
) -> dict:
    """Legacy single-guess LMTD sweep with an explicit HTC reference flow.

    Mirrors ``legacy_cold_plate_reference.cold_plate_fluid_exchange`` term for
    term; only the HTC normalisation flow is a parameter instead of a module
    global. With ``reference_mass_flow = NOMINAL_COOLANT_MASS_FLOW_KG_S`` the
    output is bit-identical to the legacy function.
    """
    walls = np.asarray(plate_wall_temperatures, dtype=float)
    if walls.ndim == 0:
        walls = np.full(N_bp, float(walls))
    inlet = float(coolant_inlet_temperature)
    mass_flow = float(mass_flow)

    h_dynamic = convective_htc(mass_flow, reference_mass_flow)

    local_temperature = inlet
    reported_heat = 0.0
    fluid_profile = np.empty(N_bp, dtype=float)
    for index in range(N_bp):
        delta_t_in = walls[index] - local_temperature
        guessed_heat = h_dynamic * A_bp_seg * delta_t_in
        guessed_outlet = local_temperature + guessed_heat / (mass_flow * cp_cool + 1e-12)
        delta_t_out = walls[index] - guessed_outlet

        if (
            abs(delta_t_in) < 1e-4
            or abs(delta_t_out) < 1e-4
            or abs(delta_t_in - delta_t_out) < 1e-4
            or delta_t_in * delta_t_out <= 0
        ):
            lmtd = 0.5 * (delta_t_in + delta_t_out)
        else:
            try:
                lmtd = (delta_t_in - delta_t_out) / np.log(delta_t_in / delta_t_out)
            except Exception:
                lmtd = 0.5 * (delta_t_in + delta_t_out)

        node_heat = h_dynamic * A_bp_seg * lmtd
        outlet = local_temperature + node_heat / (mass_flow * cp_cool + 1e-12)
        fluid_profile[index] = 0.5 * (local_temperature + outlet)
        reported_heat += node_heat
        local_temperature = outlet

    return {
        "T_out": local_temperature,
        "Q_total": reported_heat,
        "T_fluid_profile": fluid_profile,
        "h_dynamic": h_dynamic,
    }


def substep_count(dt: float, internal_dt_s: float) -> int:
    """Number of internal Euler steps, never fewer than one."""
    return max(1, int(np.ceil(float(dt) / float(internal_dt_s) - _SUBSTEP_TOLERANCE)))


class HarmonizedColdPlateAdapter:
    """Validation-only 13-node reference with an explicit HTC reference flow."""

    coolant_density = rho_cool
    coolant_cp = cp_cool
    node_count = N_bp

    def __init__(
        self,
        reference_mass_flow: float = DEFAULT_REFERENCE_MASS_FLOW_KG_S,
        initial_temperature_c: float = 25.0,
        internal_dt_s: float = DEFAULT_INTERNAL_DT_S,
    ) -> None:
        reference_mass_flow = float(reference_mass_flow)
        if not np.isfinite(reference_mass_flow) or reference_mass_flow <= 0.0:
            raise ValueError("reference_mass_flow must be positive and finite in kg/s")
        internal_dt_s = float(internal_dt_s)
        if not np.isfinite(internal_dt_s) or internal_dt_s <= 0.0:
            raise ValueError("internal_dt_s must be positive and finite in s")

        self.reference_mass_flow = reference_mass_flow
        self.internal_dt_s = internal_dt_s
        self.plate_temperatures = np.full(N_bp, float(initial_temperature_c) + 273.15)
        self.node_heat_capacity = PLATE_NODE_HEAT_CAPACITY_TOTAL / float(N_bp)

    def _advance(
        self,
        dt: float,
        coolant_inlet_temperature: float,
        mass_flow: float,
        battery_heat: np.ndarray,
        flow_direction: int,
    ) -> dict:
        """One explicit-Euler step. Returns instantaneous (un-averaged) values."""
        old_temperatures = self.plate_temperatures.copy()
        traversal = old_temperatures if flow_direction == 1 else old_temperatures[::-1]
        exchange = cold_plate_fluid_exchange(
            traversal, coolant_inlet_temperature, mass_flow, self.reference_mass_flow
        )
        fluid_means = np.asarray(exchange["T_fluid_profile"], dtype=float)
        if flow_direction == -1:
            fluid_means = fluid_means[::-1]

        h_dynamic = convective_htc(mass_flow, self.reference_mass_flow)
        # This is the heat that actually leaves the wall states.
        node_energy_heat = h_dynamic * A_bp_seg * (old_temperatures - fluid_means)
        self.plate_temperatures = old_temperatures + dt * (
            battery_heat - node_energy_heat
        ) / self.node_heat_capacity
        return {
            "outlet": float(exchange["T_out"]),
            "reported": float(exchange["Q_total"]),
            "energy": float(node_energy_heat.sum()),
            "node_energy_heat": node_energy_heat,
            "fluid_means": fluid_means,
            "h_dynamic": h_dynamic,
        }

    def step(
        self,
        dt: float,
        coolant_inlet_temperature: float,
        coolant_mass_flow: float,
        q_battery_to_plate_nodes: np.ndarray,
        flow_direction: int = 1,
    ) -> dict:
        """Advance by ``dt`` using internal steps of at most ``internal_dt_s``.

        Reported fluxes and the coolant outlet are outer-step averages, so both
        energy paths close exactly over ``dt``. The wall temperatures are the
        true end-of-step states.
        """
        dt = float(dt)
        mass_flow = float(coolant_mass_flow)
        inlet = float(coolant_inlet_temperature)
        battery_heat = np.asarray(q_battery_to_plate_nodes, dtype=float)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        if not np.isfinite(mass_flow) or mass_flow <= 0.0:
            raise ValueError("coolant_mass_flow must be positive and finite in kg/s")
        if not np.isfinite(inlet):
            raise ValueError("coolant_inlet_temperature must be finite")
        if battery_heat.shape != (N_bp,) or not np.all(np.isfinite(battery_heat)):
            raise ValueError(f"q_battery_to_plate_nodes must be finite with shape ({N_bp},)")
        if flow_direction not in (-1, 1):
            raise ValueError("flow_direction must be 1 or -1")

        n_steps = substep_count(dt, self.internal_dt_s)
        sub_dt = dt / n_steps
        outlet_sum = 0.0
        reported_sum = 0.0
        energy_sum = 0.0
        fluid_mean_sum = np.zeros(N_bp, dtype=float)
        node_heat_sum = np.zeros(N_bp, dtype=float)
        h_dynamic = HTC_FLOOR_W_M2_K

        for _ in range(n_steps):
            result = self._advance(sub_dt, inlet, mass_flow, battery_heat, flow_direction)
            outlet_sum += result["outlet"]
            reported_sum += result["reported"]
            energy_sum += result["energy"]
            fluid_mean_sum += result["fluid_means"]
            node_heat_sum += result["node_energy_heat"]
            h_dynamic = result["h_dynamic"]

        outlet = outlet_sum / n_steps
        reported = reported_sum / n_steps
        energy = energy_sum / n_steps
        fluid_means = fluid_mean_sum / n_steps
        node_energy_heat = node_heat_sum / n_steps

        if not (
            np.all(np.isfinite(self.plate_temperatures))
            and np.isfinite(outlet)
            and np.isfinite(reported)
            and np.isfinite(energy)
        ):
            raise FloatingPointError("HarmonizedColdPlateAdapter produced a non-finite state")

        return {
            "plate_temperatures": self.plate_temperatures.copy(),
            "coolant_outlet_temperature": outlet,
            "coolant_mean_temperatures": fluid_means,
            # historical single-guess LMTD output, kept for continuity only
            "q_plate_to_fluid_reported": reported,
            # energy-consistent: identical to the heat removed from the states
            "q_plate_to_fluid_energy": energy,
            "q_plate_state_loss_per_node": node_energy_heat,
            "energy_closure_error_W": energy - reported,
            "h_dynamic": h_dynamic,
            "internal_steps": n_steps,
            "internal_dt_s": sub_dt,
        }
