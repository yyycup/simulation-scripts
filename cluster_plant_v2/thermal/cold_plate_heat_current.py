"""Heat-current (epsilon-NTU) cold-plate model, parallel to ReducedColdPlate.

Stage 1 of the Heat-Current refactor. This module expresses the same
three-zone wall-node physics as ``thermal/cold_plate_rom.py`` using the
exact constant-wall epsilon-NTU solution written as a heat current through
an equivalent resistance, instead of the LMTD guess-and-correct iteration:

    G   = m_dot * cp                        coolant capacity rate [W/K]
    K_j = h(m_dot) * A_j                    zone conductance [W/K]
    R_j = 1 / (G * (1 - exp(-K_j / G)))     heat-current resistance [K/W]
    Q_j = (T_wall_j - T_in_j) / R_j         plate-to-fluid heat current [W]

Formal configuration
--------------------
**Three zones ``[4, 5, 4]`` is the frozen formal configuration.** The n-zone
generality below exists only so segmentation sensitivity studies can be run;
Stage 1.5 measured n = 1/3/5/7/13 and found the error saturates at n ~ 5, but
the validated, shipped configuration stays at [4, 5, 4] to match the incumbent
ROM and the 16-state NMPC model.

Internal sub-stepping (Stage 1.6)
---------------------------------
The wall states are advanced with explicit Euler, whose stability limit is

    dt < 2 C_j / (dQ_j / dT_wall_j)

At the design HTC and high coolant flow that limit drops below the 5 s
plant / MPC interface step, so this model sub-divides the incoming ``dt`` into
internal steps of at most ``internal_dt_s`` (default 1 s) and then reports
**outer-step averages**. The external interface is unchanged -- callers still
pass one 5 s step and get one set of outputs back -- and the plant / MPC time
base is not touched.

Averaging is what keeps the interface energy-consistent. With per-sub-step
fluxes Q_j and outlets T_out_j,

    C * (T_end - T_start) / dt = Q_battery - mean_j(Q_j)
    mean_j(Q_j)               = G * (mean_j(T_out_j) - T_in)

so the plate-side balance and the fluid-side balance both close exactly across
the outer step, exactly as they do for a single Euler step. Reporting the last
sub-step instead would break the first identity. Concretely:

* ``plate_temperatures``         -- true state at the END of the outer step
* ``q_plate_to_fluid``           -- mean over the sub-steps
* ``coolant_outlet_temperature`` -- mean over the sub-steps (carries the energy)
* ``q_battery_to_plate``         -- held constant across the sub-steps
  (zero-order hold; it is sampled upstream at the outer rate)

Correspondence to the validated ROM
-----------------------------------
- All parameters are inherited frozen from ``parameters.py`` and the
  validated ROM; nothing is re-fitted. The HTC relation is the same
  ``h = max(50, h_nominal * (m_dot / m_ref) ** 0.8)`` with
  ``m_ref = COLD_PLATE_REFERENCE_MASS_FLOW_KG_S`` = 0.1428 kg/s, the per-plate
  design flow. That value is a *correlation anchor* only, not the operating
  flow.
- Same zone mapping: ``ZONE_COLUMN_COUNTS = [4, 5, 4]`` of the 13 columns.
- Same interface contract as ``ReducedColdPlate.step``:
  ``q_battery_to_plate`` is an upstream input (the pack ROM owns the
  battery-to-plate conductance), the plate owns the wall-temperature
  states, and the coolant flows through the zones in series.
- Flow direction flips only the coolant traversal order (1->2->3 forward,
  3->2->1 reverse). The state indices keep their physical zone identity in
  both directions, exactly as in the validated ROM.
- Difference: the validated ROM approximates each zone with a single LMTD
  linearization step, while this model evaluates the analytic
  constant-wall solution, so Q_total is slightly larger (the LMTD guess
  underestimates the effectiveness). Magnitudes are compared in
  ``tests/test_cold_plate_heat_current.py``.

Zero-flow contract
------------------
Identical to the validated ROM: ``m_dot`` must be positive and finite; the
plant guarantees this through ``MIN_PUMP_SPEED_RPM``. No epsilon padding is
used. The multiplication form ``Q = G * (1 - exp(-K/G)) * dT`` (evaluated
with ``expm1``) stays well conditioned as ``G -> 0+``: the effectiveness
term rises to 1, R rises to 1/G, and Q falls to 0 -- the physically correct
limit (no flow, no convective removal).

Zone-count generality
---------------------
The zone count follows the length of ``zone_column_counts``; ``[4, 5, 4]``
is the frozen default inherited from the validated ROM. Segmentation
sensitivity studies (n = 1, 3, 5, 7, 13) pass different column counts without
any code change; totals are always partitioned by ``counts / counts.sum()``.
Those runs are sensitivity evidence only -- they do not change the shipped
three-zone configuration.
"""

from __future__ import annotations

import numpy as np

from cluster_plant_v2.parameters import (
    BATTERY_PLATE_AREA_M2,
    BATTERY_PLATE_NOMINAL_HTC_W_M2_K,
    COLD_PLATE_REFERENCE_MASS_FLOW_KG_S,
    COOLANT_DENSITY_KG_M3,
    COOLANT_SPECIFIC_HEAT_J_KG_K,
    PLATE_NODE_HEAT_CAPACITY_TOTAL,
)
from cluster_plant_v2.thermal.cold_plate_rom import ZONE_COLUMN_COUNTS

HTC_FLOOR_W_M2_K = 50.0
HTC_FLOW_EXPONENT = 0.8
#: Internal Euler step used when the caller passes a longer interface step.
DEFAULT_INTERNAL_DT_S = 1.0
#: Floating-point slack in ceil(dt / internal_dt_s).
_SUBSTEP_TOLERANCE = 1e-9


class ColdPlateHeatCurrent:
    """Cold plate with serial coolant zones in heat-current formulation."""

    coolant_density = COOLANT_DENSITY_KG_M3
    coolant_cp = COOLANT_SPECIFIC_HEAT_J_KG_K

    def __init__(
        self,
        initial_temperature_c: float = 25.0,
        zone_column_counts: np.ndarray | None = None,
        internal_dt_s: float = DEFAULT_INTERNAL_DT_S,
    ) -> None:
        if zone_column_counts is None:
            counts = ZONE_COLUMN_COUNTS.copy()
        else:
            counts = np.asarray(zone_column_counts, dtype=int)
        if counts.ndim != 1 or counts.size == 0 or np.any(counts < 1):
            raise ValueError(
                "zone_column_counts must be a non-empty 1-D array of positive"
                " column counts"
            )
        internal_dt_s = float(internal_dt_s)
        if not np.isfinite(internal_dt_s) or internal_dt_s <= 0.0:
            raise ValueError("internal_dt_s must be positive and finite in s")

        self.zone_column_counts = counts
        self.internal_dt_s = internal_dt_s
        fractions = counts.astype(float) / counts.sum()
        self.zone_heat_capacities = fractions * PLATE_NODE_HEAT_CAPACITY_TOTAL
        self.zone_areas = fractions * BATTERY_PLATE_AREA_M2
        self.plate_temperatures = np.full(
            counts.size, float(initial_temperature_c) + 273.15
        )
        self.initial_zone_energy_J = (
            self.zone_heat_capacities * self.plate_temperatures
        )
        self.coolant_outlet_temperature = float(self.plate_temperatures[0])
        self.coolant_mean_temperatures = self.plate_temperatures.copy()
        self.q_plate_to_fluid = np.zeros(counts.size, dtype=float)
        self.q_plate_to_fluid_total = 0.0
        self.r_plate_to_fluid = np.full(counts.size, np.inf)
        self.h_dynamic = HTC_FLOOR_W_M2_K

    @property
    def dynamic_state_count(self) -> int:
        return int(self.zone_column_counts.size)

    def substep_count(self, dt: float) -> int:
        """Internal Euler steps used for an outer step of ``dt``, at least one."""
        return max(
            1,
            int(np.ceil(float(dt) / self.internal_dt_s - _SUBSTEP_TOLERANCE)),
        )

    def _advance(
        self,
        dt: float,
        inlet: float,
        mass_flow: float,
        battery_heat: np.ndarray,
        flow_direction: int,
    ) -> dict:
        """One explicit-Euler step; returns instantaneous (un-averaged) values."""
        # Same HTC relation as the validated ROM (inherited, not re-fitted).
        self.h_dynamic = max(
            HTC_FLOOR_W_M2_K,
            BATTERY_PLATE_NOMINAL_HTC_W_M2_K
            * (mass_flow / COLD_PLATE_REFERENCE_MASS_FLOW_KG_S) ** HTC_FLOW_EXPONENT,
        )
        capacity_rate = mass_flow * self.coolant_cp
        traversal = (
            range(self.dynamic_state_count)
            if flow_direction == 1
            else range(self.dynamic_state_count - 1, -1, -1)
        )
        fluid_temperature = inlet
        fluid_means = np.empty(self.dynamic_state_count, dtype=float)
        heat_to_fluid = np.empty(self.dynamic_state_count, dtype=float)
        resistances = np.empty(self.dynamic_state_count, dtype=float)
        old_plate_temperatures = self.plate_temperatures.copy()

        for zone in traversal:
            conductance = self.h_dynamic * self.zone_areas[zone]
            # 1 - exp(-K/G), evaluated stably: exact for small K/G, -> 1 as
            # K/G -> infinity (expm1 underflows gracefully to -1).
            effectiveness = -np.expm1(-conductance / capacity_rate)
            resistance = 1.0 / (capacity_rate * effectiveness)
            heat = (old_plate_temperatures[zone] - fluid_temperature) / resistance
            outlet = fluid_temperature + heat / capacity_rate
            fluid_means[zone] = 0.5 * (fluid_temperature + outlet)
            heat_to_fluid[zone] = heat
            resistances[zone] = resistance
            fluid_temperature = outlet

        self.plate_temperatures = old_plate_temperatures + dt * (
            battery_heat - heat_to_fluid
        ) / self.zone_heat_capacities
        return {
            "outlet": float(fluid_temperature),
            "fluid_means": fluid_means,
            "heat_to_fluid": heat_to_fluid,
            "resistances": resistances,
        }

    def step(
        self,
        dt: float,
        coolant_inlet_temperature: float,
        coolant_mass_flow: float,
        q_battery_to_plate: np.ndarray,
        flow_direction: int = 1,
    ) -> dict[str, float | np.ndarray]:
        dt = float(dt)
        mass_flow = float(coolant_mass_flow)
        inlet = float(coolant_inlet_temperature)
        battery_heat = np.asarray(q_battery_to_plate, dtype=float)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        if not np.isfinite(mass_flow) or mass_flow <= 0.0:
            raise ValueError("coolant_mass_flow must be positive and finite in kg/s")
        if not np.isfinite(inlet):
            raise ValueError("coolant_inlet_temperature must be finite")
        if battery_heat.shape != (self.dynamic_state_count,) or not np.all(
            np.isfinite(battery_heat)
        ):
            raise ValueError(
                "q_battery_to_plate must be finite with shape"
                f" ({self.dynamic_state_count},)"
            )
        if flow_direction not in (-1, 1):
            raise ValueError("flow_direction must be 1 or -1")

        n_steps = self.substep_count(dt)
        sub_dt = dt / n_steps
        outlet_sum = 0.0
        fluid_mean_sum = np.zeros(self.dynamic_state_count, dtype=float)
        heat_sum = np.zeros(self.dynamic_state_count, dtype=float)
        resistances = np.full(self.dynamic_state_count, np.inf)

        for _ in range(n_steps):
            result = self._advance(sub_dt, inlet, mass_flow, battery_heat, flow_direction)
            outlet_sum += result["outlet"]
            fluid_mean_sum += result["fluid_means"]
            heat_sum += result["heat_to_fluid"]
            resistances = result["resistances"]

        # Outer-step averages: this is what makes both energy paths close
        # exactly across dt (see the module docstring).
        fluid_means = fluid_mean_sum / n_steps
        heat_to_fluid = heat_sum / n_steps
        outlet = outlet_sum / n_steps

        if not (
            np.all(np.isfinite(self.plate_temperatures))
            and np.all(np.isfinite(fluid_means))
            and np.all(np.isfinite(heat_to_fluid))
            and np.all(np.isfinite(resistances))
            and np.isfinite(outlet)
        ):
            raise FloatingPointError("ColdPlateHeatCurrent produced a non-finite state")

        self.coolant_outlet_temperature = outlet
        self.coolant_mean_temperatures = fluid_means
        self.q_plate_to_fluid = heat_to_fluid
        self.q_plate_to_fluid_total = float(heat_to_fluid.sum())
        self.r_plate_to_fluid = resistances
        return {
            "plate_temperatures": self.plate_temperatures.copy(),
            "coolant_outlet_temperature": self.coolant_outlet_temperature,
            "coolant_mean_temperatures": self.coolant_mean_temperatures.copy(),
            "q_plate_to_fluid": self.q_plate_to_fluid.copy(),
            "q_plate_to_fluid_total": self.q_plate_to_fluid_total,
            "r_plate_to_fluid": self.r_plate_to_fluid.copy(),
            "h_dynamic": self.h_dynamic,
            "internal_steps": n_steps,
            "internal_dt_s": sub_dt,
        }
