"""Coolant-side heat-current evaporator (Stage 2B, interface decoupling only).

Why this module exists
----------------------
Stage 2B does **not** invent a new evaporator model. The audit in
``validation/results/heat_current_stage2_evaporator_audit_20260906/``
established that the incumbent ``refrigeration.ClosedR134aCycle`` already
evaluates exactly the constant-wall epsilon-NTU relation

    G_c = m_dot_c * cp_c
    eps = 1 - exp(-UA_e / G_c)
    Q   = eps * G_c * (T_c,in - T_e)

so rewriting it as a heat current

    R_e,c = 1 / ( G_c * (1 - exp(-UA_e / G_c)) )
    Q_HC  = (T_c,in - T_e) / R_e,c

is an algebraic identity, not new physics. The value of this module is
**boundary separation**: it isolates the heat-exchanger relation into an
independently testable, reusable unit so the refrigeration cycle and the
heat-current exchanger can be reasoned about (and later swapped) separately.

Boundary contract
-----------------
This module owns **only** the coolant-side exchanger relation. It does NOT:

* call ``ClosedR134aCycle.solve()`` (no reverse dependency on the cycle),
* own a refrigerant state, a compressor map, or an expansion valve,
* add an evaporator thermal capacity ``C_e`` (see below),
* modify ``refrigeration.py`` in any way.

The caller (upper layer) supplies the refrigerant-side boundary:

    T_e   -- evaporating saturation temperature [K], from the cycle solver
    UA_e  -- evaporator overall conductance [W/K], from the cycle solver

Three distinct heat quantities (do not conflate)
-----------------------------------------------
The audit found the incumbent code computes several ``Q`` values. They mean
different things and this module keeps them separate:

* ``Q_HC``       -- steady heat-transfer *capability* of the exchanger.
* ``Q_cycle``    -- steady refrigerating capacity of the cycle; the solver
                    forces ``Q_cycle - Q_HC = 0`` (measured residual 0.000%
                    over 10 operating points), so no explicit ``min()`` is
                    needed and none is applied here.
* ``Q_applied``  -- the instantaneous heat actually acting on the coolant
                    after the incumbent 45 s first-order lag. This is the
                    quantity the dynamic plant uses.

Consequently two outlet temperatures are offered:

    T_out_ss      = T_in - Q_HC / G_c        (steady exchanger capability)
    T_out_applied = T_in - Q_applied / G_c   (what the dynamic plant uses)

The incumbent ``refrigeration.py`` keeps reporting its historical
``coolant_outlet_temperature_k`` built from ``Q_cycle``; this module does not
touch that field.

No new thermal state
--------------------
``dynamic_state_count == 0``. The incumbent ``EvaporatorThermalDynamics``
already applies ``tau_e = 45 s`` to ``Q_cycle``, and the audit concluded that
those 45 s are the lumped expression of evaporator + circuit inertia
(the cycle itself is quasi-static with zero dynamic states). Adding
``C_e * dT_e/dt`` here would double-count that same inertia. Splitting the
45 s into physical ``C_e`` plus valve dynamics is deferred to a separate
stage and would require recalibration.

Numerical conditioning
----------------------
``1 - exp(-x)`` is evaluated as ``-expm1(-x)``, which is accurate for small
``x`` and underflows gracefully to 1 as ``x -> inf``. If the effectiveness
degenerates to zero the resistance is reported as ``inf`` and the heat
current as zero, which is the correct physical limit.
"""

from __future__ import annotations

import numpy as np

from cluster_plant_v2.parameters import COOLANT_SPECIFIC_HEAT_J_KG_K


class EvaporatorHeatCurrent:
    """Coolant-side heat-current exchanger with no internal state."""

    coolant_cp = COOLANT_SPECIFIC_HEAT_J_KG_K
    #: No thermal capacity is modelled: the incumbent 45 s lag owns that.
    dynamic_state_count = 0

    @staticmethod
    def capacity_rate(coolant_mass_flow_kg_s: float) -> float:
        """Coolant capacity rate ``G_c = m_dot * cp`` [W/K]."""
        mass_flow = float(coolant_mass_flow_kg_s)
        if not np.isfinite(mass_flow) or mass_flow <= 0.0:
            raise ValueError("coolant_mass_flow_kg_s must be finite and positive")
        return float(mass_flow * COOLANT_SPECIFIC_HEAT_J_KG_K)

    @staticmethod
    def resistance(capacity_rate_w_k: float, evaporator_ua_w_k: float) -> float:
        """Heat-current resistance ``R_e,c = 1 / (G_c (1 - exp(-UA_e/G_c)))``."""
        capacity = float(capacity_rate_w_k)
        ua = float(evaporator_ua_w_k)
        if not np.isfinite(capacity) or capacity <= 0.0:
            raise ValueError("capacity_rate_w_k must be finite and positive")
        if not np.isfinite(ua) or ua <= 0.0:
            raise ValueError("evaporator_ua_w_k must be finite and positive")
        effectiveness = float(-np.expm1(-ua / capacity))
        if effectiveness <= 0.0:
            return float("inf")
        return float(1.0 / (capacity * effectiveness))

    def evaluate(
        self,
        *,
        coolant_inlet_temperature_k: float,
        coolant_mass_flow_kg_s: float,
        evaporating_temperature_k: float,
        evaporator_ua_w_k: float,
    ) -> dict[str, float]:
        """Steady coolant-side heat current for a given refrigerant boundary.

        Parameters
        ----------
        coolant_inlet_temperature_k:
            Coolant temperature entering the evaporator. In the incumbent
            plant this is the **tank** temperature, not the cluster return.
        coolant_mass_flow_kg_s:
            Coolant mass flow through the evaporator [kg/s].
        evaporating_temperature_k:
            Evaporating saturation temperature ``T_e`` [K], supplied by the
            refrigeration-cycle solver. Not solved here.
        evaporator_ua_w_k:
            Evaporator overall conductance ``UA_e`` [W/K], supplied by the
            cycle solver (it depends on refrigerant flow, hence on ``T_e``).
        """
        inlet = float(coolant_inlet_temperature_k)
        evaporating = float(evaporating_temperature_k)
        if not np.isfinite(inlet) or inlet <= 0.0:
            raise ValueError("coolant_inlet_temperature_k must be finite and positive")
        if not np.isfinite(evaporating) or evaporating <= 0.0:
            raise ValueError("evaporating_temperature_k must be finite and positive")

        capacity_rate = self.capacity_rate(coolant_mass_flow_kg_s)
        resistance = self.resistance(capacity_rate, evaporator_ua_w_k)
        heat_w = (inlet - evaporating) / resistance
        effectiveness = 0.0 if np.isinf(resistance) else 1.0 / (
            resistance * capacity_rate
        )
        outlet_ss = inlet - heat_w / capacity_rate

        values = {
            "coolant_capacity_rate_w_k": capacity_rate,
            "evaporator_effectiveness": effectiveness,
            "evaporator_resistance_k_w": resistance,
            "q_hc_w": heat_w,
            "coolant_outlet_temperature_ss_k": outlet_ss,
        }
        if not all(np.isfinite(float(v)) for v in values.values()):
            raise FloatingPointError("EvaporatorHeatCurrent produced a non-finite value")
        return values

    def outlet_temperature_from_applied_heat(
        self,
        *,
        coolant_inlet_temperature_k: float,
        coolant_mass_flow_kg_s: float,
        q_applied_w: float,
    ) -> float:
        """Dynamic outlet ``T_out,applied = T_in - Q_applied / G_c``.

        This is the caliber the running plant uses: ``Q_applied`` is the lagged
        heat from ``EvaporatorThermalDynamics`` (tau = 45 s), not the steady
        cycle solution. Energy closes exactly by construction:
        ``G_c * (T_in - T_out_applied) == Q_applied``.
        """
        inlet = float(coolant_inlet_temperature_k)
        heat = float(q_applied_w)
        if not np.isfinite(inlet) or inlet <= 0.0:
            raise ValueError("coolant_inlet_temperature_k must be finite and positive")
        if not np.isfinite(heat):
            raise ValueError("q_applied_w must be finite")
        capacity_rate = self.capacity_rate(coolant_mass_flow_kg_s)
        outlet = inlet - heat / capacity_rate
        if not np.isfinite(outlet):
            raise FloatingPointError("EvaporatorHeatCurrent produced a non-finite outlet")
        return float(outlet)
