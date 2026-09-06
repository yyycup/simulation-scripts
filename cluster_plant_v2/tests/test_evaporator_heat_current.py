"""Tests for the Stage 2B coolant-side evaporator heat-current interface.

The contract under test is deliberately narrow. ``EvaporatorHeatCurrent`` must
reproduce the incumbent epsilon-NTU relation *exactly* (it is an algebraic
rewrite, not new physics), must not reverse-depend on the cycle solver, and
must not introduce a thermal state.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

from cluster_plant_v2.parameters import COOLANT_SPECIFIC_HEAT_J_KG_K
from cluster_plant_v2.refrigeration import ClosedR134aCycle
from cluster_plant_v2.thermal import evaporator_heat_current as hc_module
from cluster_plant_v2.thermal.evaporator_heat_current import EvaporatorHeatCurrent

MODULE_PATH = Path(hc_module.__file__)

#: Coolant inlet temperatures [K] -- realistic tank temperatures.
INLET_TEMPERATURES_K = (293.15, 298.15, 303.15)
#: Evaporating saturation temperatures [K], inside the audited envelope.
EVAPORATING_TEMPERATURES_K = (281.15, 285.15, 288.15)
#: Coolant mass flows [kg/s] -- from half to full cluster flow.
MASS_FLOWS_KG_S = (0.4, 0.8, 1.2)


def _cycle_point(
    *,
    coolant_inlet_temperature_k: float,
    coolant_mass_flow_kg_s: float,
    evaporating_saturation_temperature_k: float,
    compressor_speed_rpm: float = 2400.0,
    fan_speed_rpm: float = 3000.0,
    ambient_temperature_k: float = 298.15,
    condensing_saturation_temperature_k: float = 318.15,
) -> dict[str, object]:
    return ClosedR134aCycle().evaluate_operating_point(
        compressor_speed_rpm=compressor_speed_rpm,
        fan_speed_rpm=fan_speed_rpm,
        coolant_inlet_temperature_k=coolant_inlet_temperature_k,
        coolant_mass_flow_kg_s=coolant_mass_flow_kg_s,
        ambient_temperature_k=ambient_temperature_k,
        evaporating_saturation_temperature_k=(
            evaporating_saturation_temperature_k
        ),
        condensing_saturation_temperature_k=condensing_saturation_temperature_k,
    )


class EvaporatorHeatCurrentIdentityTests(unittest.TestCase):
    """Q_HC must equal Q_NTU for identical T_e / UA_e / m_dot_c / T_in."""

    def setUp(self) -> None:
        self.model = EvaporatorHeatCurrent()

    def test_heat_current_equals_ntu_across_operating_grid(self) -> None:
        worst_relative = 0.0
        worst_absolute_w = 0.0
        for inlet in INLET_TEMPERATURES_K:
            for evaporating in EVAPORATING_TEMPERATURES_K:
                if evaporating >= inlet:
                    continue
                for mass_flow in MASS_FLOWS_KG_S:
                    point = _cycle_point(
                        coolant_inlet_temperature_k=inlet,
                        coolant_mass_flow_kg_s=mass_flow,
                        evaporating_saturation_temperature_k=evaporating,
                    )
                    reference_q = float(point["q_evaporator_ntu_w"])
                    result = self.model.evaluate(
                        coolant_inlet_temperature_k=inlet,
                        coolant_mass_flow_kg_s=mass_flow,
                        evaporating_temperature_k=evaporating,
                        evaporator_ua_w_k=float(point["evaporator_ua_w_k"]),
                    )
                    worst_absolute_w = max(
                        worst_absolute_w, abs(result["q_hc_w"] - reference_q)
                    )
                    worst_relative = max(
                        worst_relative,
                        abs(result["q_hc_w"] - reference_q)
                        / max(abs(reference_q), 1e-9),
                    )

        # Algebraic identity: only floating-point associativity may differ.
        self.assertLess(worst_relative, 1e-12)
        self.assertLess(worst_absolute_w, 1e-9)

    def test_heat_current_equals_ntu_at_the_solved_cycle_boundary(self) -> None:
        """Upper layer solves the cycle, then hands T_e and UA_e down.

        This pins the integration contract: the new module accepts whatever
        boundary the incumbent solver produces and agrees with it.
        """
        cycle = ClosedR134aCycle()
        for compressor_speed_rpm in (1800.0, 3000.0, 4200.0):
            for mass_flow in (0.6, 1.2):
                solution = cycle.solve(
                    compressor_speed_rpm=compressor_speed_rpm,
                    fan_speed_rpm=3000.0,
                    coolant_inlet_temperature_k=298.15,
                    coolant_mass_flow_kg_s=mass_flow,
                    ambient_temperature_k=298.15,
                )
                self.assertTrue(
                    bool(solution["solver_success"]),
                    msg=str(solution["solver_message"]),
                )
                result = self.model.evaluate(
                    coolant_inlet_temperature_k=298.15,
                    coolant_mass_flow_kg_s=mass_flow,
                    evaporating_temperature_k=float(
                        solution["evaporating_saturation_temperature_k"]
                    ),
                    evaporator_ua_w_k=float(solution["evaporator_ua_w_k"]),
                )
                reference_q = float(solution["q_evaporator_ntu_w"])
                self.assertLess(
                    abs(result["q_hc_w"] - reference_q)
                    / max(abs(reference_q), 1e-9),
                    1e-12,
                )
                # At the solved boundary the cycle and the exchanger agree,
                # so Q_cycle is also reproduced to solver tolerance.
                self.assertLess(
                    abs(result["q_hc_w"] - float(solution["q_evaporator_w"]))
                    / max(abs(float(solution["q_evaporator_w"])), 1e-9),
                    5e-3,
                )

    def test_resistance_is_reciprocal_capacity_times_effectiveness(self) -> None:
        for mass_flow in MASS_FLOWS_KG_S:
            capacity = self.model.capacity_rate(mass_flow)
            for ua in (500.0, 5000.0, 20000.0):
                resistance = self.model.resistance(capacity, ua)
                effectiveness = float(-np.expm1(-ua / capacity))
                self.assertAlmostEqual(
                    resistance,
                    1.0 / (capacity * effectiveness),
                    delta=1e-12 * max(resistance, 1.0),
                )

    def test_effectiveness_report_matches_analytic_epsilon_ntu(self) -> None:
        point = _cycle_point(
            coolant_inlet_temperature_k=298.15,
            coolant_mass_flow_kg_s=1.2,
            evaporating_saturation_temperature_k=285.15,
        )
        ua = float(point["evaporator_ua_w_k"])
        result = self.model.evaluate(
            coolant_inlet_temperature_k=298.15,
            coolant_mass_flow_kg_s=1.2,
            evaporating_temperature_k=285.15,
            evaporator_ua_w_k=ua,
        )
        expected = float(
            -np.expm1(-ua / result["coolant_capacity_rate_w_k"])
        )
        self.assertAlmostEqual(result["evaporator_effectiveness"], expected, places=12)


class EvaporatorHeatCurrentEnergyTests(unittest.TestCase):
    """Both outlet calibers must close energy by construction."""

    def setUp(self) -> None:
        self.model = EvaporatorHeatCurrent()

    def test_steady_outlet_closes_energy(self) -> None:
        for mass_flow in MASS_FLOWS_KG_S:
            result = self.model.evaluate(
                coolant_inlet_temperature_k=298.15,
                coolant_mass_flow_kg_s=mass_flow,
                evaporating_temperature_k=285.15,
                evaporator_ua_w_k=8000.0,
            )
            recovered = result["coolant_capacity_rate_w_k"] * (
                298.15 - result["coolant_outlet_temperature_ss_k"]
            )
            self.assertAlmostEqual(recovered, result["q_hc_w"], delta=1e-9)

    def test_applied_outlet_closes_energy_exactly(self) -> None:
        """G_c * (T_in - T_out,applied) == Q_applied."""
        for mass_flow in MASS_FLOWS_KG_S:
            capacity = self.model.capacity_rate(mass_flow)
            for q_applied in (0.0, 1500.0, 9000.0, 20000.0):
                outlet = self.model.outlet_temperature_from_applied_heat(
                    coolant_inlet_temperature_k=298.15,
                    coolant_mass_flow_kg_s=mass_flow,
                    q_applied_w=q_applied,
                )
                recovered = capacity * (298.15 - outlet)
                self.assertAlmostEqual(
                    recovered,
                    q_applied,
                    delta=max(1e-9, 1e-12 * abs(q_applied)),
                )

    def test_applied_outlet_matches_incumbent_plant_formula(self) -> None:
        """Reproduce ``T_in - Q_applied / (m_dot * cp)`` used by the plant."""
        mass_flow = 0.9
        q_applied = 7345.6
        outlet = self.model.outlet_temperature_from_applied_heat(
            coolant_inlet_temperature_k=295.35,
            coolant_mass_flow_kg_s=mass_flow,
            q_applied_w=q_applied,
        )
        expected = 295.35 - q_applied / (
            mass_flow * COOLANT_SPECIFIC_HEAT_J_KG_K
        )
        self.assertAlmostEqual(outlet, expected, places=12)

    def test_applied_heat_of_zero_returns_inlet_temperature(self) -> None:
        outlet = self.model.outlet_temperature_from_applied_heat(
            coolant_inlet_temperature_k=300.0,
            coolant_mass_flow_kg_s=1.0,
            q_applied_w=0.0,
        )
        self.assertAlmostEqual(outlet, 300.0, places=12)

    def test_applied_outlet_accepts_negative_heat(self) -> None:
        """Reverse flow of heat (e.g. startup) stays well defined."""
        outlet = self.model.outlet_temperature_from_applied_heat(
            coolant_inlet_temperature_k=300.0,
            coolant_mass_flow_kg_s=1.0,
            q_applied_w=-5000.0,
        )
        self.assertGreater(outlet, 300.0)


class EvaporatorHeatCurrentLimitTests(unittest.TestCase):
    """Physical limits of the heat-current form."""

    def setUp(self) -> None:
        self.model = EvaporatorHeatCurrent()

    def test_infinite_area_drives_outlet_to_evaporating_temperature(self) -> None:
        result = self.model.evaluate(
            coolant_inlet_temperature_k=298.15,
            coolant_mass_flow_kg_s=1.2,
            evaporating_temperature_k=285.15,
            evaporator_ua_w_k=1.0e9,
        )
        # epsilon -> 1, so T_out -> T_e and Q -> G_c (T_in - T_e).
        self.assertAlmostEqual(
            result["coolant_outlet_temperature_ss_k"], 285.15, places=6
        )
        self.assertAlmostEqual(
            result["q_hc_w"],
            result["coolant_capacity_rate_w_k"] * (298.15 - 285.15),
            delta=1e-6 * result["coolant_capacity_rate_w_k"],
        )
        self.assertAlmostEqual(result["evaporator_effectiveness"], 1.0, places=12)

    def test_vanishing_area_drives_heat_to_zero(self) -> None:
        result = self.model.evaluate(
            coolant_inlet_temperature_k=298.15,
            coolant_mass_flow_kg_s=1.2,
            evaporating_temperature_k=285.15,
            evaporator_ua_w_k=1.0e-9,
        )
        self.assertGreater(result["evaporator_resistance_k_w"], 0.0)
        self.assertLess(result["q_hc_w"], 1e-6)
        self.assertAlmostEqual(
            result["coolant_outlet_temperature_ss_k"], 298.15, places=6
        )

    def test_heat_grows_monotonically_with_area(self) -> None:
        heats = []
        for ua in (1000.0, 3000.0, 8000.0, 20000.0):
            result = self.model.evaluate(
                coolant_inlet_temperature_k=298.15,
                coolant_mass_flow_kg_s=1.2,
                evaporating_temperature_k=285.15,
                evaporator_ua_w_k=ua,
            )
            heats.append(result["q_hc_w"])
        self.assertTrue(all(b > a for a, b in zip(heats, heats[1:])))

    def test_expm1_form_stays_accurate_at_tiny_ntu(self) -> None:
        """1 - exp(-x) must not catastrophically cancel for x -> 0."""
        capacity = self.model.capacity_rate(1.2)
        resistance = self.model.resistance(capacity, 1e-12)
        self.assertTrue(np.isfinite(resistance))
        self.assertGreater(resistance, 0.0)


class EvaporatorHeatCurrentBoundaryTests(unittest.TestCase):
    """The module must not grow physics it does not own."""

    def test_no_internal_thermal_state(self) -> None:
        self.assertEqual(EvaporatorHeatCurrent.dynamic_state_count, 0)
        model = EvaporatorHeatCurrent()
        self.assertNotIn("q_evap_applied_w", vars(model))

    def test_module_never_imports_or_calls_the_cycle_solver(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(
                        "refrigeration",
                        alias.name,
                        msg="module must not import the refrigeration cycle",
                    )
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn(
                    "refrigeration",
                    node.module or "",
                    msg="module must not import the refrigeration cycle",
                )
            elif isinstance(node, ast.Attribute):
                self.assertNotEqual(
                    node.attr,
                    "solve",
                    msg="module must not call solve() on the cycle",
                )

    def test_evaluate_output_keys_are_the_agreed_contract(self) -> None:
        result = EvaporatorHeatCurrent().evaluate(
            coolant_inlet_temperature_k=298.15,
            coolant_mass_flow_kg_s=1.2,
            evaporating_temperature_k=285.15,
            evaporator_ua_w_k=8000.0,
        )
        self.assertEqual(
            set(result),
            {
                "coolant_capacity_rate_w_k",
                "evaporator_effectiveness",
                "evaporator_resistance_k_w",
                "q_hc_w",
                "coolant_outlet_temperature_ss_k",
            },
        )


class EvaporatorHeatCurrentValidationTests(unittest.TestCase):
    """Input guards."""

    def setUp(self) -> None:
        self.model = EvaporatorHeatCurrent()

    def test_nonpositive_mass_flow_rejected(self) -> None:
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.model.capacity_rate(bad)

    def test_nonpositive_ua_rejected(self) -> None:
        for bad in (0.0, -100.0, float("nan")):
            with self.assertRaises(ValueError):
                self.model.resistance(5000.0, bad)

    def test_nonpositive_capacity_rate_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.model.resistance(0.0, 5000.0)

    def test_nonfinite_temperatures_rejected(self) -> None:
        for bad_inlet in (float("nan"), float("inf"), 0.0, -5.0):
            with self.assertRaises(ValueError):
                self.model.evaluate(
                    coolant_inlet_temperature_k=bad_inlet,
                    coolant_mass_flow_kg_s=1.0,
                    evaporating_temperature_k=285.0,
                    evaporator_ua_w_k=8000.0,
                )
        for bad_evaporating in (float("nan"), float("inf"), 0.0, -5.0):
            with self.assertRaises(ValueError):
                self.model.evaluate(
                    coolant_inlet_temperature_k=298.0,
                    coolant_mass_flow_kg_s=1.0,
                    evaporating_temperature_k=bad_evaporating,
                    evaporator_ua_w_k=8000.0,
                )

    def test_nonfinite_applied_heat_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.model.outlet_temperature_from_applied_heat(
                coolant_inlet_temperature_k=298.0,
                coolant_mass_flow_kg_s=1.0,
                q_applied_w=float("nan"),
            )

    def test_capacity_rate_uses_coolant_specific_heat(self) -> None:
        self.assertAlmostEqual(
            self.model.capacity_rate(1.0), COOLANT_SPECIFIC_HEAT_J_KG_K, places=9
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
