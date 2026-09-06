"""Tests for the harmonized 13-node cold-plate reference (Stage 1.6).

The hard contract here is the *bit-exact* equivalence with the legacy
reference path when the HTC reference flow is set to the legacy value
(1.2 kg/s). Any drift between the two would silently invalidate the
frozen validation caliber, and a pinned test is the only way to catch it.
"""

from __future__ import annotations

import unittest

import numpy as np

from cluster_plant_v2.parameters import NOMINAL_COOLANT_MASS_FLOW_KG_S
from cluster_plant_v2.validation import legacy_cold_plate_reference as legacy
from cluster_plant_v2.validation.harmonized_cold_plate_reference import (
    DEFAULT_REFERENCE_MASS_FLOW_KG_S,
    HarmonizedColdPlateAdapter,
    cold_plate_fluid_exchange,
    convective_htc,
    substep_count,
)
from cluster_plant_v2.validation.validate_cold_plate_rom import (
    DetailedColdPlateAdapter,
    COOLANT_INLET_K,
    battery_heat_profile,
    liters_per_minute_to_mass_flow,
)


INLET_K = COOLANT_INLET_K


class HarmonizedColdPlateReferenceBitExactTests(unittest.TestCase):
    """Bit-exact equivalence with the legacy path at the legacy reference flow."""

    def test_substep_count_floors_at_one_and_handles_short_outer_steps(self) -> None:
        self.assertEqual(substep_count(5.0, 1.0), 5)
        self.assertEqual(substep_count(1.0, 1.0), 1)
        self.assertEqual(substep_count(0.5, 1.0), 1)
        self.assertEqual(substep_count(5.0, 2.0), 3)

    def test_default_reference_flow_matches_cold_plate_rom_anchor(self) -> None:
        from cluster_plant_v2.parameters import COLD_PLATE_REFERENCE_MASS_FLOW_KG_S

        self.assertEqual(
            DEFAULT_REFERENCE_MASS_FLOW_KG_S, COLD_PLATE_REFERENCE_MASS_FLOW_KG_S
        )
        self.assertEqual(DEFAULT_REFERENCE_MASS_FLOW_KG_S, 0.1428)

    def test_exchange_function_is_bit_exact_with_legacy_at_legacy_anchor(self) -> None:
        walls = np.linspace(295.15, 305.15, 13)
        mass_flow = liters_per_minute_to_mass_flow(21.0)

        legacy_exchange = legacy.cold_plate_fluid_exchange(
            walls, INLET_K, mass_flow
        )
        harmonized_exchange = cold_plate_fluid_exchange(
            walls, INLET_K, mass_flow, NOMINAL_COOLANT_MASS_FLOW_KG_S
        )

        self.assertEqual(legacy_exchange["T_out"], harmonized_exchange["T_out"])
        self.assertEqual(legacy_exchange["Q_total"], harmonized_exchange["Q_total"])
        np.testing.assert_array_equal(
            legacy_exchange["T_fluid_profile"],
            harmonized_exchange["T_fluid_profile"],
        )

    def test_adapter_matches_detailed_cold_plate_adapter_when_anchor_matches(self) -> None:
        mass_flow = liters_per_minute_to_mass_flow(21.0)
        node_heat = battery_heat_profile("uniform", 0.0)

        legacy_adapter = DetailedColdPlateAdapter()
        harmonized_adapter = HarmonizedColdPlateAdapter(
            reference_mass_flow=NOMINAL_COOLANT_MASS_FLOW_KG_S,
            internal_dt_s=5.0,  # disable sub-stepping for a clean comparison
        )
        for _ in range(40):
            legacy_result = legacy_adapter.outer_loop_equivalent = legacy_adapter.step(
                5.0, INLET_K, mass_flow, node_heat, 1
            )
            harmonized_result = harmonized_adapter.step(
                5.0, INLET_K, mass_flow, node_heat, 1
            )
            np.testing.assert_array_equal(
                legacy_result["plate_temperatures"],
                harmonized_result["plate_temperatures"],
            )
            self.assertEqual(
                legacy_result["coolant_outlet_temperature"],
                harmonized_result["coolant_outlet_temperature"],
            )
            np.testing.assert_array_equal(
                legacy_result["coolant_mean_temperatures"],
                harmonized_result["coolant_mean_temperatures"],
            )

    def test_harmonized_at_design_anchor_uses_higher_htc_than_legacy(self) -> None:
        mass_flow = liters_per_minute_to_mass_flow(21.0)
        h_legacy = convective_htc(mass_flow, NOMINAL_COOLANT_MASS_FLOW_KG_S)
        h_harmonized = convective_htc(mass_flow, DEFAULT_REFERENCE_MASS_FLOW_KG_S)
        self.assertGreater(h_harmonized, h_legacy)
        self.assertAlmostEqual(
            h_harmonized / h_legacy,
            (NOMINAL_COOLANT_MASS_FLOW_KG_S / DEFAULT_REFERENCE_MASS_FLOW_KG_S) ** 0.8,
            places=6,
        )


class HarmonizedColdPlateReferenceEnergyConsistencyTests(unittest.TestCase):
    """Q_ref_energy must be exactly the heat removed from the wall states."""

    def test_q_ref_energy_drives_exact_plate_state_balance(self) -> None:
        # Q_ref_energy is by construction the sum of node-wise h*A*(T_p-T_fmean)
        # which is exactly what advances the wall states, so over a single
        # sub-step (dt=1s, no accumulation loop) the identity is bit-exact.
        # Over 120 sub-stepped outer steps the only residual is the floating-
        # point associativity of the (h*A_seg*(T_p-T_fmean)) accumulation;
        # the spec demands only that the bias is bounded, not that it is zero.
        mass_flow = liters_per_minute_to_mass_flow(21.0)
        adapter = HarmonizedColdPlateAdapter(
            reference_mass_flow=DEFAULT_REFERENCE_MASS_FLOW_KG_S,
            internal_dt_s=1.0,
        )
        node_heat = battery_heat_profile("uniform", 0.0)
        max_residual_W = 0.0
        for _ in range(120):
            previous_energy = (adapter.node_heat_capacity * adapter.plate_temperatures).sum()
            result = adapter.step(5.0, INLET_K, mass_flow, node_heat, 1)
            new_energy = (adapter.node_heat_capacity * adapter.plate_temperatures).sum()
            energy_change_rate = (new_energy - previous_energy) / 5.0
            residual_W = abs(
                float(energy_change_rate)
                - (float(node_heat.sum()) - result["q_plate_to_fluid_energy"])
            )
            max_residual_W = max(max_residual_W, residual_W)
        # Bias bound: floating-point accumulation over 5 sub-steps + 13 nodes.
        # At Q ~ 1300 W, an rtol=1e-9 combined with atol=1e-9 W is well below
        # the measured ~5e-11 W associativity limit.
        self.assertLess(max_residual_W, 1e-9)

    def test_legacy_reported_q_under_reports_q_energy_at_steady_state(self) -> None:
        mass_flow = liters_per_minute_to_mass_flow(5.0)
        adapter = HarmonizedColdPlateAdapter(
            reference_mass_flow=DEFAULT_REFERENCE_MASS_FLOW_KG_S,
            internal_dt_s=1.0,
        )
        node_heat = battery_heat_profile("uniform", 0.0)
        # Long enough to flush the transient away.
        for _ in range(3000):
            result = adapter.step(5.0, INLET_K, mass_flow, node_heat, 1)
        self.assertGreater(result["q_plate_to_fluid_reported"], 0.0)
        self.assertGreater(result["q_plate_to_fluid_energy"], 0.0)
        self.assertGreater(result["q_plate_to_fluid_energy"], result["q_plate_to_fluid_reported"])
        # Steady-state defect (Stage 1.5 measured ~1.16 % at 5 L/min).
        deficit_pct = (
            1.0 - result["q_plate_to_fluid_reported"] / result["q_plate_to_fluid_energy"]
        ) * 100.0
        self.assertGreater(deficit_pct, 0.3)
        self.assertLess(deficit_pct, 2.5)


class HarmonizedColdPlateReferenceValidationTests(unittest.TestCase):
    """Configuration and constructor guards."""

    def test_invalid_reference_flow_is_rejected(self) -> None:
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(reference_mass_flow=bad):
                with self.assertRaises(ValueError):
                    HarmonizedColdPlateAdapter(reference_mass_flow=bad)

    def test_invalid_internal_dt_is_rejected(self) -> None:
        for bad in (0.0, -1.0, float("nan")):
            with self.subTest(internal_dt_s=bad):
                with self.assertRaises(ValueError):
                    HarmonizedColdPlateAdapter(internal_dt_s=bad)

    def test_zero_and_negative_mass_flow_are_rejected(self) -> None:
        adapter = HarmonizedColdPlateAdapter()
        node_heat = battery_heat_profile("uniform", 0.0)
        for bad in (0.0, -0.1):
            with self.subTest(mass_flow=bad):
                with self.assertRaises(ValueError):
                    adapter.step(5.0, INLET_K, bad, node_heat, 1)


if __name__ == "__main__":
    unittest.main()