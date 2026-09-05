"""Contract tests for the coupled reduced battery and cold plate."""

from __future__ import annotations

import unittest

import numpy as np

from cluster_plant_v2.thermal.reduced_pack import ReducedPack


MASS_FLOW_5_LPM = 5.0 / 1000.0 / 60.0 * 1071.0


class ReducedPackTests(unittest.TestCase):
    def test_components_and_dynamic_state_shapes_are_preserved(self) -> None:
        pack = ReducedPack()

        self.assertEqual(pack.battery.temps.shape, (4, 3))
        self.assertEqual(pack.cold_plate.plate_temperatures.shape, (3,))
        self.assertEqual(pack.dynamic_state_count, 43)

    def test_one_step_uses_one_exact_battery_to_plate_heat_array(self) -> None:
        pack = ReducedPack()

        result = pack.step(
            5.0, 560.0, 293.15, MASS_FLOW_5_LPM, 308.15, flow_direction=1
        )

        self.assertAlmostEqual(float(result["branch_currents"].sum()), 560.0)
        self.assertTrue(np.all(result["soc"] < 0.95))
        self.assertGreater(result["q_gen_total"], 0.0)
        self.assertEqual(result["q_battery_to_plate_branch_zone"].shape, (4, 3))
        self.assertEqual(result["q_battery_to_plate_zones"].shape, (3,))
        np.testing.assert_array_equal(
            result["q_battery_to_plate_zones"],
            result["q_plate_gain_zones"],
        )
        np.testing.assert_array_equal(result["coupling_energy_residual"], 0.0)
        self.assertGreater(result["q_plate_to_fluid_total"], 0.0)
        self.assertGreater(result["coolant_outlet_temperature"], 293.15)

    def test_whole_pack_energy_balance_closes_at_floating_point_precision(self) -> None:
        pack = ReducedPack()

        result = pack.step(
            5.0, 560.0, 293.15, MASS_FLOW_5_LPM, 308.15, flow_direction=1
        )

        self.assertLess(abs(result["whole_pack_energy_residual_J"]), 1e-6)
        self.assertLess(abs(result["whole_pack_energy_relative_error"]), 1e-9)

    def test_forward_and_reverse_flow_are_finite_and_traverse_opposite_directions(self) -> None:
        forward = ReducedPack()
        reverse = ReducedPack()

        forward_result = forward.step(
            5.0, 560.0, 293.15, MASS_FLOW_5_LPM, 308.15, flow_direction=1
        )
        reverse_result = reverse.step(
            5.0, 560.0, 293.15, MASS_FLOW_5_LPM, 308.15, flow_direction=-1
        )

        self.assertTrue(
            np.all(np.diff(forward_result["coolant_mean_temperatures"]) > 0.0)
        )
        self.assertTrue(
            np.all(np.diff(reverse_result["coolant_mean_temperatures"]) < 0.0)
        )
        for result in (forward_result, reverse_result):
            for value in result.values():
                if isinstance(value, np.ndarray):
                    self.assertTrue(np.all(np.isfinite(value)))
                elif isinstance(value, (int, float, np.number)):
                    self.assertTrue(np.isfinite(value))

    def test_soc_direction_matches_current_sign(self) -> None:
        discharge = ReducedPack()
        charge = ReducedPack()

        discharge.step(5.0, 560.0, 293.15, MASS_FLOW_5_LPM, 308.15)
        charge.step(5.0, -560.0, 293.15, MASS_FLOW_5_LPM, 308.15)

        self.assertTrue(np.all(discharge.battery.soc_branch < 0.95))
        # Charging saturates at the HPPC lookup support (SOC <= 0.95): the
        # ROM clamps Coulomb counting to the calibrated table domain.
        np.testing.assert_allclose(charge.battery.soc_branch, 0.95)


if __name__ == "__main__":
    unittest.main()
