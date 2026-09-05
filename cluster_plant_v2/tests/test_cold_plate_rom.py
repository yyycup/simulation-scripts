"""Tests for the isolated three-zone cold-plate ROM."""

from __future__ import annotations

import unittest

import numpy as np

from cluster_plant_v2.thermal.cold_plate_rom import ReducedColdPlate


class ReducedColdPlateTests(unittest.TestCase):
    def test_structure_state_heat_capacity_and_area_are_conservative(self) -> None:
        plate = ReducedColdPlate()

        self.assertEqual(plate.zone_column_counts.tolist(), [4, 5, 4])
        self.assertEqual(plate.plate_temperatures.shape, (3,))
        self.assertAlmostEqual(float(plate.zone_heat_capacities.sum()), 6000.0)
        self.assertAlmostEqual(float(plate.zone_areas.sum()), 0.5)

    def test_zero_temperature_difference_and_zero_load_leave_state_unchanged(self) -> None:
        plate = ReducedColdPlate(initial_temperature_c=20.0)

        result = plate.step(5.0, 293.15, 0.1, np.zeros(3))

        np.testing.assert_allclose(plate.plate_temperatures, 293.15)
        self.assertAlmostEqual(result["coolant_outlet_temperature"], 293.15)
        self.assertAlmostEqual(result["q_plate_to_fluid_total"], 0.0)

    def test_reference_flow_recovers_nominal_heat_transfer_coefficient(self) -> None:
        plate = ReducedColdPlate(initial_temperature_c=25.0)

        result = plate.step(5.0, 293.15, 0.1428, np.zeros(3))

        self.assertAlmostEqual(result["h_dynamic"], 2000.0)

    def test_hot_plate_heats_coolant_and_uses_same_heat_for_energy_balance(self) -> None:
        plate = ReducedColdPlate(initial_temperature_c=25.0)
        mass_flow = 0.1

        result = plate.step(5.0, 293.15, mass_flow, np.zeros(3))

        self.assertGreater(result["coolant_outlet_temperature"], 293.15)
        coolant_gain = mass_flow * plate.coolant_cp * (
            result["coolant_outlet_temperature"] - 293.15
        )
        self.assertAlmostEqual(coolant_gain, result["q_plate_to_fluid_total"], places=9)
        np.testing.assert_allclose(
            plate.initial_zone_energy_J - plate.zone_energy_J,
            5.0 * result["q_plate_to_fluid"],
        )

    def test_forward_and_reverse_flow_visit_physical_zones_in_opposite_order(self) -> None:
        forward = ReducedColdPlate(initial_temperature_c=25.0)
        reverse = ReducedColdPlate(initial_temperature_c=25.0)

        forward_result = forward.step(5.0, 293.15, 0.1, np.zeros(3), flow_direction=1)
        reverse_result = reverse.step(5.0, 293.15, 0.1, np.zeros(3), flow_direction=-1)

        self.assertTrue(np.all(np.diff(forward_result["coolant_mean_temperatures"]) > 0.0))
        self.assertTrue(np.all(np.diff(reverse_result["coolant_mean_temperatures"]) < 0.0))
        np.testing.assert_allclose(forward.plate_temperatures, reverse.plate_temperatures[::-1])

    def test_nonpositive_mass_flow_and_invalid_direction_are_rejected(self) -> None:
        plate = ReducedColdPlate()

        for mass_flow in (0.0, -0.1):
            with self.subTest(mass_flow=mass_flow):
                with self.assertRaises(ValueError):
                    plate.step(5.0, 293.15, mass_flow, np.zeros(3))
        with self.assertRaises(ValueError):
            plate.step(5.0, 293.15, 0.1, np.zeros(3), flow_direction=0)

    def test_finite_step_with_battery_heat_input(self) -> None:
        plate = ReducedColdPlate()

        result = plate.step(5.0, 293.15, 0.1, np.array([400.0, 500.0, 400.0]))

        self.assertTrue(np.all(np.isfinite(plate.plate_temperatures)))
        self.assertTrue(np.all(np.isfinite(result["q_plate_to_fluid"])))
        self.assertTrue(np.isfinite(result["coolant_outlet_temperature"]))


if __name__ == "__main__":
    unittest.main()
