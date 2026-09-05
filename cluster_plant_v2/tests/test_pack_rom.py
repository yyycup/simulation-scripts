"""Contract tests for the 4-by-3 reduced battery pack."""

from __future__ import annotations

import unittest

import numpy as np

from cluster_plant_v2.pack_reference import ReferenceBatteryPack
from cluster_plant_v2.thermal.pack_rom import (
    ZONE_CELL_COUNTS,
    ZONE_GROUPS,
    ReducedBatteryPack,
)


class ReducedBatteryPackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rom = ReducedBatteryPack()

    def test_structure_and_state_shapes_are_4_by_3(self) -> None:
        rom = self.rom

        self.assertEqual(ZONE_GROUPS, ((0, 1, 2, 3), (4, 5, 6, 7, 8), (9, 10, 11, 12)))
        np.testing.assert_array_equal(ZONE_CELL_COUNTS, [4, 5, 4])
        self.assertEqual(rom.soc_branch.shape, (4,))
        self.assertEqual(rom.eta1.shape, (4, 3))
        self.assertEqual(rom.eta2.shape, (4, 3))
        self.assertEqual(rom.temps.shape, (4, 3))
        self.assertEqual(rom.q_gen_zones.shape, (4, 3))
        self.assertEqual(rom.branch_currents.shape, (4,))
        self.assertEqual(rom.dynamic_state_count, 40)

    def test_cell_count_heat_capacity_and_plate_conductance_are_conserved(self) -> None:
        rom = self.rom

        self.assertEqual(4 * int(ZONE_CELL_COUNTS.sum()), 52)
        self.assertEqual(float(rom.zone_heat_capacities.sum()), 52 * 4747.0)
        np.testing.assert_array_equal(rom.zone_plate_conductances[0], [40.0, 50.0, 40.0])

    def test_zone_rc_scaling_preserves_cell_time_constants(self) -> None:
        rom = self.rom
        soc = np.full((4, 3), 0.95)
        temperature = np.full((4, 3), 298.15)
        current = np.full((4, 3), 140.0)

        cell = rom._cell_parameters(soc, temperature, current)
        zone = rom._zone_parameters(soc, temperature, current)

        np.testing.assert_allclose(zone["r1"] * zone["c1"], cell["r1"] * cell["c1"])
        np.testing.assert_allclose(zone["r2"] * zone["c2"], cell["r2"] * cell["c2"])

    def test_branch_current_split_conserves_pack_current(self) -> None:
        currents = self.rom.calculate_branch_currents(560.0)

        self.assertAlmostEqual(float(currents.sum()), 560.0, places=10)
        np.testing.assert_allclose(currents, 140.0, atol=1e-9, rtol=0.0)

    def test_soc_sign_matches_pybamm_convention(self) -> None:
        discharge = ReducedBatteryPack()
        charge = ReducedBatteryPack()

        discharge.step(5.0, 560.0, np.full(3, 298.15), 298.15)
        charge.step(5.0, -560.0, np.full(3, 298.15), 298.15)

        self.assertTrue(np.all(discharge.soc_branch < 0.95))
        # Charging saturates at the HPPC lookup support (SOC <= 0.95): the
        # ROM clamps Coulomb counting to the calibrated table domain so
        # parameter lookups never extrapolate.
        np.testing.assert_allclose(charge.soc_branch, 0.95)

    def test_zone_heat_uses_frozen_pybamm_definition(self) -> None:
        rom = ReducedBatteryPack()
        rom.step(5.0, 560.0, np.full(3, 298.15), 298.15)
        current = rom.branch_currents[:, None]
        expected = (
            current**2 * rom.last_zone_parameters["r0"]
            - current * rom.eta1
            - current * rom.eta2
        )

        np.testing.assert_allclose(rom.q_gen_zones, expected, rtol=0.0, atol=1e-12)
        self.assertTrue(np.all(np.isfinite(rom.q_gen_zones)))

    def test_step_exposes_the_exact_plate_and_air_heat_used_by_thermal_update(self) -> None:
        rom = ReducedBatteryPack()
        old_temperatures = rom.temps.copy()
        plate_temperatures = np.array([296.15, 297.15, 298.15])
        ambient_temperature = 308.15

        rom.step(5.0, 560.0, plate_temperatures, ambient_temperature)

        expected_plate = rom.zone_plate_conductances * (
            old_temperatures - plate_temperatures[None, :]
        )
        expected_air = rom.zone_air_conductances * (
            old_temperatures - ambient_temperature
        )
        np.testing.assert_array_equal(
            rom.get_battery_to_plate_heat_branch_zone(), expected_plate
        )
        np.testing.assert_array_equal(
            rom.get_battery_to_plate_heat(), expected_plate.sum(axis=0)
        )
        np.testing.assert_array_equal(rom.get_air_heat_loss(), expected_air)

    def test_neighbor_heat_is_directional_and_conservative(self) -> None:
        rom = self.rom
        uniform = np.full((4, 3), 298.15)
        np.testing.assert_array_equal(rom.calculate_neighbor_heat(uniform), np.zeros((4, 3)))
        hot = uniform.copy()
        hot[1, 1] += 10.0

        heat = rom.calculate_neighbor_heat(hot)

        self.assertLess(heat[1, 1], 0.0)
        self.assertGreater(heat[1, 0], 0.0)
        self.assertGreater(heat[1, 2], 0.0)
        self.assertGreater(heat[0, 1], 0.0)
        self.assertGreater(heat[2, 1], 0.0)
        self.assertAlmostEqual(float(heat.sum()), 0.0, places=12)

    def test_reconstruction_expands_4_by_3_to_4_by_13(self) -> None:
        rom = ReducedBatteryPack()
        rom.temps = np.arange(12, dtype=float).reshape(4, 3)

        reconstructed = rom.reconstruct_cell_temperatures()

        self.assertEqual(reconstructed.shape, (4, 13))
        for branch in range(4):
            for zone, columns in enumerate(ZONE_GROUPS):
                np.testing.assert_array_equal(reconstructed[branch, list(columns)], rom.temps[branch, zone])

    def test_reference_state_mapping_sums_zone_overpotentials(self) -> None:
        reference = ReferenceBatteryPack()
        reference.step(5.0, 560.0, np.full(13, 298.15), 298.15)
        rom = ReducedBatteryPack()

        rom.initialize_from_reference(reference)

        reference_soc = reference.socs.reshape(4, 13)
        reference_temp = reference.temps.reshape(4, 13)
        self.assertTrue(np.allclose(rom.soc_branch, reference_soc.mean(axis=1)))
        for branch in range(4):
            for zone, columns in enumerate(ZONE_GROUPS):
                self.assertAlmostEqual(
                    rom.temps[branch, zone], reference_temp[branch, list(columns)].mean()
                )
                expected_eta1 = sum(
                    float(reference.sims[branch * 13 + column].solution["Element-1 overpotential [V]"].data[-1])
                    for column in columns
                )
                self.assertAlmostEqual(rom.eta1[branch, zone], expected_eta1)

    def test_one_five_second_step_is_finite(self) -> None:
        rom = ReducedBatteryPack()
        rom.step(5.0, 1120.0, np.full(3, 298.15), 298.15)

        for state in (rom.soc_branch, rom.eta1, rom.eta2, rom.temps, rom.q_gen_zones):
            self.assertTrue(np.all(np.isfinite(state)))


if __name__ == "__main__":
    unittest.main()
