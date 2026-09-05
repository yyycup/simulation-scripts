"""Contract tests for the full-order 4P13S reference battery pack."""

from __future__ import annotations

import unittest

import numpy as np

from cluster_plant_v2.pack_reference import ReferenceBatteryPack


class ReferenceBatteryPackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pack = ReferenceBatteryPack()

    def test_topology_and_public_state_shapes_are_4p13s(self) -> None:
        pack = self.pack

        self.assertEqual((pack.rows, pack.cols, pack.Ns), (4, 13, 52))
        self.assertEqual(pack.cell_index(0, 0), 0)
        self.assertEqual(pack.cell_index(3, 12), 51)
        self.assertEqual(pack.temps.shape, (52,))
        self.assertEqual(pack.socs.shape, (52,))
        self.assertEqual(pack.q_gen_cells.shape, (52,))
        self.assertEqual(pack.branch_currents.shape, (4,))

    def test_pybamm_model_really_contains_two_rc_elements(self) -> None:
        model = self.pack.model_template
        rhs_state_names = {variable.name for variable in model.rhs}

        self.assertEqual(model.options["number of rc elements"], 2)
        self.assertIn("Element-1 overpotential [V]", model.variables)
        self.assertIn("Element-2 overpotential [V]", model.variables)
        self.assertIn("Element-1 overpotential [V]", rhs_state_names)
        self.assertIn("Element-2 overpotential [V]", rhs_state_names)
        self.assertNotIn("Cell temperature [K]", rhs_state_names)
        self.assertIn("Total heat generation [W]", model.variables)

    def test_branch_current_split_conserves_total_current(self) -> None:
        branch_currents = self.pack.calculate_branch_currents(560.0)

        self.assertEqual(branch_currents.shape, (4,))
        self.assertAlmostEqual(float(branch_currents.sum()), 560.0, places=10)
        np.testing.assert_allclose(branch_currents, 140.0, rtol=0.0, atol=1e-9)

    def test_recommended_read_interfaces_return_pack_state(self) -> None:
        pack = self.pack

        self.assertEqual(pack.get_max_temp(), float(np.max(pack.temps)))
        self.assertEqual(pack.get_min_temp(), float(np.min(pack.temps)))
        self.assertEqual(pack.get_delta_temp(), float(np.ptp(pack.temps)))
        np.testing.assert_array_equal(pack.get_soc_array(), pack.socs)

    def test_uniform_temperature_has_zero_neighbor_heat(self) -> None:
        neighbor_heat = self.pack.calculate_neighbor_heat(
            np.full(52, 298.15, dtype=float)
        )

        np.testing.assert_array_equal(neighbor_heat, np.zeros(52))
        self.assertEqual(float(neighbor_heat.sum()), 0.0)

    def test_temperature_gradient_drives_pairwise_conservative_heat(self) -> None:
        temperatures = np.full(52, 298.15, dtype=float)
        hot_index = self.pack.cell_index(1, 6)
        temperatures[hot_index] += 10.0

        neighbor_heat = self.pack.calculate_neighbor_heat(temperatures)

        self.assertLess(neighbor_heat[hot_index], 0.0)
        for row, col in ((0, 6), (2, 6), (1, 5), (1, 7)):
            self.assertGreater(neighbor_heat[self.pack.cell_index(row, col)], 0.0)
        self.assertAlmostEqual(float(neighbor_heat.sum()), 0.0, places=12)

    def test_zero_and_loaded_five_second_steps_are_finite(self) -> None:
        pack = ReferenceBatteryPack()
        plate_temperature = np.full(13, 298.15, dtype=float)

        pack.step(
            dt=5.0,
            total_current=0.0,
            plate_temperatures=plate_temperature,
            ambient_temperature=298.15,
        )
        np.testing.assert_allclose(pack.q_gen_cells, 0.0, rtol=0.0, atol=1e-9)

        pack.step(
            dt=5.0,
            total_current=560.0,
            plate_temperatures=plate_temperature,
            ambient_temperature=298.15,
        )

        self.assertEqual(pack.q_gen_cells.shape, (52,))
        self.assertTrue(np.all(np.isfinite(pack.q_gen_cells)))
        self.assertTrue(np.all(pack.q_gen_cells > 0.0))
        self.assertTrue(np.all(np.isfinite(pack.temps)))
        self.assertTrue(np.all(np.isfinite(pack.socs)))
        self.assertTrue(np.all(np.isfinite(pack.branch_currents)))
        self.assertGreater(pack.get_max_temperature(), pack.get_min_temperature() - 1e-12)
        self.assertAlmostEqual(
            pack.get_temperature_delta(),
            pack.get_max_temperature() - pack.get_min_temperature(),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
