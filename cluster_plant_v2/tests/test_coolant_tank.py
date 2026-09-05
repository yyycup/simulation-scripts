import unittest

from cluster_plant_v2.hydraulics import CoolantTank


class CoolantTankTests(unittest.TestCase):
    def test_five_second_step_matches_hand_calculation(self) -> None:
        tank = CoolantTank(initial_temperature_k=293.15)
        mass_flow = 0.44625
        return_temperature = 295.15
        dt_s = 5.0
        expected_delta = (
            mass_flow
            * (return_temperature - 293.15)
            * dt_s
            / (tank.coolant_density_kg_m3 * tank.volume_m3)
        )

        result = tank.step(dt_s, return_temperature, mass_flow)

        self.assertAlmostEqual(
            result["tank_temperature_after_k"], 293.15 + expected_delta, places=12
        )
        self.assertAlmostEqual(result["tank_energy_residual_j"], 0.0, places=8)

    def test_hotter_return_heats_tank(self) -> None:
        tank = CoolantTank(initial_temperature_k=293.15)
        result = tank.step(5.0, 294.15, 0.3)
        self.assertGreater(result["tank_temperature_after_k"], 293.15)

    def test_colder_return_cools_tank(self) -> None:
        tank = CoolantTank(initial_temperature_k=293.15)
        result = tank.step(5.0, 292.15, 0.3)
        self.assertLess(result["tank_temperature_after_k"], 293.15)

    def test_equal_return_temperature_has_no_advective_change(self) -> None:
        tank = CoolantTank(initial_temperature_k=293.15)
        result = tank.step(5.0, 293.15, 0.3)
        self.assertEqual(result["tank_temperature_after_k"], 293.15)
        self.assertEqual(result["return_to_tank_heat_w"], 0.0)
        self.assertEqual(result["tank_energy_change_j"], 0.0)

    def test_zero_flow_preserves_temperature(self) -> None:
        tank = CoolantTank(initial_temperature_k=293.15)
        result = tank.step(5.0, 320.0, 0.0)
        self.assertEqual(result["tank_temperature_after_k"], 293.15)


if __name__ == "__main__":
    unittest.main()
