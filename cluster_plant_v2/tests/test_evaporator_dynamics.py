import math
import unittest

import numpy as np

try:
    from cluster_plant_v2.refrigeration import EvaporatorThermalDynamics
except ImportError:
    EvaporatorThermalDynamics = None


class EvaporatorThermalDynamicsTests(unittest.TestCase):
    def test_constant_target_at_equilibrium_remains_constant(self) -> None:
        self.assertIsNotNone(EvaporatorThermalDynamics)
        dynamics = EvaporatorThermalDynamics(
            initial_q_evap_applied_w=1500.0,
            time_constant_s=45.0,
        )

        for _ in range(45):
            result = dynamics.step(dt_s=5.0, q_evap_cycle_w=1500.0)

        self.assertEqual(result["q_evap_applied_w"], 1500.0)
        self.assertEqual(result["evaporator_buffer_energy_j"], 0.0)
        self.assertEqual(result["evaporator_dynamic_energy_residual_j"], 0.0)
        self.assertEqual(dynamics.dynamic_state_count, 1)
        self.assertEqual(dynamics.diagnostic_state_count, 1)

    def test_upward_step_matches_exact_response_at_five_one_three_and_five_tau(
        self,
    ) -> None:
        self.assertIsNotNone(EvaporatorThermalDynamics)
        dynamics = EvaporatorThermalDynamics(
            initial_q_evap_applied_w=1000.0,
            time_constant_s=45.0,
        )
        samples = {}

        for step in range(1, 46):
            result = dynamics.step(dt_s=5.0, q_evap_cycle_w=2000.0)
            if step in (1, 9, 27, 45):
                samples[step * 5] = result["q_evap_applied_w"]
            self.assertLess(
                abs(result["evaporator_dynamic_energy_residual_j"]), 1e-9
            )

        for time_s in (5.0, 45.0, 135.0, 225.0):
            expected = 2000.0 - 1000.0 * math.exp(-time_s / 45.0)
            self.assertAlmostEqual(samples[int(time_s)], expected, places=12)

    def test_downward_step_is_monotonic_without_overshoot(self) -> None:
        self.assertIsNotNone(EvaporatorThermalDynamics)
        dynamics = EvaporatorThermalDynamics(
            initial_q_evap_applied_w=2000.0,
            time_constant_s=45.0,
        )

        values = [
            dynamics.step(dt_s=5.0, q_evap_cycle_w=1000.0)[
                "q_evap_applied_w"
            ]
            for _ in range(45)
        ]

        self.assertTrue(np.all(np.diff(values) < 0.0))
        self.assertTrue(np.all(np.asarray(values) > 1000.0))
        self.assertAlmostEqual(
            values[-1], 1000.0 + 1000.0 * math.exp(-5.0), places=12
        )

    def test_exact_discretization_is_independent_of_time_step(self) -> None:
        self.assertIsNotNone(EvaporatorThermalDynamics)
        final_values = []

        for dt_s in (5.0, 2.5, 1.0):
            dynamics = EvaporatorThermalDynamics(
                initial_q_evap_applied_w=1000.0,
                time_constant_s=45.0,
            )
            for _ in range(int(225.0 / dt_s)):
                result = dynamics.step(dt_s=dt_s, q_evap_cycle_w=2000.0)
            final_values.append(result["q_evap_applied_w"])

        expected = 2000.0 - 1000.0 * math.exp(-5.0)
        for value in final_values:
            self.assertAlmostEqual(value, expected, places=10)

    def test_invalid_initial_state_time_constant_and_step_are_rejected(self) -> None:
        self.assertIsNotNone(EvaporatorThermalDynamics)
        with self.assertRaisesRegex(ValueError, "initial_q_evap_applied_w"):
            EvaporatorThermalDynamics(
                initial_q_evap_applied_w=-1.0,
                time_constant_s=45.0,
            )
        with self.assertRaisesRegex(ValueError, "time_constant_s"):
            EvaporatorThermalDynamics(
                initial_q_evap_applied_w=1000.0,
                time_constant_s=0.0,
            )

        dynamics = EvaporatorThermalDynamics(
            initial_q_evap_applied_w=1000.0,
            time_constant_s=45.0,
        )
        with self.assertRaisesRegex(ValueError, "dt_s"):
            dynamics.step(dt_s=0.0, q_evap_cycle_w=2000.0)
        with self.assertRaisesRegex(ValueError, "q_evap_cycle_w"):
            dynamics.step(dt_s=5.0, q_evap_cycle_w=-1.0)


if __name__ == "__main__":
    unittest.main()
