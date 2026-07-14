import unittest

from thermal_system import pump_model, run_refrigeration_cycle


class RefrigerationCycleLimitsTest(unittest.TestCase):
    def test_cycle_exposes_heat_exchanger_and_refrigerant_capacity_limits(self):
        m_dot_cool, _ = pump_model(3000.0)
        result = run_refrigeration_cycle(3000.0, 3000.0, 298.15, m_dot_cool, 308.15)

        self.assertIn("Q_hx_potential", result)
        self.assertIn("Q_ref_max", result)
        self.assertAlmostEqual(result["Q_evap"], min(result["Q_hx_potential"], result["Q_ref_max"]), places=6)


if __name__ == "__main__":
    unittest.main()
