import unittest

from thermal_system import pump_model, run_refrigeration_cycle


class RefrigerationCycleLimitsTest(unittest.TestCase):
    def test_cycle_exposes_heat_exchanger_and_refrigerant_capacity_limits(self):
        m_dot_cool, _ = pump_model(3000.0)
        result = run_refrigeration_cycle(3000.0, 3000.0, 298.15, m_dot_cool, 308.15)

        self.assertIn("Q_hx_potential", result)
        self.assertIn("Q_ref_max", result)
        self.assertAlmostEqual(result["Q_evap"], min(result["Q_hx_potential"], result["Q_ref_max"]), places=6)

    def test_cycle_is_off_below_map_and_active_at_2000_rpm_and_15c(self):
        m_dot_cool, _ = pump_model(1600.0)

        below_map = run_refrigeration_cycle(
            1999.0,
            1000.0,
            288.15,
            m_dot_cool,
            308.15,
        )
        active = run_refrigeration_cycle(
            2000.0,
            2000.0,
            288.15,
            m_dot_cool,
            308.15,
        )

        self.assertEqual(below_map["Q_evap"], 0.0)
        self.assertEqual(below_map["W_comp"], 0.0)
        self.assertGreater(active["Q_evap"], 0.0)


if __name__ == "__main__":
    unittest.main()
