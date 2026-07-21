import unittest

from thermal_loop import staged_fan_speed
from thermal_system import (
    clear_refrigeration_cycle_cache,
    compressor_efficiencies,
    pump_model,
    run_refrigeration_cycle,
)


class RefrigerationCycleLimitsTest(unittest.TestCase):
    def test_cycle_exposes_heat_exchanger_and_refrigerant_capacity_limits(self):
        m_dot_cool, _ = pump_model(3000.0)
        result = run_refrigeration_cycle(3000.0, 3000.0, 298.15, m_dot_cool, 308.15)

        self.assertIn("Q_hx_potential", result)
        self.assertIn("Q_ref_max", result)
        self.assertAlmostEqual(result["Q_evap"], min(result["Q_hx_potential"], result["Q_ref_max"]), places=6)

    def test_cycle_is_off_at_off_command_and_active_from_1000_rpm(self):
        m_dot_cool, _ = pump_model(1600.0)

        off = run_refrigeration_cycle(
            300.0,
            0.0,
            288.15,
            m_dot_cool,
            308.15,
        )
        active = []
        for speed in (1000.0, 1500.0, 1999.0, 2000.0):
            clear_refrigeration_cycle_cache()
            active.append(run_refrigeration_cycle(
                speed,
                staged_fan_speed(speed),
                288.15,
                m_dot_cool,
                308.15,
            ))

        self.assertEqual(off["Q_evap"], 0.0)
        self.assertEqual(off["W_comp"], 0.0)
        self.assertTrue(all(point["Q_evap"] > 0.0 for point in active))
        self.assertTrue(all(point["W_comp"] > 0.0 for point in active))
        self.assertTrue(all(point["compressor_low_speed_extrapolated"] for point in active[:-1]))
        self.assertFalse(active[-1]["compressor_low_speed_extrapolated"])

    def test_low_speed_efficiency_extrapolation_is_bounded_and_continuous(self):
        low = compressor_efficiencies(1000.0, 6.4)
        edge = compressor_efficiencies(2000.0, 6.4)

        self.assertAlmostEqual(low["eta_vol"], 0.65)
        self.assertAlmostEqual(low["eta_is"], 0.41)
        self.assertAlmostEqual(edge["eta_vol"], 0.76)
        self.assertAlmostEqual(edge["eta_is"], 0.50)
        self.assertTrue(low["low_speed_extrapolated"])
        self.assertFalse(edge["low_speed_extrapolated"])

    def test_cycle_is_continuous_at_original_2000_rpm_map_boundary(self):
        m_dot_cool, _ = pump_model(3000.0)
        clear_refrigeration_cycle_cache()
        below = run_refrigeration_cycle(
            1999.0, staged_fan_speed(1999.0), 298.15, m_dot_cool, 308.15
        )
        clear_refrigeration_cycle_cache()
        edge = run_refrigeration_cycle(
            2000.0, staged_fan_speed(2000.0), 298.15, m_dot_cool, 308.15
        )

        self.assertLess(abs(below["Q_evap"] - edge["Q_evap"]) / edge["Q_evap"], 0.01)
        self.assertLess(abs(below["W_comp"] - edge["W_comp"]) / edge["W_comp"], 0.01)


if __name__ == "__main__":
    unittest.main()
