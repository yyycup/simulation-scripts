import unittest

import numpy as np

from mpc_flow_direction_strategies import (
    COMPRESSOR_POWER_DOMAIN,
    compressor_power_normalization_w,
    compressor_power_value,
)


class MpcCompressorPowerModelTest(unittest.TestCase):
    def test_power_is_positive_and_monotonic_over_the_mpc_domain(self):
        speeds = np.linspace(*COMPRESSOR_POWER_DOMAIN["n_comp_rpm"], 101)
        for coolant in COMPRESSOR_POWER_DOMAIN["t_cool_c"]:
            for ambient in COMPRESSOR_POWER_DOMAIN["t_ambient_c"]:
                power = np.array(
                    [
                        compressor_power_value(speed, coolant, ambient)
                        for speed in speeds
                    ]
                )
                self.assertTrue(np.all(power > 0.0))
                self.assertTrue(np.all(np.diff(power) >= 0.0))

    def test_zero_speed_has_zero_power(self):
        self.assertEqual(compressor_power_value(0.0, 25.0, 35.0), 0.0)

    def test_normalization_is_the_maximum_domain_corner(self):
        values = [
            compressor_power_value(speed, coolant, ambient)
            for speed in COMPRESSOR_POWER_DOMAIN["n_comp_rpm"]
            for coolant in COMPRESSOR_POWER_DOMAIN["t_cool_c"]
            for ambient in COMPRESSOR_POWER_DOMAIN["t_ambient_c"]
        ]
        self.assertAlmostEqual(compressor_power_normalization_w(), max(values))


if __name__ == "__main__":
    unittest.main()
