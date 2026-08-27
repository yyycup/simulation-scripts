import unittest
import sys

from single_pack_plant.controllers.dompc.config import PhysicsPDoMPCConfig, PhysicsPDoMPCWeights


class PhysicsPDoMPCWeightTests(unittest.TestCase):
    def test_temperature_error_scale_defaults_to_one_degree(self):
        """The explicit normalization must preserve the frozen objective."""
        self.assertEqual(PhysicsPDoMPCWeights().temperature_error_scale_c, 1.0)

    def test_temperature_error_scale_must_be_positive(self):
        self.assertLess(0.0, PhysicsPDoMPCWeights(temperature_error_scale_c=0.5).temperature_error_scale_c)

    def test_peak_settings_do_not_depend_on_gekko(self):
        config = PhysicsPDoMPCConfig.for_scene("peak")
        self.assertEqual(config.horizon, 60)
        self.assertEqual(config.dmax_pump_rpm, 300.0)
        self.assertNotIn("single_pack_plant.controllers.gekko.mpc", sys.modules)

    def test_control_interval_is_configured_independently_of_horizon(self):
        config = PhysicsPDoMPCConfig.for_scene("peak")
        self.assertEqual(config.control_interval_steps, 1)

    def test_uniform_move_blocking_counts_control_moves_and_physical_steps(self):
        config = PhysicsPDoMPCConfig.for_scene("peak")
        config = config.__class__(**{**config.__dict__, "control_horizon": 12}).validated()
        self.assertEqual(config.control_moves, 12)
        self.assertEqual(config.terminal_hold_steps, 48)

    def test_control_horizon_holds_final_move_to_prediction_end(self):
        config = PhysicsPDoMPCConfig.for_scene("peak")
        config = config.__class__(**{**config.__dict__, "control_horizon": 50}).validated()
        self.assertEqual(config.control_moves, 50)
        self.assertEqual(config.terminal_hold_steps, 10)


if __name__ == "__main__":
    unittest.main()
