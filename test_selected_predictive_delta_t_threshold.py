import unittest

from thermal_batch_config import MPC_PRED_DELTA_T_SWITCH_C


class SelectedPredictiveDeltaTThresholdTest(unittest.TestCase):
    def test_selected_threshold_is_point_45_celsius(self):
        self.assertEqual(MPC_PRED_DELTA_T_SWITCH_C, 0.45)


if __name__ == "__main__":
    unittest.main()
