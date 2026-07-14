import unittest

from predictive_delta_t_flow_mpc import PredictiveDeltaTGate, PredictiveDeltaTReversalLatch


class PredictiveDeltaTReversalLatchTests(unittest.TestCase):
    def test_latch_prevents_repeated_reversal_until_predicted_crossing_disappears(self):
        gate = PredictiveDeltaTGate(threshold_c=0.50, buffer_s=50.0, min_hold_s=200.0)
        latch = PredictiveDeltaTReversalLatch()
        crossing = gate.evaluate(
            [0.30, 0.51], dt_s=25.0, current_time_s=500.0, last_switch_time_s=0.0
        )
        no_crossing = gate.evaluate(
            [0.30, 0.40], dt_s=25.0, current_time_s=800.0, last_switch_time_s=500.0
        )

        self.assertTrue(latch.update(crossing, crossing_exists=True))
        self.assertFalse(latch.update(crossing, crossing_exists=True))
        self.assertFalse(latch.update(no_crossing, crossing_exists=False))
        self.assertTrue(latch.update(crossing, crossing_exists=True))


if __name__ == "__main__":
    unittest.main()
