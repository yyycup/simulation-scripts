import unittest

from predictive_delta_t_flow_mpc import PredictiveDeltaTGate


class PredictiveDeltaTGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = PredictiveDeltaTGate(
            threshold_c=0.50,
            buffer_s=50.0,
            min_hold_s=200.0,
        )

    def test_triggers_when_predicted_crossing_is_inside_window(self):
        decision = self.gate.evaluate(
            delta_t_series_c=[0.30, 0.42, 0.51, 0.60],
            dt_s=25.0,
            current_time_s=500.0,
            last_switch_time_s=0.0,
        )

        self.assertTrue(decision.trigger)
        self.assertEqual(decision.crossing_time_s, 50.0)

    def test_does_not_trigger_when_crossing_is_after_window(self):
        decision = self.gate.evaluate(
            delta_t_series_c=[0.30, 0.42, 0.48, 0.51],
            dt_s=25.0,
            current_time_s=500.0,
            last_switch_time_s=0.0,
        )

        self.assertFalse(decision.trigger)
        self.assertEqual(decision.crossing_time_s, 75.0)

    def test_does_not_trigger_during_minimum_hold_time(self):
        decision = self.gate.evaluate(
            delta_t_series_c=[0.30, 0.51],
            dt_s=25.0,
            current_time_s=150.0,
            last_switch_time_s=0.0,
        )

        self.assertFalse(decision.trigger)
        self.assertFalse(decision.hold_time_satisfied)

    def test_zero_window_only_accepts_current_predicted_temperature_spread(self):
        gate = PredictiveDeltaTGate(threshold_c=0.50, buffer_s=0.0, min_hold_s=0.0)

        later_crossing = gate.evaluate([0.40, 0.60], dt_s=25.0, current_time_s=0.0, last_switch_time_s=-1.0)
        current_crossing = gate.evaluate([0.60, 0.40], dt_s=25.0, current_time_s=0.0, last_switch_time_s=-1.0)

        self.assertFalse(later_crossing.trigger)
        self.assertTrue(current_crossing.trigger)


if __name__ == "__main__":
    unittest.main()
