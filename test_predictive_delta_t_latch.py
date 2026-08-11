import unittest
from types import SimpleNamespace

from predictive_delta_t_flow_controller import SinglePredictiveDeltaTMPC


class PredictiveDeltaTReversalLatchTests(unittest.TestCase):
    def test_controller_prevents_repeat_switch_until_crossing_disappears(self):
        """Exercise the latch state that now lives in the real controller."""
        controller = SinglePredictiveDeltaTMPC.__new__(SinglePredictiveDeltaTMPC)
        controller.direction = 1
        controller.switch_armed = True
        controller.last_switch_time = -1e12
        controller.last_flow_info = {}
        controller.threshold_c = 0.50
        controller.buffer_s = 50.0
        controller.min_hold_s = 200.0

        controller._solve_for_direction = lambda *args, **kwargs: {
            "n_comp": 1000.0,
            "n_pump": 2000.0,
        }

        def flow_metrics(delta_t, *, crossing_exists, trigger):
            return {
                "delta_t_pred_max": delta_t,
                "delta_t_pred_buffer_max": delta_t,
                "delta_t_pred_cross_s": 25.0 if crossing_exists else float("nan"),
                "predictive_switch_gate": trigger,
                "hold_time_satisfied": True,
                "crossing_exists": crossing_exists,
            }

        metrics = iter(
            [
                # First command: switch, then evaluate the new direction.
                flow_metrics(0.51, crossing_exists=True, trigger=True),
                flow_metrics(0.52, crossing_exists=True, trigger=False),
                # Second command: crossing persists, so no repeated switch.
                flow_metrics(0.53, crossing_exists=True, trigger=True),
                # Third command: crossing clears and rearms the controller.
                flow_metrics(0.40, crossing_exists=False, trigger=False),
                # Fourth command: a new crossing can switch again.
                flow_metrics(0.51, crossing_exists=True, trigger=True),
                flow_metrics(0.49, crossing_exists=False, trigger=False),
            ]
        )
        controller._flow_metrics = lambda *args, **kwargs: next(metrics)

        def remember_flow_info(_solution, _delta_t, switched=False, extra_info=None):
            controller.last_flow_info = {
                "switched": bool(switched),
                **(extra_info or {}),
            }

        controller._set_flow_info = remember_flow_info
        controller._remember_control = lambda _solution: None
        pack = SimpleNamespace()

        controller.command(0, pack, 298.15, 298.15, current_time=500.0)
        self.assertEqual(controller.direction, -1)
        self.assertFalse(controller.switch_armed)

        controller.command(1, pack, 298.15, 298.15, current_time=600.0)
        self.assertEqual(controller.direction, -1)
        self.assertFalse(controller.switch_armed)

        controller.command(2, pack, 298.15, 298.15, current_time=800.0)
        self.assertEqual(controller.direction, -1)
        self.assertTrue(controller.switch_armed)

        controller.command(3, pack, 298.15, 298.15, current_time=900.0)
        self.assertEqual(controller.direction, 1)
        self.assertFalse(controller.switch_armed)


if __name__ == "__main__":
    unittest.main()
