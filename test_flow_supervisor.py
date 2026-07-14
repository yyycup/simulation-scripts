import unittest

import numpy as np

from thermal_loop import SupervisoryFlowController


class SupervisoryFlowControllerTest(unittest.TestCase):
    def test_supervisor_switches_on_comprehensive_reversal_index(self):
        supervisor = SupervisoryFlowController(dt=5.0, min_hold_time=0.0)
        grid_c = np.full((4, 13), 25.0)
        grid_c[:, 8:] = 26.4

        switched = supervisor.update(
            temps_c=grid_c,
            current_time=5.0,
            flow_enabled=True,
        )

        self.assertTrue(switched)
        self.assertTrue(supervisor.is_reversed)
        self.assertEqual(supervisor.last_flow_info["flow_direction_d"], -1)
        self.assertGreater(supervisor.last_flow_info["j_rev_pred_max"], supervisor.j_rev_on)

    def test_supervisor_keeps_forward_when_flow_is_disabled(self):
        supervisor = SupervisoryFlowController(dt=5.0, min_hold_time=0.0)
        grid_c = np.full((4, 13), 25.0)
        grid_c[:, 8:] = 26.0

        switched = supervisor.update(
            temps_c=grid_c,
            current_time=5.0,
            flow_enabled=False,
        )

        self.assertFalse(switched)
        self.assertFalse(supervisor.is_reversed)
        self.assertEqual(supervisor.last_flow_info["flow_direction_d"], 1)


if __name__ == "__main__":
    unittest.main()
