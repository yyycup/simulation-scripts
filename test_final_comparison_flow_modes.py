import unittest

import predictive_delta_t_flow_controller as predictive_controller
from mpc_flow_direction_strategies import create_mpc_flow_controller
from thermal_case_simulator import flow_reversal_enabled


class FinalComparisonFlowModeTest(unittest.TestCase):
    def test_flow_reversal_enabled_accepts_all_bidirectional_labels(self):
        for label in ("双向", "反向", "double", "bidirectional", "reversed"):
            with self.subTest(label=label):
                self.assertTrue(flow_reversal_enabled(label))

    def test_flow_reversal_enabled_rejects_single_flow_labels(self):
        for label in ("单向", "single", "forward", ""):
            with self.subTest(label=label):
                self.assertFalse(flow_reversal_enabled(label))

    def test_factory_constructs_single_predictive_delta_t_controller(self):
        original = predictive_controller.SinglePredictiveDeltaTMPC

        class FakeSinglePredictiveDeltaTMPC:
            def __init__(self, current_profile, **kwargs):
                self.current_profile = current_profile
                self.kwargs = kwargs

        predictive_controller.SinglePredictiveDeltaTMPC = FakeSinglePredictiveDeltaTMPC
        try:
            controller = create_mpc_flow_controller(
                [560.0, 560.0],
                dt=5.0,
                target_temp_c=25.0,
                mpc_flow_mode="single_predictive_delta_t",
                case_name="调频 双向",
            )
        finally:
            predictive_controller.SinglePredictiveDeltaTMPC = original

        self.assertIsInstance(controller, FakeSinglePredictiveDeltaTMPC)
        self.assertEqual(controller.kwargs["case_name"], "调频 双向")
        self.assertTrue(controller.kwargs["mpc_params"].name.startswith("freq_mpc_params"))
        self.assertTrue(controller.kwargs["mpc_params"].terminal_cost_enabled)
        self.assertEqual(controller.kwargs["mpc_params"].w_terminal_temp, 5e5)


if __name__ == "__main__":
    unittest.main()
