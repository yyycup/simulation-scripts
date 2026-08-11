import unittest

import pandas as pd

from mpc_flow_direction_strategies import runtime_mpc_params_for_scene
from run_p_mpc_controller_tuning import _candidate_overrides, _select_candidates


class PhysicsPMpcControllerTuningTests(unittest.TestCase):
    def test_dmax_stage_uses_absolute_p_only_candidate(self):
        base = runtime_mpc_params_for_scene("peak")

        overrides = _candidate_overrides(
            stage="dmax",
            base_params=base,
            factor=1000.0,
            selected_comp_weight=9000.0,
        )

        self.assertEqual(overrides["dmax_comp"], 1000.0)
        self.assertEqual(overrides["w_energy_comp"], 9000.0)
        self.assertEqual(overrides["w_terminal_temp"], base.w_terminal_temp)

    def test_selection_marks_domain_violating_fallback_unqualified(self):
        common = {
            "scene": "peak",
            "solve_success_rate": 1.0,
            "pred_terminal_mean_c": 25.0,
            "temperature_mae_c": 0.02,
            "compressor_saturation_rate": 0.0,
            "energy_kwh": 0.001,
            "solve_time_mean_s": 1.0,
        }
        summary = pd.DataFrame(
            [
                {
                    **common,
                    "factor": 6000.0,
                    "is_baseline": True,
                    "pred_coolant_domain_under_max_c": 4.0,
                },
                {
                    **common,
                    "factor": 500.0,
                    "is_baseline": False,
                    "pred_coolant_domain_under_max_c": 3.0,
                },
            ]
        )

        selected = _select_candidates(summary).iloc[0]

        self.assertFalse(bool(selected["selection_qualified"]))


if __name__ == "__main__":
    unittest.main()
