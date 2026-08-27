from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from single_pack_plant.pid_param_search import (
    DEFAULT_PARAM_RANGES,
    candidate_passes_constraints,
    pid_tracking_metrics,
    selected_pid_cases,
)


class PidPsoSupportTests(unittest.TestCase):
    def test_tracking_score_uses_only_mae_rmse_and_oscillation(self):
        frame = pd.DataFrame(
            {
                "Average temperature": [25.0, 26.0, 25.0],
                "Cumulative energy consumption": [0.0, 0.1, 0.2],
                "Compressor Speed": [1000.0, 2000.0, 1000.0],
            }
        )

        metrics = pid_tracking_metrics(frame, t_ref_c=25.0)

        expected = metrics["MAE"] + 0.5 * metrics["RMSE"] + 2.0 * metrics["OSC"]
        self.assertAlmostEqual(metrics["score"], expected)

    def test_constraints_keep_latest_agreed_temperature_limits(self):
        base = {
            "T_avg_min": 24.1,
            "T_avg_max": 27.9,
            "energy": 1.0,
            "compressor_action_count": 1.0,
        }
        self.assertTrue(candidate_passes_constraints(base, "peak"))
        self.assertFalse(candidate_passes_constraints({**base, "T_avg_min": 23.9}, "peak"))
        self.assertTrue(candidate_passes_constraints({**base, "T_avg_min": 23.1}, "freq"))
        self.assertFalse(candidate_passes_constraints({**base, "T_avg_min": 22.9}, "freq"))
        self.assertFalse(candidate_passes_constraints({**base, "T_avg_max": 28.1}, "freq"))

    def test_each_scene_evaluates_single_and_double_flow(self):
        cases = selected_pid_cases("freq", source_root=Path("/prepared_inputs"))

        self.assertEqual([case.flow for case in cases], ["单向", "双向"])
        self.assertTrue(all(case.scene == "调频" for case in cases))
        self.assertTrue(all(case.source_csv.name == "调频输出单向mpc.csv" for case in cases))
        self.assertEqual(DEFAULT_PARAM_RANGES["kp"], (0.2, 5.0))


if __name__ == "__main__":
    unittest.main()
