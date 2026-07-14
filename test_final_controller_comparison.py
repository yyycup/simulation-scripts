import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from run_final_controller_comparison import (
    build_run_matrix,
    load_pid_selection,
    summarize_dataframe,
)


class FinalControllerComparisonTest(unittest.TestCase):
    def test_matrix_has_single_and_double_flow_for_every_controller(self):
        matrix = build_run_matrix()

        self.assertEqual(len(matrix), 12)
        self.assertEqual(sum(case["flow_key"] == "single" for case in matrix), 6)
        double_cases = [case for case in matrix if case["flow_key"] == "double"]
        self.assertEqual(len(double_cases), 6)
        self.assertEqual({case["control"] for case in double_cases}, {"on-off", "pid", "mpc"})
        self.assertEqual({case["scene_key"] for case in double_cases}, {"peak", "freq"})

    def test_load_pid_selection_returns_numeric_tuples(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "pid.json"
            path.write_text(json.dumps({"peak": [1.2, 0.01, 0.1], "freq": [0.2, 0, 0]}), encoding="utf-8")
            selection = load_pid_selection(path)

        self.assertEqual(selection["peak"], (1.2, 0.01, 0.1))
        self.assertEqual(selection["freq"], (0.2, 0.0, 0.0))

    def test_summary_reports_temperature_energy_and_reversal_metrics(self):
        df = pd.DataFrame(
            {
                "Time": [0.0, 5.0, 10.0],
                "Average temperature": [25.0, 25.2, 24.8],
                "Maximum temperature difference": [0.1, 0.3, 0.2],
                "Cumulative energy consumption": [0.0, 0.01, 0.02],
                "Total power": [2.0, 2.2, 2.1],
                "Flow direction d": [1, -1, -1],
            }
        )

        summary = summarize_dataframe(df, target_temp_c=25.0)

        self.assertAlmostEqual(summary["temperature_MAE_C"], 0.1333333333)
        self.assertAlmostEqual(summary["energy_kWh"], 0.02)
        self.assertAlmostEqual(summary["max_delta_T_C"], 0.3)
        self.assertEqual(summary["reversal_count"], 1)


if __name__ == "__main__":
    unittest.main()
