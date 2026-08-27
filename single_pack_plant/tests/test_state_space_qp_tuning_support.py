from __future__ import annotations

import inspect
import unittest

import pandas as pd

from single_pack_plant.runners import short_validate_fixed_qp as validation


class StateSpaceQPTuningSupportTests(unittest.TestCase):
    def test_short_validation_accepts_named_weight_case(self):
        parameters = inspect.signature(validation.run_short_validation).parameters
        self.assertIn("weights", parameters)
        self.assertIn("case_label", parameters)

    def test_summary_reports_saturation_and_pump_variation(self):
        self.assertTrue(hasattr(validation, "summarize_validation_dataframe"))
        data = pd.DataFrame(
            {
                "Average temperature": [25.0, 25.1, 25.2, 25.3],
                "T_cell_max_C": [25.0, 25.1, 25.2, 25.3],
                "Delta_T_cell_C": [0.0, 0.1, 0.1, 0.2],
                "Coolant temperature": [35.0, 34.0, 33.0, 32.0],
                "Compressor command": [6000.0, 5900.0, 5700.0, 5800.0],
                "Compressor Speed": [5800.0, 5850.0, 5750.0, 5800.0],
                "Pump command": [1600.0, 1900.0, 1600.0, 1900.0],
                "Pump Speed (RPM)": [1700.0, 1800.0, 1750.0, 1775.0],
                "MPC_Solved": [True, True, True, False],
                "MPC solve time": [0.1, 0.2, 0.3, 0.4],
                "Cumulative energy consumption": [0.01, 0.02, 0.03, 0.04],
            }
        )
        summary = validation.summarize_validation_dataframe(data)
        self.assertEqual(summary["compressor_above_5800_ratio"], 0.5)
        self.assertEqual(summary["pump_total_variation_rpm"], 900.0)
        self.assertEqual(summary["pump_direction_changes"], 2)
        self.assertIn("actual_pump_total_variation_rpm", summary)
        self.assertEqual(summary["actual_pump_total_variation_rpm"], 175.0)
        self.assertEqual(summary["actual_pump_direction_changes"], 2)
        self.assertEqual(summary["fallback_count"], 1)


if __name__ == "__main__":
    unittest.main()
