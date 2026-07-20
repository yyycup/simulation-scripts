import unittest

import pandas as pd

from evaluate_dual_p_shadow import evaluate_dual_shadow


class DualShadowEvaluationTests(unittest.TestCase):
    def test_predictions_align_with_post_step_target_at_horizon_minus_one_row(self):
        frame = pd.DataFrame(
            {
                "Time": [0.0, 5.0, 10.0],
                "Average temperature": [25.0, 26.0, 27.0],
                "Coolant temperature": [20.0, 21.0, 22.0],
                "P_Shadow_T_Plate_Actual_C": [23.0, 24.0, 25.0],
                "Supply pipe coolant temperature": [18.0, 19.0, 20.0],
                "Return pipe coolant temperature": [22.0, 23.0, 24.0],
                "Evaporator cooling rate (kW)": [1.0, 2.0, 3.0],
                "P_Shadow_Old_Command_Assumption": ["provided_plan_hold_last"] * 3,
                "P_Shadow_New_Command_Assumption": ["provided_plan_hold_last"] * 3,
            }
        )
        actual_columns = {
            "T_Batt": "Average temperature",
            "T_Cool": "Coolant temperature",
            "T_Plate": "P_Shadow_T_Plate_Actual_C",
            "T_Supply": "Supply pipe coolant temperature",
            "T_Return": "Return pipe coolant temperature",
            "Q_Evap": "Evaporator cooling rate (kW)",
        }
        for model in ("Old", "New"):
            for state, actual_column in actual_columns.items():
                suffix = "W" if state == "Q_Evap" else "C"
                actual = frame[actual_column] * (1000.0 if state == "Q_Evap" else 1.0)
                frame[f"P_Shadow_{model}_{state}_Pred_10s_{suffix}"] = actual.shift(-1)

        details, summary = evaluate_dual_shadow(
            frame,
            model_labels=("Old", "New"),
            dt_s=5.0,
            horizons_s=(10.0,),
        )

        self.assertEqual(len(details), 2 * 2 * 6)
        self.assertTrue((details["target_index"] - details["origin_index"] == 1).all())
        self.assertTrue((details["abs_error"] == 0.0).all())
        self.assertTrue((summary["mae"] == 0.0).all())


if __name__ == "__main__":
    unittest.main()
