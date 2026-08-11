import math
import unittest

import numpy as np
import pandas as pd

from experiments.physics_p.identification.evaluate_mpc_predictors import (
    capacity_metrics,
    error_metrics,
)


class ErrorMetricsTest(unittest.TestCase):
    def test_returns_signed_and_absolute_error_metrics(self):
        result = error_metrics(actual=[0.0, 2.0], predicted=[1.0, 0.0])

        self.assertEqual(result["mae"], 1.5)
        self.assertAlmostEqual(result["rmse"], math.sqrt(2.5))
        self.assertEqual(result["mean_bias"], -0.5)
        self.assertEqual(result["p95_abs"], 1.95)

    def test_rejects_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            error_metrics(actual=[1.0], predicted=[1.0, 2.0])

    def test_rejects_empty_arrays(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            error_metrics(actual=[], predicted=[])

    def test_rejects_nonfinite_values(self):
        for actual, predicted in (
            ([np.nan], [1.0]),
            ([1.0], [np.inf]),
        ):
            with self.subTest(actual=actual, predicted=predicted):
                with self.assertRaisesRegex(ValueError, "finite"):
                    error_metrics(actual=actual, predicted=predicted)

    def test_rejects_misaligned_series_indexes(self):
        actual = pd.Series([1.0, 2.0], index=["first", "second"])
        predicted = pd.Series([1.0, 2.0], index=["second", "first"])

        with self.assertRaisesRegex(ValueError, "index"):
            error_metrics(actual=actual, predicted=predicted)

    def test_rejects_series_index_metadata_mismatch(self):
        actual = pd.Series([1.0], index=pd.Index([0], name="actual_row"))
        predicted = pd.Series([1.0], index=pd.Index([0], name="predicted_row"))

        with self.assertRaisesRegex(ValueError, "index"):
            error_metrics(actual=actual, predicted=predicted)

    def test_rejects_nonfinite_metrics_from_finite_extreme_inputs(self):
        with self.assertRaisesRegex(ValueError, "finite metrics|overflow"):
            error_metrics(actual=[1e308], predicted=[-1e308])


class CapacityMetricsTest(unittest.TestCase):
    def test_separates_low_speed_and_active_capacity_metrics(self):
        frame = pd.DataFrame(
            {
                "n_comp_cmd_rpm": [1000.0, 2000.0],
                "q_evap_ss_w": [0.0, 1000.0],
                "q_pred_w": [20.0, 900.0],
            }
        )

        result = capacity_metrics(frame)

        self.assertEqual(result["low_speed_mae_w"], 20.0)
        self.assertEqual(result["active_mape_percent"], 10.0)

    def test_rejects_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "required columns"):
            capacity_metrics(pd.DataFrame({"n_comp_cmd_rpm": [1000.0]}))

    def test_rejects_nonfinite_values(self):
        frame = pd.DataFrame(
            {
                "n_comp_cmd_rpm": [1000.0, 2000.0],
                "q_evap_ss_w": [0.0, np.nan],
                "q_pred_w": [20.0, 900.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            capacity_metrics(frame)

    def test_requires_both_speed_regions(self):
        for speeds, region in (([1000.0], "active"), ([2000.0], "low-speed")):
            with self.subTest(speeds=speeds):
                frame = pd.DataFrame(
                    {
                        "n_comp_cmd_rpm": speeds,
                        "q_evap_ss_w": [100.0],
                        "q_pred_w": [90.0],
                    }
                )
                with self.assertRaisesRegex(ValueError, region):
                    capacity_metrics(frame)

    def test_rejects_zero_actual_capacity_in_active_region(self):
        for predicted in (0.0, 100.0):
            with self.subTest(predicted=predicted):
                frame = pd.DataFrame(
                    {
                        "n_comp_cmd_rpm": [1000.0, 2000.0],
                        "q_evap_ss_w": [0.0, 0.0],
                        "q_pred_w": [20.0, predicted],
                    },
                    index=["low", "active-zero"],
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "active.*nonzero.*active-zero|active-zero.*active.*nonzero",
                ):
                    capacity_metrics(frame)


if __name__ == "__main__":
    unittest.main()
