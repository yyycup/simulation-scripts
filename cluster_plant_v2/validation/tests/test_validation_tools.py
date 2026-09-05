"""Contract tests for pack comparison case construction and metrics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_plant_v2.validation.analyze_comparison import (
    branch_current_nrmse,
    electrical_diagnostics,
    error_metrics,
)
from cluster_plant_v2.validation.compare_legacy_reference import (
    build_case_specs,
    load_regd_profile,
)


class ValidationToolsTests(unittest.TestCase):
    def test_case_matrix_contains_five_currents_at_two_boundaries(self) -> None:
        cases = build_case_specs()

        self.assertEqual(len(cases), 10)
        self.assertEqual(len({case.case_id for case in cases}), 10)
        self.assertEqual(
            {(case.plate_temp_c, case.ambient_temp_c) for case in cases},
            {(25.0, 25.0), (20.0, 35.0)},
        )
        self.assertEqual(sum(case.current_kind == "regd" for case in cases), 2)

    def test_regd_loader_interpolates_existing_signal_in_amperes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "regd.csv"
            pd.DataFrame(
                {"Seconds": [0.0, 2.0, 4.0, 6.0], "RegD": [0.0, 0.5, -0.5, 1.0]}
            ).to_csv(path, index=False)

            times, currents = load_regd_profile(path, duration_s=10.0, dt=5.0)

        np.testing.assert_array_equal(times, [0.0, 5.0])
        np.testing.assert_allclose(currents, [0.0, 280.0])

    def test_error_metrics_use_entire_time_series(self) -> None:
        metrics = error_metrics(
            legacy=np.array([1.0, 2.0, 3.0]),
            reference=np.array([2.0, 2.0, 5.0]),
        )

        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(5.0 / 3.0))
        self.assertEqual(metrics["max_abs_error"], 2.0)
        self.assertEqual(metrics["final_error"], 2.0)

    def test_branch_nrmse_uses_legacy_rms_scale(self) -> None:
        legacy = np.array([[10.0, -10.0], [10.0, -10.0]])
        reference = np.array([[11.0, -9.0], [9.0, -11.0]])

        self.assertAlmostEqual(branch_current_nrmse(legacy, reference), 0.1)
        self.assertEqual(
            branch_current_nrmse(np.zeros((2, 4)), np.zeros((2, 4))), 0.0
        )

    def test_electrical_diagnostics_integrate_heat_and_preserve_voltage_spread(self) -> None:
        frame = pd.DataFrame(
            {
                "legacy_Q_gen_total_W": [10.0, 10.0],
                "reference_Q_gen_total_W": [12.0, 12.0],
                **{
                    f"legacy_branch_voltage_{index}_V": [40.0, 40.0]
                    for index in range(1, 5)
                },
                **{
                    f"reference_branch_voltage_{index}_V": [40.0 + index, 40.0 + index]
                    for index in range(1, 5)
                },
            }
        )

        result = electrical_diagnostics(frame, dt_s=5.0)

        self.assertEqual(result["qgen_integral_legacy_J"], 100.0)
        self.assertEqual(result["qgen_integral_reference_J"], 120.0)
        self.assertAlmostEqual(result["qgen_integral_difference_pct"], 20.0)
        self.assertEqual(result["reference_branch_voltage_spread_max_V"], 3.0)


if __name__ == "__main__":
    unittest.main()
