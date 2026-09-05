"""Contract tests for the long-horizon Reference-to-ROM validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cluster_plant_v2.validation.validate_pack_rom_long_horizon import (
    CaseSpec,
    aggregate_reference_zones,
    build_case_specs,
    build_plate_boundaries,
    error_metrics,
    normalized_rmse,
    run_case,
    weighted_zone_average,
)


class PackRomLongHorizonTests(unittest.TestCase):
    def test_case_matrix_has_four_currents_at_three_boundaries(self) -> None:
        cases = build_case_specs()

        self.assertEqual(len(cases), 12)
        self.assertEqual(len({case.case_id for case in cases}), 12)
        self.assertEqual(
            {case.boundary_kind for case in cases},
            {"uniform", "forward", "reverse"},
        )
        self.assertEqual(sum(case.current_kind == "regd" for case in cases), 3)

    def test_gradient_boundaries_use_exact_reference_zone_averages(self) -> None:
        for boundary_kind, endpoints in (
            ("forward", (22.0, 25.0)),
            ("reverse", (25.0, 22.0)),
        ):
            with self.subTest(boundary=boundary_kind):
                reference, rom = build_plate_boundaries(boundary_kind)
                expected_c = np.linspace(*endpoints, 13)

                np.testing.assert_allclose(reference - 273.15, expected_c)
                np.testing.assert_allclose(
                    rom - 273.15,
                    [expected_c[0:4].mean(), expected_c[4:9].mean(), expected_c[9:13].mean()],
                )

    def test_reference_temperatures_aggregate_to_four_by_three(self) -> None:
        cells = np.arange(52.0).reshape(4, 13)

        zones = aggregate_reference_zones(cells)

        self.assertEqual(zones.shape, (4, 3))
        np.testing.assert_allclose(zones[0], [1.5, 6.0, 10.5])
        np.testing.assert_allclose(zones[3], [40.5, 45.0, 49.5])

    def test_weighted_average_uses_four_five_four_zone_counts(self) -> None:
        zones = np.tile([10.0, 20.0, 30.0], (4, 1))

        self.assertAlmostEqual(weighted_zone_average(zones), 20.0)

    def test_metrics_cover_full_series_and_handle_zero_nrmse_scale(self) -> None:
        metrics = error_metrics(
            np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 5.0])
        )

        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(5.0 / 3.0))
        self.assertEqual(metrics["max_abs_error"], 2.0)
        self.assertEqual(metrics["final_error"], 2.0)
        self.assertEqual(normalized_rmse(np.zeros(3), np.zeros(3)), 0.0)

    def test_one_step_case_writes_required_timeseries_contract(self) -> None:
        case = CaseSpec("280a_uniform", "constant", 280.0, "uniform")
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            result = run_case(
                case,
                duration_s=5.0,
                dt=5.0,
                regd_file=Path("unused.csv"),
                output_dir=output_dir,
                make_plot=False,
            )
            columns = set(result["timeseries"].columns)

            self.assertTrue((output_dir / "case_280a_uniform_timeseries.csv").exists())
            self.assertEqual(len(result["timeseries"]), 1)
            self.assertTrue(
                {
                    "Ref_Tavg_C",
                    "ROM_Tavg_C",
                    "Ref_zone_Tmax_C",
                    "ROM_Tmax_C",
                    "Ref_cell_Tmax_C",
                    "hotspot_loss_C",
                    "Ref_Qgen_total_W",
                    "ROM_Qgen_total_W",
                    "Ref_T_b3_z2_C",
                    "ROM_T_b3_z2_C",
                    "Ref_SOC_branch_4",
                    "ROM_SOC_branch_4",
                    "Ref_Ibranch_4_A",
                    "ROM_Ibranch_4_A",
                }.issubset(columns)
            )
            self.assertEqual(len(result["zone_metrics"]), 12)


if __name__ == "__main__":
    unittest.main()
