"""Tests for Stage 7B independent and integrated validation helpers."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from cluster_plant_v2.hydraulics import DESIGN_PACK_MASS_FLOW_KG_S
from cluster_plant_v2.validation.validate_hydraulic_network import (
    build_cluster_case_specs,
    build_network_case_specs,
    run_cluster_case,
    run_network_case,
)


TOTAL_MASS_FLOW_KG_S = 5.0 * DESIGN_PACK_MASS_FLOW_KG_S


class HydraulicNetworkValidationTests(unittest.TestCase):
    def test_independent_matrix_covers_two_degeneracies_and_two_header_levels(self) -> None:
        cases = build_network_case_specs()

        self.assertEqual(
            [(case.case_id, case.header_ratio, case.branch_factors) for case in cases],
            [
                ("H0_zero_header_equal", 0.0, (1.0, 1.0, 1.0, 1.0, 1.0)),
                ("H1_zero_header_branch_nonuniform", 0.0, (1.0, 1.05, 0.95, 1.10, 0.90)),
                ("H2_small_header_equal", 0.001, (1.0, 1.0, 1.0, 1.0, 1.0)),
                ("H3_medium_header_equal", 0.005, (1.0, 1.0, 1.0, 1.0, 1.0)),
            ],
        )

    def test_uniform_zero_header_is_exact_stage_7a_degeneracy(self) -> None:
        result = run_network_case(build_network_case_specs()[0], TOTAL_MASS_FLOW_KG_S)

        self.assertLess(result["summary"]["stage7a_max_flow_difference_kg_s"], 1e-13)
        self.assertLess(result["summary"]["max_node_mass_residual_kg_s"], 1e-13)
        self.assertLess(result["summary"]["max_path_pressure_residual_pa"], 1e-6)

    def test_nonuniform_zero_header_recovers_stage_7a_analytic_flows(self) -> None:
        result = run_network_case(build_network_case_specs()[1], TOTAL_MASS_FLOW_KG_S)

        self.assertLess(result["summary"]["stage7a_max_flow_difference_kg_s"], 1e-10)
        self.assertLess(result["summary"]["max_path_pressure_residual_pa"], 1e-5)

    def test_nonzero_header_case_exposes_all_pack_and_segment_diagnostics(self) -> None:
        result = run_network_case(build_network_case_specs()[2], TOTAL_MASS_FLOW_KG_S)

        self.assertEqual(len(result["packs"]), 5)
        self.assertEqual(len(result["segments"]), 10)
        self.assertGreater(result["summary"]["flow_cv"], 0.0)
        self.assertGreater(result["summary"]["flow_nonuniformity"], 0.0)
        self.assertTrue(result["summary"]["all_gates_pass"])
        flows = np.array([row["mass_flow_kg_s"] for row in result["packs"]])
        self.assertAlmostEqual(float(flows.sum()), TOTAL_MASS_FLOW_KG_S, places=12)

    def test_cluster_matrix_contains_only_b0_through_b4(self) -> None:
        cases = build_cluster_case_specs()

        self.assertEqual(
            [(case.case_id, case.header_ratio, case.direction, case.branch_factors) for case in cases],
            [
                ("B0_zero_header_equal_forward", 0.0, "forward", (1.0, 1.0, 1.0, 1.0, 1.0)),
                ("B1_zero_header_branch_nonuniform_forward", 0.0, "forward", (1.0, 1.05, 0.95, 1.10, 0.90)),
                ("B2_small_header_equal_forward", 0.001, "forward", (1.0, 1.0, 1.0, 1.0, 1.0)),
                ("B3_medium_header_equal_forward", 0.005, "forward", (1.0, 1.0, 1.0, 1.0, 1.0)),
                ("B4_medium_header_equal_reverse", 0.005, "reverse", (1.0, 1.0, 1.0, 1.0, 1.0)),
            ],
        )

    def test_b0_and_b1_short_runs_recover_stage_7a_cluster(self) -> None:
        for case in build_cluster_case_specs()[:2]:
            with self.subTest(case=case.case_id), tempfile.TemporaryDirectory() as directory:
                result = run_cluster_case(
                    case,
                    duration_s=10.0,
                    dt_s=5.0,
                    output_dir=Path(directory),
                )
                self.assertLess(
                    result["summary"]["max_stage7a_regression_difference"], 1e-9
                )
                self.assertTrue(result["summary"]["all_gates_pass"])

    def test_medium_header_six_hundred_seconds_are_finite_and_solved(self) -> None:
        case = build_cluster_case_specs()[3]
        with tempfile.TemporaryDirectory() as directory:
            result = run_cluster_case(
                case,
                duration_s=600.0,
                dt_s=5.0,
                output_dir=Path(directory),
            )

        self.assertEqual(len(result["timeseries"]), 120)
        self.assertTrue(result["summary"]["all_states_finite"])
        self.assertEqual(result["summary"]["solver_failure_count"], 0)
        self.assertGreater(result["summary"]["flow_cv"], 0.0)
        self.assertTrue(result["summary"]["all_gates_pass"])


if __name__ == "__main__":
    unittest.main()
