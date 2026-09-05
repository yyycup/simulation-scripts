"""Tests for the Stage 7A cluster validation workflow."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cluster_plant_v2.validation.validate_cluster import (
    CaseSpec,
    build_case_specs,
    compare_one_step,
    run_case,
)


class ClusterValidationTests(unittest.TestCase):
    def test_formal_matrix_contains_only_the_five_stage_7a_cases(self) -> None:
        cases = build_case_specs()

        self.assertEqual(len(cases), 5)
        self.assertEqual(
            [(case.current_a, case.direction, case.resistance_factors) for case in cases],
            [
                (560.0, "forward", (1.0, 1.0, 1.0, 1.0, 1.0)),
                (280.0, "forward", (1.0, 1.0, 1.0, 1.0, 1.0)),
                (1120.0, "forward", (1.0, 1.0, 1.0, 1.0, 1.0)),
                (560.0, "reverse", (1.0, 1.0, 1.0, 1.0, 1.0)),
                (560.0, "forward", (1.0, 1.05, 0.95, 1.10, 0.90)),
            ],
        )

    def test_one_step_gate_matches_stage_6_pack_at_machine_precision(self) -> None:
        gate = compare_one_step()

        self.assertEqual(gate["mass_flow_conservation_residual_kg_s"], 0.0)
        self.assertEqual(gate["inter_pack_delta_temperature_k"], 0.0)
        self.assertEqual(gate["max_symmetric_pack_difference"], 0.0)
        self.assertEqual(gate["max_stage6_pack_regression_difference"], 0.0)
        self.assertEqual(gate["return_mixing_residual_k"], 0.0)
        self.assertTrue(gate["all_states_finite"])

    def test_short_nonuniform_case_conserves_flow_and_recomputes_return(self) -> None:
        case = CaseSpec(
            "test_nonuniform",
            560.0,
            "forward",
            (1.0, 1.05, 0.95, 1.10, 0.90),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_case(
                case,
                duration_s=10.0,
                dt_s=5.0,
                output_dir=Path(directory),
            )

        self.assertEqual(len(result["timeseries"]), 2)
        self.assertLessEqual(
            abs(result["summary"]["max_abs_mass_flow_residual_kg_s"]), 1e-15
        )
        self.assertLessEqual(
            abs(result["summary"]["max_abs_return_mixing_residual_k"]), 1e-12
        )
        flows = np.array([row["mean_mass_flow_kg_s"] for row in result["hydraulics"]])
        factors = np.array(case.resistance_factors)
        self.assertLess(flows[np.argmax(factors)], flows[np.argmin(factors)])
        self.assertTrue(result["summary"]["all_states_finite"])

    def test_six_hundred_seconds_remain_finite_without_symmetry_drift(self) -> None:
        case = build_case_specs()[0]
        with tempfile.TemporaryDirectory() as directory:
            result = run_case(
                case,
                duration_s=600.0,
                dt_s=5.0,
                output_dir=Path(directory),
            )

        self.assertEqual(len(result["timeseries"]), 120)
        self.assertTrue(result["summary"]["all_states_finite"])
        self.assertEqual(result["summary"]["max_inter_pack_delta_temperature_k"], 0.0)
        self.assertEqual(result["summary"]["max_symmetric_pack_difference"], 0.0)
        self.assertEqual(
            result["summary"]["max_stage6_pack_regression_difference"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
