import math
import unittest

import numpy as np

try:
    from cluster_plant_v2.validation.validate_dynamic_refrigerated_cluster_cooling_loop import (
        build_case_specs,
        run_case,
        run_one_step_gate,
    )
except ImportError:
    build_case_specs = None
    run_case = None
    run_one_step_gate = None


class DynamicRefrigeratedClusterCoolingLoopValidationTests(unittest.TestCase):
    def test_case_matrix_covers_hold_step_up_and_step_down(self) -> None:
        self.assertIsNotNone(build_case_specs)

        cases = build_case_specs()

        self.assertEqual(
            [case.case_id for case in cases],
            ["C1_hold_4000", "C2_step_up_2000_to_4000", "C3_step_down_4000_to_2000"],
        )
        self.assertEqual(
            [(case.initial_speed_rpm, case.speed_command_rpm) for case in cases],
            [(4000.0, 4000.0), (2000.0, 4000.0), (4000.0, 2000.0)],
        )

    def test_one_step_gate_matches_exact_tau_five_response(self) -> None:
        self.assertIsNotNone(run_one_step_gate)

        result = run_one_step_gate()

        expected = 4000.0 - 2000.0 * math.exp(-1.0)
        self.assertTrue(result["all_gates_pass"])
        self.assertAlmostEqual(result["compressor_speed_rpm"], expected, places=12)
        self.assertLess(abs(result["speed_model_residual_rpm"]), 1e-12)
        self.assertLess(abs(result["cycle_energy_residual_w"]), 1e-8)
        self.assertLess(abs(result["total_energy_residual_j"]), 1e-5)

    def test_short_cases_are_finite_monotonic_and_conservative(self) -> None:
        self.assertIsNotNone(run_case)

        for case in build_case_specs():
            result = run_case(case, duration_s=15.0)
            summary = result["summary"]
            speeds = np.asarray(
                [row["compressor_speed_rpm"] for row in result["timeseries"]]
            )

            self.assertEqual(summary["steps_completed"], 3)
            self.assertEqual(summary["solver_failure_count"], 0)
            self.assertTrue(summary["all_states_finite"])
            self.assertTrue(summary["all_gates_pass"])
            if case.speed_command_rpm > case.initial_speed_rpm:
                self.assertTrue(np.all(np.diff(speeds) > 0.0))
                self.assertTrue(np.all(speeds < case.speed_command_rpm))
            elif case.speed_command_rpm < case.initial_speed_rpm:
                self.assertTrue(np.all(np.diff(speeds) < 0.0))
                self.assertTrue(np.all(speeds > case.speed_command_rpm))
            else:
                self.assertTrue(np.all(speeds == case.initial_speed_rpm))


if __name__ == "__main__":
    unittest.main()
