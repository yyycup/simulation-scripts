import math
import unittest

try:
    from cluster_plant_v2.validation.validate_thermally_dynamic_refrigerated_cluster_cooling_loop import (
        build_case_specs,
        compressor_command_at_time,
        run_case,
        run_component_gates,
        run_one_step_gate,
    )
except ImportError:
    build_case_specs = None
    compressor_command_at_time = None
    run_case = None
    run_component_gates = None
    run_one_step_gate = None


class ThermallyDynamicCoolingLoopValidationTests(unittest.TestCase):
    def test_case_matrix_is_exactly_h0_h1_h2(self) -> None:
        self.assertIsNotNone(build_case_specs)
        cases = build_case_specs()

        self.assertEqual(
            [case.case_id for case in cases],
            ["H0_equilibrium", "H1_step_up", "H2_step_down"],
        )
        self.assertEqual(
            [
                (
                    case.initial_compressor_speed_rpm,
                    case.final_compressor_speed_rpm,
                    case.step_time_s,
                )
                for case in cases
            ],
            [(4000.0, 4000.0, 100.0), (2000.0, 4000.0, 100.0), (4000.0, 2000.0, 100.0)],
        )

    def test_command_changes_at_one_hundred_seconds(self) -> None:
        self.assertIsNotNone(compressor_command_at_time)
        case = build_case_specs()[1]

        self.assertEqual(compressor_command_at_time(case, 95.0), 2000.0)
        self.assertEqual(compressor_command_at_time(case, 100.0), 4000.0)

    def test_component_gates_report_exact_response_and_dt_independence(self) -> None:
        self.assertIsNotNone(run_component_gates)
        result = run_component_gates()

        self.assertTrue(result["all_gates_pass"])
        self.assertAlmostEqual(
            result["up_5s_w"],
            2000.0 - 1000.0 * math.exp(-5.0 / 45.0),
            places=12,
        )
        self.assertAlmostEqual(
            result["up_45s_w"], 2000.0 - 1000.0 * math.exp(-1.0), places=12
        )
        self.assertAlmostEqual(
            result["up_135s_w"], 2000.0 - 1000.0 * math.exp(-3.0), places=12
        )
        self.assertAlmostEqual(
            result["up_225s_w"], 2000.0 - 1000.0 * math.exp(-5.0), places=12
        )
        self.assertLess(result["dt_independence_spread_w"], 1e-9)

    def test_one_step_and_short_cases_close_all_energy_boundaries(self) -> None:
        self.assertIsNotNone(run_one_step_gate)
        result = run_one_step_gate()
        self.assertTrue(result["all_gates_pass"])

        for key in (
            "cycle_energy_residual_w",
            "evaporator_coolant_residual_w",
            "evaporator_dynamic_energy_residual_j",
            "bpt_energy_residual_j",
            "bpt_evaporator_energy_residual_j",
        ):
            self.assertLess(abs(result[key]), 1e-5)

        for case in build_case_specs():
            case_result = run_case(case, duration_s=10.0)
            self.assertEqual(case_result["summary"]["steps_completed"], 2)
            self.assertEqual(case_result["summary"]["solver_failure_count"], 0)
            self.assertTrue(case_result["summary"]["all_states_finite"])
            self.assertTrue(case_result["summary"]["all_gates_pass"])
            if case.case_id == "H0_equilibrium":
                self.assertTrue(case_result["summary"]["stage8c1_reference_run"])
            else:
                self.assertFalse(case_result["summary"]["stage8c1_reference_run"])
                self.assertIsNone(
                    case_result["summary"][
                        "max_abs_stage8c1_q_evap_difference_w"
                    ]
                )


if __name__ == "__main__":
    unittest.main()
