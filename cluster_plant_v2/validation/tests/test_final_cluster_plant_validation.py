import unittest

import numpy as np

try:
    from cluster_plant_v2.validation.validate_final_cluster_plant import (
        build_case_specs,
        load_regd_case_currents,
        run_case,
        run_delay_gates,
    )
except ImportError:
    build_case_specs = None
    load_regd_case_currents = None
    run_case = None
    run_delay_gates = None


class FinalClusterPlantValidationTests(unittest.TestCase):
    def test_case_matrix_is_exactly_t0_t1_t2(self) -> None:
        self.assertIsNotNone(build_case_specs)
        cases = build_case_specs()

        self.assertEqual(
            [case.case_id for case in cases],
            ["T0_constant", "T1_compressor_step", "T2_regd"],
        )
        self.assertEqual(
            [case.current_kind for case in cases],
            ["constant", "constant", "regd"],
        )
        self.assertEqual(
            [
                (
                    case.initial_compressor_speed_rpm,
                    case.final_compressor_speed_rpm,
                )
                for case in cases
            ],
            [(4000.0, 4000.0), (2000.0, 4000.0), (4000.0, 4000.0)],
        )

    def test_delay_gates_measure_exactly_fifteen_and_twenty_seconds(self) -> None:
        self.assertIsNotNone(run_delay_gates)
        result = run_delay_gates()

        self.assertTrue(result["all_gates_pass"])
        self.assertEqual(result["supply_delay_steps"], 3)
        self.assertEqual(result["return_delay_steps"], 4)
        self.assertEqual(result["measured_supply_delay_s"], 15.0)
        self.assertEqual(result["measured_return_delay_s"], 20.0)

    def test_regd_case_uses_existing_zero_to_six_hundred_second_profile(self) -> None:
        self.assertIsNotNone(load_regd_case_currents)
        times, currents = load_regd_case_currents(duration_s=600.0, dt_s=5.0)

        self.assertEqual(len(times), 120)
        self.assertEqual(times[0], 0.0)
        self.assertEqual(times[-1], 595.0)
        self.assertTrue(np.all(np.isfinite(currents)))
        self.assertLess(currents.min(), 0.0)
        self.assertGreater(currents.max(), 0.0)

    def test_short_cases_are_finite_and_preserve_local_balances(self) -> None:
        self.assertIsNotNone(run_case)
        for case in build_case_specs():
            result = run_case(case, duration_s=10.0)
            summary = result["summary"]

            self.assertEqual(summary["steps_completed"], 2)
            self.assertEqual(summary["solver_failure_count"], 0)
            self.assertTrue(summary["all_states_finite"])
            self.assertTrue(summary["all_gates_pass"])
            self.assertFalse(summary["cross_delay_energy_balance_is_modeled"])


if __name__ == "__main__":
    unittest.main()
