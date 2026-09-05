import unittest

import numpy as np

from cluster_plant_v2.validation.validate_refrigeration_cycle import (
    build_cycle_case_specs,
    run_cycle_case,
)


class RefrigerationCycleValidationTests(unittest.TestCase):
    def test_matrix_covers_six_speeds_and_three_coolant_flows(self) -> None:
        cases = build_cycle_case_specs()

        self.assertEqual(len(cases), 18)
        self.assertEqual(
            sorted({case.compressor_speed_rpm for case in cases}),
            [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0],
        )
        self.assertEqual(
            sorted({case.coolant_volume_flow_l_min for case in cases}),
            [16.8, 25.2, 33.6],
        )

    def test_nominal_case_reports_finite_closed_cycle_diagnostics(self) -> None:
        case = next(
            case
            for case in build_cycle_case_specs()
            if case.compressor_speed_rpm == 4000.0
            and case.coolant_volume_flow_l_min == 25.2
        )
        result = run_cycle_case(case)

        self.assertTrue(result["all_states_finite"])
        self.assertTrue(result["state_connections_pass"])
        self.assertTrue(result["all_gates_pass"], result["solver_message"])
        self.assertLess(result["mass_flow_relative_residual"], 1e-3)
        self.assertLess(result["evaporator_relative_residual"], 5e-3)
        self.assertLess(result["condenser_relative_residual"], 5e-3)
        self.assertLess(result["cycle_energy_relative_residual"], 5e-3)
        numeric = np.array(
            [
                result["q_evaporator_w"],
                result["q_condenser_w"],
                result["compressor_shaft_power_w"],
                result["coefficient_of_performance"],
            ]
        )
        self.assertTrue(np.all(np.isfinite(numeric)))
        self.assertTrue(np.all(numeric > 0.0))


if __name__ == "__main__":
    unittest.main()
