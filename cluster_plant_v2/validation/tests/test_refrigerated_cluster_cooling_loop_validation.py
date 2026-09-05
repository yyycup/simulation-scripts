import unittest

import numpy as np

from cluster_plant_v2.validation.validate_refrigerated_cluster_cooling_loop import (
    build_case_specs,
    run_case,
    run_nominal_cycle_interface_regression,
    run_one_step_gate,
)


class RefrigeratedClusterCoolingLoopValidationTests(unittest.TestCase):
    def test_case_matrix_is_exactly_r1_through_r5(self) -> None:
        cases = build_case_specs()

        self.assertEqual([case.case_id[:2] for case in cases], ["R1", "R2", "R3", "R4", "R5"])
        self.assertEqual(
            [case.compressor_speed_rpm for case in cases],
            [2000.0, 3000.0, 4000.0, 4000.0, 4000.0],
        )
        self.assertEqual(
            [case.fan_speed_rpm for case in cases],
            [800.0, 800.0, 1200.0, 1200.0, 1200.0],
        )
        self.assertEqual(cases[3].cluster_current_a, 1120.0)
        self.assertEqual(cases[4].direction, "reverse")

    def test_nominal_cycle_interface_recovers_cluster_sized_8d4_point(self) -> None:
        result = run_nominal_cycle_interface_regression()

        self.assertTrue(result["all_gates_pass"])
        # Stage 8D4b nominal point (pump rescaled x2: 50.4 L/min at 3600 rpm;
        # supersedes the 8D4 point 15595.39 / 18693.24 / 3097.86 / 3391.35).
        self.assertAlmostEqual(result["coolant_volume_flow_l_min"], 50.4, places=9)
        self.assertAlmostEqual(result["q_evaporator_w"], 18009.272475241098, places=6)
        self.assertAlmostEqual(result["q_condenser_w"], 21210.502824009964, places=6)
        self.assertAlmostEqual(
            result["refrigerant_compression_power_w"],
            3201.2303487688687,
            places=6,
        )
        self.assertAlmostEqual(
            result["compressor_shaft_power_w"], 3504.5219203425813, places=6
        )
        self.assertAlmostEqual(result["cop_shaft"], 5.138867122132487, places=9)
        self.assertAlmostEqual(
            result["evaporating_saturation_temperature_c"],
            18.9043151082804,
            places=6,
        )
        self.assertAlmostEqual(
            result["condensing_saturation_temperature_c"],
            50.30203846010471,
            places=6,
        )

    def test_one_step_gate_closes_entire_thermal_chain_without_double_cooling(
        self,
    ) -> None:
        result = run_one_step_gate()

        self.assertTrue(result["all_gates_pass"])
        self.assertLess(result["supply_temperature_k"], result["tank_temperature_before_k"])
        self.assertAlmostEqual(
            result["tank_energy_change_j"],
            result["q_tank_w"] * result["dt_s"],
            places=8,
        )
        for key in (
            "evaporator_coolant_residual_w",
            "cycle_energy_residual_w",
            "cluster_fluid_residual_w",
            "thermal_chain_residual_w",
            "total_energy_residual_j",
        ):
            self.assertLess(abs(result[key]), 1e-6)

    def test_short_forward_and_reverse_cases_remain_finite(self) -> None:
        cases = build_case_specs()
        for case in (cases[2], cases[4]):
            result = run_case(case, duration_s=10.0)
            self.assertEqual(len(result["timeseries"]), 2)
            self.assertEqual(result["summary"]["solver_failure_count"], 0)
            self.assertTrue(result["summary"]["all_states_finite"])
            self.assertTrue(result["summary"]["all_gates_pass"])
            numeric = np.asarray(
                [
                    result["summary"]["final_tank_temperature_c"],
                    result["summary"]["final_supply_temperature_c"],
                    result["summary"]["final_return_temperature_c"],
                ]
            )
            self.assertTrue(np.all(np.isfinite(numeric)))


if __name__ == "__main__":
    unittest.main()
