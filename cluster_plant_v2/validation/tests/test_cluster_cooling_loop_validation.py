import unittest

import numpy as np

from cluster_plant_v2.validation.validate_cluster_cooling_loop import (
    build_cluster_case_specs,
    build_pump_case_speeds,
    run_cluster_case,
    run_pump_case,
    run_tank_one_step,
)


class ClusterCoolingLoopValidationTests(unittest.TestCase):
    def test_pump_matrix_uses_audited_speed_range(self) -> None:
        self.assertEqual(
            build_pump_case_speeds(),
            [1600.0, 2400.0, 3200.0, 4000.0, 4800.0],
        )

    def test_cluster_matrix_contains_only_p1_through_p5(self) -> None:
        cases = build_cluster_case_specs()
        self.assertEqual([case.case_id[:2] for case in cases], ["P1", "P2", "P3", "P4", "P5"])
        self.assertEqual([case.pump_speed_rpm for case in cases[:3]], [2400.0, 3600.0, 4800.0])
        self.assertEqual(cases[3].current_a, 1120.0)
        self.assertEqual(cases[4].direction, "reverse")

    def test_tank_gate_matches_five_second_hand_calculation(self) -> None:
        result = run_tank_one_step()
        self.assertLess(abs(result["temperature_difference_from_hand_k"]), 1e-12)
        self.assertLess(abs(result["tank_energy_residual_j"]), 1e-8)

    def test_independent_pump_points_are_monotonic_and_balanced(self) -> None:
        points = [run_pump_case(speed) for speed in build_pump_case_speeds()]
        flows = np.array([point["total_mass_flow_kg_s"] for point in points])
        powers = np.array([point["pump_power_w"] for point in points])
        self.assertTrue(np.all(np.diff(flows) > 0.0))
        self.assertTrue(np.all(np.diff(powers) > 0.0))
        self.assertLess(
            max(abs(point["pressure_balance_residual_pa"]) for point in points),
            1e-6,
        )

    def test_short_cluster_case_exposes_finite_coupling_diagnostics(self) -> None:
        result = run_cluster_case(build_cluster_case_specs()[1], duration_s=10.0)
        self.assertEqual(len(result["timeseries"]), 2)
        self.assertTrue(result["summary"]["all_states_finite"])
        self.assertEqual(result["summary"]["solver_failure_count"], 0)
        self.assertLess(result["summary"]["max_pressure_residual_pa"], 1e-6)
        self.assertLess(result["summary"]["max_tank_energy_residual_j"], 1e-8)


if __name__ == "__main__":
    unittest.main()
