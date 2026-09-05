import unittest

from cluster_plant_v2.validation.validate_cluster_plant_v21 import (
    _direction,
    build_v21_cases,
)


class ClusterPlantV21ValidationTests(unittest.TestCase):
    def test_case_matrix_contains_required_stage8d2_cases(self) -> None:
        cases = build_v21_cases()

        self.assertEqual(
            [case.case_id for case in cases[:5]],
            [
                "T0_constant",
                "T1_compressor_step",
                "T2_regd",
                "T3_560a_thermal_balance",
                "T4_1120a_stress",
            ],
        )
        self.assertEqual(cases[3].duration_s, 600.0)
        self.assertEqual(cases[4].compressor_initial_rpm, 6000.0)

    def test_direction_schedule_covers_forward_reverse_and_switch(self) -> None:
        cases = {case.case_id: case for case in build_v21_cases()}

        self.assertEqual(_direction(cases["T0_constant"], 0.0), "forward")
        self.assertEqual(_direction(cases["D_reverse"], 0.0), "reverse")
        switch = cases["D_forward_to_reverse"]
        self.assertEqual(_direction(switch, 295.0), "forward")
        self.assertEqual(_direction(switch, 300.0), "reverse")


if __name__ == "__main__":
    unittest.main()
