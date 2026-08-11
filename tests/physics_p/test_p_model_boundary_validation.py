import unittest

from experiments.physics_p.evaluation.run_p_model_boundary_validation import (
    DEFAULT_SPEEDS_RPM,
    build_boundary_table,
    summarize_boundary_table,
)
from run_p_mpc_operational import DEFAULT_OPERATIONAL_P_ARTIFACT


class PhysicsPBoundaryValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = build_boundary_table(
            artifact_path=DEFAULT_OPERATIONAL_P_ARTIFACT,
        )
        cls.summary = summarize_boundary_table(cls.frame)

    def test_boundary_grid_contains_off_startup_and_active_points(self):
        self.assertEqual(
            tuple(self.frame["n_comp_cmd_rpm"].to_numpy(dtype=float)),
            DEFAULT_SPEEDS_RPM,
        )
        indexed = self.frame.set_index("n_comp_cmd_rpm")
        self.assertEqual(indexed.loc[300.0, "startup_fraction"], 0.0)
        self.assertEqual(indexed.loc[650.0, "startup_fraction"], 0.5)
        self.assertEqual(indexed.loc[1000.0, "startup_fraction"], 1.0)

    def test_off_and_1000rpm_continuity_boundary_qualifies(self):
        self.assertTrue(self.summary["qualified"], self.summary)
        self.assertLessEqual(self.summary["p_999_to_1000_relative_jump"], 0.01)
        self.assertLessEqual(self.summary["plant_999_to_1000_relative_jump"], 0.01)
        self.assertLess(self.summary["active_capacity_mape"], 0.12)


if __name__ == "__main__":
    unittest.main()
