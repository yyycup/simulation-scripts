import unittest

from mpc_flow_direction_strategies import _planned_control_values


class PlannedControlExportTests(unittest.TestCase):
    def test_plan_drops_current_state_and_forces_first_executed_command(self):
        plan = _planned_control_values([1000.0, 2100.0, 2200.0], 2050.0)
        self.assertEqual(plan, [2050.0, 2200.0])

    def test_empty_or_scalar_solver_values_fall_back_to_executed_command(self):
        self.assertEqual(_planned_control_values([], 2050.0), [2050.0])
        self.assertEqual(_planned_control_values([1000.0], 2050.0), [2050.0])


if __name__ == "__main__":
    unittest.main()
