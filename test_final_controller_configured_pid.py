import unittest

from run_final_controller_comparison import configured_pid_selection


class FinalControllerConfiguredPidTest(unittest.TestCase):
    def test_uses_three_fixed_pid_values_for_each_scene(self):
        selection = configured_pid_selection()

        self.assertEqual(len(selection["peak"]), 3)
        self.assertEqual(len(selection["freq"]), 3)
        self.assertTrue(all(isinstance(value, float) for value in selection["peak"]))

    def test_uses_locally_refined_1000rpm_pid_candidates(self):
        selection = configured_pid_selection()

        self.assertEqual(selection["peak"], (1.7, 0.002, 0.04))
        self.assertEqual(selection["freq"], (2.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
