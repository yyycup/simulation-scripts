"""Acceptance test for one-step Reference-to-ROM equivalence."""

from __future__ import annotations

import unittest

from cluster_plant_v2.validation.validate_pack_rom_one_step import compare_one_step


class PackRomOneStepTests(unittest.TestCase):
    def test_three_uniform_current_cases_meet_first_step_gate(self) -> None:
        for current in (280.0, 560.0, 1120.0):
            with self.subTest(current_A=current):
                result = compare_one_step(current)
                self.assertLessEqual(result["branch_current_max_abs_error_A"], 1e-9)
                self.assertLessEqual(result["soc_max_abs_error"], 1e-12)
                self.assertLess(result["q_total_relative_error"], 1e-3)
                self.assertLess(result["weighted_Tavg_abs_error_K"], 1e-4)


if __name__ == "__main__":
    unittest.main()
