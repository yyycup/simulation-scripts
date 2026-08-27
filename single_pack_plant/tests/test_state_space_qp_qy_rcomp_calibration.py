from __future__ import annotations

from dataclasses import asdict
import importlib
import importlib.util
import unittest

import pandas as pd


MODULE_NAME = "single_pack_plant.runners.calibrate_qy_rcomp"


def _calibration_module():
    spec = importlib.util.find_spec(MODULE_NAME)
    if spec is None:
        raise AssertionError(f"missing calibration runner: {MODULE_NAME}")
    return importlib.import_module(MODULE_NAME)


class StateSpaceQPQyRcompCalibrationTests(unittest.TestCase):
    def test_requested_cases_only_change_qy_and_rcomp(self):
        calibration = _calibration_module()
        expected = {
            "baseline": (1.0, 3.0),
            "qy1_rcomp2": (1.0, 2.0),
            "qy1_rcomp2p5": (1.0, 2.5),
            "qy1p5_rcomp3": (1.5, 3.0),
            "qy2_rcomp3": (2.0, 3.0),
        }
        self.assertEqual(
            {name: (q_y, r_comp) for name, q_y, r_comp in calibration.SCREEN_CASES},
            expected,
        )
        for _, q_y, r_comp in calibration.SCREEN_CASES:
            weights = asdict(calibration.calibration_weights(q_y, r_comp))
            self.assertEqual(weights["q_y"], q_y)
            self.assertEqual(weights["r_comp"], r_comp)
            self.assertEqual(weights["r_pump"], 0.001)
            self.assertEqual(weights["r_delta_comp"], 0.01)
            self.assertEqual(weights["r_delta_pump"], 0.01)
            self.assertEqual(weights["p_f"], 1.0)

    def test_screen_profile_has_one_load_step_and_full_preview(self):
        calibration = _calibration_module()
        case = calibration.build_case(
            "current_step", "qp", duration_s=300.0
        )
        self.assertEqual(case.current_profile.size, 120)
        self.assertTrue((case.current_profile[:30] == 280.0).all())
        self.assertTrue((case.current_profile[30:] == 840.0).all())

    def test_followup_cases_only_reduce_compressor_weight(self):
        calibration = _calibration_module()
        self.assertEqual(
            getattr(calibration, "FOLLOWUP_CASES", None),
            (
                ("qy1_rcomp1p5", 1.0, 1.5),
                ("qy1_rcomp1", 1.0, 1.0),
            ),
        )
        for _, q_y, r_comp in calibration.FOLLOWUP_CASES:
            weights = asdict(calibration.calibration_weights(q_y, r_comp))
            self.assertEqual(weights["q_y"], 1.0)
            self.assertEqual(weights["r_comp"], r_comp)
            self.assertEqual(weights["r_pump"], 0.001)
            self.assertEqual(weights["r_delta_comp"], 0.01)
            self.assertEqual(weights["r_delta_pump"], 0.01)
            self.assertEqual(weights["p_f"], 1.0)

    def test_confirmation_gate_requires_all_three_temperature_limits(self):
        calibration = _calibration_module()
        candidate = pd.DataFrame(
            {
                "profile": ["constant", "current_step", "regd"],
                "t_max_max_c": [26.05, 25.74, 26.12],
                "total_energy_kwh": [0.03, 0.04, 0.03],
                "solved_rate": [1.0, 1.0, 1.0],
                "domain_valid_rate": [1.0, 1.0, 1.0],
                "constraint_max_violation_rpm": [0.0, 0.0, 0.0],
            }
        )
        gekko = pd.DataFrame(
            {
                "profile": ["constant", "current_step", "regd"],
                "t_max_max_c": [26.00, 25.65, 26.04],
                "total_energy_kwh": [0.05, 0.06, 0.04],
            }
        )
        result = calibration.evaluate_confirmation(candidate, gekko)
        self.assertTrue(result["accepted"])
        candidate.loc[candidate["profile"] == "regd", "t_max_max_c"] = 26.15
        result = calibration.evaluate_confirmation(candidate, gekko)
        self.assertFalse(result["accepted"])


if __name__ == "__main__":
    unittest.main()
