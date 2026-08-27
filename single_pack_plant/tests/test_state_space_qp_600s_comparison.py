from __future__ import annotations

from dataclasses import asdict
import importlib
import importlib.util
import unittest

import numpy as np
import pandas as pd


MODULE_NAME = "single_pack_plant.runners.validate_fixed_qp_600s"


def _comparison_module():
    spec = importlib.util.find_spec(MODULE_NAME)
    if spec is None:
        raise AssertionError(f"missing comparison runner: {MODULE_NAME}")
    return importlib.import_module(MODULE_NAME)


class StateSpaceQP600SComparisonTests(unittest.TestCase):
    def test_runner_exposes_frozen_qp_weights(self):
        comparison = _comparison_module()
        self.assertEqual(
            asdict(comparison.frozen_qp_weights()),
            {
                "q_y": 1.0,
                "r_comp": 3.0,
                "r_pump": 0.001,
                "r_delta_comp": 0.01,
                "r_delta_pump": 0.01,
                "p_f": 1.0,
            },
        )

    def test_constant_and_step_profiles_have_identical_controller_inputs(self):
        comparison = _comparison_module()
        for profile in ("constant", "current_step"):
            qp_case = comparison.build_case(profile, "qp", duration_s=600.0)
            gekko_case = comparison.build_case(profile, "gekko", duration_s=600.0)
            self.assertEqual(qp_case.scene, gekko_case.scene)
            self.assertEqual(qp_case.flow, gekko_case.flow)
            self.assertEqual(qp_case.dt, gekko_case.dt)
            self.assertEqual(qp_case.duration_s, gekko_case.duration_s)
            self.assertEqual(qp_case.predictor, gekko_case.predictor)
            np.testing.assert_array_equal(qp_case.current_profile, gekko_case.current_profile)
            self.assertEqual(qp_case.current_profile.size, 180)
        constant = comparison.build_case("constant", "qp", duration_s=600.0)
        step = comparison.build_case("current_step", "qp", duration_s=600.0)
        self.assertTrue(np.all(constant.current_profile == 560.0))
        self.assertTrue(np.all(step.current_profile[:60] == 280.0))
        self.assertTrue(np.all(step.current_profile[60:] == 840.0))

    def test_summary_uses_physical_energy_and_solver_columns(self):
        comparison = _comparison_module()
        frame = pd.DataFrame(
            {
                "Average temperature": [25.0, 25.2],
                "T_cell_max_C": [25.1, 25.4],
                "Delta_T_cell_C": [0.1, 0.2],
                "Compressor command": [1800.0, 2100.0],
                "Pump command": [1800.0, 1600.0],
                "Compressor Speed": [1700.0, 2000.0],
                "Pump Speed (RPM)": [1700.0, 1650.0],
                "Compressor power (kW)": [2.0, 4.0],
                "Pump power (kW)": [0.1, 0.2],
                "Total power": [2.6, 4.7],
                "Cumulative energy consumption": [2.6 * 5 / 3600, 7.3 * 5 / 3600],
                "MPC_Solved": [True, False],
                "MPC solve recovery used": [False, True],
                "MPC prediction domain valid": [True, False],
                "MPC solve time": [0.2, 0.4],
            }
        )
        result = comparison.summarize_frame(
            frame, profile="constant", controller="qp", dt=5.0
        )
        self.assertAlmostEqual(result["compressor_energy_kwh"], 6.0 * 5 / 3600)
        self.assertAlmostEqual(result["pump_energy_kwh"], 0.3 * 5 / 3600)
        self.assertAlmostEqual(result["total_energy_kwh"], 7.3 * 5 / 3600)
        self.assertEqual(result["solved_rate"], 0.5)
        self.assertEqual(result["fallback_count"], 1)
        self.assertEqual(result["domain_valid_rate"], 0.5)
        self.assertAlmostEqual(result["solve_time_median_s"], 0.3)


if __name__ == "__main__":
    unittest.main()
