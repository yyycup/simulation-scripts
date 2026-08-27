from __future__ import annotations

import importlib
import importlib.util
import unittest

import pandas as pd


MODULE_NAME = "single_pack_plant.runners.sensitivity_qp_runtime"


def _sensitivity_module():
    spec = importlib.util.find_spec(MODULE_NAME)
    if spec is None:
        raise AssertionError(f"missing sensitivity runner: {MODULE_NAME}")
    return importlib.import_module(MODULE_NAME)


class StateSpaceQPRuntimeSensitivityTests(unittest.TestCase):
    def test_cases_vary_only_temperature_and_compressor_multipliers(self):
        sensitivity = _sensitivity_module()
        self.assertEqual(
            sensitivity.SENSITIVITY_CASES,
            (
                ("alpha_t_0p7", 0.7, 1.0),
                ("baseline", 1.0, 1.0),
                ("alpha_t_1p3", 1.3, 1.0),
                ("alpha_comp_0p7", 1.0, 0.7),
                ("alpha_comp_1p3", 1.0, 1.3),
            ),
        )
        self.assertEqual(sensitivity.ALPHA_PUMP, 1.0)

    def test_monotonic_three_profile_tradeoff_is_usable(self):
        sensitivity = _sensitivity_module()
        rows = []
        for profile in sensitivity.PROFILE_NAMES:
            for label, alpha_t, alpha_comp, tmax, energy in (
                ("alpha_t_0p7", 0.7, 1.0, 26.2, 0.03),
                ("baseline", 1.0, 1.0, 26.1, 0.04),
                ("alpha_t_1p3", 1.3, 1.0, 26.0, 0.05),
                ("alpha_comp_0p7", 1.0, 0.7, 26.0, 0.05),
                ("alpha_comp_1p3", 1.0, 1.3, 26.2, 0.03),
            ):
                rows.append(
                    {
                        "profile": profile,
                        "label": label,
                        "alpha_t": alpha_t,
                        "alpha_comp": alpha_comp,
                        "alpha_pump": 1.0,
                        "t_max_max_c": tmax,
                        "total_energy_kwh": energy,
                        "solved_rate": 1.0,
                        "domain_valid_rate": 1.0,
                        "constraint_max_violation_rpm": 0.0,
                        "all_core_values_finite": True,
                    }
                )
        result = sensitivity.evaluate_sensitivity(pd.DataFrame(rows))
        self.assertTrue(result["td3_action_usable"])
        self.assertTrue(result["alpha_t_monotonic"])
        self.assertTrue(result["alpha_comp_monotonic"])

    def test_nonmonotonic_temperature_axis_is_not_usable(self):
        sensitivity = _sensitivity_module()
        rows = []
        for profile in sensitivity.PROFILE_NAMES:
            for label, alpha_t, tmax, energy in (
                ("alpha_t_0p7", 0.7, 26.2, 0.03),
                ("baseline", 1.0, 26.1, 0.04),
                ("alpha_t_1p3", 1.3, 26.3, 0.05),
            ):
                rows.append(
                    {
                        "profile": profile,
                        "label": label,
                        "alpha_t": alpha_t,
                        "alpha_comp": 1.0,
                        "alpha_pump": 1.0,
                        "t_max_max_c": tmax,
                        "total_energy_kwh": energy,
                        "solved_rate": 1.0,
                        "domain_valid_rate": 1.0,
                        "constraint_max_violation_rpm": 0.0,
                        "all_core_values_finite": True,
                    }
                )
            for label, alpha_comp, tmax, energy in (
                ("alpha_comp_0p7", 0.7, 26.0, 0.05),
                ("alpha_comp_1p3", 1.3, 26.2, 0.03),
            ):
                rows.append(
                    {
                        "profile": profile,
                        "label": label,
                        "alpha_t": 1.0,
                        "alpha_comp": alpha_comp,
                        "alpha_pump": 1.0,
                        "t_max_max_c": tmax,
                        "total_energy_kwh": energy,
                        "solved_rate": 1.0,
                        "domain_valid_rate": 1.0,
                        "constraint_max_violation_rpm": 0.0,
                        "all_core_values_finite": True,
                    }
                )
        result = sensitivity.evaluate_sensitivity(pd.DataFrame(rows))
        self.assertFalse(result["alpha_t_monotonic"])
        self.assertFalse(result["td3_action_usable"])


if __name__ == "__main__":
    unittest.main()
