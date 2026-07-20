import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import calibrate_p_actual_replay as calibration
from calibrate_p_actual_replay import (
    PHYSICAL_TANK_HEAT_CAPACITY_J_K,
    STRUCTURAL_DELAY_CORRECTIONS,
    THERMAL_CALIBRATION_KEYS,
    apply_structural_evap_correction,
    thermal_candidate,
)
from mpc_physics_predictor import (
    DEFAULT_PHYSICS_ARTIFACT,
    PHYSICAL_COOLANT_CP_J_KG_K,
)


class ActualReplayCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.base = copy.deepcopy(DEFAULT_PHYSICS_ARTIFACT)

    def test_structural_evap_correction_freezes_capacity_and_thermal_model(self):
        candidate = apply_structural_evap_correction(
            self.base,
            tau_cond_s=45.0,
            tau_evap_s=1.0,
        )

        self.assertEqual(candidate["gate"], self.base["gate"])
        self.assertEqual(candidate["capacity"], self.base["capacity"])
        self.assertEqual(candidate["thermal"], self.base["thermal"])
        for name, value in STRUCTURAL_DELAY_CORRECTIONS.items():
            self.assertEqual(candidate["dynamic"][name], value)
        self.assertEqual(candidate["dynamic"]["tau_cond_s"], 45.0)
        self.assertEqual(candidate["dynamic"]["tau_evap_s"], 1.0)

    def test_thermal_candidate_changes_only_declared_lumped_parameters(self):
        values = np.array([0.15, 15000.0, 150.0, 1.0, 0.1, 0.25])
        candidate = thermal_candidate(self.base, values)

        self.assertEqual(candidate["gate"], self.base["gate"])
        self.assertEqual(candidate["capacity"], self.base["capacity"])
        self.assertEqual(candidate["dynamic"], self.base["dynamic"])
        for name, original in self.base["thermal"].items():
            expected = values[THERMAL_CALIBRATION_KEYS.index(name)] if name in THERMAL_CALIBRATION_KEYS else original
            self.assertEqual(candidate["thermal"][name], expected)
        self.assertEqual(
            candidate["thermal"]["coolant_cp_j_kg_k"],
            PHYSICAL_COOLANT_CP_J_KG_K,
        )
        self.assertGreater(PHYSICAL_TANK_HEAT_CAPACITY_J_K, 10000.0)

    def test_thermal_candidate_rejects_wrong_parameter_count(self):
        with self.assertRaises(ValueError):
            thermal_candidate(self.base, np.ones(len(THERMAL_CALIBRATION_KEYS) - 1))

    def test_validation_targets_cannot_change_selected_parameters(self):
        detail_rows = []
        for horizon in (50.0, 100.0, 300.0):
            for state in ("Q_Evap", "T_Batt", "T_Cool", "T_Plate", "T_Supply", "T_Return"):
                detail_rows.append(
                    {
                        "horizon_s": horizon,
                        "state": state,
                        "error": 0.1,
                        "abs_error": 0.1,
                    }
                )
        fixed_details = pd.DataFrame(detail_rows)

        def calibrate_with_validation(validation_value):
            optimizer = SimpleNamespace(
                success=True,
                x=(calibration.THERMAL_START - calibration.THERMAL_LOWER)
                / (calibration.THERMAL_UPPER - calibration.THERMAL_LOWER),
                cost=1.0,
                nfev=1,
                message="mock",
            )
            with patch.object(calibration, "_details", return_value=fixed_details), patch.object(
                calibration,
                "_q_score",
                side_effect=lambda details: 1.0,
            ), patch.object(
                calibration,
                "_thermal_score",
                side_effect=[2.0, 1.0],
            ), patch.object(
                calibration,
                "_summary_records",
                return_value=[],
            ), patch.object(
                calibration,
                "least_squares",
                return_value=optimizer,
            ):
                artifact, _report = calibration.calibrate_actual_replay(
                    self.base,
                    pd.DataFrame({"marker": [0.0]}),
                    pd.DataFrame({"target": [validation_value]}),
                )
            return artifact

        first = calibrate_with_validation(-1e9)
        second = calibrate_with_validation(1e9)
        self.assertEqual(first["dynamic"], second["dynamic"])
        self.assertEqual(first["thermal"], second["thermal"])
        self.assertEqual(first["capacity"], second["capacity"])


if __name__ == "__main__":
    unittest.main()
