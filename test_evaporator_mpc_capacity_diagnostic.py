import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import run_evaporator_mpc_capacity_diagnostic as diagnostic


class EvaporatorMpcCapacityDiagnosticTest(unittest.TestCase):
    @staticmethod
    def _candidate_b_calibration():
        return {
            "model_type": "candidate_b",
            "coefficients": {"b0": 0.2, "b1": 0.1, "b2": 0.01},
            "n_pump_ref_rpm": 2000.0,
            "q_evap_upper_bound_w": 5000.0,
        }

    @patch.object(diagnostic, "run_refrigeration_cycle", return_value={"Q_evap": 800.0})
    @patch.object(diagnostic, "staged_fan_speed", return_value=1500.0)
    @patch.object(diagnostic, "pump_model", return_value=(0.25, 10.0))
    @patch.object(diagnostic, "evaluate_capacity", return_value=750.0)
    @patch.object(diagnostic, "load_capacity_calibration")
    def test_default_diagnostic_uses_candidate_b_capacity_model(
        self,
        load_calibration,
        evaluate_capacity,
        _pump_model,
        _staged_fan_speed,
        _run_refrigeration_cycle,
    ):
        calibration = self._candidate_b_calibration()
        load_calibration.return_value = calibration

        frame = diagnostic.build_comparison([3000.0], 2000.0, 25.0)

        load_calibration.assert_called_once()
        evaluate_capacity.assert_called_once_with(
            calibration, 3000.0, 2000.0, 25.0
        )
        self.assertEqual(frame.loc[0, "Q_evap_mpc_W"], 750.0)
        self.assertFalse(hasattr(diagnostic, "mpc_q_evap_steady_w"))

    @patch.object(diagnostic, "run_refrigeration_cycle")
    def test_non_candidate_b_artifact_is_rejected_before_plant(self, run_cycle):
        for model_type in ("candidate_a", "candidate_c"):
            with self.subTest(model_type=model_type), TemporaryDirectory() as tmp:
                artifact_path = Path(tmp) / f"{model_type}.json"
                artifact_path.write_text(
                    json.dumps({"model_type": model_type}), encoding="utf-8"
                )

                with self.assertRaises(ValueError) as raised:
                    diagnostic.build_comparison(
                        [3000.0],
                        2000.0,
                        25.0,
                        predictor="candidate_b",
                        predictor_artifact=artifact_path,
                    )

                message = str(raised.exception)
                self.assertIn(str(artifact_path), message)
                self.assertIn(model_type, message)
                self.assertIn("candidate_b", message)
        run_cycle.assert_not_called()

    @patch.object(diagnostic, "run_refrigeration_cycle", return_value={"Q_evap": 800.0})
    @patch.object(diagnostic, "staged_fan_speed", return_value=1500.0)
    @patch.object(diagnostic, "pump_model", return_value=(0.25, 10.0))
    def test_valid_candidate_b_artifact_calls_real_evaluator(
        self, _pump_model, _staged_fan_speed, _run_cycle
    ):
        with TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "candidate_b.json"
            artifact_path.write_text(
                json.dumps(self._candidate_b_calibration()), encoding="utf-8"
            )

            frame = diagnostic.build_comparison(
                [3000.0],
                2000.0,
                25.0,
                predictor_artifact=artifact_path,
            )

        self.assertAlmostEqual(frame.loc[0, "Q_evap_mpc_W"], 900.0)

    def test_candidate_b_schema_rejects_invalid_fields(self):
        valid = self._candidate_b_calibration()
        invalid_cases = (
            ("root", [], "JSON object"),
            ("coefficients", {**valid, "coefficients": []}, "coefficients"),
            ("b0_bool", {**valid, "coefficients": {**valid["coefficients"], "b0": True}}, "b0"),
            ("b1_nan", {**valid, "coefficients": {**valid["coefficients"], "b1": float("nan")}}, "b1"),
            ("b2_missing", {**valid, "coefficients": {"b0": 0.2, "b1": 0.1}}, "b2"),
            ("pump_zero", {**valid, "n_pump_ref_rpm": 0.0}, "n_pump_ref_rpm"),
            ("upper_bool", {**valid, "q_evap_upper_bound_w": False}, "q_evap_upper_bound_w"),
            ("upper_negative", {**valid, "q_evap_upper_bound_w": -1.0}, "q_evap_upper_bound_w"),
        )
        for name, calibration, field in invalid_cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, field):
                    diagnostic._validate_candidate_b_calibration(
                        calibration, Path(f"{name}.json")
                    )


if __name__ == "__main__":
    unittest.main()
