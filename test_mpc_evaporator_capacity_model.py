import unittest
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mpc_evaporator_capacity_model import (
    evaluate_capacity,
    load_capacity_calibration,
    smooth_mpc_power_gate,
)
from run_evaporator_mpc_capacity_diagnostic import build_comparison
from run_mpc_evaporator_capacity_calibration import (
    T_COOL_ALL,
    T_COOL_TRAIN,
    T_COOL_VALIDATION,
)


class MpcEvaporatorCapacityModelTest(unittest.TestCase):
    @staticmethod
    def _candidate_b_calibration():
        return {
            "model_type": "candidate_b",
            "coefficients": {"b0": 0.2, "b1": 0.1, "b2": 0.01},
            "n_pump_ref_rpm": 2000.0,
            "q_evap_upper_bound_w": 5000.0,
            "minimum_active_rpm": 2000.0,
            "mpc_power_gate": {"center_rpm": 1950.0, "width_rpm": 10.0},
        }

    def test_candidate_b_uses_pump_ratio_and_coolant_temperature(self):
        calibration = {
            "model_type": "candidate_b",
            "coefficients": {"b0": 0.5, "b1": 0.1, "b2": 0.01},
            "n_pump_ref_rpm": 2000.0,
            "q_evap_upper_bound_w": 10000.0,
        }

        value = evaluate_capacity(calibration, n_comp_rpm=4000.0, n_pump_rpm=4000.0, t_cool_c=30.0)

        self.assertAlmostEqual(value, 2400.0)

    def test_candidate_b_is_off_below_physical_compressor_map(self):
        calibration = self._candidate_b_calibration()

        self.assertEqual(
            evaluate_capacity(calibration, 1999.0, 2400.0, 25.0),
            0.0,
        )
        self.assertGreater(
            evaluate_capacity(calibration, 2000.0, 2400.0, 25.0),
            0.0,
        )

    def test_mpc_power_gate_smoothly_approximates_the_2000_rpm_hard_boundary(self):
        calibration = self._candidate_b_calibration()

        self.assertLess(smooth_mpc_power_gate(calibration, 1900.0), 0.02)
        self.assertAlmostEqual(smooth_mpc_power_gate(calibration, 1950.0), 0.5)
        self.assertGreater(smooth_mpc_power_gate(calibration, 2000.0), 0.99)

    def test_canonical_candidate_b_covers_15c_with_holdout_temperatures(self):
        calibration = load_capacity_calibration()

        self.assertEqual(min(calibration["data_range"]["T_cool_in_C"]), 15.0)
        self.assertEqual(calibration["data_range"]["T_cool_train_C"], list(T_COOL_TRAIN))
        self.assertEqual(
            calibration["data_range"]["T_cool_validation_C"],
            list(T_COOL_VALIDATION),
        )
        self.assertEqual(tuple(calibration["data_range"]["T_cool_in_C"]), T_COOL_ALL)
        self.assertTrue(calibration["validation_target_met"])
        self.assertEqual(
            calibration["optimizer_low_speed_policy"]["capacity"],
            "continuous_relaxation_for_nlp",
        )

    def test_capacity_is_bounded_nonnegative(self):
        calibration = {
            "model_type": "candidate_a",
            "coefficients": {"kq": -1.0},
            "q_evap_upper_bound_w": 3000.0,
        }

        self.assertEqual(evaluate_capacity(calibration, 4000.0, 3000.0, 25.0), 0.0)

    @patch("run_evaporator_mpc_capacity_diagnostic.run_refrigeration_cycle")
    @patch("run_evaporator_mpc_capacity_diagnostic.staged_fan_speed")
    @patch("run_evaporator_mpc_capacity_diagnostic.pump_model")
    @patch("run_evaporator_mpc_capacity_diagnostic.evaluate_capacity")
    @patch("run_evaporator_mpc_capacity_diagnostic.load_capacity_calibration")
    def test_capacity_diagnostic_dispatches_to_candidate_b_once(
        self,
        load_calibration,
        evaluate,
        pump,
        fan,
        refrigeration_cycle,
    ):
        calibration = self._candidate_b_calibration()
        load_calibration.return_value = calibration
        evaluate.return_value = 750.0
        pump.return_value = (0.25, 10.0)
        fan.return_value = 1500.0
        refrigeration_cycle.return_value = {"Q_evap": 800.0}

        frame = build_comparison([3000.0], 2000.0, 25.0)

        load_calibration.assert_called_once()
        evaluate.assert_called_once_with(calibration, 3000.0, 2000.0, 25.0)
        self.assertEqual(frame.loc[0, "Q_evap_mpc_W"], 750.0)

    def test_unpromoted_predictors_are_rejected(self):
        for predictor in ("physics_p", "lpv_l"):
            with self.subTest(predictor=predictor):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{predictor}.*(?:not promoted|provisional)",
                ):
                    build_comparison(
                        [3000.0],
                        2000.0,
                        25.0,
                        predictor=predictor,
                    )

    def test_cli_reports_unpromoted_predictor_without_traceback(self):
        with patch(
            "sys.argv",
            [
                "run_evaporator_mpc_capacity_diagnostic.py",
                "--predictor",
                "physics_p",
            ],
        ), patch("sys.stderr", new_callable=StringIO) as stderr:
            from run_evaporator_mpc_capacity_diagnostic import main

            with self.assertRaises(SystemExit):
                main()

        message = stderr.getvalue()
        self.assertIn("physics_p", message)
        self.assertRegex(message, "not promoted|provisional")
        self.assertNotIn("Traceback", message)

    def test_cli_reports_candidate_b_artifact_errors_without_traceback(self):
        with TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            missing = temp_root / "missing.json"
            bad_json = temp_root / "bad.json"
            bad_json.write_text("{broken", encoding="utf-8")
            wrong_model = temp_root / "wrong.json"
            wrong_model.write_text(
                json.dumps({"model_type": "candidate_a"}), encoding="utf-8"
            )

            for artifact_path, expected in (
                (missing, "missing.json"),
                (bad_json, "bad.json"),
                (wrong_model, "candidate_b"),
            ):
                with self.subTest(artifact=artifact_path.name), patch(
                    "sys.argv",
                    [
                        "run_evaporator_mpc_capacity_diagnostic.py",
                        "--predictor",
                        "candidate_b",
                        "--predictor-artifact",
                        str(artifact_path),
                    ],
                ), patch("sys.stderr", new_callable=StringIO) as stderr, patch(
                    "run_evaporator_mpc_capacity_diagnostic.run_refrigeration_cycle"
                ) as run_cycle:
                    from run_evaporator_mpc_capacity_diagnostic import main

                    with self.assertRaises(SystemExit) as raised:
                        main()

                self.assertEqual(raised.exception.code, 2)
                message = stderr.getvalue()
                self.assertIn(expected, message)
                self.assertNotIn("Traceback", message)
                run_cycle.assert_not_called()


if __name__ == "__main__":
    unittest.main()
