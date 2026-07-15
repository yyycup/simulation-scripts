import unittest
from io import StringIO
from unittest.mock import patch

from mpc_evaporator_capacity_model import evaluate_capacity
from run_evaporator_mpc_capacity_diagnostic import build_comparison


class MpcEvaporatorCapacityModelTest(unittest.TestCase):
    def test_candidate_b_uses_pump_ratio_and_coolant_temperature(self):
        calibration = {
            "model_type": "candidate_b",
            "coefficients": {"b0": 0.5, "b1": 0.1, "b2": 0.01},
            "n_pump_ref_rpm": 2000.0,
            "q_evap_upper_bound_w": 10000.0,
        }

        value = evaluate_capacity(calibration, n_comp_rpm=4000.0, n_pump_rpm=4000.0, t_cool_c=30.0)

        self.assertAlmostEqual(value, 2400.0)

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
        calibration = {"model_type": "candidate_b"}
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


if __name__ == "__main__":
    unittest.main()
