import math
import unittest

from mpc_flow_direction_strategies import apply_reduced_model_calibration
from mpc_reduced_model_calibration import (
    THETA0,
    build_samples_from_rows,
    evaluate_theta,
    summarize_errors,
)


class ReducedModelCalibrationTest(unittest.TestCase):
    def _rows(self):
        rows = []
        for i in range(70):
            t = 25.0 + 0.01 * i
            rows.append(
                {
                    "Time": str(5.0 * i),
                    "Average temperature": str(t),
                    "Coolant temperature": "30.0",
                    "Cold plate inlet coolant temperature": "28.0",
                    "Cold plate outlet coolant temperature": "29.0",
                    "Evaporator cooling rate (kW)": "1.0",
                    "Compressor Speed": "2200.0",
                    "Pump Speed (RPM)": "2400.0",
                    "Total current": "400.0",
                    "Flow direction d": "1",
                }
            )
        return rows

    def test_build_samples_keeps_future_inputs_fixed(self):
        samples = build_samples_from_rows(
            self._rows(),
            case_name="synthetic",
            start_stride=10,
            horizons_s=(50, 100, 300),
        )

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample.case_name, "synthetic")
        self.assertEqual(sample.start_index, 0)
        self.assertEqual(sample.horizon_steps, {50: 10, 100: 20, 300: 60})
        self.assertEqual(len(sample.input_seq), 61)
        self.assertEqual(sample.input_seq[0].n_comp_rpm, 2200.0)
        self.assertEqual(sample.input_seq[0].n_pump_rpm, 2400.0)
        self.assertEqual(sample.input_seq[0].flow_direction, 1)
        self.assertAlmostEqual(sample.real_future_c[300], 25.6)

    def test_evaluate_theta_uses_real_minus_predicted_bias(self):
        samples = build_samples_from_rows(
            self._rows(),
            case_name="synthetic",
            start_stride=10,
            horizons_s=(50, 100, 300),
        )

        result = evaluate_theta(THETA0, samples)

        for horizon_s in (50, 100, 300):
            self.assertIn(horizon_s, result.error_by_horizon)
            errors = result.error_by_horizon[horizon_s]
            self.assertTrue(errors)
            stats = summarize_errors(errors)
            self.assertAlmostEqual(stats["mean_bias_c"], sum(errors) / len(errors))
        self.assertTrue(math.isfinite(result.objective))
        self.assertGreaterEqual(result.objective, 0.0)

    def test_apply_reduced_model_calibration_only_changes_prediction_scales(self):
        params = {
            "C1": 100.0,
            "C2": 200.0,
            "h1_ref": 10.0,
            "kq": 1.0,
            "tau_evap_s": 45.0,
            "tau_plate_s": 20.0,
            "T_set": 25.0,
        }

        calibrated = apply_reduced_model_calibration(
            params,
            {
                "C1_scale": 2.0,
                "C2_scale": 1.5,
                "h1_scale": 0.8,
                "kq_scale": 0.7,
                "tau_evap_scale": 1.2,
                "tau_plate_scale": 1.1,
                "Cplate_scale": 2.5,
            },
        )

        self.assertEqual(params["C1"], 100.0)
        self.assertEqual(calibrated["C1"], 200.0)
        self.assertEqual(calibrated["C2"], 300.0)
        self.assertEqual(calibrated["h1_ref"], 8.0)
        self.assertEqual(calibrated["kq"], 0.7)
        self.assertEqual(calibrated["tau_evap_s"], 54.0)
        self.assertEqual(calibrated["tau_plate_s"], 22.0)
        self.assertEqual(calibrated["Cplate_scale"], 2.5)


if __name__ == "__main__":
    unittest.main()
