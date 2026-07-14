import unittest

from mpc_evaporator_capacity_model import evaluate_capacity


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


if __name__ == "__main__":
    unittest.main()
