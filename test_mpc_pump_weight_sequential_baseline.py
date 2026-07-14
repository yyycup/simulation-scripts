import unittest

import run_mpc_sensitivity_60 as sensitivity


class MPCPumpWeightSequentialBaselineTest(unittest.TestCase):
    def test_pump_weight_scan_inherits_selected_sequential_baseline(self):
        peak = sensitivity.build_params(
            sensitivity.base_params_for_case("peak"),
            "final_pump_weight_scan",
            0.5,
            "baseline",
            case="peak",
        )
        freq = sensitivity.build_params(
            sensitivity.base_params_for_case("freq"),
            "final_pump_weight_scan",
            0.5,
            "baseline",
            case="freq",
        )

        self.assertEqual((peak.cv_band_half_width, freq.cv_band_half_width), (0.30, 0.45))
        self.assertEqual((peak.w_high_temp, peak.w_cold_temp), (5e6, 5e6))
        self.assertEqual((freq.w_high_temp, freq.w_cold_temp), (5e7, 5e7))
        self.assertEqual((peak.dmax_comp, peak.w_energy_comp), (6000.0, 600.0))
        self.assertEqual((freq.dmax_comp, freq.w_energy_comp), (6000.0, 300.0))
        self.assertEqual((peak.w_energy_pump, freq.w_energy_pump), (5000.0, 7500.0))
        self.assertEqual((peak.terminal_cost_enabled, peak.w_terminal_temp), (True, 1e6))
        self.assertEqual((freq.terminal_cost_enabled, freq.w_terminal_temp), (True, 5e5))


if __name__ == "__main__":
    unittest.main()
