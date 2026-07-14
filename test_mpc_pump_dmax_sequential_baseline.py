import unittest

import run_mpc_sensitivity_60 as sensitivity


class MPCPumpDmaxSequentialBaselineTest(unittest.TestCase):
    def test_pump_dmax_scan_inherits_selected_sequential_baseline(self):
        peak = sensitivity.build_params(
            sensitivity.base_params_for_case("peak"),
            "final_pump_dmax_scan",
            0.5,
            "baseline",
            case="peak",
        )
        freq = sensitivity.build_params(
            sensitivity.base_params_for_case("freq"),
            "final_pump_dmax_scan",
            4.0,
            "baseline",
            case="freq",
        )

        self.assertEqual((peak.cv_band_half_width, freq.cv_band_half_width), (0.30, 0.45))
        self.assertEqual((peak.w_high_temp, freq.w_high_temp), (5e6, 5e7))
        self.assertEqual((peak.dmax_comp, freq.dmax_comp), (6000.0, 6000.0))
        self.assertEqual((peak.w_energy_comp, freq.w_energy_comp), (600.0, 300.0))
        self.assertEqual((peak.w_energy_pump, freq.w_energy_pump), (10000.0, 15000.0))
        self.assertEqual((peak.dmax_pump, freq.dmax_pump), (150.0, 1200.0))
        self.assertEqual((peak.terminal_cost_enabled, peak.w_terminal_temp), (True, 1e6))
        self.assertEqual((freq.terminal_cost_enabled, freq.w_terminal_temp), (True, 5e5))

    def test_off_uses_full_pump_speed_range_per_step(self):
        params = sensitivity.build_params(
            sensitivity.base_params_for_case("peak"),
            "final_pump_dmax_scan",
            "off",
            "baseline",
            case="peak",
        )
        self.assertEqual(params.dmax_pump, float(sensitivity.N_PUMP_MAX_RPM))


if __name__ == "__main__":
    unittest.main()
