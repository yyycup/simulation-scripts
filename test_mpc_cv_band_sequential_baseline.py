import unittest

import run_mpc_sensitivity_60 as sensitivity


class MPCCVBandSequentialBaselineTest(unittest.TestCase):
    def test_cv_band_scan_uses_wide_supplement_candidates(self):
        self.assertEqual(
            sensitivity.FINAL_CV_BAND_VALUES,
            (0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
        )

    def test_cv_band_scan_inherits_selected_sequential_baseline(self):
        peak = sensitivity.build_params(
            sensitivity.base_params_for_case("peak"),
            "final_cv_band_scan",
            0.1,
            "baseline",
            case="peak",
        )
        freq = sensitivity.build_params(
            sensitivity.base_params_for_case("freq"),
            "final_cv_band_scan",
            0.1,
            "baseline",
            case="freq",
        )

        self.assertEqual((peak.w_high_temp, peak.w_cold_temp), (5e6, 5e6))
        self.assertEqual((freq.w_high_temp, freq.w_cold_temp), (5e7, 5e7))
        self.assertEqual((peak.dmax_comp, peak.w_energy_comp), (6000.0, 600.0))
        self.assertEqual((freq.dmax_comp, freq.w_energy_comp), (6000.0, 300.0))
        self.assertEqual((peak.terminal_cost_enabled, peak.w_terminal_temp), (True, 1e6))
        self.assertEqual((freq.terminal_cost_enabled, freq.w_terminal_temp), (True, 5e5))
        self.assertEqual((peak.cv_band_half_width, freq.cv_band_half_width), (0.1, 0.1))


if __name__ == "__main__":
    unittest.main()
