import unittest

import run_mpc_sensitivity_60 as sensitivity


class MPCCVWeightSequentialBaselineTest(unittest.TestCase):
    def test_cv_weight_scan_inherits_selected_compressor_settings_and_terminal_cost(self):
        peak = sensitivity.build_params(
            sensitivity.base_params_for_case("peak"),
            "final_cv_weight_scan",
            1e8,
            "baseline",
            case="peak",
        )
        freq = sensitivity.build_params(
            sensitivity.base_params_for_case("freq"),
            "final_cv_weight_scan",
            1e8,
            "baseline",
            case="freq",
        )

        self.assertEqual((peak.dmax_comp, peak.w_energy_comp), (6000.0, 600.0))
        self.assertEqual((freq.dmax_comp, freq.w_energy_comp), (6000.0, 300.0))
        self.assertEqual((peak.terminal_cost_enabled, peak.w_terminal_temp), (True, 1e6))
        self.assertEqual((freq.terminal_cost_enabled, freq.w_terminal_temp), (True, 5e5))


if __name__ == "__main__":
    unittest.main()
