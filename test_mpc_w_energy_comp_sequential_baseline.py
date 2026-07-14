import unittest

import run_mpc_sensitivity_60 as sensitivity


class MPCCompressorWeightSequentialBaselineTest(unittest.TestCase):
    def test_compressor_weight_scan_freezes_selected_dmax_and_terminal_cost(self):
        peak = sensitivity.build_params(
            sensitivity.base_params_for_case("peak"),
            "w_energy_comp",
            0.5,
            "baseline",
            case="peak",
        )
        freq = sensitivity.build_params(
            sensitivity.base_params_for_case("freq"),
            "w_energy_comp",
            0.5,
            "baseline",
            case="freq",
        )

        self.assertEqual(peak.dmax_comp, 6000.0)
        self.assertTrue(peak.terminal_cost_enabled)
        self.assertEqual(peak.w_terminal_temp, 1e6)
        self.assertEqual(freq.dmax_comp, 6000.0)
        self.assertTrue(freq.terminal_cost_enabled)
        self.assertEqual(freq.w_terminal_temp, 5e5)


if __name__ == "__main__":
    unittest.main()
