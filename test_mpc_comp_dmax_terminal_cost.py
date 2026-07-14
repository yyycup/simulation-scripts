import unittest

import run_mpc_sensitivity_60 as sensitivity


class MPCCompDmaxTerminalCostTest(unittest.TestCase):
    def test_comp_dmax_scan_explicitly_uses_final_terminal_cost(self):
        peak = sensitivity.build_params(
            sensitivity.base_params_for_case("peak"),
            "comp_dmax_scan",
            2400.0,
            "baseline",
            case="peak",
        )
        freq = sensitivity.build_params(
            sensitivity.base_params_for_case("freq"),
            "comp_dmax_scan",
            6000.0,
            "baseline",
            case="freq",
        )

        self.assertTrue(peak.terminal_cost_enabled)
        self.assertEqual(peak.w_terminal_temp, 1e6)
        self.assertTrue(freq.terminal_cost_enabled)
        self.assertEqual(freq.w_terminal_temp, 5e5)


if __name__ == "__main__":
    unittest.main()
