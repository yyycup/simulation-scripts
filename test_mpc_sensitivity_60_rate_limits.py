import sys
import unittest

import mpc_flow_direction_strategies as flow_mpc
from mpc_flow_direction_strategies import freq_mpc_params, peak_mpc_params
import run_mpc_sensitivity_60 as sensitivity
from run_mpc_sensitivity_60 import apply_rate_limit_mode


class MPCSensitivity60RateLimitTest(unittest.TestCase):
    def test_latest_comp_rate_timeseries_dmax_values_are_scene_specific(self):
        peak_params = apply_rate_limit_mode(peak_mpc_params, "dmax_240rpmps")
        freq_params = apply_rate_limit_mode(freq_mpc_params, "dmax_480rpmps")

        self.assertEqual(peak_params.dmax_comp, 240.0 * flow_mpc.SIM_DT)
        self.assertEqual(freq_params.dmax_comp, 480.0 * flow_mpc.SIM_DT)
        self.assertEqual(peak_params.w_dcomp, 0.0)
        self.assertEqual(freq_params.w_dcomp, 0.0)

    def test_default_sweep_type_runs_adaptive_dmax_w_energy_comp(self):
        original_argv = sys.argv
        sys.argv = ["run_mpc_sensitivity_60.py"]
        try:
            args = sensitivity.parse_args()
        finally:
            sys.argv = original_argv

        self.assertEqual(args.sweep_type, "adaptive_dmax_w_energy_comp")


if __name__ == "__main__":
    unittest.main()



