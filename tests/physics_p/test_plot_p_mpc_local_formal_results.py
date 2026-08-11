import unittest
from pathlib import Path

import pandas as pd

from figures.gen_fig_p_mpc_local_formal import case_csv_paths, summarize_case


class PhysicsPMpcFormalPlotTest(unittest.TestCase):
    def test_case_paths_match_formal_run_layout(self):
        paths = case_csv_paths(Path("result_root"))

        self.assertEqual(
            paths["peak"],
            Path("result_root/peak/mpc/peak_physics_p_steps1280_horizon60.csv"),
        )
        self.assertEqual(
            paths["freq"],
            Path("result_root/freq/mpc/freq_physics_p_steps720_horizon45.csv"),
        )

    def test_summary_reports_tracking_energy_solve_and_domain_metrics(self):
        frame = pd.DataFrame(
            {
                "Average temperature": [25.0, 25.2],
                "Compressor command": [1000.0, 6000.0],
                "Pump command": [1600.0, 2200.0],
                "MPC solve time": [1.0, 6.0],
                "Cumulative energy consumption": [0.0, 0.1],
                "MPC_Solved": [True, True],
                "MPC prediction domain valid": [True, True],
                "MPC solve recovery used": [False, True],
                "MPC predicted minimum coolant temperature": [24.0, 15.2],
                "MPC coolant prediction domain violation": [0.0, 0.1],
            }
        )

        summary = summarize_case(frame, "peak")

        self.assertAlmostEqual(summary["temperature_mae_c"], 0.1)
        self.assertEqual(summary["temperature_min_c"], 25.0)
        self.assertEqual(summary["temperature_max_c"], 25.2)
        self.assertEqual(summary["energy_kwh"], 0.1)
        self.assertEqual(summary["compressor_off_rate"], 0.0)
        self.assertEqual(summary["compressor_startup_transition_rate"], 0.0)
        self.assertEqual(summary["compressor_low_speed_rate"], 0.5)
        self.assertEqual(summary["compressor_saturation_rate"], 0.5)
        self.assertEqual(summary["solve_success_rate"], 1.0)
        self.assertEqual(summary["prediction_domain_valid_rate"], 1.0)
        self.assertEqual(summary["predicted_coolant_min_c"], 15.2)
        self.assertEqual(summary["coolant_domain_violation_max_c"], 0.1)
        self.assertEqual(summary["solve_recovery_count"], 1)
        self.assertEqual(summary["deadline_exceed_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
