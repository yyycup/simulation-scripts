from copy import deepcopy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from experiments.physics_p.evaluation.compare_mpc_predictor_formal_results import (
    ARCHIVE_ID,
    CURRENT_CANDIDATE_B_ID,
    PHYSICS_P_ID,
    build_current_operational_pairwise_summary,
    build_result_specs,
    historical_comparability_reasons,
    plot_comparison,
    summarize_case,
    validate_current_control_profiles,
    validate_formal_status_payload,
)


class FormalPredictorComparisonTest(unittest.TestCase):
    def test_result_specs_use_formal_current_layout_and_historical_single_flow(self):
        specs = build_result_specs(
            Path("p_root"),
            Path("b_root"),
            Path("archive_root"),
        )
        indexed = {(item["scene"], item["model_id"]): item for item in specs}

        self.assertEqual(
            indexed[("peak", PHYSICS_P_ID)]["csv_path"],
            Path("p_root/peak/mpc/peak_physics_p_steps1280_horizon60.csv"),
        )
        self.assertEqual(
            indexed[("freq", CURRENT_CANDIDATE_B_ID)]["csv_path"],
            Path("b_root/freq/mpc/freq_candidate_b_steps720_horizon45.csv"),
        )
        self.assertEqual(
            indexed[("peak", ARCHIVE_ID)]["csv_path"],
            Path(
                "archive_root/mpc/"
                "\u8c03\u5cf0_\u5355\u5411_mpc_\u5b8c\u6574.csv"
            ),
        )
        self.assertEqual(
            indexed[("freq", ARCHIVE_ID)]["comparison_tier"],
            "historical_context_only",
        )

    def test_summary_reports_tracking_energy_speed_and_solver_metrics(self):
        frame = pd.DataFrame(
            {
                "Time": [0.0, 5.0, 10.0, 15.0],
                "Average temperature": [25.0, 25.2, 24.8, 25.1],
                "Maximum temperature difference": [0.1, 0.2, 0.3, 0.4],
                "Coolant temperature": [25.0, 24.9, 24.8, 24.7],
                "Compressor command": [300.0, 650.0, 1500.0, 6000.0],
                "Pump command": [1600.0, 1700.0, 1800.0, 1900.0],
                "Total power": [0.0, 0.1, 0.2, 0.3],
                "Cumulative energy consumption": [0.0, 0.01, 0.04, 0.1],
                "MPC solve time": [1.0, 2.0, 6.0, 4.0],
                "MPC_Solved": [True, True, True, True],
                "MPC solve recovery used": [False, True, False, False],
                "MPC prediction domain valid": [True, True, True, True],
                "MPC predicted minimum coolant temperature": [24.0, 23.0, 22.0, 21.0],
                "MPC coolant prediction domain violation": [0.0, 0.0, 0.0, 0.0],
            }
        )

        summary = summarize_case(
            frame,
            scene="peak",
            model_id=PHYSICS_P_ID,
            comparison_tier="current_operational_candidate",
            source_csv=Path("physics_p.csv"),
        )

        self.assertAlmostEqual(summary["temperature_mae_c"], 0.125)
        self.assertAlmostEqual(summary["temperature_rmse_c"], 0.15)
        self.assertEqual(summary["temperature_min_c"], 24.8)
        self.assertEqual(summary["temperature_max_c"], 25.2)
        self.assertEqual(summary["energy_kwh"], 0.1)
        self.assertEqual(summary["compressor_off_rate"], 0.25)
        self.assertEqual(summary["compressor_startup_transition_rate"], 0.25)
        self.assertEqual(summary["compressor_low_speed_rate"], 0.25)
        self.assertEqual(summary["compressor_saturation_rate"], 0.25)
        self.assertEqual(summary["solve_success_rate"], 1.0)
        self.assertEqual(summary["solve_recovery_count"], 1)
        self.assertEqual(summary["deadline_exceed_rate"], 0.25)
        self.assertEqual(summary["predicted_coolant_min_c"], 21.0)
        self.assertEqual(summary["initial_coolant_c"], 25.0)

        blank_optional = frame.copy()
        blank_optional["MPC prediction domain valid"] = [None] * len(frame)
        blank_optional["MPC solve recovery used"] = [None] * len(frame)
        blank_summary = summarize_case(
            blank_optional,
            scene="peak",
            model_id=CURRENT_CANDIDATE_B_ID,
            comparison_tier="current_operational_candidate",
            source_csv=Path("candidate_b.csv"),
        )
        self.assertTrue(pd.isna(blank_summary["prediction_domain_valid_rate"]))
        self.assertEqual(blank_summary["solve_recovery_count"], 0)

    def test_pairwise_summary_compares_only_current_p_and_current_candidate_b(self):
        summary = pd.DataFrame(
            [
                {
                    "scene": "peak",
                    "model_id": PHYSICS_P_ID,
                    "comparison_tier": "current_operational_candidate",
                    "temperature_mae_c": 0.2,
                    "temperature_final_c": 25.1,
                    "energy_kwh": 0.3,
                    "solve_time_mean_s": 2.0,
                    "solve_success_rate": 1.0,
                },
                {
                    "scene": "peak",
                    "model_id": CURRENT_CANDIDATE_B_ID,
                    "comparison_tier": "current_operational_candidate",
                    "temperature_mae_c": 0.3,
                    "temperature_final_c": 25.2,
                    "energy_kwh": 0.2,
                    "solve_time_mean_s": 1.0,
                    "solve_success_rate": 0.99,
                },
                {
                    "scene": "peak",
                    "model_id": ARCHIVE_ID,
                    "comparison_tier": "historical_context_only",
                    "temperature_mae_c": 0.05,
                    "temperature_final_c": 25.0,
                    "energy_kwh": 0.1,
                    "solve_time_mean_s": 0.5,
                    "solve_success_rate": 1.0,
                },
            ]
        )

        pairwise = build_current_operational_pairwise_summary(summary)

        self.assertEqual(len(pairwise), 1)
        self.assertEqual(
            pairwise.iloc[0]["comparison_scope"],
            "operational_candidate_not_predictor_only",
        )
        self.assertAlmostEqual(
            pairwise.iloc[0]["p_minus_b_temperature_mae_c"], -0.1
        )
        self.assertAlmostEqual(pairwise.iloc[0]["p_minus_b_energy_kwh"], 0.1)
        self.assertAlmostEqual(
            pairwise.iloc[0]["p_over_b_solve_time_ratio"], 2.0
        )

    def test_historical_provenance_explicitly_blocks_fair_interpretation(self):
        reasons = historical_comparability_reasons()
        joined = " ".join(reasons)

        self.assertIn("35", joined)
        self.assertIn("2000", joined)
        self.assertIn("DMAX", joined)
        self.assertIn("context", joined.lower())

    def test_formal_status_requires_complete_matching_predictor(self):
        validate_formal_status_payload(
            {"status": "COMPLETE", "predictor": "physics_p", "completed_scenes": ["peak", "freq"]},
            expected_predictor="physics_p",
        )
        with self.assertRaises(RuntimeError):
            validate_formal_status_payload(
                {"status": "COMPLETE", "predictor": "physics_p"},
                expected_predictor="physics_p",
            )

        with self.assertRaises(RuntimeError):
            validate_formal_status_payload(
                {"status": "RUNNING", "predictor": "physics_p"},
                expected_predictor="physics_p",
            )
        with self.assertRaises(RuntimeError):
            validate_formal_status_payload(
                {"status": "COMPLETE", "predictor": "candidate_b"},
                expected_predictor="physics_p",
            )

    def test_current_control_profiles_must_match_and_use_formal_lengths(self):
        profiles = {
            "peak": {
                "simulation_steps": 1280,
                "prediction_horizon_steps": 60,
                "initial_thermal_state_c": 25.0,
                "predictor_specific_mpc_overrides": False,
                "compressor_dmax_rpm_per_step": 6000.0,
            },
            "freq": {
                "simulation_steps": 720,
                "prediction_horizon_steps": 45,
                "initial_thermal_state_c": 25.0,
                "predictor_specific_mpc_overrides": False,
                "compressor_dmax_rpm_per_step": 6000.0,
            },
        }
        p_status = {"formal_scene_profiles": profiles}
        b_status = {"formal_scene_profiles": deepcopy(profiles)}
        self.assertEqual(
            validate_current_control_profiles(p_status, b_status),
            profiles,
        )

        mismatched = deepcopy(profiles)
        mismatched["peak"]["compressor_dmax_rpm_per_step"] = 500.0
        with self.assertRaises(RuntimeError):
            validate_current_control_profiles(
                p_status, {"formal_scene_profiles": mismatched}
            )

        wrong_horizon = deepcopy(profiles)
        wrong_horizon["freq"]["prediction_horizon_steps"] = 6
        with self.assertRaises(RuntimeError):
            validate_current_control_profiles(
                {"formal_scene_profiles": wrong_horizon},
                {"formal_scene_profiles": deepcopy(wrong_horizon)},
            )

    def test_plot_writes_png_and_pdf_for_all_three_result_families(self):
        frame = pd.DataFrame(
            {
                "Time": [0.0, 5.0],
                "Average temperature": [25.0, 25.1],
                "Cumulative energy consumption": [0.0, 0.01],
                "Compressor command": [1000.0, 1500.0],
                "MPC solve time": [1.0, 2.0],
            }
        )
        frames = {
            (scene, model_id): frame
            for scene in ("peak", "freq")
            for model_id in (
                PHYSICS_P_ID,
                CURRENT_CANDIDATE_B_ID,
                ARCHIVE_ID,
            )
        }
        with TemporaryDirectory(dir=Path("C:/tmp")) as tmp:
            png_path, pdf_path = plot_comparison(frames, Path(tmp))
            self.assertTrue(png_path.is_file())
            self.assertTrue(pdf_path.is_file())
            self.assertGreater(png_path.stat().st_size, 0)
            self.assertGreater(pdf_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
