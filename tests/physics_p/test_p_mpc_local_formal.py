import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


import pandas as pd
import experiments.physics_p.tuning.run_p_mpc_local_formal as formal_runner
from mpc_predictor_selection import CANDIDATE_B, PHYSICS_P
from experiments.physics_p.tuning.run_p_mpc_local_formal import (
    FORMAL_LOCAL_CASES,
    FORMAL_PROGRESS_INTERVAL_STEPS,
    formal_local_cases,
    formal_scene_control_profiles,
    parse_progress_message,
)
from run_p_mpc_operational import DEFAULT_OPERATIONAL_P_ARTIFACT


class PhysicsPMpcLocalFormalTest(unittest.TestCase):
    def test_default_cases_use_formal_horizons_and_full_fallback_lengths(self):
        self.assertEqual(
            FORMAL_LOCAL_CASES,
            (
                {"scene": "peak", "horizon": 60, "steps": 1280},
                {"scene": "freq", "horizon": 45, "steps": 720},
            ),
        )

    def test_step_counts_can_be_shortened_without_changing_horizons(self):
        self.assertEqual(
            formal_local_cases(peak_steps=3, freq_steps=4),
            (
                {"scene": "peak", "horizon": 60, "steps": 3},
                {"scene": "freq", "horizon": 45, "steps": 4},
            ),
        )

    def test_formal_scene_profiles_record_frozen_runtime_controls(self):
        profiles = formal_scene_control_profiles(peak_steps=2, freq_steps=3)

        peak = profiles["peak"]
        self.assertEqual(peak["simulation_steps"], 2)
        self.assertEqual(peak["prediction_horizon_steps"], 60)
        self.assertEqual(peak["controller_forecast_profile_steps"], 61)
        self.assertEqual(peak["initial_thermal_state_c"], 25.0)
        self.assertEqual(peak["cv_band_half_width_c"], 0.30)
        self.assertEqual(peak["compressor_dmax_rpm_per_step"], 6000.0)
        self.assertEqual(peak["pump_dmax_rpm_per_step"], 300.0)
        self.assertEqual(peak["w_energy_comp"], 600.0)
        self.assertEqual(peak["w_energy_pump"], 10000.0)
        self.assertTrue(peak["terminal_cost_enabled"])
        self.assertEqual(peak["w_terminal_temp"], 1e6)
        self.assertFalse(peak["predictor_specific_mpc_overrides"])

        freq = profiles["freq"]
        self.assertEqual(freq["simulation_steps"], 3)
        self.assertEqual(freq["prediction_horizon_steps"], 45)
        self.assertEqual(freq["controller_forecast_profile_steps"], 46)
        self.assertEqual(freq["cv_band_half_width_c"], 0.45)
        self.assertEqual(freq["compressor_dmax_rpm_per_step"], 6000.0)
        self.assertEqual(freq["pump_dmax_rpm_per_step"], 600.0)
        self.assertEqual(freq["w_energy_comp"], 300.0)
        self.assertEqual(freq["w_energy_pump"], 15000.0)
        self.assertTrue(freq["terminal_cost_enabled"])
        self.assertEqual(freq["w_terminal_temp"], 5e5)
        self.assertFalse(freq["predictor_specific_mpc_overrides"])

    def test_progress_messages_update_machine_readable_step_counts(self):
        self.assertEqual(FORMAL_PROGRESS_INTERVAL_STEPS, 10)
        self.assertEqual(
            parse_progress_message(
                "PROGRESS mpc ?? ?? 120/720 t=595s elapsed=1.0s"
            ),
            (120, 720),
        )
        self.assertIsNone(parse_progress_message("FORMAL_LOCAL_START"))

    def test_explicit_candidate_b_uses_formal_scene_horizons_and_records_predictor(self):
        calls = []

        def fake_run_comparison(**kwargs):
            calls.append(kwargs)

        with patch.object(
            formal_runner,
            "run_comparison",
            side_effect=fake_run_comparison,
        ):
            with TemporaryDirectory(dir=Path("C:/tmp")) as tmp:
                output_root = Path(tmp) / "candidate_b"
                status_path = formal_runner.run_local_formal(
                    output_root=output_root,
                    predictor=CANDIDATE_B,
                    peak_steps=2,
                    freq_steps=2,
                )
                status = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(status["status"], "COMPLETE")
        self.assertEqual(status["predictor"], CANDIDATE_B)
        self.assertEqual(status["experiment_scope"], "native_controller_candidate")
        self.assertEqual(status["resolved_compressor_command_lower_rpm"], 1000.0)
        self.assertEqual(status["resolved_compressor_command_upper_rpm"], 6000.0)
        provenance = status["run_provenance"]
        self.assertIsNone(provenance["physics_p_artifact"])
        source_hash = provenance["source_files"][
            "experiments/physics_p/tuning/run_p_mpc_local_formal.py"
        ]["sha256"]
        self.assertEqual(len(source_hash), 64)
        self.assertEqual(provenance["predictor"], CANDIDATE_B)
        self.assertEqual(
            status["formal_scene_profiles"],
            formal_scene_control_profiles(peak_steps=2, freq_steps=2),
        )
        self.assertEqual(
            [
                (call["scenes"], call["predictors"], call["horizon"])
                for call in calls
            ],
            [
                (("peak",), (CANDIDATE_B,), 60),
                (("freq",), (CANDIDATE_B,), 45),
            ],
        )
        self.assertEqual(
            [call["output_root"] for call in calls],
            [output_root / "peak", output_root / "freq"],
        )
        self.assertEqual(
            [call["forecast_profile_steps"] for call in calls], [61, 46]
        )

    def test_explicit_physics_p_records_native_bounds(self):
        calls = []

        def fake_run_comparison(**kwargs):
            calls.append(kwargs)
            scene = kwargs["scenes"][0]
            fake_root = Path(kwargs["output_root"])
            fake_root.mkdir(parents=True, exist_ok=True)
            summary_path = fake_root / "scene_summary.csv"
            pd.DataFrame(
                [{"scene": scene, "predictor": kwargs["predictors"][0]}]
            ).to_csv(summary_path, index=False, encoding="utf-8-sig")
            return summary_path, fake_root / "pairwise.csv"

        with patch.object(
            formal_runner,
            "run_comparison",
            side_effect=fake_run_comparison,
        ):
            with TemporaryDirectory(dir=Path("C:/tmp")) as tmp:
                output_root = Path(tmp) / "physics_p"
                status_path = formal_runner.run_local_formal(
                    output_root=output_root,
                    predictor=PHYSICS_P,
                    artifact_path=DEFAULT_OPERATIONAL_P_ARTIFACT,
                    peak_steps=2,
                    freq_steps=2,
                )
                status = json.loads(status_path.read_text(encoding="utf-8"))
                combined_summary = pd.read_csv(
                    status["formal_summary_csv"], encoding="utf-8-sig"
                )

        self.assertEqual(status["status"], "COMPLETE")
        self.assertEqual(status["predictor"], PHYSICS_P)
        self.assertEqual(status["experiment_scope"], "native_controller_candidate")
        self.assertEqual(status["resolved_compressor_command_lower_rpm"], 300.0)
        self.assertEqual(status["resolved_compressor_command_upper_rpm"], 6000.0)
        provenance = status["run_provenance"]
        artifact_record = provenance["physics_p_artifact"]
        self.assertTrue(artifact_record["exists"])
        self.assertEqual(len(artifact_record["sha256"]), 64)
        self.assertEqual(
            provenance["initialization_and_solver_semantics"][
                "physics_p_each_cycle_time_shift"
            ],
            0,
        )
        self.assertIn(
            "cv_temp uses only the scalar measurement",
            provenance["initialization_and_solver_semantics"][
                "physics_p_first_cycle"
            ],
        )
        self.assertEqual(set(combined_summary["scene"]), {"peak", "freq"})
        self.assertEqual(
            [call["predictors"] for call in calls],
            [(PHYSICS_P,), (PHYSICS_P,)],
        )
        self.assertEqual(
            [call["output_root"] for call in calls],
            [output_root / "peak", output_root / "freq"],
        )

    def test_strict_ablation_records_common_scope_and_propagates_flag(self):
        calls = []

        def fake_run_comparison(**kwargs):
            calls.append(kwargs)

        with patch.object(
            formal_runner,
            "run_comparison",
            side_effect=fake_run_comparison,
        ):
            with TemporaryDirectory(dir=Path("C:/tmp")) as tmp:
                output_root = Path(tmp) / "strict_physics_p"
                status_path = formal_runner.run_local_formal(
                    output_root=output_root,
                    predictor=PHYSICS_P,
                    artifact_path=DEFAULT_OPERATIONAL_P_ARTIFACT,
                    peak_steps=2,
                    freq_steps=2,
                    strict_predictor_ablation=True,
                )
                status = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(status["status"], "COMPLETE")
        self.assertEqual(status["experiment_scope"], "strict_predictor_ablation")
        self.assertTrue(status["strict_predictor_ablation"])
        self.assertEqual(status["resolved_compressor_command_lower_rpm"], 1000.0)
        self.assertEqual(status["resolved_compressor_command_upper_rpm"], 6000.0)
        self.assertEqual(
            [call["strict_predictor_ablation"] for call in calls],
            [True, True],
        )
        solver_semantics = status["run_provenance"][
            "initialization_and_solver_semantics"
        ]
        self.assertEqual(
            solver_semantics["candidate_b_solver_behavior"],
            "common strict-ablation recovery contract",
        )

    def test_stdout_failure_does_not_turn_completed_run_into_failure(self):
        def fake_run_comparison(**kwargs):
            kwargs["log_func"]("PROGRESS mpc scene flow 2/2 t=5s elapsed=1.0s")

        with patch.object(
            formal_runner,
            "run_comparison",
            side_effect=fake_run_comparison,
        ), patch("builtins.print", side_effect=OSError(22, "closed stdout")):
            with TemporaryDirectory(dir=Path("C:/tmp")) as tmp:
                output_root = Path(tmp) / "stdout_failure"
                status_path = formal_runner.run_local_formal(
                    output_root=output_root,
                    predictor=PHYSICS_P,
                    artifact_path=DEFAULT_OPERATIONAL_P_ARTIFACT,
                    peak_steps=2,
                    freq_steps=2,
                )
                status = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(status["status"], "COMPLETE")
        self.assertEqual(status["completed_scenes"], ["peak", "freq"])

    def test_predictor_selection_is_required(self):
        with TemporaryDirectory(dir=Path("C:/tmp")) as tmp:
            with self.assertRaisesRegex(ValueError, "explicit predictor"):
                formal_runner.run_local_formal(
                    output_root=Path(tmp) / "missing_predictor",
                    peak_steps=2,
                    freq_steps=2,
                )



if __name__ == "__main__":
    unittest.main()
