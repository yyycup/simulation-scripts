import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import thermal_system

from run_p_mpc_operational import (
    DEFAULT_OPERATIONAL_P_ARTIFACT,
    OPERATIONAL_HORIZON_STEPS_BY_SCENE,
    operational_profile_for_scene,
    parse_args,
    run_operational,
)
from thermal_batch_config import INITIAL_TEMP_C


class PhysicsPMpcOperationalTest(unittest.TestCase):
    def test_operational_runner_explicitly_uses_validated_25c_thermal_initial_state(self):
        with TemporaryDirectory() as tmp:
            with patch(
                "run_p_mpc_operational._simulate_case_with_displacement",
                side_effect=RuntimeError("stop after simulator boundary"),
            ) as simulate:
                with self.assertRaisesRegex(RuntimeError, "simulator boundary"):
                    run_operational(
                        scenes=["peak"],
                        steps=1,
                        output_root=tmp,
                        force=True,
                    )

        self.assertEqual(
            simulate.call_args.kwargs.get("initial_thermal_temp_c"),
            25.0,
        )
        self.assertEqual(INITIAL_TEMP_C, 25.0)

    def test_peak_profile_enables_online_temperature_bias_compensation(self):
        profile = operational_profile_for_scene("peak")

        self.assertEqual(profile["physics_p_temp_bias_gain"], 2.0)
        self.assertEqual(
            profile["mpc_overrides"]["physics_p_temp_bias_gain"],
            2.0,
        )

    def test_peak_profile_can_override_temperature_bias_gain(self):
        profile = operational_profile_for_scene(
            "peak",
            physics_p_temp_bias_gain=2.0,
        )

        self.assertEqual(profile["physics_p_temp_bias_gain"], 2.0)
        self.assertEqual(
            profile["mpc_overrides"]["physics_p_temp_bias_gain"],
            2.0,
        )

    def test_displacement_variant_scales_plant_only_inside_context(self):
        try:
            from run_p_mpc_operational import (
                displacement_scale_from_artifact,
                patched_plant_compressor_displacement,
            )
        except ImportError as exc:
            self.fail(f"missing displacement synchronization helpers: {exc}")

        original = thermal_system.V_disp_m3_per_rev
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "variant.json"
            path.write_text(
                json.dumps(
                    {"thermal": {"compressor_displacement_scale": 0.8}}
                ),
                encoding="utf-8",
            )
            scale = displacement_scale_from_artifact(path)
            self.assertEqual(scale, 0.8)
            with patched_plant_compressor_displacement(scale):
                self.assertAlmostEqual(
                    thermal_system.V_disp_m3_per_rev,
                    0.8 * original,
                )
        self.assertEqual(thermal_system.V_disp_m3_per_rev, original)

    def test_default_artifact_is_the_original_displacement_p_model(self):
        from run_p_mpc_operational import displacement_scale_from_artifact

        self.assertEqual(
            DEFAULT_OPERATIONAL_P_ARTIFACT.name,
            "physics_p_heat_generation_corrected.json",
        )
        self.assertIn(
            "thermal_bias_correction_v3",
            DEFAULT_OPERATIONAL_P_ARTIFACT.parts,
        )
        self.assertTrue(DEFAULT_OPERATIONAL_P_ARTIFACT.is_file())
        self.assertEqual(
            displacement_scale_from_artifact(DEFAULT_OPERATIONAL_P_ARTIFACT),
            1.0,
        )

    def test_peak_profile_uses_validated_short_horizon_settings(self):
        profile = operational_profile_for_scene("peak")

        self.assertEqual(
            OPERATIONAL_HORIZON_STEPS_BY_SCENE["peak"],
            14,
        )
        self.assertEqual(
            profile["horizon_steps"],
            OPERATIONAL_HORIZON_STEPS_BY_SCENE["peak"],
        )
        self.assertEqual(
            profile["mpc_overrides"],
            {"w_dcomp": 0.0, "physics_p_temp_bias_gain": 2.0},
        )
        self.assertEqual(profile["dmax_comp_rpm_per_step"], 6000.0)
        self.assertEqual(profile["w_energy_comp"], 600.0)
        self.assertEqual(profile["w_dcomp"], 0.0)

    def test_peak_profile_can_override_horizon_for_local_diagnosis(self):
        profile = operational_profile_for_scene(
            "peak",
            horizon_steps_override=10,
        )

        self.assertEqual(profile["horizon_steps"], 10)

        with self.assertRaisesRegex(ValueError, "horizon_steps_override"):
            operational_profile_for_scene(
                "peak",
                horizon_steps_override=1,
            )

    def test_frequency_profile_keeps_separate_energy_weight(self):
        profile = operational_profile_for_scene("freq")

        self.assertEqual(profile["horizon_steps"], 12)
        self.assertIsNone(profile["mpc_overrides"])
        self.assertEqual(profile["dmax_comp_rpm_per_step"], 6000.0)
        self.assertEqual(profile["w_energy_comp"], 300.0)
        self.assertEqual(profile["w_dcomp"], 0.001)

    def test_peak_profile_can_override_only_compressor_move_cost(self):
        profile = operational_profile_for_scene("peak", w_dcomp=0.1)

        self.assertEqual(
            profile["mpc_overrides"],
            {"w_dcomp": 0.1, "physics_p_temp_bias_gain": 2.0},
        )
        self.assertEqual(profile["w_dcomp"], 0.1)
        self.assertEqual(profile["dmax_comp_rpm_per_step"], 6000.0)
        self.assertEqual(profile["w_energy_comp"], 600.0)

    def test_peak_profile_can_override_terminal_temperature_weight(self):
        profile = operational_profile_for_scene(
            "peak",
            w_terminal_temp=3.0e5,
        )

        self.assertEqual(
            profile["mpc_overrides"],
            {
                "w_dcomp": 0.0,
                "physics_p_temp_bias_gain": 2.0,
                "w_terminal_temp": 3.0e5,
            },
        )
        self.assertEqual(profile["w_terminal_temp"], 3.0e5)

        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "w_terminal_temp"):
                    operational_profile_for_scene(
                        "peak",
                        w_terminal_temp=value,
                    )

    def test_operational_profile_rejects_invalid_move_cost(self):
        for value in (-0.1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "w_dcomp"):
                    operational_profile_for_scene("peak", w_dcomp=value)

    def test_cli_accepts_explicit_compressor_move_cost(self):
        original_argv = sys.argv
        sys.argv = ["run_p_mpc_operational.py", "--scenes", "peak", "--w-dcomp", "0.1"]
        try:
            args = parse_args()
        finally:
            sys.argv = original_argv

        self.assertEqual(args.w_dcomp, 0.1)

    def test_peak_profile_adds_quadratic_move_cost_to_zero_linear_dcost(self):
        profile = operational_profile_for_scene(
            "peak",
            w_dcomp_quadratic=300.0,
        )

        self.assertEqual(
            profile["mpc_overrides"],
            {
                "w_dcomp": 0.0,
                "physics_p_temp_bias_gain": 2.0,
                "w_dcomp_quadratic": 300.0,
            },
        )
        self.assertEqual(profile["w_dcomp"], 0.0)
        self.assertEqual(profile["w_dcomp_quadratic"], 300.0)

        explicit = operational_profile_for_scene(
            "peak",
            w_dcomp=0.1,
            w_dcomp_quadratic=300.0,
        )
        self.assertEqual(
            explicit["mpc_overrides"],
            {
                "w_dcomp": 0.1,
                "physics_p_temp_bias_gain": 2.0,
                "w_dcomp_quadratic": 300.0,
            },
        )

    def test_operational_profile_rejects_invalid_quadratic_move_cost(self):
        for value in (-0.1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "w_dcomp_quadratic"):
                    operational_profile_for_scene(
                        "peak",
                        w_dcomp_quadratic=value,
                    )

    def test_cli_accepts_explicit_quadratic_move_cost(self):
        original_argv = sys.argv
        sys.argv = [
            "run_p_mpc_operational.py",
            "--scenes",
            "peak",
            "--w-dcomp-quadratic",
            "300",
        ]
        try:
            args = parse_args()
        finally:
            sys.argv = original_argv

        self.assertEqual(args.w_dcomp_quadratic, 300.0)

    def test_peak_profile_can_enable_lightweight_command_filter(self):
        profile = operational_profile_for_scene(
            "peak",
            comp_command_filter_alpha=0.5,
        )

        self.assertEqual(
            profile["mpc_overrides"],
            {
                "w_dcomp": 0.0,
                "physics_p_temp_bias_gain": 2.0,
                "comp_command_filter_alpha": 0.5,
            },
        )
        self.assertEqual(profile["comp_command_filter_alpha"], 0.5)

    def test_operational_profile_rejects_invalid_command_filter_alpha(self):
        for value in (0.0, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "comp_command_filter_alpha",
                ):
                    operational_profile_for_scene(
                        "peak",
                        comp_command_filter_alpha=value,
                    )

    def test_cli_accepts_explicit_command_filter_alpha(self):
        original_argv = sys.argv
        sys.argv = [
            "run_p_mpc_operational.py",
            "--scenes",
            "peak",
            "--comp-command-filter-alpha",
            "0.5",
        ]
        try:
            args = parse_args()
        finally:
            sys.argv = original_argv

        self.assertEqual(args.comp_command_filter_alpha, 0.5)

    def test_cli_accepts_explicit_terminal_temperature_weight(self):
        original_argv = sys.argv
        sys.argv = [
            "run_p_mpc_operational.py",
            "--scenes",
            "peak",
            "--w-terminal-temp",
            "300000",
        ]
        try:
            args = parse_args()
        finally:
            sys.argv = original_argv

        self.assertEqual(args.w_terminal_temp, 3.0e5)

    def test_cli_accepts_explicit_horizon_override(self):
        original_argv = sys.argv
        sys.argv = [
            "run_p_mpc_operational.py",
            "--scenes",
            "peak",
            "--horizon-steps",
            "10",
        ]
        try:
            args = parse_args()
        finally:
            sys.argv = original_argv

        self.assertEqual(args.horizon_steps, 10)

    def test_unknown_scene_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported scene"):
            operational_profile_for_scene("unknown")


if __name__ == "__main__":
    unittest.main()
