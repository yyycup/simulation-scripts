import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from td3_btms.env import (
    BTMSTd3Env,
    build_current_profile,
    canonical_scene,
    compute_reward,
    map_action_to_rpm,
    profile_sha256,
)
from run_td3_training import create_run_directory, write_metadata


class CurrentProfileTests(unittest.TestCase):
    def test_scene_names_are_strict(self):
        self.assertEqual(canonical_scene("peak"), "peak")
        self.assertEqual(canonical_scene(" FREQ "), "freq")
        with self.assertRaisesRegex(ValueError, "scene must be 'peak' or 'freq'"):
            canonical_scene("调峰")

    def test_peak_profile_uses_existing_560_amp_boundary(self):
        profile = build_current_profile("peak", dt=5.0, max_steps=3)
        np.testing.assert_array_equal(profile, np.array([560.0, 560.0, 560.0]))

    def test_injected_profile_is_validated_and_trimmed(self):
        profile = build_current_profile(
            "freq",
            dt=5.0,
            current_profile=[10.0, 20.0, 30.0],
            max_steps=2,
        )
        np.testing.assert_array_equal(profile, np.array([10.0, 20.0]))
        self.assertEqual(profile_sha256(profile), profile_sha256(profile.copy()))
        with self.assertRaisesRegex(ValueError, "finite"):
            build_current_profile("peak", current_profile=[1.0, np.nan])

    def test_frequency_file_must_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing_td3_agc.csv"
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                build_current_profile("freq", agc_data_file=missing, max_steps=2)


class ActionAndResetTests(unittest.TestCase):
    def test_action_endpoints_map_to_existing_actuator_bounds(self):
        self.assertEqual(
            map_action_to_rpm(np.array([-1.0, -1.0])),
            (300.0, 1600.0),
        )
        self.assertEqual(
            map_action_to_rpm(np.array([1.0, 1.0])),
            (6000.0, 4800.0),
        )

    def test_action_is_clipped_before_mapping(self):
        self.assertEqual(
            map_action_to_rpm(np.array([-2.0, 2.0])),
            (300.0, 4800.0),
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            map_action_to_rpm(np.array([0.0]))

    def test_reset_returns_nine_finite_float32_observations(self):
        env = BTMSTd3Env(scene="peak", current_profile=[560.0, 560.0])
        observation, info = env.reset(seed=7)

        self.assertEqual(observation.shape, (9,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(observation)))
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info["scene"], "peak")
        self.assertEqual(info["step_index"], 0)
        self.assertAlmostEqual(info["mean_soc"], 0.95)

    def test_frequency_reset_uses_regulation_soc(self):
        env = BTMSTd3Env(scene="freq", current_profile=[0.0, 10.0])
        first, _ = env.reset(seed=11)
        second, _ = env.reset(seed=11)

        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(env.last_info["mean_soc"], 0.55)


class RewardAndStepTests(unittest.TestCase):
    def test_reward_penalizes_temperature_spread_and_high_temperature(self):
        safe, _ = compute_reward(
            mean_temp_c=25.0,
            max_temp_c=26.0,
            delta_temp_c=0.4,
            total_power_w=0.0,
        )
        unsafe, unsafe_terms = compute_reward(
            mean_temp_c=25.0,
            max_temp_c=29.0,
            delta_temp_c=1.0,
            total_power_w=0.0,
        )

        self.assertEqual(safe, 0.0)
        self.assertLess(unsafe, safe)
        self.assertGreater(
            unsafe_terms["delta_temperature_violation_cost"],
            0.0,
        )
        self.assertGreater(
            unsafe_terms["high_temperature_violation_cost"],
            0.0,
        )

    def test_peak_environment_advances_real_physics(self):
        env = BTMSTd3Env(scene="peak", current_profile=[560.0, 560.0])
        before, _ = env.reset(seed=3)
        after, reward, terminated, truncated, info = env.step(
            np.array([0.0, 0.0])
        )

        self.assertEqual(after.shape, before.shape)
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["step_index"], 1)
        self.assertAlmostEqual(info["n_comp_cmd_rpm"], 3150.0)
        self.assertAlmostEqual(info["n_pump_cmd_rpm"], 3200.0)
        self.assertGreaterEqual(info["total_power_w"], 0.0)

    def test_profile_end_truncates_the_episode(self):
        env = BTMSTd3Env(scene="freq", current_profile=[0.0])
        env.reset(seed=5)
        _, _, terminated, truncated, info = env.step(
            np.array([-1.0, -1.0])
        )

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["end_reason"], "profile_complete")
        with self.assertRaisesRegex(RuntimeError, "reset"):
            env.step(np.array([-1.0, -1.0]))

    def test_high_temperature_terminates_with_fixed_penalty(self):
        env = BTMSTd3Env(scene="peak", current_profile=[560.0, 560.0])
        env.reset(seed=9)
        env.pack.temps[:] = 45.0 + 273.15
        _, reward, terminated, truncated, info = env.step(
            np.array([-1.0, -1.0])
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["end_reason"], "temperature_limit")
        self.assertLessEqual(reward, -100.0)

    def test_gymnasium_contract(self):
        env = BTMSTd3Env(scene="peak", current_profile=[560.0] * 8)
        check_env(env, skip_render_check=True)


class TrainingEntrypointTests(unittest.TestCase):
    def test_run_directory_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "run"
            self.assertEqual(create_run_directory(target), target.resolve())
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                create_run_directory(target)

    def test_metadata_records_profile_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metadata.json"
            profile = np.array([1.0, 2.0])
            write_metadata(
                path,
                scene="peak",
                seed=4,
                total_timesteps=16,
                dt=5.0,
                profile=profile,
                agc_data_file=None,
            )

            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["scene"], "peak")
            self.assertEqual(data["profile_sha256"], profile_sha256(profile))
            self.assertEqual(data["reward_weights"]["power"], 0.05)


if __name__ == "__main__":
    unittest.main()
