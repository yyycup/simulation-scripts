import tempfile
import unittest
from pathlib import Path

import numpy as np

from td3_btms.env import (
    BTMSTd3Env,
    build_current_profile,
    canonical_scene,
    map_action_to_rpm,
    profile_sha256,
)


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


if __name__ == "__main__":
    unittest.main()
