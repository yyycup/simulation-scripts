import tempfile
import unittest
from pathlib import Path

import numpy as np

from td3_btms.env import build_current_profile, canonical_scene, profile_sha256


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


if __name__ == "__main__":
    unittest.main()
