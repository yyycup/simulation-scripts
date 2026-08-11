import copy
import unittest

from mpc_physics_predictor import DEFAULT_PHYSICS_ARTIFACT


class BuildPCompressorDisplacementVariantTests(unittest.TestCase):
    def test_builder_scales_capacity_and_declares_matching_power_scale(self):
        try:
            from experiments.physics_p.calibration.build_p_compressor_displacement_variant import (
                build_displacement_variant,
            )
        except ImportError as exc:
            self.fail(f"missing displacement variant builder: {exc}")

        base = copy.deepcopy(DEFAULT_PHYSICS_ARTIFACT)
        variant = build_displacement_variant(base, scale=0.8)

        self.assertEqual(
            variant["thermal"]["compressor_displacement_scale"],
            0.8,
        )
        self.assertEqual(
            variant["capacity"]["q_upper_w"],
            0.8 * base["capacity"]["q_upper_w"],
        )
        self.assertEqual(
            variant["capacity"]["coefficients"],
            [0.8 * value for value in base["capacity"]["coefficients"]],
        )
        self.assertEqual(variant["dynamic"], base["dynamic"])
        self.assertNotIn("compressor_displacement_scale", base["thermal"])

    def test_builder_rejects_non_reducing_scale(self):
        try:
            from experiments.physics_p.calibration.build_p_compressor_displacement_variant import (
                build_displacement_variant,
            )
        except ImportError as exc:
            self.fail(f"missing displacement variant builder: {exc}")

        for scale in (0.0, 1.0, 1.1):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(ValueError, "scale"):
                    build_displacement_variant(
                        copy.deepcopy(DEFAULT_PHYSICS_ARTIFACT),
                        scale=scale,
                    )


if __name__ == "__main__":
    unittest.main()
