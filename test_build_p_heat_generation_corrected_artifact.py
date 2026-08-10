import copy
import unittest

from build_p_heat_generation_corrected_artifact import (
    DEFAULT_SCALE,
    build_corrected_artifact,
)
from mpc_physics_predictor import DEFAULT_PHYSICS_ARTIFACT


class BuildPHeatGenerationCorrectedArtifactTests(unittest.TestCase):
    def test_default_scale_matches_validated_heat_generation_correction(self):
        self.assertEqual(DEFAULT_SCALE, 0.74)

    def test_build_adds_only_heat_generation_scale_to_model_parameters(self):
        base = copy.deepcopy(DEFAULT_PHYSICS_ARTIFACT)
        base["thermal"].pop("battery_heat_generation_scale", None)

        corrected = build_corrected_artifact(
            base,
            scale=0.825,
            calibration_source="current_v3_actual_command_replay",
        )

        self.assertNotIn("battery_heat_generation_scale", base["thermal"])
        self.assertEqual(corrected["thermal"]["battery_heat_generation_scale"], 0.825)
        self.assertEqual(corrected["capacity"], base["capacity"])
        self.assertEqual(corrected["dynamic"], base["dynamic"])
        for name, value in base["thermal"].items():
            self.assertEqual(corrected["thermal"][name], value, name)
        metadata = corrected["fit"]["battery_heat_generation_correction"]
        self.assertTrue(metadata["capacity_model_frozen"])
        self.assertTrue(metadata["dynamic_model_frozen"])
        self.assertEqual(metadata["added_states"], 0)
        self.assertEqual(metadata["added_equations"], 0)

    def test_build_rejects_nonphysical_scale(self):
        with self.assertRaisesRegex(ValueError, "scale"):
            build_corrected_artifact(
                copy.deepcopy(DEFAULT_PHYSICS_ARTIFACT),
                scale=0.0,
                calibration_source="test",
            )

    def test_build_can_align_reference_flow_without_changing_other_parameters(self):
        base = copy.deepcopy(DEFAULT_PHYSICS_ARTIFACT)

        corrected = build_corrected_artifact(
            base,
            scale=0.74,
            coolant_mass_flow_ref_kg_s=0.3899064402993154,
            calibration_source="current_v3_actual_command_replay",
        )

        self.assertEqual(corrected["thermal"]["battery_heat_generation_scale"], 0.74)
        self.assertEqual(
            corrected["thermal"]["coolant_mass_flow_ref_kg_s"],
            0.3899064402993154,
        )
        for name, value in base["thermal"].items():
            if name not in {
                "battery_heat_generation_scale",
                "coolant_mass_flow_ref_kg_s",
            }:
                self.assertEqual(corrected["thermal"][name], value, name)
        metadata = corrected["fit"]["battery_heat_generation_correction"]
        self.assertFalse(metadata["other_thermal_parameters_frozen"])
        self.assertEqual(
            metadata["corrected_thermal_parameters"],
            ["battery_heat_generation_scale", "coolant_mass_flow_ref_kg_s"],
        )


if __name__ == "__main__":
    unittest.main()
