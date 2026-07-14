import json
import tempfile
import unittest
from pathlib import Path

from mpc_predictor_selection import (
    CANDIDATE_B,
    LPV_L,
    PHYSICS_P,
    PredictorArtifactError,
    load_predictor_artifact,
    normalize_predictor_name,
)


class PredictorSelectionTests(unittest.TestCase):
    def test_empty_name_defaults_to_candidate_b(self):
        self.assertEqual(normalize_predictor_name(None), CANDIDATE_B)
        self.assertEqual(normalize_predictor_name(""), CANDIDATE_B)

    def test_supported_names_are_normalized(self):
        self.assertEqual(normalize_predictor_name("physics_p"), PHYSICS_P)
        self.assertEqual(normalize_predictor_name("lpv_l"), LPV_L)

    def test_unsupported_name_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_predictor_name("hybrid_h")

    def test_artifact_type_must_match_expected_type(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "artifact.json"
            artifact_path.write_text(
                json.dumps({"model_type": "physics_p", "schema_version": 1}),
                encoding="utf-8",
            )

            artifact = load_predictor_artifact(artifact_path, expected_type=PHYSICS_P)
            self.assertEqual(artifact["model_type"], PHYSICS_P)

            with self.assertRaises(PredictorArtifactError):
                load_predictor_artifact(artifact_path, expected_type=LPV_L)


if __name__ == "__main__":
    unittest.main()
