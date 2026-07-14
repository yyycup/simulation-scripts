import unittest
import pandas as pd
from predictor_identification_data import (
    REQUIRED_COLUMNS,
    assign_scenario_splits,
    validate_identification_frame,
)


class PredictorIdentificationDataTest(unittest.TestCase):
    def test_split_is_deterministic_and_grouped_by_scenario(self):
        ids = [f"scenario_{index:03d}" for index in range(10)]
        first = assign_scenario_splits(ids, seed=20260714)
        second = assign_scenario_splits(reversed(ids), seed=20260714)
        self.assertEqual(first, second)
        self.assertEqual(sum(value == "train" for value in first.values()), 6)
        self.assertEqual(sum(value == "validation" for value in first.values()), 2)
        self.assertEqual(sum(value == "test" for value in first.values()), 2)

    def test_validation_rejects_missing_columns_and_split_leakage(self):
        frame = pd.DataFrame([{name: 0.0 for name in REQUIRED_COLUMNS}])
        frame["scenario_id"] = "same"
        frame["split"] = "train"
        validate_identification_frame(frame)
        leaked = pd.concat([frame, frame.assign(split="test")], ignore_index=True)
        with self.assertRaises(ValueError):
            validate_identification_frame(leaked)
        with self.assertRaises(ValueError):
            validate_identification_frame(frame.drop(columns=["q_evap_eff_w"]))


if __name__ == "__main__":
    unittest.main()
