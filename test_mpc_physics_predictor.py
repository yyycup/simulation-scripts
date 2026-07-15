import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fit_mpc_physics_predictor import fit_physics_artifact, fit_physics_predictor
from mpc_physics_predictor import (
    PhysicsArtifactError,
    evaluate_physics_capacity,
    load_physics_artifact,
    smooth_gate,
    validate_physics_artifact,
)
from predictor_identification_data import REQUIRED_COLUMNS


KNOWN_ARTIFACT = {
    "model_type": "physics_p",
    "schema_version": 1,
    "gate": {"n_on_rpm": 1950.0, "width_rpm": 25.0},
    "capacity": {
        "coefficients": [0.6, -0.1, 0.015, -0.002, 0.0, 0.0],
        "n_pump_ref_rpm": 2000.0,
        "q_upper_w": 4800.0,
    },
}


def make_row(scenario_id, split, n_comp, n_pump, t_cool, t_ambient, target):
    row = {column: 0.0 for column in REQUIRED_COLUMNS}
    row.update(
        scenario_id=scenario_id,
        split=split,
        n_comp_cmd_rpm=float(n_comp),
        n_pump_cmd_rpm=float(n_pump),
        n_comp_eff_rpm=float(n_comp),
        n_pump_eff_rpm=float(n_pump),
        t_cool_c=float(t_cool),
        t_ambient_c=float(t_ambient),
        q_evap_ss_w=float(target),
    )
    return row


def synthetic_frame(include_validation=True, test_target=123.0):
    rows = []
    cases = [
        (1000, 1600, 25, 35),
        (1900, 2000, 30, 20),
        (2000, 1600, 25, 35),
        (2600, 2400, 27.5, 30),
        (3600, 3200, 35, 40),
        (4800, 4000, 20, 20),
        (6000, 4800, 35, 20),
    ]
    for index, values in enumerate(cases):
        target = evaluate_physics_capacity(*values, artifact=KNOWN_ARTIFACT)
        rows.append(make_row(f"train_{index}", "train", *values, target))
    if include_validation:
        for index, values in enumerate(
            [(1800, 1800, 32, 35), (2200, 2200, 24, 25), (4200, 3600, 30, 40)]
        ):
            target = evaluate_physics_capacity(*values, artifact=KNOWN_ARTIFACT)
            rows.append(make_row(f"validation_{index}", "validation", *values, target))
    rows.append(make_row("test_0", "test", 5500, 4400, 34, 42, test_target))
    return pd.DataFrame(rows)


class PhysicsCapacityTests(unittest.TestCase):
    def test_capacity_is_gated_bounded_and_nonnegative(self):
        self.assertLess(
            evaluate_physics_capacity(1000, 1600, 25, 35, artifact=KNOWN_ARTIFACT),
            25.0,
        )
        self.assertGreaterEqual(
            evaluate_physics_capacity(2000, 1600, 25, 35, artifact=KNOWN_ARTIFACT),
            0.0,
        )
        self.assertLessEqual(
            evaluate_physics_capacity(6000, 4800, 35, 20, artifact=KNOWN_ARTIFACT),
            4800.0,
        )

    def test_known_active_profile_is_monotonic_in_compressor_speed(self):
        values = [
            evaluate_physics_capacity(speed, 2400, 27.5, 30, artifact=KNOWN_ARTIFACT)
            for speed in np.linspace(2000, 6000, 21)
        ]
        self.assertTrue(np.all(np.diff(values) >= -1e-9))

    def test_smooth_gate_limits_and_width_validation(self):
        self.assertAlmostEqual(smooth_gate(1950, 1950, 25), 0.5)
        self.assertLess(smooth_gate(1000, 1950, 25), 1e-6)
        self.assertGreater(smooth_gate(3000, 1950, 25), 1.0 - 1e-6)
        for width in (0, -1):
            with self.subTest(width=width):
                with self.assertRaisesRegex(ValueError, "width_rpm"):
                    smooth_gate(2000, 1950, width)

    def test_invalid_numeric_inputs_are_rejected_and_valid_outputs_are_finite(self):
        for field, values in {
            "n_comp": (math.nan, math.inf, True, "2000"),
            "n_pump": (0, -1, math.nan, math.inf, True),
            "t_cool": (math.nan, math.inf, True),
            "t_ambient": (math.nan, math.inf, True),
        }.items():
            for value in values:
                arguments = dict(
                    n_comp_rpm=2000.0,
                    n_pump_rpm=1600.0,
                    t_cool_c=25.0,
                    t_ambient_c=35.0,
                    artifact=KNOWN_ARTIFACT,
                )
                arguments[f"{field}_rpm" if field in {"n_comp", "n_pump"} else f"{field}_c"] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaises((TypeError, ValueError)):
                        evaluate_physics_capacity(**arguments)
        for speed in np.linspace(0, 10000, 41):
            value = evaluate_physics_capacity(speed, 1600, 25, 35, artifact=KNOWN_ARTIFACT)
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 4800.0)


class PhysicsArtifactTests(unittest.TestCase):
    def test_valid_artifact_round_trips_through_loader(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "physics.json"
            path.write_text(json.dumps(KNOWN_ARTIFACT), encoding="utf-8")
            loaded = load_physics_artifact(path)
        self.assertEqual(loaded, KNOWN_ARTIFACT)

    def test_artifact_fields_are_strictly_validated(self):
        mutations = {
            "model_type": lambda value: value.update(model_type="lpv_l"),
            "schema_version": lambda value: value.update(schema_version=True),
            "gate": lambda value: value.update(gate=[]),
            "n_on_rpm": lambda value: value["gate"].update(n_on_rpm=1800),
            "width_rpm": lambda value: value["gate"].update(width_rpm=100),
            "coefficients": lambda value: value["capacity"].update(coefficients=[0.0] * 5),
            "coefficient_bool": lambda value: value["capacity"].update(
                coefficients=[True, 0, 0, 0, 0, 0]
            ),
            "coefficient_inf": lambda value: value["capacity"].update(
                coefficients=[math.inf, 0, 0, 0, 0, 0]
            ),
            "pump_ref": lambda value: value["capacity"].update(n_pump_ref_rpm=0),
            "q_upper": lambda value: value["capacity"].update(q_upper_w=-1),
        }
        for expected_text, mutate in mutations.items():
            artifact = json.loads(json.dumps(KNOWN_ARTIFACT))
            mutate(artifact)
            with self.subTest(field=expected_text):
                with self.assertRaises(PhysicsArtifactError) as raised:
                    validate_physics_artifact(artifact)
                self.assertTrue(str(raised.exception))

    def test_loader_reports_invalid_p1_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "physics.json"
            artifact = json.loads(json.dumps(KNOWN_ARTIFACT))
            artifact["capacity"]["q_upper_w"] = math.nan
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(PhysicsArtifactError, "q_upper_w"):
                load_physics_artifact(path)


class PhysicsFitTests(unittest.TestCase):
    def test_synthetic_fit_is_valid_bounded_monotonic_and_accurate(self):
        artifact = fit_physics_artifact(synthetic_frame())
        validate_physics_artifact(artifact)
        self.assertTrue(1900 <= artifact["gate"]["n_on_rpm"] <= 2000)
        self.assertTrue(10 <= artifact["gate"]["width_rpm"] <= 80)
        self.assertTrue(all(-2 <= value <= 2 for value in artifact["capacity"]["coefficients"]))
        self.assertLess(artifact["fit"]["validation_metrics"]["mae_w"], 150.0)
        values = [
            evaluate_physics_capacity(speed, 2400, 27.5, 30, artifact=artifact)
            for speed in np.linspace(2000, 6000, 41)
        ]
        self.assertTrue(np.all(np.diff(values) >= -1e-7))

    def test_test_targets_do_not_change_fit_or_metadata(self):
        first = fit_physics_artifact(synthetic_frame(test_target=-1e9))
        second = fit_physics_artifact(synthetic_frame(test_target=1e9))
        self.assertEqual(first["gate"], second["gate"])
        self.assertEqual(first["capacity"], second["capacity"])
        self.assertNotIn("test", json.dumps(first["fit"]).lower())

    def test_empty_validation_smoke_produces_valid_artifact(self):
        frame = synthetic_frame(include_validation=False).iloc[[2, -1]].copy()
        artifact = fit_physics_artifact(frame)
        validate_physics_artifact(artifact)
        self.assertFalse(artifact["fit"]["validation_available"])
        self.assertIsNone(artifact["fit"]["validation_metrics"])
        value = evaluate_physics_capacity(1000, 1600, 25, 35, artifact=artifact)
        self.assertTrue(math.isfinite(value))
        self.assertLess(value, 25.0)

    def test_core_fit_requires_explicit_train_and_validation_splits(self):
        frame = synthetic_frame()
        train = frame[frame["split"] == "train"]
        validation = frame[frame["split"] == "validation"]
        fit_physics_predictor(train, validation)
        with self.assertRaisesRegex(ValueError, "train"):
            fit_physics_predictor(validation, validation)
        with self.assertRaisesRegex(ValueError, "validation"):
            fit_physics_predictor(train, frame[frame["split"] == "test"])

    def test_bad_frames_are_rejected(self):
        frame = synthetic_frame()
        bad_frames = [
            frame[frame["split"] != "train"],
            frame.drop(columns=["q_evap_ss_w"]),
            frame.assign(q_evap_ss_w=np.nan),
            pd.concat(
                [frame, frame.iloc[[0]].assign(split="validation")], ignore_index=True
            ),
        ]
        for bad_frame in bad_frames:
            with self.subTest(columns=tuple(bad_frame.columns), rows=len(bad_frame)):
                with self.assertRaises(ValueError):
                    fit_physics_artifact(bad_frame)


if __name__ == "__main__":
    unittest.main()
