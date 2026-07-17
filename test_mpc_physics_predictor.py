import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from fit_mpc_physics_predictor import (
    _validate_output_path,
    fit_physics_artifact,
    fit_physics_dynamic,
    fit_physics_predictor,
)
from mpc_physics_predictor import (
    DEFAULT_INPUT_DOMAIN,
    DEFAULT_DYNAMIC_PARAMETERS,
    DEFAULT_THERMAL_PARAMETERS,
    PhysicsArtifactError,
    PhysicsPredictorState,
    consumed_parameter_names,
    evaluate_physics_capacity,
    initialize_physics_state,
    lag_step,
    load_physics_artifact,
    step_physics_predictor,
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


def synthetic_dynamic_frame():
    artifact = json.loads(json.dumps(KNOWN_ARTIFACT))
    artifact["dynamic"] = json.loads(json.dumps(DEFAULT_DYNAMIC_PARAMETERS))
    artifact["thermal"] = json.loads(json.dumps(DEFAULT_THERMAL_PARAMETERS))
    rows = []
    for split, scenario_id, offset in (
        ("train", "dynamic_train", 0.0),
        ("validation", "dynamic_validation", 150.0),
        ("test", "dynamic_test", 300.0),
    ):
        state = initialize_physics_state(
            2200.0, 2400.0, 300.0, 200.0, 27.0, 28.0, 28.5, 30.0, 27.5
        )
        for step in range(65):
            n_comp = 2200.0 + offset + (1800.0 if 12 <= step < 40 else 0.0)
            n_pump = 2400.0 + (1200.0 if 25 <= step < 50 else 0.0)
            q_gen = 450.0 + (200.0 if step >= 35 else 0.0)
            state = step_physics_predictor(
                state, n_comp, n_pump, q_gen, 35.0, 5.0, artifact
            )
            row = make_row(
                scenario_id,
                split,
                n_comp,
                n_pump,
                state.t_cool_c,
                35.0,
                0.0,
            )
            row.update(
                time_s=5.0 * (step + 1),
                q_gen_w=q_gen,
                n_comp_eff_rpm=state.n_comp_eff_rpm,
                n_pump_eff_rpm=state.n_pump_eff_rpm,
                q_cond_eff_w=state.q_cond_w,
                q_evap_eff_w=state.q_evap_w,
                t_supply_c=state.t_supply_c,
                t_plate_c=state.t_plate_c,
                t_return_c=state.t_return_c,
                t_batt_c=state.t_batt_c,
                t_cool_c=state.t_cool_c,
                dt_s=5.0,
            )
            rows.append(row)
    return pd.DataFrame(rows)


class PhysicsCapacityTests(unittest.TestCase):
    def test_gate_is_applied_after_active_capacity_is_clipped(self):
        artifact = json.loads(json.dumps(KNOWN_ARTIFACT))
        artifact["capacity"]["coefficients"] = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        value = evaluate_physics_capacity(
            artifact["gate"]["n_on_rpm"], 1600, 35, 40, artifact=artifact
        )
        self.assertAlmostEqual(value, 0.5 * artifact["capacity"]["q_upper_w"])

    def test_inputs_are_clipped_to_artifact_domain_before_evaluation(self):
        boundaries = {
            "n_comp_rpm": (1000.0, 6000.0),
            "n_pump_rpm": (1600.0, 4800.0),
            "t_cool_c": (20.0, 35.0),
            "t_ambient_c": (20.0, 40.0),
        }
        base = dict(
            n_comp_rpm=3000.0,
            n_pump_rpm=2400.0,
            t_cool_c=27.5,
            t_ambient_c=30.0,
            artifact=KNOWN_ARTIFACT,
        )
        for field, (lower, upper) in boundaries.items():
            for outside, boundary in ((lower - 9999.0, lower), (upper + 9999.0, upper)):
                outside_args = dict(base, **{field: outside})
                boundary_args = dict(base, **{field: boundary})
                with self.subTest(field=field, outside=outside):
                    self.assertEqual(
                        evaluate_physics_capacity(**outside_args),
                        evaluate_physics_capacity(**boundary_args),
                    )

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
            "n_pump": (math.nan, math.inf, True),
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


class PhysicsDynamicTests(unittest.TestCase):
    def dynamic_artifact(self):
        artifact = json.loads(json.dumps(KNOWN_ARTIFACT))
        artifact["dynamic"] = json.loads(json.dumps(DEFAULT_DYNAMIC_PARAMETERS))
        artifact["thermal"] = json.loads(json.dumps(DEFAULT_THERMAL_PARAMETERS))
        return artifact

    def test_lag_step_uses_project_first_order_convention(self):
        self.assertEqual(lag_step(2.0, 10.0, 5.0, 0.0), 10.0)
        self.assertAlmostEqual(lag_step(2.0, 10.0, 5.0, 15.0), 4.0)

    def test_equilibrium_step_is_finite_and_remains_at_equilibrium(self):
        artifact = self.dynamic_artifact()
        state = initialize_physics_state(
            n_comp_eff_rpm=1000.0,
            n_pump_eff_rpm=2000.0,
            q_cond_w=0.0,
            q_evap_w=0.0,
            t_supply_c=30.0,
            t_plate_c=30.0,
            t_return_c=30.0,
            t_batt_c=30.0,
            t_cool_c=30.0,
        )
        result = step_physics_predictor(
            state,
            n_comp_cmd_rpm=1000.0,
            n_pump_cmd_rpm=2000.0,
            q_gen_w=0.0,
            t_ambient_c=30.0,
            dt_s=5.0,
            artifact=artifact,
        )
        self.assertIsInstance(result, PhysicsPredictorState)
        for name, value in vars(result).items():
            values = value if isinstance(value, tuple) else (value,)
            self.assertTrue(all(math.isfinite(item) for item in values), name)
        self.assertAlmostEqual(result.t_supply_c, 30.0)
        self.assertAlmostEqual(result.t_plate_c, 30.0)
        self.assertAlmostEqual(result.t_return_c, 30.0)
        self.assertAlmostEqual(result.t_batt_c, 30.0)
        self.assertAlmostEqual(result.t_cool_c, 30.0)

    def test_more_cooling_lowers_supply_temperature(self):
        artifact = self.dynamic_artifact()
        artifact["dynamic"]["supply_delay_s"] = 0.0
        state = initialize_physics_state(
            n_comp_eff_rpm=6000.0,
            n_pump_eff_rpm=3200.0,
            q_cond_w=4800.0,
            q_evap_w=4800.0,
            t_supply_c=30.0,
            t_plate_c=30.0,
            t_return_c=30.0,
            t_batt_c=30.0,
            t_cool_c=30.0,
        )
        cooled = step_physics_predictor(
            state, 6000.0, 3200.0, 0.0, 30.0, 5.0, artifact
        )
        off_state = initialize_physics_state(
            n_comp_eff_rpm=1000.0,
            n_pump_eff_rpm=3200.0,
            q_cond_w=0.0,
            q_evap_w=0.0,
            t_supply_c=30.0,
            t_plate_c=30.0,
            t_return_c=30.0,
            t_batt_c=30.0,
            t_cool_c=30.0,
        )
        off = step_physics_predictor(
            off_state, 1000.0, 3200.0, 0.0, 30.0, 5.0, artifact
        )
        self.assertLess(cooled.t_supply_c, off.t_supply_c)

    def test_all_exported_dynamic_and_thermal_parameters_are_consumed(self):
        artifact = self.dynamic_artifact()
        exported = set(artifact["dynamic"]) | set(artifact["thermal"])
        self.assertEqual(exported, consumed_parameter_names())

    def test_each_exported_numeric_parameter_changes_the_rollout(self):
        def rollout(artifact):
            state = initialize_physics_state(
                2200.0, 2400.0, 200.0, 100.0, 28.0, 29.0, 29.5, 31.0, 28.5
            )
            values = []
            for step in range(24):
                state = step_physics_predictor(
                    state,
                    5000.0 if step >= 3 else 2200.0,
                    4000.0 if step >= 8 else 2400.0,
                    700.0,
                    35.0,
                    5.0,
                    artifact,
                )
                values.extend(
                    value
                    for value in vars(state).values()
                    if not isinstance(value, tuple)
                )
            return np.asarray(values)

        artifact = self.dynamic_artifact()
        baseline = rollout(artifact)
        for group in ("dynamic", "thermal"):
            for name, value in artifact[group].items():
                if not isinstance(value, (int, float)):
                    continue
                changed = json.loads(json.dumps(artifact))
                increment = 5.1 if name.endswith("delay_s") else max(0.1, 0.1 * value)
                changed[group][name] = value + increment
                with self.subTest(group=group, name=name):
                    self.assertFalse(np.array_equal(baseline, rollout(changed)))
        scheduled = self.dynamic_artifact()
        scheduled["dynamic"]["time_constant_model"] = "scheduled"
        for name in (
            "tau_comp_schedule_s",
            "tau_pump_schedule_s",
            "tau_cond_schedule_s",
            "tau_evap_schedule_s",
        ):
            scheduled["dynamic"][name] = 0.0
        scheduled_baseline = rollout(scheduled)
        self.assertEqual(
            set(scheduled["dynamic"]) | set(scheduled["thermal"]),
            consumed_parameter_names(scheduled),
        )
        for name in set(scheduled["dynamic"]) - set(DEFAULT_DYNAMIC_PARAMETERS):
            changed = json.loads(json.dumps(scheduled))
            changed["dynamic"][name] = 5.0
            with self.subTest(group="scheduled", name=name):
                self.assertFalse(np.array_equal(scheduled_baseline, rollout(changed)))


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

    def test_input_domain_is_strictly_validated_when_present(self):
        for bad_range in ([1.0], [2.0, 1.0], [1.0, math.inf], [True, 2.0]):
            artifact = json.loads(json.dumps(KNOWN_ARTIFACT))
            artifact["input_domain"] = json.loads(json.dumps(DEFAULT_INPUT_DOMAIN))
            artifact["input_domain"]["n_comp_rpm"] = bad_range
            with self.subTest(bad_range=bad_range):
                with self.assertRaisesRegex(PhysicsArtifactError, "input_domain"):
                    validate_physics_artifact(artifact)

    def test_require_validated_loader_rejects_smoke_and_accepts_validated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "physics.json"
            artifact = json.loads(json.dumps(KNOWN_ARTIFACT))
            artifact["fit"] = {"fit_status": "mechanical_smoke_unvalidated"}
            path.write_text(json.dumps(artifact), encoding="utf-8")
            self.assertEqual(load_physics_artifact(path), artifact)
            with self.assertRaisesRegex(PhysicsArtifactError, "validated"):
                load_physics_artifact(path, require_validated=True)
            artifact["fit"]["fit_status"] = "validated"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            self.assertEqual(
                load_physics_artifact(path, require_validated=True), artifact
            )


class PhysicsFitTests(unittest.TestCase):
    def test_synthetic_fit_is_valid_bounded_monotonic_and_accurate(self):
        artifact = fit_physics_artifact(synthetic_frame())
        validate_physics_artifact(artifact)
        self.assertTrue(1900 <= artifact["gate"]["n_on_rpm"] <= 2000)
        self.assertTrue(10 <= artifact["gate"]["width_rpm"] <= 80)
        self.assertTrue(all(-2 <= value <= 2 for value in artifact["capacity"]["coefficients"]))
        self.assertLess(artifact["fit"]["validation_metrics"]["mae_w"], 150.0)
        self.assertEqual(artifact["fit"]["fit_status"], "validated")
        self.assertEqual(artifact["fit"]["selection_metric"], "validation_mae_w")
        self.assertIn("selected_features", artifact["fit"])
        self.assertIn("selected_mask", artifact["fit"])
        self.assertIn("ridge", artifact["fit"])
        self.assertIn("start_index", artifact["fit"])
        self.assertGreater(artifact["fit"]["candidate_count"], 1)
        self.assertIn("max_low_speed_w", artifact["fit"])
        self.assertLessEqual(artifact["fit"]["max_low_speed_w"], 25.0 + 1e-7)
        self.assertGreaterEqual(
            artifact["fit"]["min_active_slope_w_per_step"], -1e-7
        )
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

    def test_test_rows_features_and_targets_do_not_change_artifact(self):
        first_frame = synthetic_frame(test_target=-1e9)
        second_frame = synthetic_frame(test_target=1e9)
        altered = second_frame[second_frame["split"] == "test"].copy()
        altered.loc[:, "scenario_id"] = ["test_extreme"]
        altered.loc[:, "n_comp_eff_rpm"] = 1e12
        altered.loc[:, "n_pump_eff_rpm"] = 1e12
        altered.loc[:, "t_cool_c"] = -1e12
        altered.loc[:, "t_ambient_c"] = 1e12
        extra = altered.copy()
        extra.loc[:, "scenario_id"] = ["test_extra"]
        second_frame = pd.concat(
            [second_frame[second_frame["split"] != "test"], altered, extra],
            ignore_index=True,
        )
        first = fit_physics_artifact(first_frame)
        second = fit_physics_artifact(second_frame)
        self.assertEqual(first["gate"], second["gate"])
        self.assertEqual(first["capacity"], second["capacity"])
        self.assertEqual(first["fit"], second["fit"])

    def test_repeated_fit_is_deterministic(self):
        first = fit_physics_artifact(synthetic_frame())
        second = fit_physics_artifact(synthetic_frame())
        self.assertEqual(first["gate"], second["gate"])
        self.assertEqual(first["capacity"], second["capacity"])
        self.assertEqual(first["fit"], second["fit"])

    def test_validation_targets_are_not_passed_to_optimizer(self):
        def optimizer_residuals(frame):
            captured = []

            def fake_optimizer(fun, x0, **_kwargs):
                residual = fun(np.asarray(x0, dtype=float))
                captured.append(residual)
                return SimpleNamespace(success=True, x=x0, cost=float(residual @ residual))

            with patch("fit_mpc_physics_predictor.least_squares", side_effect=fake_optimizer):
                fit_physics_artifact(frame)
            return captured

        first = synthetic_frame()
        second = first.copy()
        second.loc[second["split"] == "validation", "q_evap_ss_w"] += 1e6
        first_residuals = optimizer_residuals(first)
        second_residuals = optimizer_residuals(second)
        self.assertEqual(len(first_residuals), len(second_residuals))
        for first_residual, second_residual in zip(first_residuals, second_residuals):
            np.testing.assert_array_equal(first_residual, second_residual)

    def test_empty_validation_smoke_produces_valid_artifact(self):
        frame = synthetic_frame(include_validation=False).iloc[[2, -1]].copy()
        artifact = fit_physics_artifact(frame)
        validate_physics_artifact(artifact)
        self.assertFalse(artifact["fit"]["validation_available"])
        self.assertIsNone(artifact["fit"]["validation_metrics"])
        self.assertEqual(
            artifact["fit"]["fit_status"], "mechanical_smoke_unvalidated"
        )
        self.assertEqual(
            artifact["fit"]["selection_method"],
            "fixed_mechanical_no_validation",
        )
        self.assertEqual(artifact["fit"]["candidate_count"], 1)
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

    def test_all_invalid_optimizer_results_fail_clearly(self):
        class Result:
            success = False
            message = "forced failure"
            x = np.zeros(8)
            cost = math.inf

        frame = synthetic_frame()
        train = frame[frame["split"] == "train"]
        validation = frame[frame["split"] == "validation"]
        with patch("fit_mpc_physics_predictor.least_squares", return_value=Result()):
            with self.assertRaisesRegex(RuntimeError, "all candidates failed"):
                fit_physics_predictor(train, validation)

    def test_nonfinite_and_out_of_bounds_optimizer_results_are_skipped(self):
        class Result:
            success = True
            message = "mock"
            cost = 0.0

            def __init__(self, x):
                self.x = x

        frame = synthetic_frame(include_validation=False)
        train = frame[frame["split"] == "train"]
        validation = frame.iloc[0:0].assign(split="validation")
        for parameters in (
            np.full(8, math.nan),
            np.array([3000, 25, 0, 0, 0, 0, 0, 0], dtype=float),
        ):
            with self.subTest(parameters=parameters):
                with patch(
                    "fit_mpc_physics_predictor.least_squares",
                    return_value=Result(parameters),
                ):
                    with self.assertRaisesRegex(RuntimeError, "all candidates failed"):
                        fit_physics_predictor(train, validation)

    def test_dynamic_fit_exports_consumed_parameters_and_horizon_metadata(self):
        capacity = fit_physics_artifact(synthetic_frame())
        artifact = fit_physics_dynamic(capacity, synthetic_dynamic_frame())
        validate_physics_artifact(artifact)
        self.assertEqual(
            set(artifact["dynamic"]) | set(artifact["thermal"]),
            consumed_parameter_names(artifact),
        )
        metadata = artifact["fit"]["dynamic_fit"]
        self.assertEqual(metadata["training_horizons_steps"], [10, 20, 60])
        self.assertEqual(metadata["selection_source"], "validation_only")
        self.assertEqual(metadata["scheduled_minimum_improvement"], 0.10)
        self.assertIn(metadata["selected_time_constant_model"], {"constant", "scheduled"})
        if metadata["selected_time_constant_model"] == "scheduled":
            self.assertGreaterEqual(metadata["scheduled_improvement"], 0.10)

    def test_dynamic_test_rows_features_and_targets_do_not_affect_fit(self):
        dynamic = synthetic_dynamic_frame()
        altered = dynamic.copy()
        test_mask = altered["split"] == "test"
        altered.loc[test_mask, "q_evap_eff_w"] = 1e9
        altered.loc[test_mask, "q_cond_eff_w"] = -1e9
        altered.loc[test_mask, "t_batt_c"] = 1e6
        altered.loc[test_mask, "n_comp_cmd_rpm"] = 1e8
        altered = pd.concat([altered, altered.loc[test_mask]], ignore_index=True)
        capacity = fit_physics_artifact(synthetic_frame())
        first = fit_physics_dynamic(capacity, dynamic)
        second = fit_physics_dynamic(capacity, altered)
        self.assertEqual(first["dynamic"], second["dynamic"])
        self.assertEqual(first["thermal"], second["thermal"])
        self.assertEqual(first["fit"]["dynamic_fit"], second["fit"]["dynamic_fit"])

    def test_failed_scheduled_candidate_falls_back_to_constant(self):
        constant = json.loads(json.dumps(KNOWN_ARTIFACT))
        constant["dynamic"] = json.loads(json.dumps(DEFAULT_DYNAMIC_PARAMETERS))
        constant["thermal"] = json.loads(json.dumps(DEFAULT_THERMAL_PARAMETERS))
        result = SimpleNamespace(success=True, cost=1.0)
        with patch(
            "fit_mpc_physics_predictor._fit_dynamic_candidate",
            side_effect=[(constant, result), RuntimeError("forced scheduled failure")],
        ):
            artifact = fit_physics_dynamic(KNOWN_ARTIFACT, synthetic_dynamic_frame())
        self.assertEqual(artifact["dynamic"]["time_constant_model"], "constant")
        self.assertEqual(
            artifact["fit"]["dynamic_fit"]["scheduled_candidate_status"], "failed"
        )

    def test_numeric_scheduled_candidate_failure_falls_back_to_constant(self):
        constant = json.loads(json.dumps(KNOWN_ARTIFACT))
        constant["dynamic"] = json.loads(json.dumps(DEFAULT_DYNAMIC_PARAMETERS))
        constant["thermal"] = json.loads(json.dumps(DEFAULT_THERMAL_PARAMETERS))
        result = SimpleNamespace(success=True, cost=1.0)
        for error in (ValueError("invalid candidate"), FloatingPointError("non-finite")):
            with self.subTest(error=type(error).__name__):
                with patch(
                    "fit_mpc_physics_predictor._fit_dynamic_candidate",
                    side_effect=[(constant, result), error],
                ):
                    artifact = fit_physics_dynamic(
                        KNOWN_ARTIFACT, synthetic_dynamic_frame()
                    )
                self.assertEqual(
                    artifact["dynamic"]["time_constant_model"], "constant"
                )
                self.assertEqual(
                    artifact["fit"]["dynamic_fit"]["scheduled_candidate_status"],
                    "failed",
                )

    def test_zero_validation_error_does_not_select_unimproved_scheduled_model(self):
        constant = json.loads(json.dumps(KNOWN_ARTIFACT))
        constant["dynamic"] = json.loads(json.dumps(DEFAULT_DYNAMIC_PARAMETERS))
        constant["thermal"] = json.loads(json.dumps(DEFAULT_THERMAL_PARAMETERS))
        scheduled = json.loads(json.dumps(constant))
        scheduled["dynamic"]["time_constant_model"] = "scheduled"
        optimizer = SimpleNamespace(success=True, cost=1.0)
        with patch(
            "fit_mpc_physics_predictor._fit_dynamic_candidate",
            side_effect=[(constant, optimizer), (scheduled, optimizer)],
        ), patch(
            "fit_mpc_physics_predictor._dynamic_validation_metric",
            side_effect=[0.0, 0.0],
        ):
            artifact = fit_physics_dynamic(KNOWN_ARTIFACT, synthetic_dynamic_frame())
        self.assertEqual(artifact["dynamic"]["time_constant_model"], "constant")
        self.assertEqual(
            artifact["fit"]["dynamic_fit"]["scheduled_improvement"], 0.0
        )

    def test_scheduled_model_requires_at_least_ten_percent_validation_improvement(self):
        constant = json.loads(json.dumps(KNOWN_ARTIFACT))
        constant["dynamic"] = json.loads(json.dumps(DEFAULT_DYNAMIC_PARAMETERS))
        constant["thermal"] = json.loads(json.dumps(DEFAULT_THERMAL_PARAMETERS))
        scheduled = json.loads(json.dumps(constant))
        scheduled["dynamic"]["time_constant_model"] = "scheduled"
        for name in (
            "tau_comp_schedule_s",
            "tau_pump_schedule_s",
            "tau_cond_schedule_s",
            "tau_evap_schedule_s",
        ):
            scheduled["dynamic"][name] = 1.0
        optimizer = SimpleNamespace(success=True, cost=1.0)
        for scheduled_metric, expected in ((0.91, "constant"), (0.90, "scheduled")):
            with self.subTest(scheduled_metric=scheduled_metric):
                with patch(
                    "fit_mpc_physics_predictor._fit_dynamic_candidate",
                    side_effect=[(constant, optimizer), (scheduled, optimizer)],
                ), patch(
                    "fit_mpc_physics_predictor._dynamic_validation_metric",
                    side_effect=[1.0, scheduled_metric],
                ):
                    artifact = fit_physics_dynamic(
                        KNOWN_ARTIFACT, synthetic_dynamic_frame()
                    )
                self.assertEqual(
                    artifact["dynamic"]["time_constant_model"], expected
                )


class PhysicsCliTests(unittest.TestCase):
    def test_cli_output_cannot_be_inside_project_model_data(self):
        project = Path(__file__).resolve().parent
        with self.assertRaisesRegex(ValueError, "model_data"):
            _validate_output_path(project / "model_data" / "physics.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            accepted = Path(temp_dir) / "outputs" / "physics.json"
            self.assertEqual(_validate_output_path(accepted), accepted.resolve())


if __name__ == "__main__":
    unittest.main()
