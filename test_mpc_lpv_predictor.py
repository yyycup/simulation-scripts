import copy
import json
import unittest

import numpy as np
import pandas as pd

from fit_mpc_lpv_predictor import (
    _rate_matrices,
    _stable,
    build_grouped_lagged_samples,
    fit_lpv_artifact,
    pump_schedule_improves,
)
from mpc_lpv_predictor import (
    DISTURBANCE_NAMES,
    INPUT_NAMES,
    STATE_NAMES,
    LpvArtifactError,
    LpvPredictor,
    augmented_state_names,
    scheduled_matrices,
    spectral_radius_grid,
    validate_lpv_artifact,
)


def synthetic_frame():
    # Stable two-state thermal core embedded in the fixed nine-state contract.
    a = np.diag([0.96, 0.92, 0.88, 0.84, 0.82, 0.70, 0.72, 0.55, 0.50])
    a[0, 1] = 0.02
    a[1, 0] = 0.01
    b = np.zeros((9, 2))
    b[:, 0] = [-1e-5, -2e-5, -1e-5, -2e-5, -1e-5, 0.35, 0.38, 0.45, 0.0]
    b[:, 1] = [-2e-6, -4e-6, -3e-6, -4e-6, -3e-6, 0.02, 0.02, 0.0, 0.45]
    e = np.zeros((9, 2))
    e[0, 0] = 1e-4
    e[0:5, 1] = [0.002, 0.004, 0.003, 0.003, 0.003]
    c = np.array([0.55, 0.35, 0.45, 0.55, 0.50, 0.0, 0.0, 50.0, 160.0])
    specifications = [
        ("train_forward_a", "train", 1, 1),
        ("train_forward_b", "train", 1, 2),
        ("train_reverse_a", "train", -1, 3),
        ("train_reverse_b", "train", -1, 4),
        ("validation_forward", "validation", 1, 5),
        ("validation_reverse", "validation", -1, 6),
        ("test_forward", "test", 1, 7),
        ("test_reverse", "test", -1, 8),
    ]
    rows = []
    for scenario_id, split, direction, phase in specifications:
        state = np.array([31, 27, 29, 26, 27, 0, 0, 1000, 3200], dtype=float)
        for step in range(36):
            n_comp_cmd = (1000.0, 1600.0, 4200.0, 6000.0)[(step // 6 + phase) % 4]
            n_pump_cmd = (1800.0, 3200.0, 4600.0)[(step // 5 + phase) % 3]
            q_gen = 300.0 + 20.0 * ((step + phase) % 5)
            ambient = 30.0 + 2.0 * np.sin(0.2 * (step + phase))
            rows.append({
                "scenario_id": scenario_id, "split": split, "time_s": 5.0 * step,
                "flow_direction": direction,
                **dict(zip(STATE_NAMES, state)),
                "n_comp_cmd_rpm": n_comp_cmd, "n_pump_cmd_rpm": n_pump_cmd,
                "q_gen_w": q_gen, "t_ambient_c": ambient,
                "q_evap_ss_w": max(0.0, 0.8 * (n_comp_cmd - 1000.0)),
                "q_cond_ss_w": max(0.0, 0.9 * (n_comp_cmd - 1000.0)),
                "dt_s": 5.0,
            })
            state = a @ state + b @ np.array([n_comp_cmd, n_pump_cmd]) + e @ np.array([q_gen, ambient]) + c
    return pd.DataFrame(rows)


class LpvContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = synthetic_frame()
        cls.artifact = fit_lpv_artifact(cls.frame)

    def test_canonical_names_and_augmented_order(self):
        self.assertEqual(len(STATE_NAMES), 9)
        self.assertEqual(INPUT_NAMES, ("n_comp_cmd_rpm", "n_pump_cmd_rpm"))
        self.assertEqual(DISTURBANCE_NAMES, ("q_gen_w", "t_ambient_c"))
        self.assertEqual(len(augmented_state_names(3)), 27)
        self.assertEqual(self.artifact["base_state_count"], 9)
        self.assertEqual(
            self.artifact["augmented_state_count"], 9 * self.artifact["order"]
        )
        if not self.artifact["use_pump_schedule"]:
            self.assertNotIn("n_pump_eff_rpm", self.artifact["normalization"])
            terms = self.artifact["directions"]["1"]["active"]["discrete"]["A"]
            self.assertNotIn("M_pump", terms)

    def test_fit_is_numerically_and_json_deterministic(self):
        again = fit_lpv_artifact(self.frame.sample(frac=1.0, random_state=123))
        self.assertEqual(
            json.dumps(self.artifact, sort_keys=True), json.dumps(again, sort_keys=True)
        )

    def test_all_schedule_grid_matrices_have_margin(self):
        radii = [row["spectral_radius"] for row in spectral_radius_grid(self.artifact)]
        self.assertTrue(radii)
        self.assertLess(max(radii), 0.995)

    def test_grouped_lags_never_cross_scenarios(self):
        train = self.frame[self.frame["split"] == "train"]
        normalization = {
            "n_comp_eff_rpm": {"center": 3500.0, "scale": 2500.0},
            "t_cool_c": {"center": 27.5, "scale": 7.5},
        }
        samples = build_grouped_lagged_samples(train, 3, normalization)
        expected = sum(max(0, len(group) - 3) for _, group in train.groupby("scenario_id"))
        self.assertEqual(len(samples), expected)
        self.assertTrue(all(sample["scenario_id"].startswith("train_") for sample in samples))

    def test_test_features_targets_and_row_count_do_not_affect_fit(self):
        changed = self.frame.copy()
        mask = changed["split"] == "test"
        changed.loc[mask, list(STATE_NAMES)] += 1e6
        changed.loc[mask, list(INPUT_NAMES)] += 1e5
        extra = changed[mask].copy()
        extra["scenario_id"] = "test_extra"
        changed = pd.concat([changed, extra], ignore_index=True)
        self.assertEqual(
            json.dumps(self.artifact, sort_keys=True),
            json.dumps(fit_lpv_artifact(changed), sort_keys=True),
        )

    def test_rate_matrices_use_exact_conversion(self):
        local = self.artifact["directions"]["1"]["active"]
        discrete = {
            name: {term: np.asarray(value) for term, value in terms.items()}
            for name, terms in local["discrete"].items()
        }
        expected = _rate_matrices(discrete, self.artifact["dt_s"])
        for name in expected:
            for term in expected[name]:
                np.testing.assert_allclose(expected[name][term], local["rate"][name][term])

    def test_public_outputs_come_from_lag_zero_and_rollout_is_finite(self):
        predictor = LpvPredictor(self.artifact)
        initial = dict(zip(STATE_NAMES, self.frame.loc[0, STATE_NAMES].to_numpy(float)))
        predictor.reset(initial)
        output = None
        for _ in range(60):
            output = predictor.step({
                "n_comp_cmd_rpm": 3500.0, "n_pump_cmd_rpm": 3200.0,
                "q_gen_w": 350.0, "t_ambient_c": 30.0, "flow_direction": 1,
            })
        self.assertEqual(tuple(output), STATE_NAMES)
        self.assertTrue(np.isfinite(list(output.values())).all())
        np.testing.assert_allclose(list(output.values()), predictor._state[:9])

    def test_scheduled_matrix_dimensions_are_explicit(self):
        matrices = scheduled_matrices(self.artifact, 1, 3500.0, 27.0, 3200.0)
        n = self.artifact["augmented_state_count"]
        self.assertEqual(matrices["A"].shape, (n, n))
        self.assertEqual(matrices["B"].shape, (n, 2))
        self.assertEqual(matrices["E"].shape, (n, 2))
        self.assertEqual(matrices["c"].shape, (n,))

    def test_bad_artifact_dimensions_are_rejected(self):
        bad = copy.deepcopy(self.artifact)
        bad["augmented_state_count"] += 1
        with self.assertRaises(LpvArtifactError):
            validate_lpv_artifact(bad)
        bad = copy.deepcopy(self.artifact)
        bad["directions"]["1"]["low"]["discrete"]["B"]["M0"] = [[0.0]]
        with self.assertRaises(LpvArtifactError):
            validate_lpv_artifact(bad)

    def test_rate_mismatch_is_rejected(self):
        bad = copy.deepcopy(self.artifact)
        bad["directions"]["1"]["low"]["rate"]["c"]["M0"][0] += 1.0
        with self.assertRaises(LpvArtifactError):
            validate_lpv_artifact(bad)

    def test_unstable_candidate_is_rejected(self):
        bad = copy.deepcopy(self.artifact)
        for direction in ("-1", "1"):
            for regime in ("low", "active"):
                local = bad["directions"][direction][regime]
                local["discrete"]["A"]["M0"][0][0] = 1.1
                local["rate"] = _rate_matrices(
                    {name: {term: np.asarray(value) for term, value in terms.items()}
                     for name, terms in local["discrete"].items()}, bad["dt_s"]
                )
        validate_lpv_artifact(bad)
        self.assertFalse(_stable(bad))

    def test_pump_schedule_threshold_is_inclusive_at_five_percent(self):
        self.assertTrue(pump_schedule_improves(10.0, 9.5))
        self.assertFalse(pump_schedule_improves(10.0, 9.500001))

    def test_reset_and_step_reject_wrong_dimensions(self):
        predictor = LpvPredictor(self.artifact)
        with self.assertRaises(ValueError):
            predictor.reset({"t_batt_c": 30.0})
        predictor.reset(dict(zip(STATE_NAMES, np.ones(9))))
        with self.assertRaises(ValueError):
            predictor.step({"n_comp_cmd_rpm": 3000.0})

    def test_fit_metadata_records_local_sample_provenance(self):
        sources = self.artifact["fit"]["sample_sources"]
        self.assertEqual(self.artifact["fit"]["fit_status"], "validated")
        for direction in ("-1", "1"):
            self.assertTrue(
                self.artifact["fit"]["validation_by_direction"][direction]["validated"]
            )
            for regime in ("low", "active"):
                self.assertGreater(sources[direction][regime]["sample_count"], 0)
                self.assertEqual(sources[direction][regime]["sample_source"], "direction_regime")


if __name__ == "__main__":
    unittest.main()
