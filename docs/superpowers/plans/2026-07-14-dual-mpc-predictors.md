# MPC Dual Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and fairly validate two independent online MPC predictors: model P, a layered physics-guided predictor, and model L, an LPV/ARX predictor, while keeping Candidate B as the unchanged default and fallback.

**Architecture:** Generate one versioned identification dataset from the unchanged full R134a plant, split it by complete scenario, and use the same train/validation/test boundaries for B0, P, and L. Model P reuses the existing physical state structure with a smooth low-speed gate and newly identified parameters; model L uses stable scheduled linear state equations. Both expose a common offline rollout contract and separate GEKKO builders, but no fusion model is created.

**Tech Stack:** Python 3.12, NumPy, Pandas, SciPy, GEKKO, CoolProp, PyBaMM, `unittest`, JSON/CSV, PowerShell, Git.

---

## Scope and execution boundaries

- Work in `C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制`.
- Use `C:\Users\24776\miniforge3\envs\btms\python.exe` for all Python commands.
- Do not modify the plant equations in `thermal_system.py` or `thermal_loop.py`.
- Do not change the default predictor from `candidate_b` during this plan.
- Do not add a P+L fusion or residual model.
- Keep large datasets and fit outputs under ignored, versioned `outputs/` directories.
- Promote artifacts to tracked `model_data/` only after their independent validation passes and the user explicitly approves promotion.
- Run long full-data fits and formal MPC cases on the server if local GEKKO/CoolProp execution becomes unstable; local smoke tests remain mandatory.

## File map

### Shared data and evaluation

- Create `predictor_identification_data.py` — schema, scenario specs, deterministic group split, metadata and leakage validation.
- Create `generate_predictor_identification_data.py` — steady grid and dynamic open-loop plant data generator.
- Create `evaluate_mpc_predictors.py` — B0/P/L metrics, thresholds, comparison tables and solve-time summary.
- Create `test_evaluate_mpc_predictors.py` — zero/active capacity metrics, horizon grouping and acceptance-summary tests.
- Create `test_predictor_identification_data.py` — schema, deterministic split, DMAX and data-leakage tests.
- Create `test_generate_predictor_identification_data.py` — steady-grid and dynamic-generator smoke tests.

### Predictor selection and fallback

- Create `mpc_predictor_selection.py` — canonical model names, artifact validation, fallback reason and common offline rollout types.
- Create `test_mpc_predictor_selection.py` — default, explicit selection, invalid artifact and fallback tests.

### Model P

- Create `mpc_physics_predictor.py` — P parameter schema, smooth gate, steady capacity, offline state rollout and GEKKO expressions.
- Create `fit_mpc_physics_predictor.py` — fit P1 capacity and P2/P3 dynamics with multi-horizon loss.
- Create `test_mpc_physics_predictor.py` — gate, bounds, monotonicity, artifact consumption and rollout tests.

### Model L

- Create `mpc_lpv_predictor.py` — L artifact schema, scheduled state transition, stability checks, offline rollout and GEKKO expressions.
- Create `fit_mpc_lpv_predictor.py` — ridge identification, order/regularization selection and stable artifact export.
- Create `test_mpc_lpv_predictor.py` — dimensions, scheduling, stable rollout, artifact and reproducibility tests.

### Existing integration points

- Modify `mpc_flow_direction_strategies.py` — dispatch Candidate B/P/L predictor equations without changing objective weights or plant boundaries.
- Modify `thermal_control_strategies.py` — pass an explicit predictor name and artifact path, defaulting to Candidate B with no external artifact.
- Modify `thermal_case_simulator.py` — accept and log `mpc_predictor_model` and `mpc_predictor_artifact_path`.
- Modify `run_final_controller_comparison.py` — expose the predictor name/artifact CLI for closed-loop and formal runs.
- Modify `run_final_controller_parallel.py` — propagate the same predictor arguments into isolated workers.
- Modify `run_evaporator_mpc_capacity_diagnostic.py` — remove stale `kq * N_comp` logic and call the selected predictor.
- Modify `README.md` — document generation, fitting, evaluation and smoke commands.
- Modify `test_model_data_paths.py` only after validated P/L artifacts are explicitly promoted.

---

### Task 1: Freeze the baseline and define predictor selection

**Files:**
- Create: `mpc_predictor_selection.py`
- Create: `test_mpc_predictor_selection.py`
- Test: existing Candidate B and MPC tests

- [ ] **Step 1: Run the pre-change baseline**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q test_mpc_evaporator_capacity_model test_mpc_evaporator_capacity_table test_mpc_reduced_model_calibration test_final_controller_comparison
```

Expected: all collected tests pass. Record the test count in the execution log.

- [ ] **Step 2: Write failing selection and fallback tests**

Create `test_mpc_predictor_selection.py`:

```python
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


class PredictorSelectionTest(unittest.TestCase):
    def test_none_and_empty_name_keep_candidate_b_default(self):
        self.assertEqual(normalize_predictor_name(None), CANDIDATE_B)
        self.assertEqual(normalize_predictor_name(""), CANDIDATE_B)

    def test_only_three_canonical_names_are_accepted(self):
        self.assertEqual(normalize_predictor_name("physics_p"), PHYSICS_P)
        self.assertEqual(normalize_predictor_name("lpv_l"), LPV_L)
        with self.assertRaises(ValueError):
            normalize_predictor_name("hybrid_h")

    def test_artifact_type_and_version_are_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            path.write_text(json.dumps({"model_type": "physics_p", "schema_version": 1}), encoding="utf-8")
            loaded = load_predictor_artifact(path, expected_type=PHYSICS_P)
            self.assertEqual(loaded["model_type"], PHYSICS_P)
            with self.assertRaises(PredictorArtifactError):
                load_predictor_artifact(path, expected_type=LPV_L)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the new test and verify RED**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -v test_mpc_predictor_selection
```

Expected: import failure because `mpc_predictor_selection.py` does not exist.

- [ ] **Step 4: Implement the minimal selection contract**

Create `mpc_predictor_selection.py` with this public surface:

```python
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

CANDIDATE_B = "candidate_b"
PHYSICS_P = "physics_p"
LPV_L = "lpv_l"
SUPPORTED_PREDICTORS = (CANDIDATE_B, PHYSICS_P, LPV_L)


class PredictorArtifactError(ValueError):
    pass


class OfflinePredictor(Protocol):
    def reset(self, row: dict[str, float]) -> None: ...
    def step(self, row: dict[str, float]) -> dict[str, float]: ...


@dataclass
class GekkoPredictorVariables:
    t_batt_k: object
    t_cool_k: object
    t_plate_c: object
    t_supply_c: object
    t_return_c: object
    n_comp: object
    n_pump: object
    q_evap: object
    q_cond: object
    q_evap_cmd_w: object


def normalize_predictor_name(value: object) -> str:
    name = CANDIDATE_B if value is None or str(value).strip() == "" else str(value).strip().lower()
    if name not in SUPPORTED_PREDICTORS:
        raise ValueError(f"Unsupported MPC predictor: {name}")
    return name


def load_predictor_artifact(path: Path | str, expected_type: str) -> dict:
    artifact_path = Path(path)
    try:
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictorArtifactError(f"Cannot load predictor artifact {artifact_path}: {exc}") from exc
    if data.get("model_type") != expected_type:
        raise PredictorArtifactError(
            f"Predictor type mismatch: expected={expected_type}, actual={data.get('model_type')}"
        )
    if data.get("schema_version") != 1:
        raise PredictorArtifactError(f"Unsupported predictor schema: {data.get('schema_version')}")
    return data
```

- [ ] **Step 5: Verify GREEN and commit**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q test_mpc_predictor_selection test_mpc_evaporator_capacity_model
git add -- mpc_predictor_selection.py test_mpc_predictor_selection.py
git commit -m "feat: 添加MPC预测模型选择契约"
```

Expected: tests pass and the commit contains only the two new files.

---

### Task 2: Add the common identification schema and deterministic split

**Files:**
- Create: `predictor_identification_data.py`
- Create: `test_predictor_identification_data.py`

- [ ] **Step 1: Write failing schema and split tests**

Create `test_predictor_identification_data.py`:

```python
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
```

- [ ] **Step 2: Run RED**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -v test_predictor_identification_data
```

Expected: import failure because the schema module does not exist.

- [ ] **Step 3: Implement the schema and split functions**

Create `predictor_identification_data.py`:

```python
import hashlib
from collections.abc import Iterable

import pandas as pd

REQUIRED_COLUMNS = (
    "scenario_id", "split", "time_s", "flow_direction",
    "n_comp_cmd_rpm", "n_pump_cmd_rpm", "n_comp_eff_rpm", "n_pump_eff_rpm",
    "q_gen_w", "t_ambient_c", "t_batt_c", "t_cool_c", "t_plate_c",
    "t_supply_c", "t_return_c", "q_evap_ss_w", "q_evap_eff_w",
    "q_cond_ss_w", "q_cond_eff_w",
)


def assign_scenario_splits(scenario_ids: Iterable[str], seed: int = 20260714) -> dict[str, str]:
    unique_ids = sorted(set(str(value) for value in scenario_ids))
    ranked = sorted(
        unique_ids,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(),
    )
    n_total = len(ranked)
    n_train = int(round(0.60 * n_total))
    n_validation = int(round(0.20 * n_total))
    result = {}
    for index, scenario_id in enumerate(ranked):
        if index < n_train:
            result[scenario_id] = "train"
        elif index < n_train + n_validation:
            result[scenario_id] = "validation"
        else:
            result[scenario_id] = "test"
    return result


def validate_identification_frame(frame: pd.DataFrame) -> None:
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing identification columns: {missing}")
    split_counts = frame.groupby("scenario_id")["split"].nunique()
    leaked = split_counts[split_counts > 1]
    if not leaked.empty:
        raise ValueError(f"Scenario split leakage: {list(leaked.index)}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Identification data contain NaN values")
```

- [ ] **Step 4: Verify GREEN and commit**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q test_predictor_identification_data
git add -- predictor_identification_data.py test_predictor_identification_data.py
git commit -m "feat: 添加预测模型辨识数据契约"
```

---

### Task 3: Generate the steady identification grid

**Files:**
- Create: `generate_predictor_identification_data.py`
- Create: `test_generate_predictor_identification_data.py`

- [ ] **Step 1: Write a failing steady-grid test using a mocked plant**

Create `test_generate_predictor_identification_data.py` with:

```python
import unittest
from unittest.mock import patch

from generate_predictor_identification_data import build_steady_grid, generate_steady_rows


class GeneratePredictorIdentificationDataTest(unittest.TestCase):
    def test_full_grid_has_1200_unique_points(self):
        grid = build_steady_grid(mode="full")
        self.assertEqual(len(grid), 12 * 5 * 4 * 5)
        self.assertEqual(len(set(grid)), len(grid))

    @patch("generate_predictor_identification_data.run_refrigeration_cycle")
    def test_steady_row_preserves_the_1999_2000_boundary(self, run_cycle):
        run_cycle.side_effect = lambda n_comp, *_args, **_kwargs: {
            "Q_evap": 0.0 if n_comp < 2000.0 else 900.0,
            "Q_cond": 0.0 if n_comp < 2000.0 else 1200.0,
            "Q_hx_potential": 900.0,
            "Q_ref_max": 1000.0,
            "W_comp": 100.0,
            "T_evap_sat": 280.15,
            "T_cond_sat": 320.15,
        }
        rows = generate_steady_rows([(1999.0, 1600.0, 25.0, 35.0), (2000.0, 1600.0, 25.0, 35.0)])
        self.assertEqual([row["q_evap_ss_w"] for row in rows], [0.0, 900.0])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run RED**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -v test_generate_predictor_identification_data
```

Expected: import failure or missing function failures.

- [ ] **Step 3: Implement the steady grid and CLI**

In `generate_predictor_identification_data.py`, define exact full axes:

```python
COMPRESSOR_LEVELS_RPM = (1000, 1400, 1800, 1900, 1950, 1999, 2000, 2200, 3000, 4000, 5000, 6000)
PUMP_LEVELS_RPM = (1600, 2400, 3200, 4000, 4800)
COOLANT_LEVELS_C = (20, 25, 30, 35)
AMBIENT_LEVELS_C = (20, 25, 30, 35, 40)


def build_steady_grid(mode="full"):
    if mode == "smoke":
        return [(1999.0, 1600.0, 25.0, 35.0), (2000.0, 1600.0, 25.0, 35.0)]
    return [
        (float(n_comp), float(n_pump), float(t_cool), float(t_ambient))
        for n_comp in COMPRESSOR_LEVELS_RPM
        for n_pump in PUMP_LEVELS_RPM
        for t_cool in COOLANT_LEVELS_C
        for t_ambient in AMBIENT_LEVELS_C
    ]
```

`generate_steady_rows()` must call `pump_model()`, `staged_fan_speed()` and `run_refrigeration_cycle()` and write explicit source fields, limit type and plant configuration values. Add CLI:

Each point receives a stable ID formatted from all four inputs, for example `steady_nc2000_np1600_tc25_ta35`. Run `assign_scenario_splits()` over all steady IDs before writing the CSV, and persist the split column. The fitters may read train and validation rows but must not read steady test rows until final evaluation.

```powershell
python generate_predictor_identification_data.py --dataset steady --mode smoke --output-root outputs/mpc_predictor_identification_v1
```

- [ ] **Step 4: Run unit and real two-point smoke tests**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q test_generate_predictor_identification_data
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' generate_predictor_identification_data.py --dataset steady --mode smoke --output-root outputs/mpc_predictor_identification_v1
```

Expected: two rows; 1999 rpm has zero plant cooling and 2000 rpm has nonnegative plant cooling; no tracked files under `outputs/`.

- [ ] **Step 5: Commit**

```powershell
git add -- generate_predictor_identification_data.py test_generate_predictor_identification_data.py
git commit -m "feat: 添加蒸发器稳态辨识网格"
```

---

### Task 4: Generate deterministic dynamic identification trajectories

**Files:**
- Modify: `generate_predictor_identification_data.py`
- Modify: `test_generate_predictor_identification_data.py`

- [ ] **Step 1: Add failing excitation tests**

Add tests for `ramp_limited_multilevel_sequence()`:

```python
def test_excitation_is_reproducible_and_respects_dmax(self):
    first = ramp_limited_multilevel_sequence((1000.0, 2000.0, 4000.0, 6000.0), 80, 600.0, seed=17)
    second = ramp_limited_multilevel_sequence((1000.0, 2000.0, 4000.0, 6000.0), 80, 600.0, seed=17)
    self.assertEqual(first.tolist(), second.tolist())
    self.assertLessEqual(float(abs(first[1:] - first[:-1]).max()), 600.0)
```

Add a mocked one-step test asserting each dynamic output row contains every `REQUIRED_COLUMNS` field.

- [ ] **Step 2: Run RED**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -v test_generate_predictor_identification_data
```

Expected: missing excitation and dynamic functions.

- [ ] **Step 3: Implement excitation and plant rollout**

Use one deterministic generator:

```python
def ramp_limited_multilevel_sequence(levels, steps, dmax, seed):
    rng = np.random.default_rng(seed)
    targets = rng.choice(np.asarray(levels, dtype=float), size=steps)
    values = np.empty(steps, dtype=float)
    values[0] = targets[0]
    for index in range(1, steps):
        delta = np.clip(targets[index] - values[index - 1], -dmax, dmax)
        values[index] = values[index - 1] + delta
    return values
```

Implement `run_dynamic_scenario(spec)` by reusing `build_pack_config()`, `BatteryPack`, `initialize_refrigeration_dynamic_state()` and `simulate_thermal_loop_step()`. Before each `pack.step()`, assign `pack.current` from the scenario current profile. Record effective states from the thermal-step return value. Record the predictor disturbance with the same current-to-heat expression used by `StandardMPC`:

```python
q_gen_w = ((float(pack.current) / 4.0) ** 2) * 0.001 * 52.0
```

`simulate_thermal_loop_step()` exposes dynamic `Q_evap_eff` and `Q_cond_eff` but not the two steady targets. For each saved row, call `pump_model(n_pump_eff)` and `run_refrigeration_cycle(n_comp_eff, staged_fan_speed(n_comp_eff), t_cool_k, m_dot_cool, t_ambient_k)` once with the same effective inputs, then record its `Q_evap` and `Q_cond` as `q_evap_ss_w` and `q_cond_ss_w`. This duplicate call is diagnostic-only and must not feed back into the plant state.

Define smoke scenarios as 20 steps and full scenarios as 120 to 180 steps. Generate separate compressor-only, pump-only, combined, forward and reverse scenario IDs. Assign train/validation/test only after all scenario IDs are known.

- [ ] **Step 4: Verify dynamic smoke and schema**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q test_predictor_identification_data test_generate_predictor_identification_data
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' generate_predictor_identification_data.py --dataset dynamic --mode smoke --output-root outputs/mpc_predictor_identification_v1
```

Expected: no missing fields, no split leakage, no DMAX violation and no NaN.

- [ ] **Step 5: Commit**

```powershell
git add -- generate_predictor_identification_data.py test_generate_predictor_identification_data.py
git commit -m "feat: 添加动态预测模型辨识轨迹"
```

---

### Task 5: Build B0 metrics and fix the stale capacity diagnostic

**Files:**
- Create: `evaluate_mpc_predictors.py`
- Create: `test_evaluate_mpc_predictors.py`
- Modify: `run_evaporator_mpc_capacity_diagnostic.py`
- Modify: `test_mpc_evaporator_capacity_model.py`

- [ ] **Step 1: Write failing metric and diagnostic tests**

Create tests for active MAPE, zero-region MAE and Candidate B dispatch:

```python
def test_capacity_metrics_separate_zero_and_active_regions(self):
    frame = pd.DataFrame({
        "n_comp_cmd_rpm": [1000.0, 2000.0],
        "q_evap_ss_w": [0.0, 1000.0],
        "q_pred_w": [20.0, 900.0],
    })
    result = capacity_metrics(frame)
    self.assertEqual(result["low_speed_mae_w"], 20.0)
    self.assertEqual(result["active_mape_percent"], 10.0)
```

In the diagnostic test, inspect `build_comparison()` and assert it calls `evaluate_capacity()` rather than `mpc_q_evap_steady_w()`.

- [ ] **Step 2: Run RED**

Run the two new/updated modules and expect missing evaluator functions plus the stale diagnostic assertion.

- [ ] **Step 3: Implement reusable metrics**

Create `evaluate_mpc_predictors.py` with:

```python
def error_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mean_bias": float(np.mean(error)),
        "p95_abs": float(np.quantile(np.abs(error), 0.95)),
    }


def capacity_metrics(frame):
    low = frame[frame["n_comp_cmd_rpm"] < 2000.0]
    active = frame[frame["n_comp_cmd_rpm"] >= 2000.0]
    result = {"low_speed_mae_w": error_metrics(low["q_evap_ss_w"], low["q_pred_w"])["mae"]}
    relative = np.abs(active["q_pred_w"] - active["q_evap_ss_w"]) / active["q_evap_ss_w"].abs().clip(lower=1e-9)
    result.update(error_metrics(active["q_evap_ss_w"], active["q_pred_w"]))
    result["active_mape_percent"] = float(100.0 * relative.mean())
    result["active_max_relative_error_percent"] = float(100.0 * relative.max())
    return result
```

- [ ] **Step 4: Replace stale diagnostic computation**

Load `DEFAULT_CALIBRATION_PATH` and call `evaluate_capacity(calibration, n_comp, n_pump, t_cool)` for the Candidate B line. Add `--predictor candidate_b|physics_p|lpv_l`, but reject P/L with a clear artifact-not-promoted message until their provisional artifacts exist.

- [ ] **Step 5: Run smoke evaluation and commit**

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q test_evaluate_mpc_predictors test_mpc_evaporator_capacity_model
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' run_evaporator_mpc_capacity_diagnostic.py --n-pump 2000 --coolant-temp-c 25 --output-root outputs/evaporator_mpc_capacity_diagnostic_candidate_b
git add -- evaluate_mpc_predictors.py test_evaluate_mpc_predictors.py run_evaporator_mpc_capacity_diagnostic.py test_mpc_evaporator_capacity_model.py
git commit -m "fix: 对齐Candidate B蒸发器诊断"
```

---

### Task 6: Implement and fit model P steady capacity

**Files:**
- Create: `mpc_physics_predictor.py`
- Create: `fit_mpc_physics_predictor.py`
- Create: `test_mpc_physics_predictor.py`

- [ ] **Step 1: Write failing gate, bounds and monotonicity tests**

Use a fixed synthetic artifact and assert:

```python
class MpcPhysicsPredictorTest(unittest.TestCase):
    def setUp(self):
        self.artifact = {
            "model_type": "physics_p", "schema_version": 1,
            "gate": {"n_on_rpm": 1950.0, "width_rpm": 25.0},
            "capacity": {
                "coefficients": [0.6, -0.1, 0.015, -0.002, 0.0, 0.0],
                "n_pump_ref_rpm": 2000.0,
                "q_upper_w": 4800.0,
            },
        }

    def test_low_speed_is_near_zero_and_output_is_bounded(self):
        self.assertLess(evaluate_physics_capacity(self.artifact, 1000, 1600, 25, 35), 25.0)
        self.assertGreaterEqual(evaluate_physics_capacity(self.artifact, 2000, 1600, 25, 35), 0.0)
        self.assertLessEqual(evaluate_physics_capacity(self.artifact, 6000, 4800, 35, 20), 4800.0)

    def test_active_capacity_is_monotone_in_compressor_speed(self):
        values = [evaluate_physics_capacity(self.artifact, speed, 2400, 25, 35) for speed in (2000, 3000, 4000, 5000, 6000)]
        self.assertEqual(values, sorted(values))
```

- [ ] **Step 2: Run RED**

Expected: model P module does not exist.

- [ ] **Step 3: Implement the exact P1 formula**

In `mpc_physics_predictor.py`:

```python
def smooth_gate(n_comp_rpm, n_on_rpm, width_rpm):
    return 0.5 * (1.0 + np.tanh((float(n_comp_rpm) - float(n_on_rpm)) / float(width_rpm)))


def active_capacity_w(coefficients, n_comp_rpm, n_pump_rpm, t_cool_c, t_ambient_c, n_pump_ref_rpm):
    c0, c1, c2, c3, c4, c5 = map(float, coefficients)
    n_comp_norm = (float(n_comp_rpm) - 4000.0) / 2000.0
    t_cool_norm = (float(t_cool_c) - 27.5) / 7.5
    t_ambient_norm = (float(t_ambient_c) - 30.0) / 10.0
    gain = (
        c0 + c1 * float(n_pump_ref_rpm) / max(float(n_pump_rpm), 1e-6)
        + c2 * t_cool_norm + c3 * t_ambient_norm
        + c4 * t_cool_norm * t_ambient_norm + c5 * n_comp_norm
    )
    return float(n_comp_rpm) * gain


def evaluate_physics_capacity(artifact, n_comp_rpm, n_pump_rpm, t_cool_c, t_ambient_c):
    gate = artifact["gate"]
    capacity = artifact["capacity"]
    raw = smooth_gate(n_comp_rpm, gate["n_on_rpm"], gate["width_rpm"]) * active_capacity_w(
        capacity["coefficients"], n_comp_rpm, n_pump_rpm, t_cool_c, t_ambient_c,
        capacity["n_pump_ref_rpm"],
    )
    return max(0.0, min(float(capacity["q_upper_w"]), raw))
```

- [ ] **Step 4: Implement constrained fitting**

In `fit_mpc_physics_predictor.py`, use `scipy.optimize.least_squares` with parameter bounds:

```python
LOWER = np.array([1900.0, 10.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0])
UPPER = np.array([2000.0, 80.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0])
```

The residual vector must concatenate training-point errors, a 10x penalty for low-speed predictions above25 W, and a 10x penalty for negative compressor-speed slopes on the validation grid. Select the fit only by validation metrics; do not inspect the test metrics before freezing the artifact.

- [ ] **Step 5: Run synthetic and smoke-data fits**

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q test_mpc_physics_predictor
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' fit_mpc_physics_predictor.py --steady-csv outputs/mpc_predictor_identification_v1/steady_smoke.csv --output outputs/mpc_predictor_comparison_v1/physics_p_smoke.json
```

Expected: artifact schema passes; low-speed predictions are below25 W; no artifact is copied to `model_data/`.

- [ ] **Step 6: Commit**

```powershell
git add -- mpc_physics_predictor.py fit_mpc_physics_predictor.py test_mpc_physics_predictor.py
git commit -m "feat: 添加分层物理蒸发器容量模型"
```

---

### Task 7: Fit model P dynamic and temperature states

**Files:**
- Modify: `mpc_physics_predictor.py`
- Modify: `fit_mpc_physics_predictor.py`
- Modify: `test_mpc_physics_predictor.py`

- [ ] **Step 1: Add failing state-rollout and parameter-consumption tests**

Define a `PhysicsPredictorState` dataclass and test that one equilibrium step remains finite, a cooling step lowers supply temperature, and every exported dynamic/thermal parameter appears in `consumed_parameter_names()`.

```python
def test_all_exported_parameters_are_consumed(self):
    artifact = load_physics_artifact(self.fixture_path)
    exported = set(artifact["dynamic"]) | set(artifact["thermal"])
    self.assertEqual(exported, consumed_parameter_names())
```

- [ ] **Step 2: Run RED**

Expected: missing state and rollout functions.

- [ ] **Step 3: Implement the offline P rollout**

Use the existing first-order convention exactly:

```python
def lag_step(previous, target, dt_s, tau_s):
    alpha = 1.0 if tau_s <= 0.0 else dt_s / (tau_s + dt_s)
    return previous + alpha * (target - previous)
```

The state step order is: effective speeds, delayed speed buffers, P1 steady capacity, `Q_cond`, `Q_evap`, supply delay, plate state, return delay, battery balance, coolant balance. Use deque-backed delay buffers offline and fixed GEKKO delay steps online.

- [ ] **Step 4: Implement multi-horizon fitting**

Fit only training trajectories. The residual vector must include standardized errors at10,20 and60 steps, parameter regularization, and dynamic step-response errors. Use validation weighted MAE to choose constant versus linearly scheduled time constants. Reject scheduled constants unless they improve the corresponding validation error by at least10%.

- [ ] **Step 5: Verify synthetic recovery and smoke rollout**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q test_mpc_physics_predictor
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' fit_mpc_physics_predictor.py --steady-csv outputs/mpc_predictor_identification_v1/steady_smoke.csv --dynamic-csv outputs/mpc_predictor_identification_v1/dynamic_smoke.csv --output outputs/mpc_predictor_comparison_v1/physics_p_smoke.json
```

Expected: finite 60-step rollout, no unused parameter and no tracked artifact.

- [ ] **Step 6: Commit**

```powershell
git add -- mpc_physics_predictor.py fit_mpc_physics_predictor.py test_mpc_physics_predictor.py
git commit -m "feat: 添加物理预测模型动态标定"
```

---

### Task 8: Implement and identify model L

**Files:**
- Create: `mpc_lpv_predictor.py`
- Create: `fit_mpc_lpv_predictor.py`
- Create: `test_mpc_lpv_predictor.py`

- [ ] **Step 1: Write failing dimension, stability and deterministic-fit tests**

Use a synthetic stable two-state system and assert the fitted artifact has correct dimensions, reproduces with the same seed/data, and every schedule-grid matrix has spectral radius below0.995.

- [ ] **Step 2: Run RED**

Expected: LPV modules do not exist.

- [ ] **Step 3: Implement artifact loading and scheduled rollout**

In `mpc_lpv_predictor.py`, use canonical base-state order:

```python
STATE_NAMES = (
    "t_batt_c", "t_cool_c", "t_plate_c", "t_supply_c", "t_return_c",
    "q_evap_eff_w", "q_cond_eff_w", "n_comp_eff_rpm", "n_pump_eff_rpm",
)
INPUT_NAMES = ("n_comp_cmd_rpm", "n_pump_cmd_rpm")
DISTURBANCE_NAMES = ("q_gen_w", "t_ambient_c")
```

For an artifact with ARX order `p`, construct `AUGMENTED_STATE_NAMES` by concatenating `STATE_NAMES` at lags0 through `p-1`. Matrices operate on all `9 * p` augmented states; public temperature/cooling outputs always come from the lag0 block. Store `order`, `base_state_count=9` and `augmented_state_count=9*p` in the artifact and validate all three values on load.

For each flow direction, store low and active local matrices. Blend them with the same smooth gate form as P. Affine scheduling uses normalized effective compressor speed and coolant temperature:

```python
M(rho) = M0 + rho_comp * M_comp + rho_temp * M_temp
```

Implement `scheduled_matrices()`, `spectral_radius_grid()`, `reset()` and `step()` with explicit dimension checks.

- [ ] **Step 4: Implement ridge identification and order selection**

Build grouped lagged samples without crossing scenario boundaries. For each direction/regime/order/regularization candidate, solve:

```python
theta = np.linalg.solve(phi.T @ phi + ridge * penalty, phi.T @ targets)
```

Candidate orders are1,2,3; ridge values are `1e-6, 1e-4, 1e-2, 1, 100`. Reject unstable candidates before validation ranking. Add pump-speed scheduling only if validation weighted temperature MAE improves at least5% and the stability grid still passes. Export both the discrete matrices and GEKKO rate matrices:

```python
A_rate = (A_discrete - np.eye(n_state)) / dt_s
B_rate = B_discrete / dt_s
E_rate = E_discrete / dt_s
c_rate = c_discrete / dt_s
```

- [ ] **Step 5: Run synthetic and smoke fits**

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q test_mpc_lpv_predictor
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' fit_mpc_lpv_predictor.py --dynamic-csv outputs/mpc_predictor_identification_v1/dynamic_smoke.csv --output outputs/mpc_predictor_comparison_v1/lpv_l_smoke.json
```

Expected: stable artifact, reproducible parameters and finite 60-step rollout.

- [ ] **Step 6: Commit**

```powershell
git add -- mpc_lpv_predictor.py fit_mpc_lpv_predictor.py test_mpc_lpv_predictor.py
git commit -m "feat: 添加LPV整体预测模型"
```

---

### Task 9: Build the unified offline B0/P/L comparison

**Files:**
- Modify: `evaluate_mpc_predictors.py`
- Modify: `test_evaluate_mpc_predictors.py`

- [ ] **Step 1: Write failing winner-independent report tests**

Assert the evaluator always emits three model rows, separate50/100/300-second rows, four scene/flow groups, solve-time placeholders and no fusion model row.

- [ ] **Step 2: Implement grouped free-rollout evaluation**

For each complete test trajectory:

1. Reset B0, P and L from the same first row.
2. Roll each model freely without future plant-state replacement.
3. Capture aligned predictions at10,20 and60 steps.
4. Separately run one-step-reset evaluation.
5. Aggregate by model, scene, flow, horizon and evaluation mode.

Write:

- `capacity_metrics.csv`;
- `temperature_metrics.csv`;
- `stability_metrics.csv`;
- `acceptance_summary.csv`;
- `comparison_metadata.json`.

The acceptance summary must calculate the exact thresholds from the approved design and must not automatically change the runtime default. Add `--baseline-only`; in that mode the CLI evaluates only the current Candidate B plus current reduced dynamics, writes the B0 files and exits before loading P or L. Implement B0 by using the same offline physical rollout code as P with the current Candidate B capacity artifact and current runtime dynamic/thermal parameters.

Also extend `run_evaporator_mpc_capacity_diagnostic.py` with `--physics-artifact` and `--lpv-artifact`. P/L are evaluated only when their explicit provisional artifact path is supplied; omission keeps Candidate B-only behavior.

- [ ] **Step 3: Verify smoke reports and commit**

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q test_evaluate_mpc_predictors
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' evaluate_mpc_predictors.py --steady-csv outputs/mpc_predictor_identification_v1/steady_smoke.csv --dynamic-csv outputs/mpc_predictor_identification_v1/dynamic_smoke.csv --physics-artifact outputs/mpc_predictor_comparison_v1/physics_p_smoke.json --lpv-artifact outputs/mpc_predictor_comparison_v1/lpv_l_smoke.json --output-root outputs/mpc_predictor_comparison_v1/smoke
git add -- evaluate_mpc_predictors.py test_evaluate_mpc_predictors.py
git commit -m "feat: 添加双预测模型统一评估"
```

---

### Task 10: Integrate model P into the existing GEKKO controller

**Files:**
- Modify: `mpc_physics_predictor.py`
- Modify: `mpc_flow_direction_strategies.py`
- Modify: `test_mpc_physics_predictor.py`
- Create: `test_mpc_predictor_runtime.py`

- [ ] **Step 1: Write a failing Candidate B invariance test**

Instantiate `MPCControllerDual` with omitted predictor and with `predictor_model="candidate_b"`; inspect source/attributes and assert both load the same Candidate B artifact and expose the same state names. Add a P construction test using a temporary artifact.

- [ ] **Step 2: Extract a physics GEKKO builder without changing equations**

Move only the current Candidate B capacity expression and current physical state equations into `mpc_physics_predictor.build_gekko_physics_backend(...)`. Return the common `GekkoPredictorVariables` dataclass defined in `mpc_predictor_selection.py`, containing the exact existing state handles:

```python
GekkoPredictorVariables(
    t_batt_k=t_batt_k,
    t_cool_k=t_cool_k,
    t_plate_c=t_plate_c,
    t_supply_c=t_supply_c,
    t_return_c=t_return_c,
    n_comp=n_comp,
    n_pump=n_pump,
    q_evap=q_evap,
    q_cond=q_cond,
    q_evap_cmd_w=q_evap_cmd_w,
)
```

Run all existing MPC tests before adding P. The extracted Candidate B path must remain behaviorally unchanged.

- [ ] **Step 3: Add the P GEKKO expressions**

Add `predictor_model="candidate_b"` and `predictor_artifact_path=None` to `MPCControllerDual.__init__`. For `physics_p`, load and validate the provisional artifact, create the smooth gate with `m.tanh`, apply P dynamic/thermal parameters and build the same returned variable contract. On load/build failure, log a structured fallback reason and rebuild Candidate B.

- [ ] **Step 4: Run runtime smoke tests**

Run Candidate B regression first, then a one-solve P smoke using a tiny fixture artifact. Expected: both solve; Candidate B output regression does not change; P exposes finite first-step cooling fields.

- [ ] **Step 5: Commit**

```powershell
git add -- mpc_physics_predictor.py mpc_flow_direction_strategies.py test_mpc_physics_predictor.py test_mpc_predictor_runtime.py
git commit -m "feat: 接入物理预测模型P"
```

---

### Task 11: Integrate model L into the existing GEKKO controller

**Files:**
- Modify: `mpc_lpv_predictor.py`
- Modify: `mpc_flow_direction_strategies.py`
- Modify: `test_mpc_lpv_predictor.py`
- Modify: `test_mpc_predictor_runtime.py`

- [ ] **Step 1: Write a failing L construction and solve test**

Use a stable9-state fixture artifact and assert `MPCControllerDual(... predictor_model="lpv_l")` builds the common variable contract, respects state dimensions and completes one local GEKKO solve.

- [ ] **Step 2: Implement the L GEKKO builder**

Create one GEKKO state variable for every `AUGMENTED_STATE_NAMES` entry. Construct affine scheduled rate matrices from GEKKO Intermediates, blend low/active local rates with a locally defined gate formula numerically equivalent to P's gate, and enforce one differential equation per augmented state:

```python
m.Equation(state_i.dt() == rate_i)
```

The offline test rollout uses forward Euler with the exported discrete matrices; add a one-step test that the discrete and rate forms agree within tolerance at `dt_s=5`. Map only the lag0 block into the common runtime contract. LPV temperature states are stored in degrees Celsius, so map `t_batt_k = t_batt_c + 273.15` and `t_cool_k = t_cool_c + 273.15` when constructing `GekkoPredictorVariables`. This keeps the existing objective and result-export code shared.

Do not call NumPy matrix multiplication on GEKKO objects; expand each row explicitly:

```python
next_expr_i = c_i + sum(a_ij * x_j for j in range(n_state)) + sum(b_ij * u_j for j in range(n_input))
```

- [ ] **Step 3: Add stable failure fallback**

If artifact validation, stability-grid validation, dimension checks, build or first solve fails, return to Candidate B and expose `predictor_fallback_reason` in `last_flow_info`.

- [ ] **Step 4: Run B/P/L runtime tests and commit**

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q test_mpc_predictor_runtime test_mpc_physics_predictor test_mpc_lpv_predictor test_mpc_evaporator_capacity_table
git add -- mpc_lpv_predictor.py mpc_flow_direction_strategies.py test_mpc_lpv_predictor.py test_mpc_predictor_runtime.py
git commit -m "feat: 接入LPV预测模型L"
```

---

### Task 12: Propagate explicit predictor selection through simulation entrypoints

**Files:**
- Modify: `thermal_control_strategies.py`
- Modify: `thermal_case_simulator.py`
- Modify: `mpc_flow_direction_strategies.py`
- Modify: `run_final_controller_comparison.py`
- Modify: `run_final_controller_parallel.py`
- Modify: `test_mpc_predictor_selection.py`
- Modify: `test_final_comparison_flow_modes.py`
- Modify: `test_final_controller_parallel.py`

- [ ] **Step 1: Write failing propagation tests**

Mock `create_mpc_flow_controller` and assert `create_controller(... mpc_predictor_model="physics_p", mpc_predictor_artifact_path=path)` forwards the exact name and path. Mock `create_controller` and assert `simulate_case(... mpc_predictor_model="lpv_l", mpc_predictor_artifact_path=path)` forwards and logs both. Verify omission still forwards `candidate_b` and `None`.

- [ ] **Step 2: Add explicit defaulted parameters**

Add `mpc_predictor_model="candidate_b"` and `mpc_predictor_artifact_path=None` to:

- `create_mpc_flow_controller()`;
- `thermal_control_strategies.create_controller()`;
- `thermal_case_simulator.simulate_case()`.

Normalize at the outermost MPC factory and pass the canonical name and resolved artifact path into both forward and reverse predictor instances. Do not encode the predictor name into `mpc_flow_mode`. P/L with a missing artifact path must log a fallback and use Candidate B; Candidate B ignores the external artifact argument.

- [ ] **Step 3: Add final-runner CLI propagation**

Add to both final runners:

```python
parser.add_argument("--mpc-predictor-model", choices=("candidate_b", "physics_p", "lpv_l"), default="candidate_b")
parser.add_argument("--mpc-predictor-artifact", type=Path, default=None)
```

`run_final_controller_comparison.py` passes both values into `simulate_case()`. `run_final_controller_parallel.py` includes both flags in every worker command, preserving the absolute artifact path. Add tests asserting worker command construction preserves the path exactly.

- [ ] **Step 4: Verify all flow modes and commit**

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q test_mpc_predictor_selection test_final_comparison_flow_modes test_final_controller_comparison test_final_controller_parallel
git add -- thermal_control_strategies.py thermal_case_simulator.py mpc_flow_direction_strategies.py run_final_controller_comparison.py run_final_controller_parallel.py test_mpc_predictor_selection.py test_final_comparison_flow_modes.py test_final_controller_parallel.py
git commit -m "feat: 贯通MPC预测模型选择参数"
```

---

### Task 13: Run full identification, fitting and independent offline validation

**Files:**
- Generated only under: `outputs/mpc_predictor_identification_v1/`
- Generated only under: `outputs/mpc_predictor_comparison_v1/`

- [ ] **Step 1: Generate the full steady and dynamic datasets**

Run locally if stable, otherwise use the same commands on the server:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' generate_predictor_identification_data.py --dataset all --mode full --seed 20260714 --output-root outputs/mpc_predictor_identification_v1
```

Expected: schema validation passes, no scenario leakage, full metadata and hashes written.

- [ ] **Step 2: Freeze B0 before fitting P or L**

Run the evaluator in `--baseline-only` mode and save B0 metrics. Copy no parameters to `model_data/`.

- [ ] **Step 3: Fit P and L independently**

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' fit_mpc_physics_predictor.py --steady-csv outputs/mpc_predictor_identification_v1/steady_full.csv --dynamic-csv outputs/mpc_predictor_identification_v1/dynamic_full.csv --output outputs/mpc_predictor_comparison_v1/physics_p_v1.json
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' fit_mpc_lpv_predictor.py --dynamic-csv outputs/mpc_predictor_identification_v1/dynamic_full.csv --output outputs/mpc_predictor_comparison_v1/lpv_l_v1.json
```

Expected: neither fitter reads test rows during fitting or validation selection.

- [ ] **Step 4: Run independent test comparison**

Run `evaluate_mpc_predictors.py` with frozen artifacts and save all five reports. Confirm there is no fusion row and no automatic model-default change.

- [ ] **Step 5: Record actual pass/fail without promoting artifacts**

Write a concise Chinese `outputs/mpc_predictor_comparison_v1/双预测模型离线结论.md` containing exact B0/P/L values, threshold failures, data hashes and recommended next runtime candidate. This report remains ignored until the user decides whether to promote a model.

---

### Task 14: Benchmark online MPC and run staged closed-loop validation

**Files:**
- Generated only under distinct versioned output roots

- [ ] **Step 1: Benchmark construction and one-step solve time**

For B0, P and L, run the same initial state and forecast at least30 times after one warmup. Record median, P95, maximum, failures and fallback count.

- [ ] **Step 2: Run open-loop dynamic checks**

Verify compressor commands 1000, 1999, 2000, 4000 and6000 rpm, plus pump and ambient variations. Compare aligned `Q_evap_cmd`, `Q_cond`, `Q_evap` and temperature trajectories.

- [ ] **Step 3: Run separate 60-step peak/frequency closed-loop smokes**

Use separate output roots:

```text
outputs/mpc_predictor_comparison_v1/closed_loop_candidate_b/
outputs/mpc_predictor_comparison_v1/closed_loop_physics_p/
outputs/mpc_predictor_comparison_v1/closed_loop_lpv_l/
```

Run each model explicitly:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' run_final_controller_comparison.py --max-steps 60 --output-root outputs/mpc_predictor_comparison_v1/closed_loop_candidate_b --mpc-predictor-model candidate_b
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' run_final_controller_comparison.py --max-steps 60 --output-root outputs/mpc_predictor_comparison_v1/closed_loop_physics_p --mpc-predictor-model physics_p --mpc-predictor-artifact outputs/mpc_predictor_comparison_v1/physics_p_v1.json
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' run_final_controller_comparison.py --max-steps 60 --output-root outputs/mpc_predictor_comparison_v1/closed_loop_lpv_l --mpc-predictor-model lpv_l --mpc-predictor-artifact outputs/mpc_predictor_comparison_v1/lpv_l_v1.json
```

Do not overwrite previous Candidate B smoke or formal results.

- [ ] **Step 4: Stop on failures before formal runs**

If either new model has NaN, GEKKO failure, fallback, P95 runtime over120% of B0, or any horizon MAE degradation over10%, mark it failed and do not run its formal server cases.

- [ ] **Step 5: Run eligible formal MPC-only server cases**

Run only eligible models on peak/freq single/double MPC cases in separate directories. Do not run PID, On-Off, PSO or a fusion model. For example, an eligible P run uses:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' run_final_controller_parallel.py --case-indices 4,5,10,11 --output-root outputs/mpc_predictor_comparison_v1/formal_physics_p --mpc-predictor-model physics_p --mpc-predictor-artifact outputs/mpc_predictor_comparison_v1/physics_p_v1.json --workers 4
```

Use the equivalent `lpv_l` command only if L passes the stop gate.

---

### Task 15: Final verification, documentation and handoff

**Files:**
- Modify: `README.md`
- Modify: `PROJECT_MAP_MIN.md` only if new entrypoints pass validation
- Test: all retained root Python files and selected regression modules

- [ ] **Step 1: Add README commands and model-state wording**

Document exact smoke/full commands, output roots, Candidate B default, manual P/L selection and the fact that P/L are independent. Do not claim either is formal until acceptance results support it.

- [ ] **Step 2: Compile all retained root Python files**

```powershell
$files = Get-ChildItem -File -Filter '*.py' | Select-Object -ExpandProperty FullName
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m py_compile @files
```

Expected: exit code0. Delete the regenerated root `__pycache__/` only if the user has authorized cache cleanup for this execution.

- [ ] **Step 3: Run the full relevant regression suite**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q test_predictor_identification_data test_generate_predictor_identification_data test_mpc_predictor_selection test_mpc_physics_predictor test_mpc_lpv_predictor test_evaluate_mpc_predictors test_mpc_predictor_runtime test_model_data_paths test_mpc_evaporator_capacity_model test_mpc_evaporator_capacity_table test_mpc_reduced_model_calibration test_final_comparison_flow_modes test_final_controller_comparison test_final_controller_configured_pid test_final_controller_parallel test_final_controller_parallel_pure test_final_controller_parallel_system_tmp test_plot_final_mpc_report
```

Expected: all tests pass with zero failures.

- [ ] **Step 4: Verify Git and artifact boundaries**

```powershell
$forbidden = git ls-files | Where-Object { $_ -match '^(outputs/|data/|_archive/|tmp|_gekko_tmp_)' }
if ($forbidden) { $forbidden; throw 'Generated data entered Git' }
git status --short --branch
```

Expected: only intended source, test and documentation changes are tracked; no provisional P/L artifacts in `model_data/`.

- [ ] **Step 5: Commit documentation**

```powershell
git add -- README.md PROJECT_MAP_MIN.md
git commit -m "docs: 说明MPC双预测模型工作流"
```

- [ ] **Step 6: Present the evidence and request a separate promotion decision**

Report:

- exact B0/P/L offline metrics;
- exact solve-time metrics;
- closed-loop/formal outcomes;
- every fallback or failed threshold;
- which model, if any, qualifies for promotion;
- the exact two provisional artifact paths.

Do not copy artifacts into `model_data/`, change the default predictor or push a formal-model selection without a new explicit user decision.
