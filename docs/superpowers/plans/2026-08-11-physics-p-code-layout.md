# Physics-P Code Layout Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Physics-P 的生产入口、正式参数资产、实验脚本和测试整理到职责明确的目录，同时保持 Candidate B/Physics-P 隔离、控制参数和闭环数值行为不变。

**Architecture:** 根目录只保留稳定生产模块和唯一正式入口；`p_mpc_run_support.py` 提供不依赖实验包的项目根路径、场景、输入定位和汇总接口。辨识、调参、评估、校准按依赖顺序迁入 `experiments.physics_p`，P 专属测试迁入 `tests.physics_p`，正式资产原字节迁入 `model_data`。

**Tech Stack:** Python 3、`unittest`、NumPy、pandas、Matplotlib、GEKKO、PowerShell、Git worktree。

---

## Implementation Context

- 工作树：`C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\.worktrees\physics-p-layout`
- 分支：`codex/physics-p-layout`
- 功能基线：`1794265622bf73823ef4ba5ef78b6bf247207b3e`
- 已批准并修正的设计：`docs/superpowers/specs/2026-08-11-physics-p-code-layout-design.md`
- Python：`C:\Users\24776\miniforge3\envs\btms\python.exe`
- 只使用 `unittest`；不要安装 `pytest`。
- 所有命令从上述 worktree 根目录执行。
- 所有内容修改使用 `apply_patch`；现有受控文件的纯路径迁移使用 `git mv`。
- 不修改 `mpc_flow_direction_strategies.py`、`thermal_loop.py`、`thermal_system.py`、Candidate B 默认值、35/25 °C 边界、P 权重、14/12 步 operational horizon 或求解恢复逻辑。
- 不触碰主工作目录的 TD3 未跟踪文件，也不改写完整 P 结果。
- 任一任务的预期测试失败时立即停在该任务，不提交红色状态，不通过改参数、删断言或 skip 测试绕过。已提交任务如需整体回滚，使用针对该任务提交的 `git revert <commit>`，不使用 `git reset --hard`。

## Locked File Structure

### Create

- `p_mpc_run_support.py`：生产安全的项目根路径、场景、输入定位和通用汇总。
- `experiments/__init__.py`：实验顶层包。
- `experiments/physics_p/__init__.py`：Physics-P 实验包。
- `experiments/physics_p/identification/__init__.py`：辨识子包。
- `experiments/physics_p/calibration/__init__.py`：校准子包。
- `experiments/physics_p/evaluation/__init__.py`：评估子包。
- `experiments/physics_p/tuning/__init__.py`：调参子包。
- `experiments/physics_p/evaluation/plot_p_mpc_local_formal_results.py`：补齐当前 clean checkout 缺失的已有正式实验绘图 helper。
- `experiments/physics_p/README.md`：唯一有效的旧命令迁移表。
- `tests/__init__.py`、`tests/physics_p/__init__.py`：显式测试包。
- `tests/physics_p/test_layout_contract.py`：生产依赖、路径和目录契约。

### Keep at repository root

- `run_p_mpc_operational.py`
- `mpc_physics_predictor.py`
- `mpc_physics_shadow.py`
- `mpc_predictor_selection.py`
- `p_mpc_run_support.py`

### Move

- 20 个根目录实验脚本迁入 `experiments/physics_p/{identification,tuning,evaluation,calibration}`。
- 21 个根目录 P 测试迁入 `tests/physics_p/`。
- 正式资产迁至 `model_data/physics_p_operational_v1.json`。

### Modify but do not move

- `README.md`
- `PROJECT_MAP_MIN.md`
- `CODEX_README_MIN.md`
- `test_model_data_paths.py`

## Import Rules

```text
run_p_mpc_operational.py -> p_mpc_run_support.py -> stable root modules
experiments.physics_p.* -> p_mpc_run_support.py / stable root modules / relative experiment imports
tests.physics_p.* -> stable root modules / experiments.physics_p.*
```

禁止生产根模块导入 `experiments.*`，禁止 `sys.path` 注入，禁止根目录兼容 wrapper。三个库模块没有 CLI：

- `experiments.physics_p.identification.predictor_identification_data`
- `experiments.physics_p.identification.mpc_lpv_predictor`
- `experiments.physics_p.identification.evaluate_mpc_predictors`

---

### Task 1: Capture the pre-move smoke and extract production run support

**Files:**
- Create: `p_mpc_run_support.py`
- Create: `experiments/__init__.py`
- Create: `experiments/physics_p/__init__.py`
- Create: `experiments/physics_p/identification/__init__.py`
- Create: `experiments/physics_p/calibration/__init__.py`
- Create: `experiments/physics_p/evaluation/__init__.py`
- Create: `experiments/physics_p/tuning/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/physics_p/__init__.py`
- Create: `tests/physics_p/test_layout_contract.py`
- Modify: `run_p_mpc_operational.py:14-44`
- Modify: `run_p_mpc_short_comparison.py:10-165`
- Test: `test_p_mpc_operational.py`
- Test: `test_p_mpc_controller_tuning.py`
- Test: `test_p_mpc_local_formal.py`

- [ ] **Step 1: Verify the implementation worktree and reserve a new baseline output path**

Run:

```powershell
git status --short --branch
$python = 'C:\Users\24776\miniforge3\envs\btms\python.exe'
$pre = 'outputs/p_mpc_layout_smoke_v1/pre_move_20260811'
if (Test-Path -LiteralPath $pre) {
    throw "Output already exists: $pre"
}
```

Expected: branch is `codex/physics-p-layout`; tracked worktree is clean; `$pre` does not exist.

- [ ] **Step 2: Generate the non-overwriting six-step pre-move reference**

Run:

```powershell
& $python run_p_mpc_operational.py `
  --scenes peak freq `
  --steps 6 `
  --output-root $pre
```

Expected:

```text
RESULT scene=peak ... qualified=True
RESULT scene=freq ... qualified=True
SUMMARY outputs\p_mpc_layout_smoke_v1\pre_move_20260811\operational_summary.csv
```

Required files:

```text
outputs/p_mpc_layout_smoke_v1/pre_move_20260811/operational_summary.csv
outputs/p_mpc_layout_smoke_v1/pre_move_20260811/mpc/peak_physics_p_operational_steps6_horizon14.csv
outputs/p_mpc_layout_smoke_v1/pre_move_20260811/mpc/freq_physics_p_operational_steps6_horizon12.csv
```

- [ ] **Step 3: Add explicit package markers and the first failing layout tests**

Create the package files with exactly these contents:

```python
# experiments/__init__.py
"""Runnable experiment packages for the BTMS repository."""

# experiments/physics_p/__init__.py
"""Physics-P identification, calibration, evaluation, and tuning experiments."""

# experiments/physics_p/identification/__init__.py
"""Physics-P and LPV identification experiments."""

# experiments/physics_p/calibration/__init__.py
"""Physics-P calibration and artifact-building experiments."""

# experiments/physics_p/evaluation/__init__.py
"""Physics-P evaluation and plotting experiments."""

# experiments/physics_p/tuning/__init__.py
"""Physics-P controller tuning and formal experiment runners."""

# tests/__init__.py
"""Repository test packages."""

# tests/physics_p/__init__.py
"""Physics-P regression tests."""
```

Create `tests/physics_p/test_layout_contract.py`:

```python
import ast
import unittest
from pathlib import Path

import p_mpc_run_support as support
import run_p_mpc_operational as operational


class PhysicsPLayoutContractTest(unittest.TestCase):
    def test_operational_runner_uses_shared_run_support(self):
        self.assertIs(operational.SCENES, support.SCENES)
        self.assertIs(operational.source_csv_for_scene, support.source_csv_for_scene)
        self.assertIs(operational.summarize_run, support.summarize_run)
        self.assertEqual(support.PROJECT_ROOT, Path(__file__).resolve().parents[2])

    def test_production_runner_does_not_import_experiment_modules(self):
        tree = ast.parse(Path(operational.__file__).read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        self.assertEqual(
            [name for name in imported if name == "experiments" or name.startswith("experiments.")],
            [],
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: Run the new test to verify it fails before support extraction**

Run:

```powershell
& $python -m unittest -v tests.physics_p.test_layout_contract
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'p_mpc_run_support'`.

- [ ] **Step 5: Implement `p_mpc_run_support.py` with the exact extracted behavior**

Create:

```python
"""Shared, production-safe support for Physics-P MPC runners."""

from pathlib import Path

import numpy as np
import pandas as pd

from thermal_batch_config import CASES


PROJECT_ROOT = Path(__file__).resolve().parent
SCENES = {
    "peak": "调峰",
    "freq": "调频",
}


def source_csv_for_scene(scene):
    matches = [
        case.source_csv
        for case in CASES
        if case.control == "mpc" and case.scene == scene and case.flow == "单向"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one MPC single-flow source for {scene}, got {len(matches)}"
        )
    return matches[0]


def _column(frame, *names):
    for name in names:
        if name in frame.columns:
            return frame[name]
    raise KeyError(f"Missing required columns: {names}")


def summarize_run(frame, *, scene_key, predictor, horizon, artifact_path, out_csv):
    temperature = _column(
        frame,
        "Average temperature",
        "Battery Temp (C)",
    ).astype(float)
    energy = _column(
        frame,
        "Cumulative energy consumption",
        "Cumulative Energy (kWh)",
    ).astype(float)
    power = _column(frame, "Total power", "Total Power (kW)").astype(float)
    solve_time = _column(frame, "MPC solve time", "MPC_Solve_Time_S").astype(float)
    solved = _column(frame, "MPC_Solved").astype(str).str.lower().isin(("true", "1"))
    n_comp = _column(
        frame,
        "Compressor command",
        "Compressor Command (RPM)",
    ).astype(float)
    n_pump = _column(frame, "Pump command", "Pump Command (RPM)").astype(float)
    target_error = temperature - 25.0
    finite_solve_time = solve_time[np.isfinite(solve_time)]
    if "MPC solve error" in frame.columns:
        solve_errors = frame["MPC solve error"].fillna("").astype(str).str.strip()
    elif "MPC_Solve_Error" in frame.columns:
        solve_errors = frame["MPC_Solve_Error"].fillna("").astype(str).str.strip()
    else:
        solve_errors = pd.Series([""] * len(frame), index=frame.index, dtype=str)
    unique_solve_errors = list(dict.fromkeys(solve_errors[solve_errors != ""]))
    recovery_used_column = next(
        (
            name
            for name in ("MPC solve recovery used", "MPC_Solve_Recovery_Used")
            if name in frame.columns
        ),
        None,
    )
    if recovery_used_column is not None:
        recovery_used = (
            frame[recovery_used_column]
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin(("true", "1"))
        )
    else:
        recovery_used = pd.Series(False, index=frame.index, dtype=bool)
    recovery_reason_column = next(
        (
            name
            for name in ("MPC solve recovery reason", "MPC_Solve_Recovery_Reason")
            if name in frame.columns
        ),
        None,
    )
    if recovery_reason_column is not None:
        recovery_reasons = (
            frame[recovery_reason_column].fillna("").astype(str).str.strip()
        )
    else:
        recovery_reasons = pd.Series([""] * len(frame), index=frame.index, dtype=str)
    unique_recovery_reasons = list(
        dict.fromkeys(recovery_reasons[recovery_reasons != ""])
    )
    if "MPC_Strict_Predictor_Ablation" in frame.columns:
        strict_predictor_ablation = bool(
            frame["MPC_Strict_Predictor_Ablation"]
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin(("true", "1"))
            .all()
        )
    else:
        strict_predictor_ablation = False
    return {
        "scene": scene_key,
        "predictor": predictor,
        "strict_predictor_ablation": strict_predictor_ablation,
        "steps": int(len(frame)),
        "horizon_steps": int(horizon),
        "temperature_mae_c": float(np.mean(np.abs(target_error))),
        "temperature_max_c": float(np.max(temperature)),
        "temperature_final_c": float(temperature.iloc[-1]),
        "energy_kwh": float(energy.iloc[-1]),
        "mean_power_kw": float(np.mean(power)),
        "mean_n_comp_rpm": float(np.mean(n_comp)),
        "mean_n_pump_rpm": float(np.mean(n_pump)),
        "solve_success_rate": float(np.mean(solved)),
        "solve_time_mean_s": (
            float(np.mean(finite_solve_time)) if len(finite_solve_time) else np.nan
        ),
        "solve_time_p95_s": (
            float(np.percentile(finite_solve_time, 95))
            if len(finite_solve_time)
            else np.nan
        ),
        "solve_recovery_count": int(np.count_nonzero(recovery_used)),
        "solve_recovery_reasons": " | ".join(unique_recovery_reasons),
        "solve_error_count": int(np.count_nonzero(solve_errors != "")),
        "solve_error_messages": " | ".join(unique_solve_errors),
        "artifact": str(artifact_path) if artifact_path is not None else "",
        "out_csv": str(out_csv),
    }
```

- [ ] **Step 6: Rewire both runners without changing their experiment-specific behavior**

In `run_p_mpc_operational.py`, replace the import from `run_p_mpc_short_comparison` and the local `PROJECT_ROOT` with:

```python
from p_mpc_run_support import (
    PROJECT_ROOT,
    SCENES,
    source_csv_for_scene,
    summarize_run,
)
```

Keep `DEFAULT_OPERATIONAL_P_ARTIFACT` on its current old path until Task 2.

In `run_p_mpc_short_comparison.py`, add:

```python
from p_mpc_run_support import (
    PROJECT_ROOT,
    SCENES,
    source_csv_for_scene,
    summarize_run,
)
```

Remove only these local definitions from `run_p_mpc_short_comparison.py`:

```text
PROJECT_ROOT
SCENES
source_csv_for_scene
_column
summarize_run
```

Also remove `from thermal_batch_config import CASES`. Keep NumPy, pandas, `DEFAULT_P_ARTIFACT`, `DEFAULT_OUTPUT_ROOT`, `build_pairwise_summary`, and `run_comparison` unchanged.

- [ ] **Step 7: Run the support and dependent runner regressions**

Run:

```powershell
& $python -m unittest `
  tests.physics_p.test_layout_contract `
  test_p_mpc_operational `
  test_p_mpc_controller_tuning `
  test_p_mpc_local_formal
```

Expected: `Ran 34 tests` and `OK`.

- [ ] **Step 8: Commit the production support boundary**

```powershell
git add -- `
  p_mpc_run_support.py `
  run_p_mpc_operational.py `
  run_p_mpc_short_comparison.py `
  experiments/__init__.py `
  experiments/physics_p/__init__.py `
  experiments/physics_p/identification/__init__.py `
  experiments/physics_p/calibration/__init__.py `
  experiments/physics_p/evaluation/__init__.py `
  experiments/physics_p/tuning/__init__.py `
  tests/__init__.py `
  tests/physics_p/__init__.py `
  tests/physics_p/test_layout_contract.py
git diff --cached --check
git commit -m "refactor: isolate Physics-P run support"
```

Expected: one focused commit; ignored smoke outputs are not staged.

---

### Task 2: Move the operational Physics-P asset into tracked model data

**Files:**
- Move: `outputs/mpc_predictor_low_speed_retrain_v1/thermal_bias_correction_v3/physics_p_heat_generation_corrected.json` → `model_data/physics_p_operational_v1.json`
- Modify: `run_p_mpc_operational.py:36-44`
- Modify: `test_p_mpc_operational.py:89-104`
- Modify: `test_model_data_paths.py:1-36`

- [ ] **Step 1: Change the tests to require the new path**

Add to `test_model_data_paths.py`:

```python
import run_p_mpc_operational as p_operational
```

Add to `test_p_mpc_operational.py`:

```python
from p_mpc_run_support import PROJECT_ROOT
```

Append to `test_default_paths_use_tracked_model_data_directory`:

```python
self.assertEqual(
    p_operational.DEFAULT_OPERATIONAL_P_ARTIFACT,
    MODEL_DATA_ROOT / "physics_p_operational_v1.json",
)
```

Append to the `required` tuple:

```python
MODEL_DATA_ROOT / "physics_p_operational_v1.json",
```

Replace the operational asset test with:

```python
def test_default_artifact_is_tracked_operational_p_model(self):
    from run_p_mpc_operational import displacement_scale_from_artifact

    self.assertEqual(
        DEFAULT_OPERATIONAL_P_ARTIFACT,
        PROJECT_ROOT / "model_data" / "physics_p_operational_v1.json",
    )
    self.assertTrue(DEFAULT_OPERATIONAL_P_ARTIFACT.is_file())
    self.assertEqual(
        displacement_scale_from_artifact(DEFAULT_OPERATIONAL_P_ARTIFACT),
        1.0,
    )
```

- [ ] **Step 2: Run the path tests and verify they fail**

Run:

```powershell
& $python -m unittest -v `
  test_p_mpc_operational.PhysicsPMpcOperationalTest.test_default_artifact_is_tracked_operational_p_model `
  test_model_data_paths.ModelDataPathTest
```

Expected: FAIL because the constant and file still use the old `outputs/...` path.

- [ ] **Step 3: Verify the source bytes, move with Git, and update the constant**

Run before the move:

```powershell
$oldAsset = 'outputs/mpc_predictor_low_speed_retrain_v1/thermal_bias_correction_v3/physics_p_heat_generation_corrected.json'
(Get-FileHash -Algorithm SHA256 -LiteralPath $oldAsset).Hash
```

Expected SHA-256:

```text
0BFE23DF636193F954EA4BFA0E457EE29F64720A88995AB818CAE9265E0109C0
```

Move:

```powershell
git mv -- $oldAsset model_data/physics_p_operational_v1.json
```

Set the constant to exactly:

```python
DEFAULT_OPERATIONAL_P_ARTIFACT = (
    PROJECT_ROOT / "model_data" / "physics_p_operational_v1.json"
)
```

Do not read/write or normalize the asset. It contains 48 historical `NaN` tokens that the existing Python loader accepts.

- [ ] **Step 4: Verify the asset and path tests pass**

Run:

```powershell
(Get-FileHash -Algorithm SHA256 -LiteralPath 'model_data/physics_p_operational_v1.json').Hash
git ls-files --error-unmatch model_data/physics_p_operational_v1.json
& $python -m unittest -v `
  test_p_mpc_operational.PhysicsPMpcOperationalTest.test_default_artifact_is_tracked_operational_p_model `
  test_model_data_paths.ModelDataPathTest
```

Expected: same SHA-256, tracked path printed, `Ran 3 tests`, `OK`.

- [ ] **Step 5: Commit the byte-preserving asset migration**

```powershell
git add -- run_p_mpc_operational.py test_p_mpc_operational.py test_model_data_paths.py
git diff --cached --check
git commit -m "refactor: move Physics-P runtime asset to model data"
```

---

### Task 3: Move the identification package and its tests

**Files:**
- Move: `predictor_identification_data.py`
- Move: `generate_predictor_identification_data.py`
- Move: `fit_mpc_physics_predictor.py`
- Move: `fit_mpc_lpv_predictor.py`
- Move: `mpc_lpv_predictor.py`
- Move: `evaluate_mpc_predictors.py`
- Move: `evaluate_p_identification_rollouts.py`
- Move tests: `test_predictor_identification_data.py`, `test_generate_predictor_identification_data.py`, `test_evaluate_mpc_predictors.py`, `test_mpc_lpv_predictor.py`, `test_mpc_physics_predictor.py`
- Modify temporarily: `refine_p_thermal_capacities.py` so it remains importable after the fit module moves

- [ ] **Step 1: Point the five tests at the future package names**

Use these exact import replacements:

```text
predictor_identification_data
  -> experiments.physics_p.identification.predictor_identification_data
generate_predictor_identification_data
  -> experiments.physics_p.identification.generate_predictor_identification_data
evaluate_mpc_predictors
  -> experiments.physics_p.identification.evaluate_mpc_predictors
fit_mpc_lpv_predictor
  -> experiments.physics_p.identification.fit_mpc_lpv_predictor
mpc_lpv_predictor
  -> experiments.physics_p.identification.mpc_lpv_predictor
fit_mpc_physics_predictor
  -> experiments.physics_p.identification.fit_mpc_physics_predictor
```

In `test_generate_predictor_identification_data.py`, replace all 86 patch-target prefixes:

```text
generate_predictor_identification_data.
  -> experiments.physics_p.identification.generate_predictor_identification_data.
```

Replace all eight simulated argv program names:

```python
"experiments.physics_p.identification.generate_predictor_identification_data"
```

Import the generator module once for source provenance:

```python
import experiments.physics_p.identification.generate_predictor_identification_data as generator_module
```

Replace the old root source read with:

```python
source = Path(generator_module.__file__).read_text(encoding="utf-8")
```

In `test_mpc_physics_predictor.py`, replace all 15 patch-target prefixes:

```text
fit_mpc_physics_predictor.
  -> experiments.physics_p.identification.fit_mpc_physics_predictor.
```

Add:

```python
from p_mpc_run_support import PROJECT_ROOT
```

and change the CLI boundary test to:

```python
project = PROJECT_ROOT
```

- [ ] **Step 2: Run one future import and verify it fails before the move**

Run:

```powershell
& $python -m unittest -v test_predictor_identification_data
```

Expected: FAIL with `ModuleNotFoundError` for `experiments.physics_p.identification.predictor_identification_data`.

- [ ] **Step 3: Move the seven modules and five tests**

```powershell
git mv -- predictor_identification_data.py experiments/physics_p/identification/predictor_identification_data.py
git mv -- generate_predictor_identification_data.py experiments/physics_p/identification/generate_predictor_identification_data.py
git mv -- fit_mpc_physics_predictor.py experiments/physics_p/identification/fit_mpc_physics_predictor.py
git mv -- fit_mpc_lpv_predictor.py experiments/physics_p/identification/fit_mpc_lpv_predictor.py
git mv -- mpc_lpv_predictor.py experiments/physics_p/identification/mpc_lpv_predictor.py
git mv -- evaluate_mpc_predictors.py experiments/physics_p/identification/evaluate_mpc_predictors.py
git mv -- evaluate_p_identification_rollouts.py experiments/physics_p/identification/evaluate_p_identification_rollouts.py

git mv -- test_predictor_identification_data.py tests/physics_p/test_predictor_identification_data.py
git mv -- test_generate_predictor_identification_data.py tests/physics_p/test_generate_predictor_identification_data.py
git mv -- test_evaluate_mpc_predictors.py tests/physics_p/test_evaluate_mpc_predictors.py
git mv -- test_mpc_lpv_predictor.py tests/physics_p/test_mpc_lpv_predictor.py
git mv -- test_mpc_physics_predictor.py tests/physics_p/test_mpc_physics_predictor.py
```

- [ ] **Step 4: Repair package-relative imports and root-sensitive defaults**

Apply these exact code changes:

```python
# generate_predictor_identification_data.py
# place this after ensure_env_library_bin_on_path()
from p_mpc_run_support import PROJECT_ROOT
from .predictor_identification_data import (
    REQUIRED_COLUMNS,
    VALID_SPLITS,
    assign_scenario_splits,
    validate_identification_frame,
)

# argparse default only; keep Path(__file__) provenance reads unchanged
default=PROJECT_ROOT / "outputs" / "mpc_predictor_identification_v1"
```

```python
# fit_mpc_physics_predictor.py
# place this after ensure_env_library_bin_on_path()
from p_mpc_run_support import PROJECT_ROOT
from .predictor_identification_data import validate_identification_frame

# inside _validate_output_path
model_data = (PROJECT_ROOT / "model_data").resolve()
```

```python
# fit_mpc_lpv_predictor.py
from .mpc_lpv_predictor import (
    DISTURBANCE_NAMES,
    INPUT_NAMES,
    STATE_NAMES,
    augmented_state_names,
    spectral_radius_grid,
    validate_lpv_artifact,
)
from .predictor_identification_data import validate_identification_frame

# the function-local import at the old line 225
from .mpc_lpv_predictor import scheduled_matrices
```

```python
# evaluate_p_identification_rollouts.py
from .predictor_identification_data import validate_identification_frame
```

Because `refine_p_thermal_capacities.py` remains at the root until Task 6, replace its fit import now with:

```python
from experiments.physics_p.identification.fit_mpc_physics_predictor import (
    _dynamic_validation_metric,
)
```

Do not alter stable root imports in these modules.

- [ ] **Step 5: Run identification imports, CLI smoke, and all five test modules**

Run:

```powershell
& $python -m experiments.physics_p.identification.generate_predictor_identification_data --help
& $python -m experiments.physics_p.identification.fit_mpc_physics_predictor --help
& $python -m experiments.physics_p.identification.fit_mpc_lpv_predictor --help
& $python -m experiments.physics_p.identification.evaluate_p_identification_rollouts --help

& $python -m unittest `
  tests.physics_p.test_predictor_identification_data `
  tests.physics_p.test_generate_predictor_identification_data `
  tests.physics_p.test_evaluate_mpc_predictors `
  tests.physics_p.test_mpc_lpv_predictor `
  tests.physics_p.test_mpc_physics_predictor
```

Expected: four help commands exit 0; `Ran 141 tests`, `OK`.

- [ ] **Step 6: Commit the identification package**

```powershell
git add -- `
  experiments/physics_p/identification `
  tests/physics_p `
  refine_p_thermal_capacities.py
git diff --cached --check
git commit -m "refactor: package Physics-P identification tools"
```

---

### Task 4: Move tuning and formal experiment runners

**Files:**
- Move: `run_p_mpc_short_comparison.py`
- Move: `run_p_mpc_controller_tuning.py`
- Move: `run_p_mpc_local_formal.py`
- Move tests: `test_p_mpc_controller_tuning.py`, `test_p_mpc_local_formal.py`
- Modify temporarily: `run_p_model_boundary_validation.py`, `refine_p_thermal_capacities.py`

- [ ] **Step 1: Change tests to future imports and future provenance keys**

Replace test imports with:

```python
from experiments.physics_p.tuning.run_p_mpc_controller_tuning import (
    _candidate_overrides,
    _select_candidates,
)
```

```python
import experiments.physics_p.tuning.run_p_mpc_local_formal as formal_runner
from experiments.physics_p.tuning.run_p_mpc_local_formal import (
    FORMAL_LOCAL_CASES,
    FORMAL_PROGRESS_INTERVAL_STEPS,
    formal_local_cases,
    formal_scene_control_profiles,
    parse_progress_message,
)
```

Import the tracked operational artifact for hermetic Physics-P test fixtures:

```python
from run_p_mpc_operational import DEFAULT_OPERATIONAL_P_ARTIFACT
```

In these three tests, pass the tracked fixture explicitly to
`formal_runner.run_local_formal(...)`:

```python
# test_explicit_physics_p_records_native_bounds
# test_strict_ablation_records_common_scope_and_propagates_flag
# test_stdout_failure_does_not_turn_completed_run_into_failure
artifact_path=DEFAULT_OPERATIONAL_P_ARTIFACT,
```

The three tests currently fail on a clean checkout because the tuning runner's
historical default artifact lives under ignored `outputs/`. Do not copy, stage,
or replace that experimental default; only make the tests hermetic with the
already tracked operational artifact.

Change the provenance lookup to:

```python
source_hash = provenance["source_files"][
    "experiments/physics_p/tuning/run_p_mpc_local_formal.py"
]["sha256"]
```

- [ ] **Step 2: Verify the future tuning import fails**

Run:

```powershell
& $python -m unittest -v test_p_mpc_controller_tuning
```

Expected: FAIL with `ModuleNotFoundError` for `experiments.physics_p.tuning.run_p_mpc_controller_tuning`.

- [ ] **Step 3: Move three runners and two tests**

```powershell
git mv -- run_p_mpc_short_comparison.py experiments/physics_p/tuning/run_p_mpc_short_comparison.py
git mv -- run_p_mpc_controller_tuning.py experiments/physics_p/tuning/run_p_mpc_controller_tuning.py
git mv -- run_p_mpc_local_formal.py experiments/physics_p/tuning/run_p_mpc_local_formal.py
git mv -- test_p_mpc_controller_tuning.py tests/physics_p/test_p_mpc_controller_tuning.py
git mv -- test_p_mpc_local_formal.py tests/physics_p/test_p_mpc_local_formal.py
```

- [ ] **Step 4: Repair tuning imports, default roots, and provenance**

In `run_p_mpc_controller_tuning.py`, use:

```python
from p_mpc_run_support import (
    PROJECT_ROOT,
    SCENES,
    source_csv_for_scene,
    summarize_run,
)
from .run_p_mpc_short_comparison import DEFAULT_P_ARTIFACT

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "p_mpc_controller_tuning_v1"
```

In `run_p_mpc_local_formal.py`, use:

```python
from p_mpc_run_support import PROJECT_ROOT
from .run_p_mpc_short_comparison import DEFAULT_P_ARTIFACT, run_comparison
```

Replace `FORMAL_SOURCE_FILES` with:

```python
FORMAL_SOURCE_FILES = (
    "experiments/physics_p/tuning/run_p_mpc_local_formal.py",
    "experiments/physics_p/tuning/run_p_mpc_short_comparison.py",
    "p_mpc_run_support.py",
    "mpc_flow_direction_strategies.py",
    "mpc_physics_predictor.py",
    "thermal_case_simulator.py",
    "thermal_loop.py",
    "thermal_batch_config.py",
)
```

In moved `run_p_mpc_short_comparison.py`, keep the Task 1 support imports, experimental `DEFAULT_P_ARTIFACT`, and output path unchanged.

The boundary and calibration scripts still at the root must remain importable between commits. Update them temporarily to:

```python
# run_p_model_boundary_validation.py
from experiments.physics_p.tuning.run_p_mpc_short_comparison import (
    DEFAULT_P_ARTIFACT,
)

# refine_p_thermal_capacities.py
from experiments.physics_p.tuning.run_p_mpc_short_comparison import (
    DEFAULT_P_ARTIFACT,
)
```

- [ ] **Step 5: Run tuning tests and module help**

```powershell
& $python -m experiments.physics_p.tuning.run_p_mpc_short_comparison --help
& $python -m experiments.physics_p.tuning.run_p_mpc_controller_tuning --help
& $python -m experiments.physics_p.tuning.run_p_mpc_local_formal --help
& $python -m unittest `
  tests.physics_p.test_p_mpc_controller_tuning `
  tests.physics_p.test_p_mpc_local_formal
& $python -c "import run_p_model_boundary_validation, refine_p_thermal_capacities"
```

Expected: three help commands exit 0; `Ran 11 tests`, `OK`.

- [ ] **Step 6: Commit the tuning package**

```powershell
git add -- `
  experiments/physics_p/tuning `
  tests/physics_p `
  run_p_model_boundary_validation.py `
  refine_p_thermal_capacities.py
git diff --cached --check
git commit -m "refactor: package Physics-P tuning runners"
```

---

### Task 5: Move evaluation tools and restore the missing tracked plotting helper

**Files:**
- Move: `evaluate_dual_p_shadow.py`
- Move: `evaluate_p_frozen_mpc_plan.py`
- Move: `evaluate_p_shadow_actual_replay.py`
- Move: `run_p_model_boundary_validation.py`
- Move: `compare_mpc_predictor_formal_results.py`
- Move: `plot_p_peak_displacement_comparison.py`
- Create from existing preserved source: `experiments/physics_p/evaluation/plot_p_mpc_local_formal_results.py`
- Move tests: `test_compare_mpc_predictor_formal_results.py`, `test_evaluate_dual_p_shadow.py`, `test_evaluate_p_frozen_mpc_plan.py`, `test_evaluate_p_shadow_actual_replay.py`, `test_p_model_boundary_validation.py`, `test_plot_p_mpc_local_formal_results.py`
- Modify temporarily: `build_p_heat_generation_corrected_artifact.py`, `calibrate_p_actual_replay.py`, `refine_p_thermal_capacities.py`

- [ ] **Step 1: Change six tests to the future evaluation modules**

Use these exact imports:

```python
from experiments.physics_p.evaluation.compare_mpc_predictor_formal_results import (
    ARCHIVE_ID,
    CURRENT_CANDIDATE_B_ID,
    PHYSICS_P_ID,
    build_current_operational_pairwise_summary,
    build_result_specs,
    historical_comparability_reasons,
    plot_comparison,
    summarize_case,
    validate_current_control_profiles,
    validate_formal_status_payload,
)
from experiments.physics_p.evaluation.evaluate_dual_p_shadow import evaluate_dual_shadow
import experiments.physics_p.evaluation.evaluate_p_frozen_mpc_plan as frozen
from experiments.physics_p.evaluation.evaluate_p_frozen_mpc_plan import (
    _extend_plan,
    evaluate_frozen_forecasts,
)
from experiments.physics_p.evaluation.evaluate_p_shadow_actual_replay import (
    evaluate_actual_command_replay,
)
from experiments.physics_p.evaluation.run_p_model_boundary_validation import (
    DEFAULT_SPEEDS_RPM,
    build_boundary_table,
    summarize_boundary_table,
)
from experiments.physics_p.evaluation.plot_p_mpc_local_formal_results import (
    case_csv_paths,
    summarize_case,
)
```

- [ ] **Step 2: Run the plotting test and verify the clean-checkout gap is visible**

Run:

```powershell
& $python -m unittest -v test_plot_p_mpc_local_formal_results
```

Expected: FAIL because `experiments.physics_p.evaluation.plot_p_mpc_local_formal_results` does not exist yet. Do not skip or delete the test.

- [ ] **Step 3: Move six evaluation scripts and six tests**

```powershell
git mv -- evaluate_dual_p_shadow.py experiments/physics_p/evaluation/evaluate_dual_p_shadow.py
git mv -- evaluate_p_frozen_mpc_plan.py experiments/physics_p/evaluation/evaluate_p_frozen_mpc_plan.py
git mv -- evaluate_p_shadow_actual_replay.py experiments/physics_p/evaluation/evaluate_p_shadow_actual_replay.py
git mv -- run_p_model_boundary_validation.py experiments/physics_p/evaluation/run_p_model_boundary_validation.py
git mv -- compare_mpc_predictor_formal_results.py experiments/physics_p/evaluation/compare_mpc_predictor_formal_results.py
git mv -- plot_p_peak_displacement_comparison.py experiments/physics_p/evaluation/plot_p_peak_displacement_comparison.py

git mv -- test_compare_mpc_predictor_formal_results.py tests/physics_p/test_compare_mpc_predictor_formal_results.py
git mv -- test_evaluate_dual_p_shadow.py tests/physics_p/test_evaluate_dual_p_shadow.py
git mv -- test_evaluate_p_frozen_mpc_plan.py tests/physics_p/test_evaluate_p_frozen_mpc_plan.py
git mv -- test_evaluate_p_shadow_actual_replay.py tests/physics_p/test_evaluate_p_shadow_actual_replay.py
git mv -- test_p_model_boundary_validation.py tests/physics_p/test_p_model_boundary_validation.py
git mv -- test_plot_p_mpc_local_formal_results.py tests/physics_p/test_plot_p_mpc_local_formal_results.py
```

- [ ] **Step 4: Recover the existing plotting helper into the tracked evaluation package**

Read this preserved, untracked source without modifying or deleting it:

```text
C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\.worktrees\dual-mpc-predictors\figures\gen_fig_p_mpc_local_formal.py
```

Verify before transplanting:

```powershell
$plotSource = '..\dual-mpc-predictors\figures\gen_fig_p_mpc_local_formal.py'
if (-not (Test-Path -LiteralPath $plotSource -PathType Leaf)) {
    throw "Missing preserved Physics-P plotting helper: $plotSource"
}
(Get-FileHash -Algorithm SHA256 -LiteralPath $plotSource).Hash
```

Expected source SHA-256:

```text
227B2FFAC5E6803697C724C705EC5888B42AD3DB9416154DF7A550BF5202B83D
```

Using `apply_patch`, create `experiments/physics_p/evaluation/plot_p_mpc_local_formal_results.py` with that source's complete content. Make only this root-path change:

```python
from p_mpc_run_support import PROJECT_ROOT
```

Remove:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
```

Keep the `Path` import because the module uses it elsewhere. Preserve all plotting thresholds, 60/45 historical formal horizons, summary formulas, CLI, filenames, and status behavior.

- [ ] **Step 5: Repair evaluation root paths and cross-package imports**

In `evaluate_p_frozen_mpc_plan.py`, add:

```python
from p_mpc_run_support import PROJECT_ROOT
```

and define:

```python
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "p_shadow_frozen_mpc_plan_active"
ARTIFACTS = {
    "Old": (
        PROJECT_ROOT
        / "outputs"
        / "mpc_predictor_accuracy_correction_v2"
        / "artifacts"
        / "physics_p_dynamic_supply_cp_fixed_v1.json"
    ),
    "New": (
        PROJECT_ROOT
        / "outputs"
        / "p_shadow_actual_command_replay"
        / "calibration"
        / "physics_p_actual_replay_corrected_v3_physical.json"
    ),
}
```

Keep the external `Path.home()/Desktop/...` source-data path unchanged.

In `run_p_model_boundary_validation.py`, use:

```python
from p_mpc_run_support import PROJECT_ROOT
from ..tuning.run_p_mpc_short_comparison import DEFAULT_P_ARTIFACT

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "p_model_boundary_validation_v1"
```

In each of these files, import `PROJECT_ROOT` from `p_mpc_run_support` and remove the local `Path(__file__)` root assignment:

```text
experiments/physics_p/evaluation/compare_mpc_predictor_formal_results.py
experiments/physics_p/evaluation/plot_p_peak_displacement_comparison.py
```

Do not change their external desktop/archive inputs or output naming.

The three calibration scripts still at the root must remain importable until Task 6. Change their actual-replay import to:

```python
from experiments.physics_p.evaluation.evaluate_p_shadow_actual_replay import (
    evaluate_actual_command_replay,
)
```

- [ ] **Step 6: Run all evaluation tests and seven CLI smoke checks**

```powershell
$evaluationModules = @(
  'experiments.physics_p.evaluation.evaluate_dual_p_shadow'
  'experiments.physics_p.evaluation.evaluate_p_frozen_mpc_plan'
  'experiments.physics_p.evaluation.evaluate_p_shadow_actual_replay'
  'experiments.physics_p.evaluation.run_p_model_boundary_validation'
  'experiments.physics_p.evaluation.compare_mpc_predictor_formal_results'
  'experiments.physics_p.evaluation.plot_p_peak_displacement_comparison'
  'experiments.physics_p.evaluation.plot_p_mpc_local_formal_results'
)
foreach ($module in $evaluationModules) {
    & $python -m $module --help
    if ($LASTEXITCODE -ne 0) { throw "CLI failed: $module" }
}

& $python -m unittest `
  tests.physics_p.test_compare_mpc_predictor_formal_results `
  tests.physics_p.test_evaluate_dual_p_shadow `
  tests.physics_p.test_evaluate_p_frozen_mpc_plan `
  tests.physics_p.test_evaluate_p_shadow_actual_replay `
  tests.physics_p.test_p_model_boundary_validation `
  tests.physics_p.test_plot_p_mpc_local_formal_results
& $python -c "import build_p_heat_generation_corrected_artifact, calibrate_p_actual_replay, refine_p_thermal_capacities"
```

Expected: all CLI checks exit 0; `Ran 20 tests`, `OK`.

- [ ] **Step 7: Commit the self-contained evaluation package**

```powershell
git add -- `
  experiments/physics_p/evaluation `
  tests/physics_p `
  build_p_heat_generation_corrected_artifact.py `
  calibrate_p_actual_replay.py `
  refine_p_thermal_capacities.py
git diff --cached --check
git commit -m "refactor: package Physics-P evaluation tools"
```

---

### Task 6: Move calibration and artifact-building tools

**Files:**
- Move: `build_p_compressor_displacement_variant.py`
- Move: `build_p_heat_generation_corrected_artifact.py`
- Move: `calibrate_p_actual_replay.py`
- Move: `refine_p_thermal_capacities.py`
- Move tests: `test_build_p_compressor_displacement_variant.py`, `test_build_p_heat_generation_corrected_artifact.py`, `test_calibrate_p_actual_replay.py`, `test_refine_p_thermal_capacities.py`

- [ ] **Step 1: Change calibration tests to future imports and stable project root**

Use these module prefixes:

```text
build_p_compressor_displacement_variant
  -> experiments.physics_p.calibration.build_p_compressor_displacement_variant
build_p_heat_generation_corrected_artifact
  -> experiments.physics_p.calibration.build_p_heat_generation_corrected_artifact
calibrate_p_actual_replay
  -> experiments.physics_p.calibration.calibrate_p_actual_replay
refine_p_thermal_capacities
  -> experiments.physics_p.calibration.refine_p_thermal_capacities
```

In `test_build_p_heat_generation_corrected_artifact.py`, add:

```python
from p_mpc_run_support import PROJECT_ROOT
```

and replace the calibration source with:

```python
absolute_source = PROJECT_ROOT / "outputs" / "calibration.csv"
```

- [ ] **Step 2: Verify a future calibration import fails**

```powershell
& $python -m unittest -v test_build_p_heat_generation_corrected_artifact
```

Expected: FAIL with `ModuleNotFoundError` for the future calibration module.

- [ ] **Step 3: Move four modules and four tests**

```powershell
git mv -- build_p_compressor_displacement_variant.py experiments/physics_p/calibration/build_p_compressor_displacement_variant.py
git mv -- build_p_heat_generation_corrected_artifact.py experiments/physics_p/calibration/build_p_heat_generation_corrected_artifact.py
git mv -- calibrate_p_actual_replay.py experiments/physics_p/calibration/calibrate_p_actual_replay.py
git mv -- refine_p_thermal_capacities.py experiments/physics_p/calibration/refine_p_thermal_capacities.py

git mv -- test_build_p_compressor_displacement_variant.py tests/physics_p/test_build_p_compressor_displacement_variant.py
git mv -- test_build_p_heat_generation_corrected_artifact.py tests/physics_p/test_build_p_heat_generation_corrected_artifact.py
git mv -- test_calibrate_p_actual_replay.py tests/physics_p/test_calibrate_p_actual_replay.py
git mv -- test_refine_p_thermal_capacities.py tests/physics_p/test_refine_p_thermal_capacities.py
```

- [ ] **Step 4: Repair calibration imports and path boundaries**

In `build_p_heat_generation_corrected_artifact.py`:

```python
from p_mpc_run_support import PROJECT_ROOT
from ..evaluation.evaluate_p_shadow_actual_replay import (
    evaluate_actual_command_replay,
)
```

Remove its local `PROJECT_ROOT = Path(__file__).resolve().parent` only.

In `calibrate_p_actual_replay.py`:

```python
from p_mpc_run_support import PROJECT_ROOT
from ..evaluation.evaluate_p_shadow_actual_replay import (
    evaluate_actual_command_replay,
)

# inside _validate_output_path
model_data = (PROJECT_ROOT / "model_data").resolve()
```

In `refine_p_thermal_capacities.py`:

```python
from p_mpc_run_support import PROJECT_ROOT
from ..evaluation.evaluate_p_shadow_actual_replay import (
    evaluate_actual_command_replay,
)
from ..identification.fit_mpc_physics_predictor import (
    _dynamic_validation_metric,
)
from ..tuning.run_p_mpc_short_comparison import DEFAULT_P_ARTIFACT
```

Remove its local `PROJECT_ROOT` assignment; leave all defaults joined to the imported root. `build_p_compressor_displacement_variant.py` keeps its stable root-module import unchanged.

- [ ] **Step 5: Run calibration CLI and tests**

```powershell
$calibrationModules = @(
  'experiments.physics_p.calibration.build_p_compressor_displacement_variant'
  'experiments.physics_p.calibration.build_p_heat_generation_corrected_artifact'
  'experiments.physics_p.calibration.calibrate_p_actual_replay'
  'experiments.physics_p.calibration.refine_p_thermal_capacities'
)
foreach ($module in $calibrationModules) {
    & $python -m $module --help
    if ($LASTEXITCODE -ne 0) { throw "CLI failed: $module" }
}

& $python -m unittest `
  tests.physics_p.test_build_p_compressor_displacement_variant `
  tests.physics_p.test_build_p_heat_generation_corrected_artifact `
  tests.physics_p.test_calibrate_p_actual_replay `
  tests.physics_p.test_refine_p_thermal_capacities
```

Expected: four CLI checks exit 0; `Ran 13 tests`, `OK`.

- [ ] **Step 6: Commit the calibration package**

```powershell
git add -- experiments/physics_p/calibration tests/physics_p
git diff --cached --check
git commit -m "refactor: package Physics-P calibration tools"
```

---

### Task 7: Move the remaining production-boundary tests and enforce the final tree

**Files:**
- Move tests: `test_mpc_physics_p_closed_loop.py`, `test_mpc_physics_shadow.py`, `test_mpc_predictor_selection.py`, `test_p_mpc_operational.py`
- Modify: `tests/physics_p/test_layout_contract.py`

- [ ] **Step 1: Add final module and file inventories to the layout contract**

Add these constants above the test class:

```python
LEGACY_EXPERIMENT_FILES = (
    "predictor_identification_data.py",
    "generate_predictor_identification_data.py",
    "fit_mpc_physics_predictor.py",
    "fit_mpc_lpv_predictor.py",
    "mpc_lpv_predictor.py",
    "evaluate_mpc_predictors.py",
    "evaluate_p_identification_rollouts.py",
    "run_p_mpc_short_comparison.py",
    "run_p_mpc_controller_tuning.py",
    "run_p_mpc_local_formal.py",
    "evaluate_dual_p_shadow.py",
    "evaluate_p_frozen_mpc_plan.py",
    "evaluate_p_shadow_actual_replay.py",
    "run_p_model_boundary_validation.py",
    "compare_mpc_predictor_formal_results.py",
    "plot_p_peak_displacement_comparison.py",
    "build_p_compressor_displacement_variant.py",
    "build_p_heat_generation_corrected_artifact.py",
    "calibrate_p_actual_replay.py",
    "refine_p_thermal_capacities.py",
)

EXPERIMENT_MODULES = (
    "experiments.physics_p.identification.predictor_identification_data",
    "experiments.physics_p.identification.generate_predictor_identification_data",
    "experiments.physics_p.identification.fit_mpc_physics_predictor",
    "experiments.physics_p.identification.fit_mpc_lpv_predictor",
    "experiments.physics_p.identification.mpc_lpv_predictor",
    "experiments.physics_p.identification.evaluate_mpc_predictors",
    "experiments.physics_p.identification.evaluate_p_identification_rollouts",
    "experiments.physics_p.tuning.run_p_mpc_short_comparison",
    "experiments.physics_p.tuning.run_p_mpc_controller_tuning",
    "experiments.physics_p.tuning.run_p_mpc_local_formal",
    "experiments.physics_p.evaluation.evaluate_dual_p_shadow",
    "experiments.physics_p.evaluation.evaluate_p_frozen_mpc_plan",
    "experiments.physics_p.evaluation.evaluate_p_shadow_actual_replay",
    "experiments.physics_p.evaluation.run_p_model_boundary_validation",
    "experiments.physics_p.evaluation.compare_mpc_predictor_formal_results",
    "experiments.physics_p.evaluation.plot_p_peak_displacement_comparison",
    "experiments.physics_p.evaluation.plot_p_mpc_local_formal_results",
    "experiments.physics_p.calibration.build_p_compressor_displacement_variant",
    "experiments.physics_p.calibration.build_p_heat_generation_corrected_artifact",
    "experiments.physics_p.calibration.calibrate_p_actual_replay",
    "experiments.physics_p.calibration.refine_p_thermal_capacities",
)

P_TEST_FILES = (
    "test_build_p_compressor_displacement_variant.py",
    "test_build_p_heat_generation_corrected_artifact.py",
    "test_calibrate_p_actual_replay.py",
    "test_compare_mpc_predictor_formal_results.py",
    "test_evaluate_dual_p_shadow.py",
    "test_evaluate_mpc_predictors.py",
    "test_evaluate_p_frozen_mpc_plan.py",
    "test_evaluate_p_shadow_actual_replay.py",
    "test_generate_predictor_identification_data.py",
    "test_mpc_lpv_predictor.py",
    "test_mpc_physics_p_closed_loop.py",
    "test_mpc_physics_predictor.py",
    "test_mpc_physics_shadow.py",
    "test_mpc_predictor_selection.py",
    "test_p_model_boundary_validation.py",
    "test_p_mpc_controller_tuning.py",
    "test_p_mpc_local_formal.py",
    "test_p_mpc_operational.py",
    "test_plot_p_mpc_local_formal_results.py",
    "test_predictor_identification_data.py",
    "test_refine_p_thermal_capacities.py",
)
```

Add imports:

```python
import importlib
```

Add three test methods:

```python
def test_legacy_experiment_root_files_are_retired(self):
    self.assertEqual(
        [name for name in LEGACY_EXPERIMENT_FILES if (support.PROJECT_ROOT / name).exists()],
        [],
    )

def test_all_experiment_modules_import_from_packages(self):
    imported = [importlib.import_module(name).__name__ for name in EXPERIMENT_MODULES]
    self.assertEqual(imported, list(EXPERIMENT_MODULES))

def test_all_existing_physics_p_tests_live_in_the_test_package(self):
    package_root = support.PROJECT_ROOT / "tests" / "physics_p"
    self.assertEqual(
        [name for name in P_TEST_FILES if not (package_root / name).is_file()],
        [],
    )
    self.assertEqual(
        [name for name in P_TEST_FILES if (support.PROJECT_ROOT / name).exists()],
        [],
    )
```

- [ ] **Step 2: Run the final tree contract and verify the four remaining root tests fail it**

```powershell
& $python -m unittest -v tests.physics_p.test_layout_contract
```

Expected: the first four layout checks pass; `test_all_existing_physics_p_tests_live_in_the_test_package` fails and names the four remaining root tests.

- [ ] **Step 3: Move the four production-boundary tests**

```powershell
git mv -- test_mpc_physics_p_closed_loop.py tests/physics_p/test_mpc_physics_p_closed_loop.py
git mv -- test_mpc_physics_shadow.py tests/physics_p/test_mpc_physics_shadow.py
git mv -- test_mpc_predictor_selection.py tests/physics_p/test_mpc_predictor_selection.py
git mv -- test_p_mpc_operational.py tests/physics_p/test_p_mpc_operational.py
```

In the moved operational test, ensure the asset assertion already uses:

```python
from p_mpc_run_support import PROJECT_ROOT

self.assertEqual(
    DEFAULT_OPERATIONAL_P_ARTIFACT,
    PROJECT_ROOT / "model_data" / "physics_p_operational_v1.json",
)
```

- [ ] **Step 4: Run the packaged P suite and the frozen 74-test boundary**

```powershell
& $python -m unittest discover -s tests/physics_p -t . -p 'test_*.py'
& $python -m unittest `
  tests.physics_p.test_mpc_predictor_selection `
  test_mpc_compressor_power_model `
  tests.physics_p.test_mpc_physics_p_closed_loop `
  tests.physics_p.test_p_mpc_operational `
  test_thermal_initial_state
```

Expected: P discovery runs 255 preserved tests plus 5 layout-contract tests, so `Ran 260 tests`; boundary command remains `Ran 74 tests`; both end with `OK`.

- [ ] **Step 5: Commit the test layout**

```powershell
git add -- tests/physics_p
git diff --cached --check
git commit -m "test: package Physics-P regressions"
```

---

### Task 8: Update active documentation and retire old commands

**Files:**
- Create: `experiments/physics_p/README.md`
- Modify: `README.md`
- Modify: `PROJECT_MAP_MIN.md`
- Modify: `CODEX_README_MIN.md`
- Modify: `tests/physics_p/test_layout_contract.py`

Do not rewrite older `docs/superpowers/plans/`, design specs, or result records; their pre-migration commands are historical evidence.

- [ ] **Step 1: Add a failing active-documentation contract**

Add this method to `PhysicsPLayoutContractTest`:

```python
def test_active_docs_route_to_supported_physics_p_commands(self):
    root_readme = (support.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    project_map = (support.PROJECT_ROOT / "PROJECT_MAP_MIN.md").read_text(
        encoding="utf-8"
    )
    codex_map = (support.PROJECT_ROOT / "CODEX_README_MIN.md").read_text(
        encoding="utf-8"
    )
    experiment_readme = (
        support.PROJECT_ROOT / "experiments" / "physics_p" / "README.md"
    ).read_text(encoding="utf-8")

    self.assertIn("run_p_mpc_operational.py", root_readme)
    self.assertIn("Candidate B", root_readme)
    self.assertIn("model_data/physics_p_operational_v1.json", project_map)
    self.assertIn("experiments/physics_p/", codex_map)
    self.assertIn("python -m experiments.physics_p", experiment_readme)
    self.assertIn("旧命令已退役，不提供根目录 wrapper", experiment_readme)
```

Run:

```powershell
& $python -m unittest -v `
  tests.physics_p.test_layout_contract.PhysicsPLayoutContractTest.test_active_docs_route_to_supported_physics_p_commands
```

Expected: FAIL because `experiments/physics_p/README.md` and the active routes are not complete.

- [ ] **Step 2: Add the Physics-P section to the root README**

Add this block after the core-structure section in `README.md`:

````markdown
## Physics-P 显式运行入口

Candidate B 仍是仓库默认 MPC 预测器。Physics-P 只通过下面的唯一正式入口显式启用：

```powershell
python run_p_mpc_operational.py --help
```

正式参数资产位于 `model_data/physics_p_operational_v1.json`。辨识、校准、评估和调参脚本位于 `experiments/physics_p/`，必须从项目根目录用 `python -m ...` 调用；旧根目录脚本命令已退役。

Physics-P 专属回归：

```powershell
python -m unittest discover -s tests/physics_p -t . -p "test_*.py"
```
````

Also add these entries to the core-structure list:

```markdown
- `run_p_mpc_operational.py`：唯一正式 Physics-P 闭环入口。
- `p_mpc_run_support.py`：P 运行器共享场景、路径和汇总支持。
- `mpc_physics_predictor.py`、`mpc_physics_shadow.py`、`mpc_predictor_selection.py`：稳定 P 预测与选择契约。
- `experiments/physics_p/`：P 辨识、校准、评估和调参。
- `tests/physics_p/`：P 专属回归。
```

- [ ] **Step 3: Replace the compact project maps with exact current routes**

Set `PROJECT_MAP_MIN.md` to:

```markdown
# Project Map Min

## 1. 一句话
储能电池热管理仿真/控制：电池热、冷板冷却液、制冷循环、PID/MPC、调峰/调频。Candidate B 仍为默认 MPC 预测器；Physics-P 仅由专用入口显式启用。

## 2. 新对话默认读取
只读核心文件：`AGENTS.md`、`PROJECT_MAP_MIN.md`、`thermal_case_simulator.py`、`thermal_batch_config.py`、`thermal_control_strategies.py`、`mpc_flow_direction_strategies.py`、`thermal_loop.py`、`thermal_system.py`、`pid_param_search.py`；模型参数按需读取 `model_data/`。

## 3. 默认主入口
`thermal_case_simulator.py -> thermal_control_strategies.py -> mpc_flow_direction_strategies.py`；参数：`thermal_batch_config.py`。这条默认链继续选择 Candidate B。

## 4. Physics-P
唯一正式入口：`run_p_mpc_operational.py`。稳定模块：`p_mpc_run_support.py`、`mpc_physics_predictor.py`、`mpc_physics_shadow.py`、`mpc_predictor_selection.py`。正式资产：`model_data/physics_p_operational_v1.json`。实验：`experiments/physics_p/`；专属回归：`tests/physics_p/`。

## 5. 当前其他入口
最终控制器与论文结果复现优先读取：`run_final_controller_comparison.py`、`run_final_controller_parallel.py`、`run_mpc_sensitivity_60.py`、`plot_final_mpc_report.py`。Candidate B 蒸发器继续读取 `mpc_evaporator_capacity_model.py` 和对应标定/诊断入口。

## 6. 不建议读取
默认不扫描：`outputs/`、`输出结果/`、`data/`、`.git/`、`__pycache__/`、`tmp*/`、`_gekko_tmp_*/`、`.codex/`、`.agents/`、`simulink-agentic-toolkit*/`。历史 `_archive/` 已永久删除。

## 7. 新对话 Prompt
请先读 `AGENTS.md` 和 `PROJECT_MAP_MIN.md`。本项目是储能热管理 PID/MPC 仿真；只从核心链路或点名的 Physics-P 入口入手，不扫描 outputs/data。
```

Set `CODEX_README_MIN.md` to:

```markdown
项目：储能电池热管理仿真/控制；电池热、冷板、制冷、PID/MPC、调峰/调频。
默认主流程：`thermal_case_simulator.py -> thermal_control_strategies.py -> mpc_flow_direction_strategies.py`；默认预测器仍是 Candidate B。
Physics-P：唯一正式入口 `run_p_mpc_operational.py`；调峰 horizon 14、调频 horizon 12；入口显式使用 25 °C，普通 `simulate_case` 默认使用环境温度 35 °C。
P 稳定模块：`p_mpc_run_support.py`、`mpc_physics_predictor.py`、`mpc_physics_shadow.py`、`mpc_predictor_selection.py`。
P 实验：`experiments/physics_p/`，从项目根目录使用 `python -m experiments.physics_p...`；不再支持旧根命令。
P 测试：`python -m unittest discover -s tests/physics_p -t . -p "test_*.py"`。
P 正式资产：`model_data/physics_p_operational_v1.json`。
读取：`AGENTS.md` -> `PROJECT_MAP_MIN.md` -> 主流程/参数文件 -> 点名文件。
禁扫：`outputs/`、`data/`、`.git/`、`__pycache__/`、`.pytest_cache/`、`.mypy_cache/`、GEKKO 目录、`simulink-agentic-toolkit*/`。
```

- [ ] **Step 4: Create the complete experiment command migration README**

Create `experiments/physics_p/README.md` with:

```markdown
# Physics-P Experiments

`run_p_mpc_operational.py` 是唯一正式 Physics-P 入口。此目录只保存辨识、校准、评估、绘图和调参实验。

所有命令必须从项目根目录执行。旧命令已退役，不提供根目录 wrapper。

## Identification

| 旧命令/文件 | 新模块 |
| --- | --- |
| `generate_predictor_identification_data.py` | `python -m experiments.physics_p.identification.generate_predictor_identification_data` |
| `fit_mpc_physics_predictor.py` | `python -m experiments.physics_p.identification.fit_mpc_physics_predictor` |
| `fit_mpc_lpv_predictor.py` | `python -m experiments.physics_p.identification.fit_mpc_lpv_predictor` |
| `evaluate_p_identification_rollouts.py` | `python -m experiments.physics_p.identification.evaluate_p_identification_rollouts` |
| `predictor_identification_data.py` | 库模块 `experiments.physics_p.identification.predictor_identification_data` |
| `mpc_lpv_predictor.py` | 库模块 `experiments.physics_p.identification.mpc_lpv_predictor` |
| `evaluate_mpc_predictors.py` | 库模块 `experiments.physics_p.identification.evaluate_mpc_predictors` |

## Calibration

| 旧命令 | 新命令 |
| --- | --- |
| `build_p_compressor_displacement_variant.py` | `python -m experiments.physics_p.calibration.build_p_compressor_displacement_variant` |
| `build_p_heat_generation_corrected_artifact.py` | `python -m experiments.physics_p.calibration.build_p_heat_generation_corrected_artifact` |
| `calibrate_p_actual_replay.py` | `python -m experiments.physics_p.calibration.calibrate_p_actual_replay` |
| `refine_p_thermal_capacities.py` | `python -m experiments.physics_p.calibration.refine_p_thermal_capacities` |

## Evaluation

| 旧命令/文件 | 新命令 |
| --- | --- |
| `evaluate_dual_p_shadow.py` | `python -m experiments.physics_p.evaluation.evaluate_dual_p_shadow` |
| `evaluate_p_frozen_mpc_plan.py` | `python -m experiments.physics_p.evaluation.evaluate_p_frozen_mpc_plan` |
| `evaluate_p_shadow_actual_replay.py` | `python -m experiments.physics_p.evaluation.evaluate_p_shadow_actual_replay` |
| `run_p_model_boundary_validation.py` | `python -m experiments.physics_p.evaluation.run_p_model_boundary_validation` |
| `compare_mpc_predictor_formal_results.py` | `python -m experiments.physics_p.evaluation.compare_mpc_predictor_formal_results` |
| `plot_p_peak_displacement_comparison.py` | `python -m experiments.physics_p.evaluation.plot_p_peak_displacement_comparison` |
| `figures/gen_fig_p_mpc_local_formal.py` | `python -m experiments.physics_p.evaluation.plot_p_mpc_local_formal_results` |

## Tuning and formal experiments

| 旧命令 | 新命令 |
| --- | --- |
| `run_p_mpc_short_comparison.py` | `python -m experiments.physics_p.tuning.run_p_mpc_short_comparison` |
| `run_p_mpc_controller_tuning.py` | `python -m experiments.physics_p.tuning.run_p_mpc_controller_tuning` |
| `run_p_mpc_local_formal.py` | `python -m experiments.physics_p.tuning.run_p_mpc_local_formal` |

每个 CLI 都可追加 `--help` 查看原有参数。历史 `docs/superpowers/` 记录保留迁移前命令，不代表当前支持接口。
```

- [ ] **Step 5: Run documentation and layout tests**

```powershell
& $python -m unittest -v tests.physics_p.test_layout_contract
```

Expected: `Ran 6 tests`, `OK`.

- [ ] **Step 6: Commit active documentation**

```powershell
git add -- `
  README.md `
  PROJECT_MAP_MIN.md `
  CODEX_README_MIN.md `
  experiments/physics_p/README.md `
  tests/physics_p/test_layout_contract.py
git diff --cached --check
git commit -m "docs: document Physics-P package commands"
```

---

### Task 9: Run final structural, regression, asset, and behavior verification

**Files:**
- Verify only; do not modify tracked files.
- Read: `outputs/p_mpc_layout_smoke_v1/pre_move_20260811/`
- Create ignored result: `outputs/p_mpc_layout_smoke_v1/post_move_20260811/`

- [ ] **Step 1: Compile and import every production and experiment module**

```powershell
& $python -m compileall -q `
  p_mpc_run_support.py `
  run_p_mpc_operational.py `
  mpc_physics_predictor.py `
  mpc_physics_shadow.py `
  mpc_predictor_selection.py `
  experiments/physics_p `
  tests/physics_p
if ($LASTEXITCODE -ne 0) { throw 'compileall failed' }

@'
import importlib
from tests.physics_p.test_layout_contract import EXPERIMENT_MODULES

for module_name in EXPERIMENT_MODULES:
    module = importlib.import_module(module_name)
    assert module.__name__ == module_name
print(f"IMPORTED {len(EXPERIMENT_MODULES)} experiment modules")
'@ | & $python -
```

Expected: `IMPORTED 21 experiment modules`.

- [ ] **Step 2: Run all 18 supported experiment CLI help commands**

```powershell
$cliModules = @(
  'experiments.physics_p.identification.generate_predictor_identification_data'
  'experiments.physics_p.identification.fit_mpc_physics_predictor'
  'experiments.physics_p.identification.fit_mpc_lpv_predictor'
  'experiments.physics_p.identification.evaluate_p_identification_rollouts'
  'experiments.physics_p.calibration.build_p_compressor_displacement_variant'
  'experiments.physics_p.calibration.build_p_heat_generation_corrected_artifact'
  'experiments.physics_p.calibration.calibrate_p_actual_replay'
  'experiments.physics_p.calibration.refine_p_thermal_capacities'
  'experiments.physics_p.evaluation.evaluate_dual_p_shadow'
  'experiments.physics_p.evaluation.evaluate_p_frozen_mpc_plan'
  'experiments.physics_p.evaluation.evaluate_p_shadow_actual_replay'
  'experiments.physics_p.evaluation.run_p_model_boundary_validation'
  'experiments.physics_p.evaluation.compare_mpc_predictor_formal_results'
  'experiments.physics_p.evaluation.plot_p_peak_displacement_comparison'
  'experiments.physics_p.evaluation.plot_p_mpc_local_formal_results'
  'experiments.physics_p.tuning.run_p_mpc_short_comparison'
  'experiments.physics_p.tuning.run_p_mpc_controller_tuning'
  'experiments.physics_p.tuning.run_p_mpc_local_formal'
)
foreach ($module in $cliModules) {
    & $python -m $module --help *> $null
    if ($LASTEXITCODE -ne 0) { throw "CLI failed: $module" }
}
"CLI_OK $($cliModules.Count)"
```

Expected: `CLI_OK 18`.

- [ ] **Step 3: Run the complete packaged suite and frozen 74-test boundary**

```powershell
& $python -m unittest discover -s tests/physics_p -t . -p 'test_*.py'
& $python -m unittest `
  tests.physics_p.test_mpc_predictor_selection `
  test_mpc_compressor_power_model `
  tests.physics_p.test_mpc_physics_p_closed_loop `
  tests.physics_p.test_p_mpc_operational `
  test_thermal_initial_state
```

Expected: `Ran 261 tests` then `Ran 74 tests`; both `OK`.

- [ ] **Step 4: Run the exact historical 277-test adjacent regression set**

```powershell
$targets = @(
  'tests.physics_p.test_build_p_compressor_displacement_variant'
  'tests.physics_p.test_build_p_heat_generation_corrected_artifact'
  'tests.physics_p.test_calibrate_p_actual_replay'
  'tests.physics_p.test_compare_mpc_predictor_formal_results'
  'tests.physics_p.test_evaluate_dual_p_shadow'
  'tests.physics_p.test_evaluate_mpc_predictors'
  'tests.physics_p.test_evaluate_p_frozen_mpc_plan'
  'tests.physics_p.test_evaluate_p_shadow_actual_replay'
  'test_evaporator_mpc_capacity_diagnostic'
  'tests.physics_p.test_generate_predictor_identification_data'
  'test_mpc_compressor_power_model'
  'test_mpc_evaporator_capacity_model'
  'test_mpc_evaporator_capacity_table'
  'tests.physics_p.test_mpc_lpv_predictor'
  'test_mpc_params'
  'tests.physics_p.test_mpc_physics_p_closed_loop'
  'tests.physics_p.test_mpc_physics_shadow'
  'test_mpc_planned_controls'
  'tests.physics_p.test_mpc_predictor_selection'
  'test_mpc_solve_recovery'
  'tests.physics_p.test_p_model_boundary_validation'
  'tests.physics_p.test_p_mpc_controller_tuning'
  'tests.physics_p.test_p_mpc_local_formal'
  'tests.physics_p.test_p_mpc_operational'
  'tests.physics_p.test_plot_p_mpc_local_formal_results'
  'test_predictive_delta_t_latch'
  'tests.physics_p.test_predictor_identification_data'
  'tests.physics_p.test_refine_p_thermal_capacities'
  'test_refrigeration_cycle_limits'
  'test_thermal_initial_state'
  'tests.physics_p.test_mpc_physics_predictor.PhysicsCapacityTests'
  'tests.physics_p.test_mpc_physics_predictor.PhysicsDynamicTests'
  'tests.physics_p.test_mpc_physics_predictor.PhysicsArtifactTests'
  'tests.physics_p.test_mpc_physics_predictor.PhysicsCliTests'
)
& $python -m unittest @targets
```

Expected: `Ran 277 tests` and `OK`. This intentionally excludes the 32 long `PhysicsFitTests`; the full P discovery in Step 3 covers them separately.

- [ ] **Step 5: Verify the asset remained byte-identical and portable**

```powershell
$asset = 'model_data/physics_p_operational_v1.json'
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $asset).Hash
if ($hash -ne '0BFE23DF636193F954EA4BFA0E457EE29F64720A88995AB818CAE9265E0109C0') {
    throw "Unexpected Physics-P asset hash: $hash"
}
git ls-files --error-unmatch $asset

@'
import json
import re
from pathlib import Path

path = Path("model_data/physics_p_operational_v1.json")
text = path.read_text(encoding="utf-8")
payload = json.loads(text)
assert payload["model_type"] == "physics_p"
assert not re.search(r"[A-Za-z]:[\\/]", text)
assert text.count("NaN") == 48
print("ASSET_OK")
'@ | & $python -
```

Expected: tracked path and `ASSET_OK`.

- [ ] **Step 6: Check for stale root imports and unsupported patch targets**

Run:

```powershell
$staleImports = rg -n `
  '^(from|import) (predictor_identification_data|generate_predictor_identification_data|fit_mpc_physics_predictor|fit_mpc_lpv_predictor|mpc_lpv_predictor|evaluate_mpc_predictors|evaluate_p_identification_rollouts|run_p_mpc_short_comparison|run_p_mpc_controller_tuning|run_p_mpc_local_formal|evaluate_dual_p_shadow|evaluate_p_frozen_mpc_plan|evaluate_p_shadow_actual_replay|run_p_model_boundary_validation|compare_mpc_predictor_formal_results|plot_p_peak_displacement_comparison|build_p_compressor_displacement_variant|build_p_heat_generation_corrected_artifact|calibrate_p_actual_replay|refine_p_thermal_capacities)' `
  -g '*.py' .
if ($LASTEXITCODE -eq 0) { throw "Stale root imports:`n$staleImports" }
if ($LASTEXITCODE -ne 1) { throw "rg failed with exit code $LASTEXITCODE" }

$stalePatch = rg -n `
  '["''](generate_predictor_identification_data|fit_mpc_physics_predictor)\.' `
  -g '*.py' .
if ($LASTEXITCODE -eq 0) { throw "Stale patch targets:`n$stalePatch" }
if ($LASTEXITCODE -ne 1) { throw "rg failed with exit code $LASTEXITCODE" }
```

Expected: both searches return no matches. Historical docs and the explicit migration README are not searched by this Python-only check.

- [ ] **Step 7: Generate the post-move six-step smoke**

```powershell
$post = 'outputs/p_mpc_layout_smoke_v1/post_move_20260811'
if (Test-Path -LiteralPath $post) {
    throw "Output already exists: $post"
}
& $python run_p_mpc_operational.py `
  --scenes peak freq `
  --steps 6 `
  --output-root $post
```

Expected: peak uses horizon 14, freq uses horizon 12, both summaries report `qualified=True`.

- [ ] **Step 8: Compare pre/post control and thermal results**

```powershell
@'
from pathlib import Path

import numpy as np
import pandas as pd

pre = Path("outputs/p_mpc_layout_smoke_v1/pre_move_20260811")
post = Path("outputs/p_mpc_layout_smoke_v1/post_move_20260811")
files = {
    "peak": "peak_physics_p_operational_steps6_horizon14.csv",
    "freq": "freq_physics_p_operational_steps6_horizon12.csv",
}
numeric_columns = (
    "Average temperature",
    "Coolant temperature",
    "Compressor command",
    "Pump command",
    "Total power",
    "Cumulative energy consumption",
    "MPC raw compressor command",
    "MPC applied compressor command",
    "MPC predicted battery temperature +1 step",
    "MPC predicted coolant temperature horizon end",
    "MPC coolant prediction domain violation",
)
exact_columns = (
    "MPC_Predictor",
    "MPC_Solved",
    "MPC solve error",
    "MPC solve recovery used",
    "MPC solve recovery reason",
    "MPC prediction domain valid",
)

for scene, filename in files.items():
    before = pd.read_csv(pre / "mpc" / filename, encoding="utf-8-sig")
    after = pd.read_csv(post / "mpc" / filename, encoding="utf-8-sig")
    assert len(before) == len(after) == 6, scene
    for column in numeric_columns:
        np.testing.assert_allclose(
            pd.to_numeric(before[column], errors="raise"),
            pd.to_numeric(after[column], errors="raise"),
            rtol=0.0,
            atol=1e-10,
            equal_nan=True,
            err_msg=f"{scene}: {column}",
        )
    for column in exact_columns:
        pd.testing.assert_series_equal(
            before[column].fillna("").astype(str),
            after[column].fillna("").astype(str),
            check_names=False,
            obj=f"{scene}: {column}",
        )

summary_before = pd.read_csv(pre / "operational_summary.csv", encoding="utf-8-sig")
summary_after = pd.read_csv(post / "operational_summary.csv", encoding="utf-8-sig")
summary_columns = (
    "scene",
    "predictor",
    "steps",
    "horizon_steps",
    "temperature_mae_c",
    "temperature_max_c",
    "temperature_final_c",
    "energy_kwh",
    "mean_power_kw",
    "mean_n_comp_rpm",
    "mean_n_pump_rpm",
    "solve_success_rate",
    "prediction_domain_valid_rate",
    "qualified",
    "compressor_displacement_scale",
)
pd.testing.assert_frame_equal(
    summary_before.loc[:, summary_columns],
    summary_after.loc[:, summary_columns],
    check_exact=False,
    rtol=0.0,
    atol=1e-10,
)
print("LAYOUT_BEHAVIOR_MATCH")
'@ | & $python -
```

Expected: `LAYOUT_BEHAVIOR_MATCH`. Solve time, GEKKO model path, `artifact`, and `out_csv` are intentionally excluded.

- [ ] **Step 9: Finish with Git integrity checks**

```powershell
git diff --check
git status --short --branch
git log --oneline --decorate -10
```

Expected: no whitespace errors; branch is clean; ignored smoke outputs and the unrelated main-worktree TD3 files are not staged or committed.
