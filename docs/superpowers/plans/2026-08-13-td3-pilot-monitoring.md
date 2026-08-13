# TD3 Pilot Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-step CSV logging, 100-step model checkpoints, and a four-panel PNG to the existing TD3 trainer, then run separate 500-step peak and frequency pilot trainings plus deterministic 100-step evaluations.

**Architecture:** A single `PilotTrainingCallback` in `run_td3_training.py` reads the one-environment Stable-Baselines3 `infos` and `dones` arrays, accumulates fixed-schema rows, and saves checkpoints. The trainer owns CSV/plot finalization in `finally`, so partial diagnostics survive training failures while exceptions still propagate.

**Tech Stack:** Python 3.12, Stable-Baselines3 2.9, NumPy, Pandas, Matplotlib Agg, unittest, PyBaMM BTMS environment.

---

## File map

- Modify `run_td3_training.py`: callback, CSV writer, training plot, CLI flags, callback integration.
- Modify `tests/td3/test_td3_btms_env.py`: callback schema/checkpoint tests, CSV/PNG tests, CLI validation.

Do not modify `td3_btms/env.py`, `run_td3_evaluation.py`, any PID/MPC/Physics-P file, or any physical model.

Use this interpreter for every command:

```powershell
$python = 'D:\conda_envs\btms_td3\python.exe'
```

### Task 1: Training-history row schema

**Files:**
- Modify: `run_td3_training.py`
- Modify: `tests/td3/test_td3_btms_env.py`

- [ ] **Step 1: Write failing row-schema tests**

Extend the training import in `tests/td3/test_td3_btms_env.py`:

```python
from run_td3_training import (
    HISTORY_COLUMNS,
    PilotTrainingCallback,
    create_run_directory,
    write_metadata,
)
```

Add this test class:

```python
class PilotTrainingCallbackTests(unittest.TestCase):
    def _info(self):
        return {
            "step_index": 3,
            "current_a": 560.0,
            "mean_soc": 0.94,
            "mean_temp_c": 25.2,
            "max_temp_c": 25.3,
            "delta_temp_c": 0.1,
            "tank_temp_c": 34.0,
            "plate_mean_temp_c": 33.0,
            "total_power_w": 1200.0,
            "n_comp_cmd_rpm": 3000.0,
            "n_pump_cmd_rpm": 2800.0,
            "n_comp_eff_rpm": 2500.0,
            "n_pump_eff_rpm": 2600.0,
        }

    def test_callback_records_fixed_schema_for_one_environment(self):
        callback = PilotTrainingCallback(checkpoint_dir=None, checkpoint_interval=0)
        callback.num_timesteps = 12
        callback.locals = {
            "infos": [self._info()],
            "rewards": np.array([-0.25]),
            "dones": np.array([False]),
        }

        self.assertTrue(callback._on_step())
        self.assertEqual(tuple(callback.records[0]), HISTORY_COLUMNS)
        self.assertEqual(callback.records[0]["training_step"], 12)
        self.assertEqual(callback.records[0]["episode_index"], 0)
        self.assertEqual(callback.records[0]["episode_step"], 3)
        self.assertEqual(callback.records[0]["reward"], -0.25)
        self.assertFalse(callback.records[0]["terminated"])
        self.assertFalse(callback.records[0]["truncated"])

    def test_callback_rejects_multiple_environments(self):
        callback = PilotTrainingCallback(checkpoint_dir=None, checkpoint_interval=0)
        callback.locals = {
            "infos": [self._info(), self._info()],
            "rewards": np.array([-0.1, -0.2]),
            "dones": np.array([False, False]),
        }
        with self.assertRaisesRegex(RuntimeError, "one environment"):
            callback._on_step()
```

- [ ] **Step 2: Run the tests and verify the import failure**

Run:

```powershell
& $python -m unittest -v tests.td3.test_td3_btms_env.PilotTrainingCallbackTests
```

Expected: import failure for `HISTORY_COLUMNS` or `PilotTrainingCallback`.

- [ ] **Step 3: Implement the fixed-schema callback**

Add imports to `run_td3_training.py`:

```python
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
```

Add the schema and callback before `create_run_directory`:

```python
HISTORY_COLUMNS = (
    "training_step",
    "episode_index",
    "episode_step",
    "reward",
    "current_a",
    "mean_soc",
    "mean_temp_c",
    "max_temp_c",
    "delta_temp_c",
    "tank_temp_c",
    "plate_mean_temp_c",
    "total_power_w",
    "n_comp_cmd_rpm",
    "n_pump_cmd_rpm",
    "n_comp_eff_rpm",
    "n_pump_eff_rpm",
    "terminated",
    "truncated",
)


class PilotTrainingCallback(BaseCallback):
    def __init__(self, *, checkpoint_dir, checkpoint_interval: int):
        super().__init__()
        self.checkpoint_dir = (
            None if checkpoint_dir is None else Path(checkpoint_dir).resolve()
        )
        self.checkpoint_interval = int(checkpoint_interval)
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be nonnegative")
        self.records = []
        self.episode_index = 0

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        rewards = np.asarray(self.locals["rewards"], dtype=float)
        dones = np.asarray(self.locals["dones"], dtype=bool)
        if len(infos) != 1 or rewards.size != 1 or dones.size != 1:
            raise RuntimeError("pilot logging supports exactly one environment")
        info = infos[0]
        end_reason = info.get("end_reason")
        row = {
            "training_step": int(self.num_timesteps),
            "episode_index": int(self.episode_index),
            "episode_step": int(info["step_index"]),
            "reward": float(rewards[0]),
            "current_a": float(info["current_a"]),
            "mean_soc": float(info["mean_soc"]),
            "mean_temp_c": float(info["mean_temp_c"]),
            "max_temp_c": float(info["max_temp_c"]),
            "delta_temp_c": float(info["delta_temp_c"]),
            "tank_temp_c": float(info["tank_temp_c"]),
            "plate_mean_temp_c": float(info["plate_mean_temp_c"]),
            "total_power_w": float(info["total_power_w"]),
            "n_comp_cmd_rpm": float(info["n_comp_cmd_rpm"]),
            "n_pump_cmd_rpm": float(info["n_pump_cmd_rpm"]),
            "n_comp_eff_rpm": float(info["n_comp_eff_rpm"]),
            "n_pump_eff_rpm": float(info["n_pump_eff_rpm"]),
            "terminated": bool(end_reason in {"nonfinite_state", "temperature_limit"}),
            "truncated": bool(end_reason == "profile_complete"),
        }
        self.records.append(row)
        if dones[0]:
            self.episode_index += 1
        return True
```

- [ ] **Step 4: Run the callback tests**

Run:

```powershell
& $python -m unittest -v tests.td3.test_td3_btms_env.PilotTrainingCallbackTests
```

Expected: `Ran 2 tests` and `OK`.

- [ ] **Step 5: Commit the row schema**

```powershell
git add run_td3_training.py tests/td3/test_td3_btms_env.py
git commit -m "feat: record TD3 pilot training history"
```

### Task 2: Checkpoints, CSV, and PNG

**Files:**
- Modify: `run_td3_training.py`
- Modify: `tests/td3/test_td3_btms_env.py`

- [ ] **Step 1: Write failing artifact tests**

Extend the training import:

```python
from run_td3_training import (
    HISTORY_COLUMNS,
    PilotTrainingCallback,
    create_run_directory,
    plot_training_history,
    write_metadata,
    write_training_history,
)
```

Add methods to `PilotTrainingCallbackTests`:

```python
    def test_callback_saves_zero_padded_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            callback = PilotTrainingCallback(
                checkpoint_dir=Path(temp_dir), checkpoint_interval=100
            )
            callback.num_timesteps = 100
            callback.locals = {
                "infos": [self._info()],
                "rewards": np.array([-0.25]),
                "dones": np.array([False]),
            }
            callback.model = unittest.mock.Mock()

            callback._on_step()

            callback.model.save.assert_called_once_with(
                Path(temp_dir).resolve() / "checkpoint_000100_steps"
            )

    def test_history_writer_and_plot_create_nonempty_artifacts(self):
        records = []
        for training_step in range(1, 4):
            row = {
                "training_step": training_step,
                "episode_index": 0,
                "episode_step": training_step,
                "reward": -0.1 * training_step,
                **self._info(),
                "terminated": False,
                "truncated": training_step == 3,
            }
            row.pop("step_index")
            records.append(row)
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "training_history.csv"
            plot_path = Path(temp_dir) / "training_curve.png"

            frame = write_training_history(records, csv_path)
            plot_training_history(frame, plot_path)

            self.assertEqual(tuple(frame.columns), HISTORY_COLUMNS)
            self.assertEqual(len(pd.read_csv(csv_path)), 3)
            self.assertGreater(csv_path.stat().st_size, 0)
            self.assertGreater(plot_path.stat().st_size, 0)

    def test_empty_history_cannot_be_plotted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "empty"):
                plot_training_history(
                    pd.DataFrame(columns=HISTORY_COLUMNS),
                    Path(temp_dir) / "empty.png",
                )
```

Add `from unittest import mock` and replace `unittest.mock.Mock()` with `mock.Mock()`.

- [ ] **Step 2: Run artifact tests and verify failure**

Run:

```powershell
& $python -m unittest -v tests.td3.test_td3_btms_env.PilotTrainingCallbackTests
```

Expected: import failure for `write_training_history` or `plot_training_history`.

- [ ] **Step 3: Implement checkpoint, CSV, and plot artifacts**

Add to `PilotTrainingCallback._on_step()` immediately after appending the row:

```python
        if (
            self.checkpoint_interval > 0
            and self.num_timesteps % self.checkpoint_interval == 0
        ):
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = self.checkpoint_dir / (
                f"checkpoint_{self.num_timesteps:06d}_steps"
            )
            self.model.save(checkpoint)
```

Update constructor validation so a positive interval requires a directory:

```python
        if self.checkpoint_interval > 0 and self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required when checkpoints are enabled")
```

Add the artifact functions:

```python
def write_training_history(records, path) -> pd.DataFrame:
    frame = pd.DataFrame(records, columns=HISTORY_COLUMNS)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


def plot_training_history(frame: pd.DataFrame, path) -> None:
    if frame.empty:
        raise ValueError("training history is empty")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = frame["training_step"]
    reward_mean = frame["reward"].rolling(50, min_periods=1).mean()
    figure, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    axes[0].plot(steps, frame["reward"], alpha=0.35, label="step reward")
    axes[0].plot(steps, reward_mean, linewidth=2.0, label="50-step mean")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(steps, frame["mean_temp_c"], label="mean temperature")
    axes[1].plot(steps, frame["max_temp_c"], label="max temperature")
    axes[1].plot(steps, frame["delta_temp_c"], label="max delta T")
    axes[1].axhline(25.0, color="black", linestyle="--", label="target 25 C")
    axes[1].axhline(0.5, color="red", linestyle=":", label="delta T limit")
    axes[1].set_ylabel("Temperature (C)")
    axes[1].legend(ncol=2)
    axes[1].grid(alpha=0.25)

    axes[2].plot(steps, frame["total_power_w"] / 1000.0)
    axes[2].set_ylabel("Power (kW)")
    axes[2].grid(alpha=0.25)

    axes[3].plot(steps, frame["n_comp_cmd_rpm"], label="compressor command")
    axes[3].plot(steps, frame["n_comp_eff_rpm"], label="compressor actual")
    axes[3].plot(steps, frame["n_pump_cmd_rpm"], label="pump command")
    axes[3].plot(steps, frame["n_pump_eff_rpm"], label="pump actual")
    axes[3].set_ylabel("Speed (rpm)")
    axes[3].set_xlabel("Training step")
    axes[3].legend(ncol=2)
    axes[3].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
```

- [ ] **Step 4: Run all callback artifact tests**

Run:

```powershell
& $python -m unittest -v tests.td3.test_td3_btms_env.PilotTrainingCallbackTests
```

Expected: `Ran 5 tests` and `OK`.

- [ ] **Step 5: Commit artifact generation**

```powershell
git add run_td3_training.py tests/td3/test_td3_btms_env.py
git commit -m "feat: save TD3 pilot checkpoints and curves"
```

### Task 3: Integrate monitoring into the training CLI

**Files:**
- Modify: `run_td3_training.py`
- Modify: `tests/td3/test_td3_btms_env.py`

- [ ] **Step 1: Write failing CLI tests**

Extend the training import with `parse_args as parse_training_args` and add:

```python
class PilotTrainingCliTests(unittest.TestCase):
    def test_formal_training_defaults_to_100_step_checkpoints(self):
        args = parse_training_args(
            ["--scene", "peak", "--total-timesteps", "500"]
        )
        self.assertEqual(args.checkpoint_interval, 100)
        self.assertFalse(args.no_training_plot)

    def test_smoke_defaults_to_disabled_checkpoints(self):
        args = parse_training_args(["--scene", "peak", "--smoke"])
        self.assertEqual(args.checkpoint_interval, 0)

    def test_checkpoint_interval_rejects_negative_values(self):
        with self.assertRaises(SystemExit):
            parse_training_args(
                [
                    "--scene",
                    "peak",
                    "--total-timesteps",
                    "500",
                    "--checkpoint-interval",
                    "-1",
                ]
            )
```

- [ ] **Step 2: Run CLI tests and verify failure**

Run:

```powershell
& $python -m unittest -v tests.td3.test_td3_btms_env.PilotTrainingCliTests
```

Expected: failures because `checkpoint_interval` and `no_training_plot` are absent.

- [ ] **Step 3: Add CLI flags and monitor finalization**

Add parser options:

```python
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--no-training-plot", action="store_true")
```

After parsing, resolve the default and validate:

```python
    if args.checkpoint_interval is None:
        args.checkpoint_interval = 0 if args.smoke else 100
    if args.checkpoint_interval < 0:
        parser.error("--checkpoint-interval must be nonnegative")
```

In `main()`, after creating `run_dir`, create the callback:

```python
    callback = PilotTrainingCallback(
        checkpoint_dir=run_dir / "checkpoints",
        checkpoint_interval=args.checkpoint_interval,
    )
```

Replace the learning call with:

```python
        model.learn(total_timesteps=total_timesteps, callback=callback)
```

In the existing `finally` block, write partial history before closing the environment:

```python
        history = write_training_history(
            callback.records, run_dir / "training_history.csv"
        )
        if not args.no_training_plot and not history.empty:
            plot_training_history(history, run_dir / "training_curve.png")
        env.close()
```

Add optional keyword-only parameters `checkpoint_interval=0` and `training_history_rows=0` to `write_metadata()`, record both integer values in the JSON output, then pass `args.checkpoint_interval` and `len(callback.records)` from `main()`. Defaults preserve existing callers and the metadata unit test.

- [ ] **Step 4: Run CLI and all TD3 tests**

Run:

```powershell
& $python -m unittest -v tests.td3.test_td3_btms_env.PilotTrainingCliTests
& $python -m unittest -v tests.td3.test_td3_btms_env
& $python run_td3_training.py --help
```

Expected: CLI tests pass, the full TD3 count is 24 tests, and help lists both new flags.

- [ ] **Step 5: Commit CLI integration**

```powershell
git add run_td3_training.py tests/td3/test_td3_btms_env.py
git commit -m "feat: integrate TD3 pilot monitoring"
```

### Task 4: Regression and monitored smoke verification

**Files:**
- Modify only if a verified defect requires it: `run_td3_training.py`, `tests/td3/test_td3_btms_env.py`

- [ ] **Step 1: Compile and run TD3 plus adjacent regressions**

Run:

```powershell
& $python -m compileall -q run_td3_training.py tests/td3
& $python -m unittest -v tests.td3.test_td3_btms_env
& $python -m unittest -v test_refrigeration_cycle_limits test_thermal_initial_state tests.physics_p.test_mpc_predictor_selection
```

Expected: compile exit 0, 24 TD3 tests pass, and 18 adjacent regressions pass.

- [ ] **Step 2: Run a fresh monitored smoke training**

Run with a new output directory:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$smoke = "outputs/td3/monitor_smoke_$stamp"
& $python run_td3_training.py `
  --scene peak `
  --total-timesteps 4 `
  --seed 7 `
  --checkpoint-interval 2 `
  --output-dir $smoke
```

Expected: exit 0; 4-row `training_history.csv`, nonempty `training_curve.png`, `model.zip`, `metadata.json`, and checkpoints at steps 2 and 4.

- [ ] **Step 3: Verify core-file isolation and Git hygiene**

Run:

```powershell
$changed = git diff --name-only a8a964b..HEAD
$changed
$forbidden = $changed | Where-Object {
    $_ -notin @('run_td3_training.py', 'tests/td3/test_td3_btms_env.py')
}
if ($forbidden) { throw "Out-of-scope files changed: $($forbidden -join ', ')" }
git diff --check a8a964b..HEAD
git status --short
```

Expected: only the two approved files changed after the spec commit; ignored smoke outputs do not appear.

### Task 5: Run peak and frequency 500-step pilots

**Files:**
- Create ignored outputs only under `outputs/td3/`

- [ ] **Step 1: Preflight the two output targets and AGC source**

Run:

```powershell
@(
    'outputs/td3/pilot_peak_500',
    'outputs/td3/pilot_freq_500',
    'outputs/td3/pilot_peak_500_eval100',
    'outputs/td3/pilot_freq_500_eval100'
) | ForEach-Object {
    if (Test-Path -LiteralPath $_) { throw "Output already exists: $_" }
}
& $python -c "from thermal_batch_config import AGC_DATA_FILE; from pathlib import Path; p=Path(AGC_DATA_FILE); print(p); assert p.is_file(), p"
```

Expected: both targets absent and the AGC file exists.

- [ ] **Step 2: Run the peak pilot**

```powershell
& $python run_td3_training.py `
  --scene peak `
  --total-timesteps 500 `
  --seed 7 `
  --checkpoint-interval 100 `
  --output-dir outputs/td3/pilot_peak_500
```

Expected: exit 0 with 500 recorded steps and five checkpoints.

- [ ] **Step 3: Run the frequency pilot**

```powershell
& $python run_td3_training.py `
  --scene freq `
  --total-timesteps 500 `
  --seed 7 `
  --checkpoint-interval 100 `
  --output-dir outputs/td3/pilot_freq_500
```

Expected: exit 0 with 500 recorded steps and five checkpoints.

- [ ] **Step 4: Validate pilot artifacts and numeric finiteness**

Run a Python check that loads both `training_history.csv` files and asserts:

```python
from pathlib import Path
import numpy as np
import pandas as pd

for scene in ("peak", "freq"):
    root = Path(f"outputs/td3/pilot_{scene}_500")
    history = pd.read_csv(root / "training_history.csv")
    assert len(history) == 500
    numeric = history.select_dtypes(include="number").to_numpy(dtype=float)
    assert np.isfinite(numeric).all()
    assert (root / "model.zip").is_file()
    assert (root / "metadata.json").is_file()
    assert (root / "training_curve.png").stat().st_size > 0
    checkpoints = sorted((root / "checkpoints").glob("*.zip"))
    assert [path.name for path in checkpoints] == [
        f"checkpoint_{step:06d}_steps.zip" for step in range(100, 501, 100)
    ]
```

Expected: exit 0.

### Task 6: Run 100-step deterministic evaluations and report

**Files:**
- Create ignored outputs only under `outputs/td3/`

- [ ] **Step 1: Run peak and frequency evaluations**

```powershell
& $python run_td3_evaluation.py `
  --model outputs/td3/pilot_peak_500/model.zip `
  --scene peak `
  --max-steps 100 `
  --output-dir outputs/td3/pilot_peak_500_eval100

& $python run_td3_evaluation.py `
  --model outputs/td3/pilot_freq_500/model.zip `
  --scene freq `
  --max-steps 100 `
  --output-dir outputs/td3/pilot_freq_500_eval100
```

Expected: both commands exit 0 and each trajectory has 100 rows.

- [ ] **Step 2: Calculate the fixed pilot report metrics**

For each scene calculate:

```python
history = pd.read_csv(training_path)
evaluation = pd.read_csv(evaluation_path)
metrics = {
    "reward_first_100_mean": float(history["reward"].head(100).mean()),
    "reward_last_100_mean": float(history["reward"].tail(100).mean()),
    "max_temp_c": float(evaluation["max_temp_c"].max()),
    "max_delta_temp_c": float(evaluation["delta_temp_c"].max()),
    "mean_power_kw": float(evaluation["total_power_w"].mean() / 1000.0),
    "compressor_lower_ratio": float((evaluation["n_comp_cmd_rpm"] <= 300.0 + 1e-9).mean()),
    "compressor_upper_ratio": float((evaluation["n_comp_cmd_rpm"] >= 6000.0 - 1e-9).mean()),
    "pump_lower_ratio": float((evaluation["n_pump_cmd_rpm"] <= 1600.0 + 1e-9).mean()),
    "pump_upper_ratio": float((evaluation["n_pump_cmd_rpm"] >= 4800.0 - 1e-9).mean()),
}
```

Also assert all numeric evaluation fields are finite and no row has `end_reason` equal to `nonfinite_state` or `temperature_limit`.

- [ ] **Step 3: Fresh final verification**

Run again after all pilot work:

```powershell
& $python -m unittest -v tests.td3.test_td3_btms_env
& $python -m unittest -v test_refrigeration_cycle_limits test_thermal_initial_state tests.physics_p.test_mpc_predictor_selection
git diff --check origin/main..HEAD
git status --short --branch
```

Expected: all tests pass; tracked changes are committed; pilot artifacts remain ignored and uncommitted.
