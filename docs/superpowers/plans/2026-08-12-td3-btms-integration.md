# BTMS TD3 Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated Gymnasium environment plus TD3 training and evaluation entrypoints for peak-shaving and frequency-regulation BTMS scenarios without changing existing PID, MPC, or plant behavior.

**Architecture:** A new `td3_btms` package composes the existing `BatteryPack` and `simulate_thermal_loop_step()` APIs into a stateful Gymnasium environment. Root-level training and evaluation scripts depend on that package, while the existing controller factory and full-run simulator remain untouched.

**Tech Stack:** Python 3.12, NumPy, Pandas, Gymnasium 1.2, Stable-Baselines3 2.9, PyTorch 2.12, PyBaMM, unittest.

---

## File map

- Create `td3_btms/__init__.py`: public package export only.
- Create `td3_btms/env.py`: scene validation, current-profile loading, action mapping, Gymnasium reset/step, reward, diagnostics.
- Create `run_td3_training.py`: CLI parsing, TD3 construction, short smoke mode, model and metadata persistence.
- Create `run_td3_evaluation.py`: deterministic model rollout, trajectory CSV, summary JSON.
- Create `tests/td3/__init__.py`: test package marker.
- Create `tests/td3/test_td3_btms_env.py`: environment, training helper, and evaluation helper tests.

Do not modify `thermal_case_simulator.py`, `thermal_control_strategies.py`, `thermal_loop.py`, `thermal_system.py`, `pack.py`, or MPC/Physics-P modules.

Use this interpreter for every command:

```powershell
$python = 'D:\conda_envs\btms_td3\python.exe'
```

### Task 1: Current-profile and scene boundary

**Files:**
- Create: `td3_btms/__init__.py`
- Create: `td3_btms/env.py`
- Create: `tests/td3/__init__.py`
- Create: `tests/td3/test_td3_btms_env.py`

- [ ] **Step 1: Write failing profile tests**

Create the package markers and add these tests to `tests/td3/test_td3_btms_env.py`:

```python
import tempfile
import unittest
from pathlib import Path

import numpy as np

from td3_btms.env import build_current_profile, canonical_scene, profile_sha256


class CurrentProfileTests(unittest.TestCase):
    def test_scene_names_are_strict(self):
        self.assertEqual(canonical_scene("peak"), "peak")
        self.assertEqual(canonical_scene(" FREQ "), "freq")
        with self.assertRaisesRegex(ValueError, "scene must be 'peak' or 'freq'"):
            canonical_scene("调峰")

    def test_peak_profile_uses_existing_560_amp_boundary(self):
        profile = build_current_profile("peak", dt=5.0, max_steps=3)
        np.testing.assert_array_equal(profile, np.array([560.0, 560.0, 560.0]))

    def test_injected_profile_is_validated_and_trimmed(self):
        profile = build_current_profile(
            "freq",
            dt=5.0,
            current_profile=[10.0, 20.0, 30.0],
            max_steps=2,
        )
        np.testing.assert_array_equal(profile, np.array([10.0, 20.0]))
        self.assertEqual(profile_sha256(profile), profile_sha256(profile.copy()))
        with self.assertRaisesRegex(ValueError, "finite"):
            build_current_profile("peak", current_profile=[1.0, np.nan])

    def test_frequency_file_must_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing_td3_agc.csv"
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                build_current_profile("freq", agc_data_file=missing, max_steps=2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify the expected import failure**

Run:

```powershell
& $python -m unittest -v tests.td3.test_td3_btms_env.CurrentProfileTests
```

Expected: failure with `ModuleNotFoundError: No module named 'td3_btms'`.

- [ ] **Step 3: Implement the minimal profile boundary**

Create `td3_btms/__init__.py`:

```python
from .env import BTMSTd3Env

__all__ = ["BTMSTd3Env"]
```

Create the initial `td3_btms/env.py`:

```python
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from thermal_batch_config import AGC_DATA_FILE, SIM_DT


DEFAULT_DURATION_S = {"peak": 6400.0, "freq": 3600.0}


def canonical_scene(scene: str) -> str:
    scene_key = str(scene).strip().lower()
    if scene_key not in DEFAULT_DURATION_S:
        raise ValueError("scene must be 'peak' or 'freq'")
    return scene_key


def _validated_profile(values) -> np.ndarray:
    profile = np.asarray(values, dtype=np.float64)
    if profile.ndim != 1 or profile.size == 0:
        raise ValueError("current profile must be a non-empty 1D array")
    if not np.all(np.isfinite(profile)):
        raise ValueError("current profile values must be finite")
    return profile


def build_current_profile(
    scene: str,
    *,
    dt: float = SIM_DT,
    current_profile=None,
    agc_data_file=None,
    max_steps=None,
) -> np.ndarray:
    scene_key = canonical_scene(scene)
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if current_profile is not None:
        profile = _validated_profile(current_profile)
    elif scene_key == "peak":
        steps = int(DEFAULT_DURATION_S[scene_key] / dt)
        profile = np.full(steps, 560.0, dtype=np.float64)
    else:
        path = Path(AGC_DATA_FILE if agc_data_file is None else agc_data_file)
        if not path.is_file():
            raise FileNotFoundError(f"AGC data file does not exist: {path}")
        frame = pd.read_csv(path)
        required = {"Seconds", "RegD"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"AGC data file is missing columns: {missing}")
        source_time = frame["Seconds"].to_numpy(dtype=float)
        source_signal = frame["RegD"].to_numpy(dtype=float)
        if source_time.size == 0 or not np.all(np.isfinite(source_time)):
            raise ValueError("AGC Seconds must be non-empty and finite")
        if source_time.size > 1 and not np.all(np.diff(source_time) > 0.0):
            raise ValueError("AGC Seconds must be strictly increasing")
        if not np.all(np.isfinite(source_signal)):
            raise ValueError("AGC RegD values must be finite")
        times = np.arange(0.0, DEFAULT_DURATION_S[scene_key], dt)
        profile = np.interp(
            times, source_time, source_signal * 1120.0, left=0.0, right=0.0
        )
    if max_steps is not None:
        max_steps = int(max_steps)
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        profile = profile[:max_steps]
    return _validated_profile(profile).copy()


def profile_sha256(profile) -> str:
    values = _validated_profile(profile).astype("<f8", copy=False)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()
```

Add a temporary class declaration at the bottom so the package export resolves until Task 2 replaces it:

```python
class BTMSTd3Env:
    pass
```

- [ ] **Step 4: Run the profile tests**

Run:

```powershell
& $python -m unittest -v tests.td3.test_td3_btms_env.CurrentProfileTests
```

Expected: `Ran 4 tests` and `OK`.

- [ ] **Step 5: Commit the profile boundary**

```powershell
git add td3_btms/__init__.py td3_btms/env.py tests/td3/__init__.py tests/td3/test_td3_btms_env.py
git commit -m "feat: add TD3 scene profile boundary"
```

### Task 2: Reset, observation, and action mapping

**Files:**
- Modify: `td3_btms/env.py`
- Modify: `tests/td3/test_td3_btms_env.py`

- [ ] **Step 1: Add failing action and reset tests**

Append to `tests/td3/test_td3_btms_env.py`:

```python
from td3_btms.env import BTMSTd3Env, map_action_to_rpm


class ActionAndResetTests(unittest.TestCase):
    def test_action_endpoints_map_to_existing_actuator_bounds(self):
        self.assertEqual(map_action_to_rpm(np.array([-1.0, -1.0])), (300.0, 1600.0))
        self.assertEqual(map_action_to_rpm(np.array([1.0, 1.0])), (6000.0, 4800.0))

    def test_action_is_clipped_before_mapping(self):
        self.assertEqual(map_action_to_rpm(np.array([-2.0, 2.0])), (300.0, 4800.0))
        with self.assertRaisesRegex(ValueError, "shape"):
            map_action_to_rpm(np.array([0.0]))

    def test_reset_returns_nine_finite_float32_observations(self):
        env = BTMSTd3Env(scene="peak", current_profile=[560.0, 560.0])
        observation, info = env.reset(seed=7)
        self.assertEqual(observation.shape, (9,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(observation)))
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info["scene"], "peak")
        self.assertEqual(info["step_index"], 0)
        self.assertAlmostEqual(info["mean_soc"], 0.95)

    def test_frequency_reset_uses_regulation_soc(self):
        env = BTMSTd3Env(scene="freq", current_profile=[0.0, 10.0])
        first, _ = env.reset(seed=11)
        second, _ = env.reset(seed=11)
        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(env.last_info["mean_soc"], 0.55)
```

- [ ] **Step 2: Verify the tests fail because the environment is incomplete**

Run:

```powershell
& $python -m unittest -v tests.td3.test_td3_btms_env.ActionAndResetTests
```

Expected: import failure for `map_action_to_rpm` or constructor failure for the placeholder environment.

- [ ] **Step 3: Replace the placeholder with the Gymnasium reset boundary**

Add these imports to `td3_btms/env.py` before importing BTMS physics modules:

```python
from btms_runtime import ensure_env_library_bin_on_path

ensure_env_library_bin_on_path()

import gymnasium as gym
from gymnasium import spaces

from pack import BatteryPack
from thermal_batch_config import (
    AMBIENT_TEMP_C,
    INITIAL_SOC_PEAK,
    INITIAL_SOC_REG,
    INITIAL_TEMP_C,
    N_COMP_MAX_RPM,
    N_COMP_OFF_RPM,
    N_PUMP_MAX_RPM,
    N_PUMP_MIN_RPM,
    PLATE_NODE_HEAT_CAPACITY_TOTAL,
)
from thermal_case_simulator import initialize_thermal_temperatures
from thermal_loop import build_pack_config, initialize_refrigeration_dynamic_state
```

Add the action mapper:

```python
def map_action_to_rpm(action) -> tuple[float, float]:
    values = np.asarray(action, dtype=float)
    if values.shape != (2,):
        raise ValueError("action must have shape (2,)")
    if not np.all(np.isfinite(values)):
        raise ValueError("action values must be finite")
    values = np.clip(values, -1.0, 1.0)
    n_comp = N_COMP_OFF_RPM + 0.5 * (values[0] + 1.0) * (
        N_COMP_MAX_RPM - N_COMP_OFF_RPM
    )
    n_pump = N_PUMP_MIN_RPM + 0.5 * (values[1] + 1.0) * (
        N_PUMP_MAX_RPM - N_PUMP_MIN_RPM
    )
    return float(n_comp), float(n_pump)
```

Replace the placeholder class with:

```python
class BTMSTd3Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scene: str,
        *,
        current_profile=None,
        agc_data_file=None,
        max_steps=None,
        dt: float = SIM_DT,
    ):
        super().__init__()
        self.scene = canonical_scene(scene)
        self.dt = float(dt)
        self.agc_data_file = (
            Path(AGC_DATA_FILE if agc_data_file is None else agc_data_file).resolve()
            if self.scene == "freq" and current_profile is None
            else None
        )
        self.current_profile = build_current_profile(
            self.scene,
            dt=self.dt,
            current_profile=current_profile,
            agc_data_file=self.agc_data_file,
            max_steps=max_steps,
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )
        self.pack = None
        self.last_info = {}
        self._done = False

    def _metrics(self) -> dict[str, float]:
        temps_c = self.pack.temps - 273.15
        dynamic = self.dynamic_state
        return {
            "current_a": float(self.current_profile[min(self.step_index, len(self.current_profile) - 1)]),
            "mean_soc": float(np.mean(self.pack.socs)),
            "mean_temp_c": float(np.mean(temps_c)),
            "max_temp_c": float(np.max(temps_c)),
            "delta_temp_c": float(np.max(temps_c) - np.min(temps_c)),
            "tank_temp_c": float(self.t_tank_k - 273.15),
            "plate_mean_temp_c": float(np.mean(self.t_plate_k) - 273.15),
            "n_comp_eff_rpm": float(dynamic["N_comp_eff"]),
            "n_pump_eff_rpm": float(dynamic["N_pump_eff"]),
        }

    def _observation(self, metrics) -> np.ndarray:
        return np.asarray(
            [
                metrics["current_a"] / 1120.0,
                2.0 * metrics["mean_soc"] - 1.0,
                (metrics["mean_temp_c"] - 25.0) / 10.0,
                (metrics["max_temp_c"] - 25.0) / 10.0,
                metrics["delta_temp_c"] / 2.0,
                (metrics["tank_temp_c"] - 25.0) / 10.0,
                (metrics["plate_mean_temp_c"] - 25.0) / 10.0,
                2.0 * (metrics["n_comp_eff_rpm"] - N_COMP_OFF_RPM)
                / (N_COMP_MAX_RPM - N_COMP_OFF_RPM) - 1.0,
                2.0 * (metrics["n_pump_eff_rpm"] - N_PUMP_MIN_RPM)
                / (N_PUMP_MAX_RPM - N_PUMP_MIN_RPM) - 1.0,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        initial_soc = INITIAL_SOC_REG if self.scene == "freq" else INITIAL_SOC_PEAK
        config = build_pack_config(
            total_current=float(self.current_profile[0]),
            initial_soc=initial_soc,
            initial_temp_c=INITIAL_TEMP_C,
        )
        self.pack = BatteryPack(config)
        self.t_tank_k, self.t_plate_k = initialize_thermal_temperatures(
            self.pack.cols, initial_temp_c=AMBIENT_TEMP_C
        )
        self.c_plate_node = PLATE_NODE_HEAT_CAPACITY_TOTAL / self.pack.cols
        self.dynamic_state = initialize_refrigeration_dynamic_state(
            N_COMP_OFF_RPM,
            N_PUMP_MIN_RPM,
            initial_temp_k=self.t_tank_k,
            dt=self.dt,
        )
        self.step_index = 0
        self._done = False
        metrics = self._metrics()
        self.last_info = {"scene": self.scene, "step_index": 0, **metrics}
        return self._observation(metrics), dict(self.last_info)
```

- [ ] **Step 4: Run reset and profile tests**

Run:

```powershell
& $python -m unittest -v tests.td3.test_td3_btms_env.CurrentProfileTests tests.td3.test_td3_btms_env.ActionAndResetTests
```

Expected: `Ran 8 tests` and `OK`.

- [ ] **Step 5: Commit the reset boundary**

```powershell
git add td3_btms/env.py tests/td3/test_td3_btms_env.py
git commit -m "feat: add TD3 reset and observation boundary"
```

### Task 3: Physics step, reward, and termination

**Files:**
- Modify: `td3_btms/env.py`
- Modify: `tests/td3/test_td3_btms_env.py`

- [ ] **Step 1: Add failing reward and physics-step tests**

Append:

```python
from gymnasium.utils.env_checker import check_env

from td3_btms.env import compute_reward


class RewardAndStepTests(unittest.TestCase):
    def test_reward_penalizes_temperature_spread_and_high_temperature(self):
        safe, safe_terms = compute_reward(
            mean_temp_c=25.0,
            max_temp_c=26.0,
            delta_temp_c=0.4,
            total_power_w=0.0,
        )
        unsafe, unsafe_terms = compute_reward(
            mean_temp_c=25.0,
            max_temp_c=29.0,
            delta_temp_c=1.0,
            total_power_w=0.0,
        )
        self.assertEqual(safe, 0.0)
        self.assertLess(unsafe, safe)
        self.assertGreater(unsafe_terms["delta_temperature_violation_cost"], 0.0)
        self.assertGreater(unsafe_terms["high_temperature_violation_cost"], 0.0)

    def test_peak_environment_advances_real_physics(self):
        env = BTMSTd3Env(scene="peak", current_profile=[560.0, 560.0])
        before, _ = env.reset(seed=3)
        after, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
        self.assertEqual(after.shape, before.shape)
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["step_index"], 1)
        self.assertAlmostEqual(info["n_comp_cmd_rpm"], 3150.0)
        self.assertAlmostEqual(info["n_pump_cmd_rpm"], 3200.0)
        self.assertGreaterEqual(info["total_power_w"], 0.0)

    def test_profile_end_truncates_the_episode(self):
        env = BTMSTd3Env(scene="freq", current_profile=[0.0])
        env.reset(seed=5)
        _, _, terminated, truncated, info = env.step(np.array([-1.0, -1.0]))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["end_reason"], "profile_complete")
        with self.assertRaisesRegex(RuntimeError, "reset"):
            env.step(np.array([-1.0, -1.0]))

    def test_high_temperature_terminates_with_fixed_penalty(self):
        env = BTMSTd3Env(scene="peak", current_profile=[560.0, 560.0])
        env.reset(seed=9)
        env.pack.temps[:] = 45.0 + 273.15
        _, reward, terminated, _, info = env.step(np.array([-1.0, -1.0]))
        self.assertTrue(terminated)
        self.assertEqual(info["end_reason"], "temperature_limit")
        self.assertLessEqual(reward, -100.0)

    def test_gymnasium_contract(self):
        env = BTMSTd3Env(scene="peak", current_profile=[560.0] * 8)
        check_env(env, skip_render_check=True)
```

- [ ] **Step 2: Run the step tests and verify failure**

Run:

```powershell
& $python -m unittest -v tests.td3.test_td3_btms_env.RewardAndStepTests
```

Expected: import failure for `compute_reward` or missing `BTMSTd3Env.step`.

- [ ] **Step 3: Implement reward and the physical transition**

Add to `td3_btms/env.py`:

```python
from thermal_batch_config import AMBIENT_TEMP_K
from thermal_loop import simulate_thermal_loop_step


REWARD_WEIGHTS = {
    "temperature_tracking": 1.0,
    "power": 0.05,
    "delta_temperature": 4.0,
    "high_temperature": 4.0,
}
TERMINATION_TEMP_C = 45.0
TERMINATION_PENALTY = 100.0


def compute_reward(
    *, mean_temp_c: float, max_temp_c: float, delta_temp_c: float, total_power_w: float
) -> tuple[float, dict[str, float]]:
    terms = {
        "temperature_tracking_cost": ((float(mean_temp_c) - 25.0) / 2.0) ** 2,
        "power_cost": float(total_power_w) / 5000.0,
        "delta_temperature_violation_cost": (
            max(0.0, float(delta_temp_c) - 0.5) / 0.5
        ) ** 2,
        "high_temperature_violation_cost": (
            max(0.0, float(max_temp_c) - 27.0) / 2.0
        ) ** 2,
    }
    reward = -(
        REWARD_WEIGHTS["temperature_tracking"]
        * terms["temperature_tracking_cost"]
        + REWARD_WEIGHTS["power"] * terms["power_cost"]
        + REWARD_WEIGHTS["delta_temperature"]
        * terms["delta_temperature_violation_cost"]
        + REWARD_WEIGHTS["high_temperature"]
        * terms["high_temperature_violation_cost"]
    )
    return float(reward), terms
```

Add this method to `BTMSTd3Env`:

```python
    def step(self, action):
        if self.pack is None or self._done:
            raise RuntimeError("call reset() before step() or after episode completion")
        n_comp_cmd, n_pump_cmd = map_action_to_rpm(action)
        step_current = float(self.current_profile[self.step_index])
        self.pack.total_current = step_current
        thermal = simulate_thermal_loop_step(
            pack=self.pack,
            T_tank_K=self.t_tank_k,
            T_plate_K_array=self.t_plate_k,
            N_comp_cmd=n_comp_cmd,
            N_pump_cmd=n_pump_cmd,
            T_outdoor=AMBIENT_TEMP_K,
            dt=self.dt,
            is_reversed=False,
            C_plate_node=self.c_plate_node,
            compressor_power_scale=1.0,
            dynamic_state=self.dynamic_state,
        )
        self.t_tank_k = float(thermal["T_tank_K"])
        self.t_plate_k = np.asarray(thermal["T_plate_K_array"], dtype=float)
        self.dynamic_state = thermal["dynamic_state"]
        self.pack.step(self.dt, T_plate=self.t_plate_k, T_cabinet=AMBIENT_TEMP_K)
        self.pack.history = [[] for _ in range(self.pack.Ns)]
        self.pack.branch_currents_history = [[] for _ in range(self.pack.rows)]
        self.step_index += 1
        metrics = self._metrics()
        total_power_w = float(
            thermal["W_comp_real"] + thermal["W_pump_val"] + thermal["W_fan_real"]
        )
        reward, reward_terms = compute_reward(
            mean_temp_c=metrics["mean_temp_c"],
            max_temp_c=metrics["max_temp_c"],
            delta_temp_c=metrics["delta_temp_c"],
            total_power_w=total_power_w,
        )
        finite_values = np.asarray([*metrics.values(), total_power_w, reward], dtype=float)
        finite_state = bool(np.all(np.isfinite(finite_values)))
        terminated = not finite_state
        end_reason = "nonfinite_state" if terminated else None
        if terminated:
            reward = -TERMINATION_PENALTY
        elif metrics["max_temp_c"] >= TERMINATION_TEMP_C:
            terminated = True
            reward -= TERMINATION_PENALTY
            end_reason = "temperature_limit"
        truncated = not terminated and self.step_index >= len(self.current_profile)
        if truncated and end_reason is None:
            end_reason = "profile_complete"
        self._done = bool(terminated or truncated)
        info = {
            "scene": self.scene,
            "step_index": self.step_index,
            "time_s": self.step_index * self.dt,
            **metrics,
            "current_a": step_current,
            "n_comp_cmd_rpm": n_comp_cmd,
            "n_pump_cmd_rpm": n_pump_cmd,
            "total_power_w": total_power_w,
            "reward": reward,
            "end_reason": end_reason,
            **reward_terms,
        }
        self.last_info = info
        observation = self._observation(metrics)
        if not finite_state:
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
            info["terminal_observation_replaced"] = True
        return observation, reward, terminated, truncated, dict(info)
```

Adjust `_metrics()` so its `current_a` lookup safely uses the last profile value after the final increment:

```python
profile_index = min(self.step_index, len(self.current_profile) - 1)
```

- [ ] **Step 4: Run all environment tests**

Run:

```powershell
& $python -m unittest -v tests.td3.test_td3_btms_env
```

Expected: `Ran 13 tests` and `OK`.

- [ ] **Step 5: Commit the physical environment**

```powershell
git add td3_btms/env.py tests/td3/test_td3_btms_env.py
git commit -m "feat: add TD3 physical step and reward"
```

### Task 4: TD3 training entrypoint and metadata

**Files:**
- Create: `run_td3_training.py`
- Modify: `tests/td3/test_td3_btms_env.py`

- [ ] **Step 1: Add failing training-helper tests**

Append:

```python
import json

from run_td3_training import create_run_directory, write_metadata


class TrainingEntrypointTests(unittest.TestCase):
    def test_run_directory_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "run"
            self.assertEqual(create_run_directory(target), target.resolve())
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                create_run_directory(target)

    def test_metadata_records_profile_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metadata.json"
            profile = np.array([1.0, 2.0])
            write_metadata(
                path,
                scene="peak",
                seed=4,
                total_timesteps=16,
                dt=5.0,
                profile=profile,
                agc_data_file=None,
            )
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["scene"], "peak")
            self.assertEqual(data["profile_sha256"], profile_sha256(profile))
            self.assertEqual(data["reward_weights"]["power"], 0.05)
```

- [ ] **Step 2: Verify the training-helper tests fail**

Run:

```powershell
& $python -m unittest -v tests.td3.test_td3_btms_env.TrainingEntrypointTests
```

Expected: `ModuleNotFoundError: No module named 'run_td3_training'`.

- [ ] **Step 3: Implement the training CLI**

Create `run_td3_training.py` with these concrete boundaries:

```python
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from td3_btms.env import (
    BTMSTd3Env,
    REWARD_WEIGHTS,
    profile_sha256,
)
from thermal_batch_config import (
    N_COMP_MAX_RPM,
    N_COMP_OFF_RPM,
    N_PUMP_MAX_RPM,
    N_PUMP_MIN_RPM,
    SIM_DT,
)


def create_run_directory(path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"output directory already exists: {resolved}") from exc
    return resolved


def write_metadata(
    path,
    *,
    scene,
    seed,
    total_timesteps,
    dt,
    profile,
    agc_data_file,
):
    data = {
        "scene": scene,
        "seed": int(seed),
        "total_timesteps": int(total_timesteps),
        "dt_s": float(dt),
        "action_bounds_rpm": {
            "compressor": [N_COMP_OFF_RPM, N_COMP_MAX_RPM],
            "pump": [N_PUMP_MIN_RPM, N_PUMP_MAX_RPM],
        },
        "reward_weights": dict(REWARD_WEIGHTS),
        "agc_data_file": None if agc_data_file is None else str(Path(agc_data_file).resolve()),
        "profile_sha256": profile_sha256(profile),
    }
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_model(env, *, seed: int, smoke: bool) -> TD3:
    noise = NormalActionNoise(
        mean=np.zeros(2, dtype=np.float32),
        sigma=np.full(2, 0.1, dtype=np.float32),
    )
    kwargs = {
        "action_noise": noise,
        "seed": int(seed),
        "verbose": 1,
    }
    if smoke:
        kwargs.update(
            buffer_size=256,
            learning_starts=1,
            batch_size=8,
            train_freq=(1, "step"),
            gradient_steps=1,
            policy_kwargs={"net_arch": [32, 32]},
        )
    return TD3("MlpPolicy", env, **kwargs)


def smoke_profile(scene: str) -> np.ndarray:
    if scene == "peak":
        return np.full(16, 560.0, dtype=float)
    return np.asarray([0.0, 280.0, 560.0, 280.0] * 4, dtype=float)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train an isolated BTMS TD3 policy")
    parser.add_argument("--scene", choices=("peak", "freq"), required=True)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--agc-data-file", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if not args.smoke and (args.total_timesteps is None or args.total_timesteps < 1):
        parser.error("formal training requires positive --total-timesteps")
    if args.total_timesteps is not None and args.total_timesteps < 1:
        parser.error("--total-timesteps must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    total_timesteps = args.total_timesteps if args.total_timesteps is not None else 16
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    requested = args.output_dir or Path("outputs") / "td3" / args.scene / timestamp
    run_dir = create_run_directory(requested)
    profile = smoke_profile(args.scene) if args.smoke else None
    env = BTMSTd3Env(
        args.scene,
        current_profile=profile,
        agc_data_file=args.agc_data_file,
    )
    try:
        model = build_model(env, seed=args.seed, smoke=args.smoke)
        model.learn(total_timesteps=total_timesteps)
        model.save(run_dir / "model")
        write_metadata(
            run_dir / "metadata.json",
            scene=args.scene,
            seed=args.seed,
            total_timesteps=total_timesteps,
            dt=SIM_DT,
            profile=env.current_profile,
            agc_data_file=env.agc_data_file,
        )
    finally:
        env.close()
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run helper tests and CLI help**

Run:

```powershell
& $python -m unittest -v tests.td3.test_td3_btms_env.TrainingEntrypointTests
& $python run_td3_training.py --help
```

Expected: `Ran 2 tests`, `OK`, and the help command exits 0 with all six options.

- [ ] **Step 5: Commit the training entrypoint**

```powershell
git add run_td3_training.py tests/td3/test_td3_btms_env.py
git commit -m "feat: add isolated TD3 training entrypoint"
```

### Task 5: Deterministic evaluation and result summaries

**Files:**
- Create: `run_td3_evaluation.py`
- Modify: `tests/td3/test_td3_btms_env.py`

- [ ] **Step 1: Add failing summary tests**

Append:

```python
import pandas as pd

from run_td3_evaluation import summarize_trajectory


class EvaluationEntrypointTests(unittest.TestCase):
    def test_summary_uses_physical_columns(self):
        frame = pd.DataFrame(
            {
                "mean_temp_c": [25.0, 26.0],
                "max_temp_c": [25.5, 27.0],
                "delta_temp_c": [0.2, 0.6],
                "total_power_w": [1000.0, 3000.0],
                "reward": [-0.1, -1.1],
            }
        )
        summary = summarize_trajectory(frame, scene="peak", dt=5.0)
        self.assertEqual(summary["steps"], 2)
        self.assertAlmostEqual(summary["temperature_mae_c"], 0.5)
        self.assertAlmostEqual(summary["max_temp_c"], 27.0)
        self.assertAlmostEqual(summary["max_delta_temp_c"], 0.6)
        self.assertAlmostEqual(summary["mean_total_power_kw"], 2.0)
        self.assertAlmostEqual(summary["total_energy_kwh"], 2000.0 * 10.0 / 3_600_000.0)
        self.assertAlmostEqual(summary["cumulative_reward"], -1.2)
```

- [ ] **Step 2: Verify the evaluation test fails**

Run:

```powershell
& $python -m unittest -v tests.td3.test_td3_btms_env.EvaluationEntrypointTests
```

Expected: `ModuleNotFoundError: No module named 'run_td3_evaluation'`.

- [ ] **Step 3: Implement deterministic evaluation**

Create `run_td3_evaluation.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import TD3

from run_td3_training import create_run_directory, smoke_profile
from td3_btms.env import BTMSTd3Env


def summarize_trajectory(frame: pd.DataFrame, *, scene: str, dt: float) -> dict:
    return {
        "scene": scene,
        "steps": int(len(frame)),
        "temperature_mae_c": float(np.mean(np.abs(frame["mean_temp_c"] - 25.0))),
        "max_temp_c": float(frame["max_temp_c"].max()),
        "max_delta_temp_c": float(frame["delta_temp_c"].max()),
        "mean_total_power_kw": float(frame["total_power_w"].mean() / 1000.0),
        "total_energy_kwh": float(frame["total_power_w"].sum() * dt / 3_600_000.0),
        "cumulative_reward": float(frame["reward"].sum()),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate an isolated BTMS TD3 policy")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scene", choices=("peak", "freq"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--agc-data-file", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"TD3 model does not exist: {model_path}")
    metadata_path = model_path.parent / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("scene") != args.scene:
            raise ValueError(
                f"model scene {metadata.get('scene')!r} does not match {args.scene!r}"
            )
    output_dir = create_run_directory(args.output_dir)
    profile = smoke_profile(args.scene) if args.smoke else None
    env = BTMSTd3Env(
        args.scene,
        current_profile=profile,
        agc_data_file=args.agc_data_file,
        max_steps=args.max_steps,
    )
    model = TD3.load(model_path, env=env)
    rows = []
    try:
        observation, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            rows.append(
                {
                    **info,
                    "action_comp_normalized": float(action[0]),
                    "action_pump_normalized": float(action[1]),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
    finally:
        env.close()
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "trajectory.csv", index=False, encoding="utf-8-sig")
    summary = summarize_trajectory(frame, scene=args.scene, dt=env.dt)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run evaluation helper tests and CLI help**

Run:

```powershell
& $python -m unittest -v tests.td3.test_td3_btms_env.EvaluationEntrypointTests
& $python run_td3_evaluation.py --help
```

Expected: `Ran 1 test`, `OK`, and evaluation help exits 0.

- [ ] **Step 5: Commit the evaluation entrypoint**

```powershell
git add run_td3_evaluation.py tests/td3/test_td3_btms_env.py
git commit -m "feat: add deterministic TD3 evaluation"
```

### Task 6: Full milestone verification

**Files:**
- Modify only if verification exposes a defect: files created in Tasks 1–5

- [ ] **Step 1: Compile and run all TD3 tests**

Run:

```powershell
& $python -m compileall -q td3_btms run_td3_training.py run_td3_evaluation.py tests/td3
& $python -m unittest -v tests.td3.test_td3_btms_env
```

Expected: compile exit 0, `Ran 16 tests`, and `OK`.

- [ ] **Step 2: Run adjacent regression boundaries**

Run:

```powershell
& $python -m unittest -v test_refrigeration_cycle_limits test_thermal_initial_state tests.physics_p.test_mpc_predictor_selection
```

Expected: all discovered tests pass. Report the actual count and duration; do not replace this result with the TD3-only test result.

- [ ] **Step 3: Run real short TD3 train/load/evaluate smoke checks for both scenes**

Run once with fresh directories:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$smokeRoot = Join-Path (Resolve-Path 'outputs').Path "td3\smoke_$stamp"
& $python run_td3_training.py --scene peak --smoke --seed 7 --output-dir "$smokeRoot\peak_train"
& $python run_td3_evaluation.py --model "$smokeRoot\peak_train\model.zip" --scene peak --smoke --max-steps 4 --output-dir "$smokeRoot\peak_eval"
& $python run_td3_training.py --scene freq --smoke --seed 7 --output-dir "$smokeRoot\freq_train"
& $python run_td3_evaluation.py --model "$smokeRoot\freq_train\model.zip" --scene freq --smoke --max-steps 4 --output-dir "$smokeRoot\freq_eval"
```

Expected:

- Four commands exit 0.
- Both training directories contain `model.zip` and `metadata.json`.
- Both evaluation directories contain `trajectory.csv` and `summary.json`.
- Each evaluation CSV contains four rows with finite reward, temperature, speed, and power fields.

- [ ] **Step 4: Verify behavior isolation and repository hygiene**

Run:

```powershell
$changed = git diff --name-only 82d8782..HEAD
$changed
$forbidden = $changed | Where-Object {
    $_ -in @(
        'thermal_case_simulator.py',
        'thermal_control_strategies.py',
        'thermal_loop.py',
        'thermal_system.py',
        'pack.py',
        'mpc_flow_direction_strategies.py',
        'mpc_physics_predictor.py'
    )
}
if ($forbidden) { throw "Core behavior files changed: $($forbidden -join ', ')" }
git diff --check 82d8782..HEAD
git status --short
```

Expected: no forbidden files, no whitespace errors, and only the user's pre-existing untracked environment artifacts remain outside the TD3 commits.

- [ ] **Step 5: Commit any verification-only correction**

If a defect was found, add only the affected new TD3 files and commit:

```powershell
git add td3_btms run_td3_training.py run_td3_evaluation.py tests/td3
git commit -m "fix: complete TD3 smoke verification"
```

If no defect was found, do not create an empty commit.
