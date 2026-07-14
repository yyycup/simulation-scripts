# PID PSO Full-Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old offline PID random-search entry scripts with one full-run PSO tuner that optimizes `peak_pid_params` and `freq_pid_params` on complete one-way plus bidirectional scenes.

**Architecture:** Keep reusable PID evaluation and constraint helpers in `pid_param_search.py`, add a new PSO-focused PyCharm runner that drives scene-level optimization, and delete the obsolete random/two-stage PyCharm entry scripts. Add a narrow unittest file for PSO helper behavior and run it red-green before changing production code.

**Tech Stack:** Python, `unittest`, `pandas`, existing thermal simulation helpers in `simulate_case`

---

### Task 1: Add Narrow Failing Tests For PSO Helper Behavior

**Files:**
- Create: `C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\test_pid_pso_search.py`
- Test: `C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\test_pid_pso_search.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest

from run_pid_pso_full_search_pycharm import (
    clamp_pid_params,
    pso_objective_value,
    update_particle_state,
)


class PidPsoHelperTest(unittest.TestCase):
    def test_objective_prefers_feasible_candidate(self):
        feasible = {"T_avg_mean_error": 0.4, "feasible": True}
        infeasible = {"T_avg_mean_error": 0.1, "feasible": False}

        self.assertLess(pso_objective_value(feasible), pso_objective_value(infeasible))

    def test_clamp_pid_params_limits_each_dimension(self):
        bounds = {"kp": (0.2, 2.0), "ki": (0.0, 0.08), "kd": (0.0, 6.0)}

        self.assertEqual(clamp_pid_params((5.0, -1.0, 9.0), bounds), (2.0, 0.0, 6.0))

    def test_update_particle_state_respects_bounds(self):
        bounds = {"kp": (0.2, 2.0), "ki": (0.0, 0.08), "kd": (0.0, 6.0)}
        position, velocity = update_particle_state(
            position=(1.9, 0.07, 5.5),
            velocity=(1.0, 0.05, 2.0),
            personal_best=(1.9, 0.07, 5.5),
            global_best=(1.9, 0.07, 5.5),
            rng_values=(0.5, 0.5),
            inertia=1.0,
            cognitive=0.0,
            social=0.0,
            bounds=bounds,
        )

        self.assertEqual(position, (2.0, 0.08, 6.0))
        self.assertEqual(velocity, (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest test_pid_pso_search.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing symbol errors because `run_pid_pso_full_search_pycharm.py` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create the PSO module with the tested helper function names and minimal implementations that make the test meaningful.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest test_pid_pso_search.py -v`
Expected: PASS

### Task 2: Implement The Full-Run PSO Runner

**Files:**
- Create: `C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\run_pid_pso_full_search_pycharm.py`
- Modify: `C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\pid_param_search.py`
- Test: `C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\test_pid_pso_search.py`

- [ ] **Step 1: Reuse existing PID evaluation helpers**

Import and use:

```python
from pid_param_search import (
    DEFAULT_PARAM_RANGES,
    _log_factory,
    candidate_passes_constraints,
    evaluate_pid_candidate,
    infer_constraint_limits,
    normalize_scene_arg,
    selected_pid_cases,
)
```

and keep full-scene evaluation through:

```python
metrics = evaluate_pid_candidate(
    params,
    cases=cases,
    scene=scene,
    output_root=candidate_root,
    log_func=log,
)
```

- [ ] **Step 2: Implement scene-level PSO search**

Define focused functions in `run_pid_pso_full_search_pycharm.py`:

```python
def clamp_pid_params(params, bounds): ...
def pso_objective_value(candidate): ...
def update_particle_state(...): ...
def evaluate_scene_particle(...): ...
def run_scene_pso(...): ...
def write_final_params(...): ...
```

with scene configs shaped like:

```python
PSO_SEARCH_RUNS = [
    {"scene": "peak", "swarm_size": 8, "iterations": 6, "output_root": Path("outputs/pid_pso_peak")},
    {"scene": "freq", "swarm_size": 8, "iterations": 6, "output_root": Path("outputs/pid_pso_freq")},
]
```

- [ ] **Step 3: Implement per-particle logging and summary writing**

Write per-scene:

```python
summary_csv = output_root / "pso_summary.csv"
log_path = output_root / "pid_pso_progress.log"
```

and record:

```python
{
    "iteration": iteration,
    "particle": particle_idx,
    "kp": ...,
    "ki": ...,
    "kd": ...,
    "T_avg_mean_error": ...,
    "T_avg_min": ...,
    "T_avg_max": ...,
    "energy": ...,
    "compressor_action_count": ...,
    "feasible": ...,
}
```

- [ ] **Step 4: Implement final output file**

Write:

```python
final_path = Path("outputs/pid_pso_final_pid_params.txt")
```

with:

```python
peak_pid_params = (...)
freq_pid_params = (...)
```

- [ ] **Step 5: Run tests to verify helper behavior still passes**

Run: `python -m unittest test_pid_pso_search.py -v`
Expected: PASS

### Task 3: Delete Old Random-Search Entrypoints

**Files:**
- Delete: `C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\run_pid_param_search_pycharm.py`
- Delete: `C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\run_pid_two_stage_search_pycharm.py`
- Delete: `C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\run_pid_two_stage_search_resilient_pycharm.py`
- Delete: `C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\run_pid_two_stage_freq_only_pycharm.py`
- Delete: `C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\run_pid_two_stage_freq_coarse_batch_pycharm.py`
- Delete: `C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\run_pid_two_stage_freq_full_review_pycharm.py`

- [ ] **Step 1: Confirm shared logic does not depend on deleted entry scripts**

Run:

```powershell
rg -n "run_pid_param_search_pycharm|run_pid_two_stage_search_pycharm|run_pid_two_stage_search_resilient_pycharm|run_pid_two_stage_freq_only_pycharm|run_pid_two_stage_freq_coarse_batch_pycharm|run_pid_two_stage_freq_full_review_pycharm" .
```

Expected: only direct references that can safely disappear.

- [ ] **Step 2: Delete the old entry scripts**

Delete only the six PyCharm random-search entry scripts above. Keep `pid_param_search.py` unless all shared helpers have been cleanly migrated.

- [ ] **Step 3: Run targeted import verification**

Run:

```powershell
python -m py_compile pid_param_search.py run_pid_pso_full_search_pycharm.py test_pid_pso_search.py
```

Expected: exit code 0

### Task 4: Smoke Verification

**Files:**
- Modify: `C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\run_pid_pso_full_search_pycharm.py`

- [ ] **Step 1: Add a tiny smoke mode**

Support:

```python
parser.add_argument("--swarm-size", type=int, default=None)
parser.add_argument("--iterations", type=int, default=None)
parser.add_argument("--scene", choices=["peak", "freq"], default=None)
```

so a very small local verification can run without committing to a huge optimization.

- [ ] **Step 2: Run smoke tests**

Run:

```powershell
python -m unittest test_pid_pso_search.py -v
python -m py_compile pid_param_search.py run_pid_pso_full_search_pycharm.py test_pid_pso_search.py
```

Expected:

- unittest PASS
- py_compile exit code 0

- [ ] **Step 3: Report exact remaining runtime caveat**

Document in the final handoff that full PSO on complete one-way plus bidirectional scenes will still be slow and that feasibility for `freq` is not guaranteed by the algorithm alone.

