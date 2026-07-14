# Final Controller Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the selected MPC weights, refine peak/frequency PID parameters on single-flow cases, and provide one reproducible full-run matrix for On-off, PID, and MPC plus MPC bidirectional cases.

**Architecture:** Keep the plant and controller implementations unchanged except for the selected MPC constants and canonical flow-label handling. A focused PID refinement runner writes selected parameters to JSON; a separate final experiment runner reads that JSON and produces eight unique full simulations and a common metric summary under a Chinese output directory.

**Tech Stack:** Python, unittest, pandas, NumPy, existing PyBaMM/GEKKO thermal simulation.

---

### Task 1: Freeze selected MPC weights

**Files:**
- Modify: `mpc_flow_direction_strategies.py`
- Modify: `test_mpc_params.py`

- [ ] **Step 1: Write the failing assertions**

Assert peak/frequency pump energy weights are `15000.0` and `22500.0`, while CV, compressor weights, DMAX limits, and terminal-cost-off defaults remain unchanged.

- [ ] **Step 2: Run the targeted test and verify RED**

Run: `python -m unittest test_mpc_params.MPCParameterSelectionTest -v`

Expected: the two pump-weight assertions fail against the old `10000.0/15000.0` constants.

- [ ] **Step 3: Apply the minimal constant update**

Set:

```python
FINAL_PEAK_W_ENERGY_PUMP = 15000.0
FINAL_FREQ_W_ENERGY_PUMP = 22500.0
```

- [ ] **Step 4: Run the test and verify GREEN**

Run: `python -m unittest test_mpc_params.MPCParameterSelectionTest -v`

Expected: all MPC parameter tests pass.

### Task 2: Canonical flow labels and single-criterion MPC mode

**Files:**
- Modify: `thermal_case_simulator.py`
- Modify: `mpc_flow_direction_strategies.py`
- Create: `test_final_comparison_flow_modes.py`

- [ ] **Step 1: Write failing tests**

Test that `flow_reversal_enabled()` accepts `双向`, `反向`, `double`, `bidirectional`, and `reversed`, rejects `单向` and `single`, and that the MPC factory can construct the `single_predictive_delta_t` mode.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest test_final_comparison_flow_modes -v`

Expected: import or behavior failures because the helper and factory branch do not yet exist.

- [ ] **Step 3: Implement minimal behavior**

Add:

```python
def flow_reversal_enabled(flow):
    return str(flow).strip().lower() in {"双向", "反向", "double", "bidirectional", "reversed"}
```

Use it in `simulate_case`. Add a lazy import branch in `create_mpc_flow_controller`:

```python
if mpc_flow_mode == "single_predictive_delta_t":
    from predictive_delta_t_flow_controller import SinglePredictiveDeltaTMPC
    return SinglePredictiveDeltaTMPC(...)
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m unittest test_final_comparison_flow_modes -v`

Expected: all tests pass.

### Task 3: PID local refinement runner

**Files:**
- Create: `run_pid_local_refinement.py`
- Create: `test_pid_local_refinement.py`

- [ ] **Step 1: Write failing helper tests**

Test deterministic coordinate candidate generation, scene-specific starting parameters, selection of the minimum feasible temperature score, and UTF-8 JSON persistence.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest test_pid_local_refinement -v`

Expected: module import failure.

- [ ] **Step 3: Implement the runner**

Use single-flow cases only. Evaluate scene-specific coordinate candidates on a 1500 s representative segment, save every evaluation to `PID局部细化汇总.csv`, choose the lowest feasible `MAE + 0.5 RMSE + 2 OSC`, and save `PID最终参数.json`. Support `--max-steps` for smoke validation and `--skip-existing` for resumability.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m unittest test_pid_local_refinement -v`

Expected: all tests pass.

### Task 4: Eight-case final experiment runner

**Files:**
- Create: `run_final_controller_comparison.py`
- Create: `test_final_controller_comparison.py`

- [ ] **Step 1: Write failing matrix and summary tests**

Assert the unique matrix contains six single-flow controller cases and two additional bidirectional MPC cases. Test common summary metrics from a small in-memory frame and PID JSON loading.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest test_final_controller_comparison -v`

Expected: module import failure.

- [ ] **Step 3: Implement the runner**

Run On-off/PID/MPC for peak and frequency in single flow; additionally run MPC peak/frequency with `single_predictive_delta_t` bidirectional flow. Use identical source time axes and plant settings within each scene, read PID parameters from `输出结果/最终控制器完整对比/PID局部优化/PID最终参数.json`, checkpoint `完整仿真指标汇总.csv` after each case, and save raw CSVs below `输出结果/最终控制器完整对比/完整仿真`.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m unittest test_final_controller_comparison -v`

Expected: all tests pass.

### Task 5: Smoke verification and handoff

**Files:**
- Verify: all files above

- [ ] **Step 1: Run static/targeted tests**

Run: `python -m unittest test_mpc_params test_final_comparison_flow_modes test_pid_local_refinement test_final_controller_comparison -v`

Expected: all tests pass.

- [ ] **Step 2: Run lightweight CLI checks**

Run both new scripts with `--help`, then run the final comparison with `--max-steps 2` using a test PID JSON.

Expected: argument parsing succeeds and eight summary rows are written without changing full-run outputs.

- [ ] **Step 3: Provide exact PowerShell commands**

Provide one command for PID refinement and one command for the full eight-case comparison, both using `C:\Users\24776\miniforge3\envs\btms\python.exe`.
