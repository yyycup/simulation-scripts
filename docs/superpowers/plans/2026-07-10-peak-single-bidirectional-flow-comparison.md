# Peak Single/Bidirectional Flow Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce fresh peak-shaving MPC data and a reproducible figure comparing single-direction flow with simple threshold-based bidirectional flow.

**Architecture:** Add one focused runner/plotting module that calls the existing `simulate_case()` twice with the code-recognized `normal` and `reversed` flow identifiers. Keep summary extraction and plotting as pure functions so they can be tested without running the expensive thermal simulation.

**Tech Stack:** Python, pandas, numpy, matplotlib, unittest, existing BTMS/PyBaMM/GEKKO simulation modules.

---

### Task 1: Test summary and reversal extraction

**Files:**
- Create: `test_peak_flow_comparison.py`
- Create: `run_peak_flow_comparison.py`

- [ ] **Step 1: Write failing tests**

Create synthetic single- and bidirectional DataFrames and assert that `summarize_case()` returns peak delta-T/time and counts direction changes, while `validate_comparison()` rejects a single-flow direction other than `+1` and accepts valid data.

- [ ] **Step 2: Verify RED**

Run: `C:\Users\24776\miniforge3\envs\btms\python.exe -m unittest test_peak_flow_comparison.py -v`

Expected: import failure because `run_peak_flow_comparison.py` does not exist.

- [ ] **Step 3: Implement pure helpers**

Implement `time_column()`, `delta_t_column()`, `direction_column()`, `reversal_indices()`, `summarize_case()`, and `validate_comparison()` using exported simulator column names (`Time`, `Cell temperature difference`, `Flow direction d`) with fallback support for internal names.

- [ ] **Step 4: Verify GREEN**

Run the unittest command again and expect all tests to pass.

### Task 2: Implement fresh simulation and plotting entrypoint

**Files:**
- Modify: `run_peak_flow_comparison.py`

- [ ] **Step 1: Add case execution**

Use the existing peak MPC source CSV when present, call `simulate_case()` with identical inputs for `flow="normal"` and `flow="reversed"`, set `temp_diff_limit_c=0.5`, `mpc_flow_mode="switching"`, and write into `outputs/peak_single_bidirectional_flow_comparison/`.

- [ ] **Step 2: Add reproducible plot export**

Create a two-panel matplotlib figure: delta-T comparison plus threshold/peak markers on top, and bidirectional `+1/-1` direction with reversal markers below. Export `peak_flow_comparison.png` at 300 DPI and `peak_flow_comparison.pdf`.

- [ ] **Step 3: Add summary export**

Write `peak_flow_comparison_summary.csv` with peak delta-T, peak time, reversal count, row count, start time, and end time.

- [ ] **Step 4: Syntax and unit verification**

Run `python -m py_compile` on the runner and test, then run the complete unit test file.

### Task 3: Run, verify, and inspect artifacts

**Files:**
- Generate: `outputs/peak_single_bidirectional_flow_comparison/**`

- [ ] **Step 1: Run full comparison**

Run the runner with `C:\Users\24776\miniforge3\envs\btms\python.exe` and wait for both full peak cases.

- [ ] **Step 2: Verify numerical artifacts**

Check equal row counts and time bounds, single direction always `+1`, bidirectional reversal intervals greater than 200 seconds, and summary maxima equal direct CSV maxima.

- [ ] **Step 3: Inspect final image**

Open the generated PNG with the local image viewer and confirm labels, legends, peaks, threshold line, and reversal markers are visible.

- [ ] **Step 4: Deliver image and key values**

Return the rendered PNG and clickable paths to CSV, PDF, summary, and plotting script.
