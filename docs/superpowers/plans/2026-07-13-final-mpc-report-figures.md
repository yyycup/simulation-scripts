# Final MPC Report Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build four Chinese internal-report figures from the existing terminal-cost 12-case CSV results without rerunning simulations.

**Architecture:** Add one standalone plotting module that owns file discovery, metadata validation, summary extraction, and four figure builders. Keep simulation and controller modules untouched. Add source-data fixture tests plus an end-to-end output test so plotted metrics remain tied to the saved CSV artifacts.

**Tech Stack:** Python 3, pandas, NumPy, Matplotlib, unittest

---

### Task 1: Define and test result loading

**Files:**
- Create: `plot_final_mpc_report.py`
- Create: `test_plot_final_mpc_report.py`

- [ ] **Step 1: Write failing loader tests**

Create tests that import `RESULT_ROOT`, `load_summary`, `load_case`, and `validate_mpc_terminal_cost`, then assert that the 12-case summary has 12 rows, the six single-flow comparison cases exist, and all four MPC cases carry `Terminal_Cost_Enabled=True` with peak/freq weights `1e6/5e5`.

- [ ] **Step 2: Run the loader tests and verify failure**

Run: `C:\Users\24776\miniforge3\envs\btms\python.exe -m unittest -v test_plot_final_mpc_report.py`

Expected: import fails because `plot_final_mpc_report` does not exist.

- [ ] **Step 3: Implement the minimal loader and validation layer**

Implement constants for the terminal-cost result root and output directory, robust Boolean parsing, `load_summary()`, `case_csv_path(scene, control, flow)`, `load_case(...)`, and `validate_mpc_terminal_cost(...)`. Reject missing files, missing columns, incorrect terminal flags, and incorrect scene weights with explicit errors.

- [ ] **Step 4: Run loader tests**

Run the same unittest command.

Expected: loader and metadata tests pass.

### Task 2: Build the four figures

**Files:**
- Modify: `plot_final_mpc_report.py`
- Modify: `test_plot_final_mpc_report.py`

- [ ] **Step 1: Write failing plotting tests**

Add tests for `plot_controller_temperature_timeseries`, `plot_controller_temperature_metrics`, `plot_mpc_flow_temperature`, and `plot_mpc_other_metrics`. Assert expected axes counts `2, 2, 4, 3`, key Chinese axis labels, and the intended number of plotted controller/flow series or bars.

- [ ] **Step 2: Run focused plotting tests and verify failure**

Run: `C:\Users\24776\miniforge3\envs\btms\python.exe -m unittest -v test_plot_final_mpc_report.py`

Expected: failures identify the four missing plotting functions.

- [ ] **Step 3: Implement shared styling and plot builders**

Implement Chinese font fallback, white background, light grid, blue/orange scenario colors, stable controller colors, minute conversion, consistent temperature limits, and these builders:

- Figure 1: peak/freq single-flow On-Off, PID, MPC average-temperature time series with MPC dynamic target.
- Figure 2: single-flow MAE and maximum cell temperature difference for On-Off, PID, MPC.
- Figure 3: peak/freq MPC single/double average temperature and cell maximum-difference time series.
- Figure 4: four-case MPC bars for cumulative energy, minimum cell temperature, and mean cell temperature difference, with value labels.

Use the saved summary columns `temperature_MAE_C`, `max_delta_T_C`, `energy_kWh`, `temperature_min_C`, and `mean_delta_T_C`. Use raw columns `Time`, `T_cell_mean_C`, `Delta_T_cell_C`, and `MPC dynamic target temperature`.

- [ ] **Step 4: Run plotting tests**

Run the same unittest command.

Expected: all plotting tests pass without opening GUI windows.

### Task 3: Export and visually verify artifacts

**Files:**
- Modify: `plot_final_mpc_report.py`
- Modify: `test_plot_final_mpc_report.py`
- Create: `输出结果/最终控制器完整对比/服务器_12组_终端代价_candidateB_20260712_134913/MPC内部汇报图/*.png`
- Create: `输出结果/最终控制器完整对比/服务器_12组_终端代价_candidateB_20260712_134913/MPC内部汇报图/*.pdf`

- [ ] **Step 1: Write the failing export test**

Test `generate_all_figures(output_dir)` with `tempfile.TemporaryDirectory`. Assert exactly four PNG and four PDF files are created and each file has nonzero size.

- [ ] **Step 2: Run the export test and verify failure**

Run the same unittest command.

Expected: failure reports that `generate_all_figures` is missing.

- [ ] **Step 3: Implement export entry point**

Implement `generate_all_figures` to validate source artifacts, create the output directory, save each figure at 220 dpi as PNG and as tightly bounded PDF, close figures, print absolute output paths, and expose a `main()` entry point.

- [ ] **Step 4: Run all tests**

Run: `C:\Users\24776\miniforge3\envs\btms\python.exe -m unittest -v test_plot_final_mpc_report.py`

Expected: all tests pass.

- [ ] **Step 5: Generate the real figures**

Run: `C:\Users\24776\miniforge3\envs\btms\python.exe plot_final_mpc_report.py`

Expected: four PNG and four PDF paths are printed under `MPC内部汇报图`.

- [ ] **Step 6: Perform visual and data QA**

Open all four PNG files and verify Chinese text renders, curves are legible, legends do not cover key data, axes are consistent, bar labels are visible, and no global titles appear. Cross-check displayed bar values against `parallel_controller_summary.csv`; run `python -m py_compile plot_final_mpc_report.py test_plot_final_mpc_report.py` and the focused unittest suite once more.
