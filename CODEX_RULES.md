# Codex Project Rules

## General Rules

- Do not run PSO unless explicitly requested.
- Do not run long simulations unless explicitly requested.
- Do not execute `git add`, `git commit`, or `git push` unless explicitly requested.
- Do not delete, move, or rename files unless explicitly requested.
- Do not modify controller, plant, PID, or main simulation logic unless the task explicitly asks for it.
- For temporary experiments, prefer independent diagnostic scripts instead of modifying the main workflow.

## Test and Diagnostic Running Rules

- Run tests and diagnostics silently by default.
- Do not continuously monitor terminal output.
- Do not repeatedly run `tail` or keep polling progress logs.
- Do not frequently report intermediate progress while a script is running.
- Write runtime logs to `outputs/<task_name>/run.log`.
- Console output should be minimal: start time, end time, and result file path.
- Only after the run finishes, read the final CSV, summary CSV if available, and the last 30 lines of `run.log`.
- If a script fails, inspect only the traceback or error region instead of repeatedly checking logs.

## Reporting Rules

Final reports should only include:

- Whether the task succeeded;
- Files added or modified;
- `py_compile` or `unittest` results if applicable;
- Result CSV path;
- Compact result table;
- Objective conclusion;
- Next recommended step.

Do not paste large logs unless explicitly requested.

## Data Output and Token Saving Rules

- Do not generate large CSV files unless explicitly requested.
- By default, each diagnostic script should output only one compact summary CSV, such as `summary_metrics.csv`.
- Do not save full time-series data by default.
- Do not save per-step, per-cell, per-node, or per-prediction-horizon data unless explicitly requested.
- If detailed time-series data is needed for debugging, save it only when a flag such as `--save-detail` or `SAVE_DETAIL = True` is enabled.
- Default output should include only aggregate metrics, for example:
  - Tmin
  - Tmean
  - Tmax
  - Tfinal
  - hot duration
  - hot overshoot
  - hot degree seconds
  - energy consumption
  - mean actuator speed
  - mean heat transfer rate
- Do not print full CSV contents in the chat.
- Do not paste large tables into the final report.
- Final reports should show only a compact table with the most important rows and columns.
- If a CSV has more than 20 rows or many columns, report only:
  - file path
  - row count
  - column count
  - selected key columns
  - objective conclusion
- Keep detailed logs and data files local. Only inspect them if a run fails or if the user explicitly asks.
- Prefer `summary.csv` for reporting and keep detailed data optional.

## Script Reuse and Minimal-Code Rules

- Do not create a new diagnostic script for every parameter sweep.
- Before adding any new script, first check whether an existing diagnostic runner can be reused or extended.
- Prefer one reusable runner with command-line arguments or a small config block over many task-specific scripts.
- New scripts should be created only when the task cannot be handled by existing runners.
- If a new script is necessary, explain why in one sentence.
- Do not create a new unittest file unless the task changes reusable logic or parsing logic.
- For simple 60-step sweeps, prefer `py_compile` plus one summary CSV over new test files.
- Do not copy large functions such as `simulate_case` repeatedly unless there is no stable import/configuration path.
- If copying is unavoidable, mark the script as temporary and keep it compact.
- Default output should be one summary CSV only.
- Do not save detailed time-series CSV unless explicitly requested.
- Do not paste full CSV contents in the chat.
- Final response should report only key rows, key metrics, and objective conclusion.

## Preferred Diagnostic Workflow

For future MPC parameter sweeps, prefer reusing a single script such as:

`run_mpc_sensitivity_60.py`

with configurable options for:

- sweep target, such as `w_energy_comp`, `WSPHI`, `DMAX`, `DCOST`;
- case type, such as `peak` or `freq`;
- initial state mode, such as `default` or `unified_25C`;
- evaporator mode, such as `baseline` or `candidate_B`;
- output mode, default `summary_only`;
- optional detail saving only with `--save-detail`.

Do not create separate scripts like `run_xxx_60.py` for every small sweep unless explicitly requested.
