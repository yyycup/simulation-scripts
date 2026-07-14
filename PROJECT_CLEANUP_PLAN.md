# Project Cleanup Plan

Generated: 2026-06-29. Scope: analysis and dry-run plan only.

This plan does not delete files, move files, modify business logic, run simulations, run PSO, or run full tests. Output/data/archive directories were inspected by directory-level statistics and selected filename listings only.

## 1. Current Directory Overview

Top-level structure:

| Path | Type | Notes |
| -- | -- | -- |
| `*.py` | source/scripts/tests | Core models, controllers, run scripts, and regression/smoke tests. |
| `outputs/` | result directory | 193 top-level result directories; newest work is 2026-06-29 WSPHI / 60-step MPC validation. |
| `data/` | input data | 2 small parameter JSON files; keep by default. |
| `_archive/` | existing archive | Existing `cleanup_20260622`; leave untouched. |
| `docs/` | documentation/plans | Existing specs/plans; keep. |
| `.agents/`, `.codex/`, `.git_empty_backup/` | tooling/history | Mark only; do not move in this cleanup. |
| `.git/`, `__pycache__/`, `tmprlrvl59z/` | git/cache/GEKKO temp | Ignore; do not scan or move in this pass. |
| `simulink-agentic-toolkit*/` | external toolkit dirs | Mark only; do not move unless separately requested. |

## 2. File Classification Table

| Path | Classification | Suggested Action | Reason | Risk |
| -- | -- | -- | -- | -- |
| `thermal_case_simulator.py` | A core main flow | KEEP | Main case simulation orchestration. | High if moved. |
| `thermal_loop.py` | A core thermal loop | KEEP | Step-level thermal/coolant loop logic. | High if moved. |
| `thermal_system.py` | A core thermal/refrigeration model | KEEP | Current system model and evaporator-related behavior. | High if moved. |
| `thermal_control_strategies.py` | A core controller factory | KEEP | Routes PID/MPC/on-off strategies. | High if moved. |
| `mpc_flow_direction_strategies.py` | A core MPC/flow-direction implementation | KEEP | Main GEKKO MPC and switching/supervisory wrappers. | High if moved. |
| `thermal_batch_config.py` | A core scenario/config file | KEEP | Current case definitions and PID/MPC constants. | High if moved. |
| `pack.py` | A core battery pack model | KEEP | Battery pack thermal/electrical model support. | High if moved. |
| `age_model.py` | A core degradation model | KEEP | Ageing model dependency. | Medium if moved. |
| `mpc_reduced_model_calibration.py` | A core calibration utility | KEEP | Current reduced-order model calibration path. | Medium if moved. |
| `pid_param_search.py` | B current PID/PSO logic | KEEP | Current constraint-filtered PID/PSO helper. | High for PSO task. |
| `run_unified_initial_state_mpc_60.py` | B current validation script | KEEP | Recent unified 25 C initial-state MPC 60-step validation. | Medium. |
| `run_current_evap_default_mpc_60.py` | B current validation script | KEEP | Recent current evaporator default MPC 60-step validation. | Medium. |
| `run_mpc_cv_wsphi_sensitivity_60.py` | B current validation script | KEEP | Recent WSPHI 60-step scan. | Medium. |
| `test_mpc_cv_wsphi_sensitivity.py` | B current smoke/regression test | KEEP | Lightweight WSPHI result validation. | Medium. |
| `run_evaporator_config_mpc_short_validation.py` | B current validation script | KEEP | Recent evaporator parameter MPC short validation. | Medium. |
| `run_evaporator_candidate_validation.py` | B current validation script | KEEP | Recent evaporator candidate validation. | Medium. |
| `run_evaporator_candidate_validation_mpc_retry.py` | B current retry script | REVIEW | Retry helper may be obsolete after successful validation. | Medium until user confirms. |
| `run_evaporator_parameter_sensitivity.py` | B current sensitivity script | REVIEW | Recent but sensitivity-style; keep if still tuning evaporator parameters. | Medium. |
| `run_pid_fixed_param_validation.py` | B current PID/MPC comparison | KEEP | Validates fixed PID parameters. | Medium. |
| `run_pid_pso_full_search_pycharm.py` | B current PSO task | KEEP | Current PSO-PID full-search runner. | High for PSO task. |
| `run_freq_pso_monitor_pycharm.py` | B current PSO monitor | KEEP | Preferred quiet PyCharm monitor for frequency PSO. | High for PSO task. |
| `run_local_freq_pso_scheduled.cmd` | B current PSO launcher | KEEP | Scheduled local PSO launcher. | Medium. |
| `run_mpc_param_cross_full.py` | B current/near-current MPC validation | KEEP | Recent full parameter cross validation. | Medium. |
| `run_mpc_supervised.py` | B reusable runner | KEEP | Canonical supervised MPC runner. | Medium. |
| `run_mpc_switching.py` | B reusable runner | KEEP | Canonical switching MPC runner. | Medium. |
| `test_mpc_params.py` | B regression test | KEEP | Checks peak/frequency MPC parameter routing. | Medium. |
| `test_pid_pso_search.py` | B regression test | KEEP | Checks current PID/PSO helpers. | Medium. |
| `test_mpc_reduced_model_calibration.py` | B regression test | KEEP | Checks calibration helper behavior. | Medium. |
| `test_flow_supervisor.py` | B regression test | KEEP | Checks flow supervisor behavior. | Medium. |
| `test_exergy_metrics.py` | C large historical test | REVIEW | Large test file; keep only if exergy metrics remain active. | Medium. |
| `run_basic_mpc_direct_benchmark.py` | C benchmark script | ARCHIVE | One-off direct benchmark, superseded by current 60-step validation. | Low. |
| `run_control_effectiveness_diagnostics.py` | C diagnostic script | REVIEW | Recent diagnostic; confirm before archiving. | Medium. |
| `run_heat_exchange_mechanism_diagnostics.py` | C diagnostic script | REVIEW | Recent diagnostic; confirm before archiving. | Medium. |
| `run_pump_mechanism_diagnostics.py` | C diagnostic script | REVIEW | Recent diagnostic; confirm before archiving. | Medium. |
| `run_tank_capacity_diagnostics.py` | C diagnostic script | REVIEW | Recent diagnostic; confirm before archiving. | Medium. |
| `run_tank_capacity_diagnostics_continue.py` | C diagnostic continuation | REVIEW | Continuation script may be obsolete if diagnosis is done. | Medium. |
| `run_tank_initial_temperature_diagnostics.py` | C diagnostic script | REVIEW | Recent diagnostic; confirm before archiving. | Medium. |
| `run_pump_weight_sensitivity.py` | C sensitivity script | REVIEW | Recent sensitivity script; likely one-off after WSPHI work. | Medium. |
| `run_mpc_objective_ablation.py` | C ablation script | ARCHIVE | Ablation-style experiment, not current task path. | Low. |
| `run_mpc_objective_ablation_pipe_delay.py` | C ablation script | ARCHIVE | Ablation-style experiment, not current task path. | Low. |
| `run_mpc_scene_param_2x2.py` | C parameter trial | ARCHIVE | Older small parameter cross trial. | Low. |
| `run_mpc_preview_reserve_comparison.py` | C comparison script | ARCHIVE | Historical preview comparison. | Low. |
| `run_pipe_delay_comparison.py` | C comparison script | ARCHIVE | Historical pipe-delay comparison. | Low. |
| `run_mpc_weight_tuning.py` | C old tuning script | ARCHIVE | Superseded by current parameter/WSPHI scripts. | Low. |
| `run_same_limits_12_cases.py` | C old validation script | ARCHIVE | Older same-limits runner. | Low. |
| `identify_open_loop_lag.py` | C old identification script | ARCHIVE | Historical lag identification workflow. | Low. |
| `run_complete_structure_batch.cmd` | C old launcher | ARCHIVE | Old batch helper. | Low. |
| `run_supervised_buffer_sensitivity.py` | C sensitivity script | REVIEW | Memory marks it as recoverable preview runner; confirm before archive. | Medium. |
| `切换式单一判据.py` | C historical/Chinese runner | REVIEW | Likely older named runner; confirm if still referenced in notes. | Medium. |
| `监督式综合判据.py` | C historical/Chinese runner | REVIEW | Likely older named runner; confirm if still referenced in notes. | Medium. |
| `m_dot_ref_nominal_sensitivity.csv` | C old root CSV | ARCHIVE | Root-level sensitivity output should live in archive/output. | Low. |
| `压缩机功耗拟合结果` | C old result artifact | ARCHIVE | Old compressor fitting artifact without extension. | Low. |
| `nul` | C accidental file | ARCHIVE | Zero-byte Windows artifact. | Low. |
| `AGC` | E input/reference artifact | REVIEW | Small root artifact; unclear whether data/reference. | Medium. |
| `.gitignore`, `AGENTS.md` | repo metadata/instructions | KEEP | Required repo hygiene and agent rules. | High if moved. |
| `docs/` | docs/plans | KEEP | Contains existing specs/plans. | Medium. |
| `data/` | E data | KEEP | Only directory-level statistics checked; retain by default. | High if moved. |
| `.agents/`, `.codex/`, `.git_empty_backup/` | F tooling/history | IGNORE | Marked only; do not move this pass. | Medium/high. |
| `simulink-agentic-toolkit/`, `simulink-agentic-toolkit-main/` | F external toolkit | IGNORE | External tools; do not move in project cleanup. | High. |
| `tmprlrvl59z/` | F GEKKO temp/access-denied dir | IGNORE | Do not scan or touch GEKKO temp dir. | High. |
| `__pycache__/` | cache | IGNORE | Do not scan or touch in this pass. | Low. |

## 3. Outputs Slimming Recommendation

Directory-level statistics only: `outputs/` currently has 193 top-level result directories.

Keep these output directories:

| Path | Reason |
| -- | -- |
| `outputs/mpc_cv_wsphi_x2_x5_x10_60steps/` | Latest WSPHI x2/x5/x10 run log. |
| `outputs/mpc_cv_wsphi_freq_x1_60steps/` | Latest WSPHI frequency x1 metrics. |
| `outputs/mpc_cv_wsphi_sensitivity_60steps/` | Current WSPHI 60-step metrics. |
| `outputs/unified_initial_state_mpc_60steps/` | Current unified 25 C initial-state validation. |
| `outputs/current_evap_default_mpc_60steps/` | Current evaporator-default validation. |
| `outputs/evaporator_config_mpc_validation_60steps/` | Current evaporator configuration validation. |
| `outputs/evaporator_candidate_validation_60steps/` | Recent evaporator candidate validation. |
| `outputs/evaporator_parameter_sensitivity/` | Recent evaporator parameter sensitivity; keep until parameter choice is finalized. |
| `outputs/pid_fixed_param_validation/` | Current fixed PID validation summary. |
| `outputs/pid_pso_freq/` | Current/recent frequency PSO output; contains progress and summary files. |
| `outputs/mpc_freq_1000floor_directdisp130_h45_temptrack400_full/` | Previously validated full frequency MPC result retained for reporting. |
| `outputs/mpc_reduced_model_calibration/` | Contains `best_theta.json`, calibration CSV, and figure. |
| `outputs/supervised_buffer_sensitivity/` | Contains summary CSVs/figure path from prior report-ready sensitivity study. |
| `outputs/strategy_24case_figures/` | Possible paper/report figures. |
| `outputs/latest_run_analysis/` | Recent summary analysis snapshot. |

Review before archiving:

| Path Pattern | Reason |
| -- | -- |
| `outputs/pid_pso_peak/`, `outputs/pid_two_stage_peak/`, `outputs/pid_two_stage_freq/` | Large PID search histories; confirm whether only summaries are needed. |
| `outputs/mpc_param_cross_no_precool_full/`, `outputs/mpc_freq_*_full/` from 2026-06-16 to 2026-06-25 | Some may be report baselines; keep only selected winners. |
| `outputs/*diagnostics*`, `outputs/*sensitivity*` from 2026-06-27 | Recent mechanism/sensitivity work; confirm if conclusions have been copied to current notes. |
| Empty `outputs/btms_gekko_*` directories | Likely GEKKO temp output roots; archive only after confirming no active process uses them. |

Archive candidates:

| Path Pattern | Suggested Action | Reason |
| -- | -- | -- |
| `outputs/*smoke*` | ARCHIVE | Smoke checks are historical once scripts/tests are stable. |
| `outputs/*diagnose*` and `outputs/diagnose_*` | ARCHIVE | Historical troubleshooting output. |
| `outputs/*preview*`, except explicitly reviewed current baselines | ARCHIVE | Preview experiments are not current results. |
| `outputs/*ablation*` | ARCHIVE | Ablation trials are historical. |
| `outputs/*old*` | ARCHIVE | Explicitly old result roots. |
| `outputs/*check*` from May/early June | ARCHIVE | Historical check runs. |
| `outputs/basic_mpc_*`, `outputs/header_smoke_*`, `outputs/cache_smoke*`, `outputs/rated_*`, `outputs/vd*_smoke`, `outputs/off_active_min_smoke` | ARCHIVE | Early one-off smoke/parameter results. |
| `outputs/pycache_check/` | ARCHIVE | Cache check output, not a retained result. |

Current task result files to retain:

| Path | Reason |
| -- | -- |
| `outputs/mpc_cv_wsphi_sensitivity_60steps/mpc_cv_wsphi_sensitivity_metrics.csv` | Current WSPHI metrics. |
| `outputs/mpc_cv_wsphi_sensitivity_60steps/mpc_cv_wsphi_sensitivity_metrics.json` | Current WSPHI metrics. |
| `outputs/mpc_cv_wsphi_sensitivity_60steps/run_log.txt` | Current WSPHI run log. |
| `outputs/mpc_cv_wsphi_freq_x1_60steps/mpc_cv_wsphi_sensitivity_metrics.csv` | Latest frequency x1 metrics. |
| `outputs/unified_initial_state_mpc_60steps/unified_initial_state_mpc_metrics.csv` | Unified initial-state validation summary. |
| `outputs/current_evap_default_mpc_60steps/current_evap_default_mpc_metrics.csv` | Current evaporator-default summary. |
| `outputs/evaporator_config_mpc_validation_60steps/evaporator_config_mpc_metrics.csv` | Evaporator config validation summary. |
| `outputs/pid_pso_freq/progress.json` | Current/recent PSO progress state. |
| `outputs/pid_pso_freq/pid_pso_progress.log` | Current/recent PSO log. |
| `outputs/pid_pso_freq/pso_summary.csv` | Current/recent PSO summary. |
| `outputs/pid_fixed_param_validation/pid_fixed_summary.csv` | Current fixed PID validation summary. |

Estimated dry-run result: archive about 176 items after confirmation: about 16 root files plus about 160 output directories. Review about 20 items manually before moving.

## 4. Recommended New-Conversation Read List

Keep the first pass under 10 files:

1. `AGENTS.md`
2. `PROJECT_MAP_MIN.md`
3. `thermal_case_simulator.py`
4. `thermal_batch_config.py`
5. `thermal_control_strategies.py`
6. `mpc_flow_direction_strategies.py`
7. `thermal_loop.py`
8. `thermal_system.py`
9. `pid_param_search.py`
10. The one current `run_*.py` file named by the task.

Avoid reading `outputs/`, `data/`, `_archive/`, `.git/`, `__pycache__/`, `node_modules/`, GEKKO temp directories, and external toolkit directories unless a specific named file is needed.

## 5. No-Change Statement

This round generated only the cleanup plan and minimal project map. No files were deleted, no files were moved, no business code was changed, no simulations were run, no PSO was run, and no full tests were run.

## 6. Dry-Run Archive Command Template

If confirmed, create a timestamped archive root:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$archive = Join-Path '_archive' "cleanup_$stamp"
New-Item -ItemType Directory -Path $archive
```

Then move only confirmed candidates, grouped by category:

```powershell
Move-Item -LiteralPath 'run_basic_mpc_direct_benchmark.py' -Destination $archive
Move-Item -LiteralPath 'run_mpc_objective_ablation.py' -Destination $archive
Move-Item -LiteralPath 'run_mpc_objective_ablation_pipe_delay.py' -Destination $archive
Move-Item -LiteralPath 'run_mpc_scene_param_2x2.py' -Destination $archive
Move-Item -LiteralPath 'run_mpc_preview_reserve_comparison.py' -Destination $archive
Move-Item -LiteralPath 'run_pipe_delay_comparison.py' -Destination $archive
Move-Item -LiteralPath 'run_mpc_weight_tuning.py' -Destination $archive
Move-Item -LiteralPath 'run_same_limits_12_cases.py' -Destination $archive
Move-Item -LiteralPath 'identify_open_loop_lag.py' -Destination $archive
Move-Item -LiteralPath 'run_complete_structure_batch.cmd' -Destination $archive
Move-Item -LiteralPath 'm_dot_ref_nominal_sensitivity.csv' -Destination $archive
Move-Item -LiteralPath '压缩机功耗拟合结果' -Destination $archive
Move-Item -LiteralPath 'nul' -Destination $archive
```

For outputs, use an explicit confirmed list rather than a broad wildcard. A safe pattern is:

```powershell
$outArchive = Join-Path $archive 'outputs'
New-Item -ItemType Directory -Path $outArchive
Move-Item -LiteralPath 'outputs\NAME_CONFIRMED_FOR_ARCHIVE' -Destination $outArchive
```

