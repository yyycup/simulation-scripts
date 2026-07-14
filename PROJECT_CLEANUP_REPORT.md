# Project Cleanup Report

Generated: 2026-06-29 10:46:32

## Actual Moved Files

Archive destination: $archiveRoot

- run_basic_mpc_direct_benchmark.py -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104632\root_files\run_basic_mpc_direct_benchmark.py
- run_mpc_objective_ablation.py -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104632\root_files\run_mpc_objective_ablation.py
- run_mpc_objective_ablation_pipe_delay.py -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104632\root_files\run_mpc_objective_ablation_pipe_delay.py
- run_mpc_scene_param_2x2.py -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104632\root_files\run_mpc_scene_param_2x2.py
- run_mpc_preview_reserve_comparison.py -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104632\root_files\run_mpc_preview_reserve_comparison.py
- run_pipe_delay_comparison.py -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104632\root_files\run_pipe_delay_comparison.py
- run_mpc_weight_tuning.py -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104632\root_files\run_mpc_weight_tuning.py
- run_same_limits_12_cases.py -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104632\root_files\run_same_limits_12_cases.py
- identify_open_loop_lag.py -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104632\root_files\identify_open_loop_lag.py
- run_complete_structure_batch.cmd -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104632\root_files\run_complete_structure_batch.cmd
- m_dot_ref_nominal_sensitivity.csv -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104632\root_files\m_dot_ref_nominal_sensitivity.csv
- 压缩机功耗拟合结果 -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104632\root_files\压缩机功耗拟合结果

## REVIEW Items Not Moved

- run_evaporator_candidate_validation_mpc_retry.py
- run_evaporator_parameter_sensitivity.py
- test_exergy_metrics.py
- run_control_effectiveness_diagnostics.py
- run_heat_exchange_mechanism_diagnostics.py
- run_pump_mechanism_diagnostics.py
- run_tank_capacity_diagnostics.py
- run_tank_capacity_diagnostics_continue.py
- run_tank_initial_temperature_diagnostics.py
- run_pump_weight_sensitivity.py
- run_supervised_buffer_sensitivity.py
- 切换式单一判据.py
- 监督式综合判据.py
- AGC

## Core File Presence

- thermal_case_simulator.py : present
- thermal_control_strategies.py : present
- mpc_flow_direction_strategies.py : present
- thermal_loop.py : present
- thermal_system.py : present
- thermal_batch_config.py : present
- pack.py : present
- age_model.py : present
- mpc_reduced_model_calibration.py : present
- pid_param_search.py : present

## Business Code Modified

No business code logic was edited. This cleanup only moved selected low-risk root files into _archive/cleanup_20260629_104632/root_files/ and generated this report.

## Failure Or Skipped Items

- 
ul was not moved. Windows treated 
ul as a reserved device name; normal LiteralPath checks did not resolve it as a movable file.

## Explicitly Untouched

- outputs/
- data/
- REVIEW items
- .git/, .codex/, .agents/
- simulink-agentic-toolkit*/
- GEKKO temporary directories
- __pycache__/ was not moved or cleaned; the requested py_compile command may refresh Python bytecode cache as a normal side effect

## Minimal Syntax Check

Command:

```powershell
python -m py_compile thermal_case_simulator.py thermal_control_strategies.py mpc_flow_direction_strategies.py thermal_loop.py thermal_system.py thermal_batch_config.py pack.py age_model.py mpc_reduced_model_calibration.py pid_param_search.py
```

Result: passed with exit code 0 and no output.



## Outputs Cleanup 2026-06-29 10:49:16

Archive destination: $archive

Moved outputs directories: 103

- outputs/basic_mpc_direct_benchmark -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\basic_mpc_direct_benchmark
- outputs/basic_mpc_min_benchmark -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\basic_mpc_min_benchmark
- outputs/btms_batch_script_check -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\btms_batch_script_check
- outputs/btms_miniforge_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\btms_miniforge_smoke
- outputs/btms_miniforge_smoke_bidir -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\btms_miniforge_smoke_bidir
- outputs/cache_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\cache_smoke
- outputs/cache_smoke_conservative -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\cache_smoke_conservative
- outputs/cache_smoke_fast -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\cache_smoke_fast
- outputs/diagnose_v4_20_direct -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\diagnose_v4_20_direct
- outputs/diagnose_v4_20_escalated_workspace -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\diagnose_v4_20_escalated_workspace
- outputs/diagnose_v4_20_from_home -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\diagnose_v4_20_from_home
- outputs/diagnose_v4_200_direct -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\diagnose_v4_200_direct
- outputs/diagnose_v4_user_error -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\diagnose_v4_user_error
- outputs/dynamics_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\dynamics_smoke
- outputs/exergy_supervised_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\exergy_supervised_smoke
- outputs/header_smoke_no_j -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\header_smoke_no_j
- outputs/header_smoke_same_limits -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\header_smoke_same_limits
- outputs/mpc_clean_objective_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_clean_objective_smoke
- outputs/mpc_clean_rate_limit_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_clean_rate_limit_smoke
- outputs/mpc_cv_normalized_100step_check -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_cv_normalized_100step_check
- outputs/mpc_cv5e8_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_cv5e8_smoke
- outputs/mpc_cv5e8_smoke_conda -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_cv5e8_smoke_conda
- outputs/mpc_delay_compare_old_freq -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_delay_compare_old_freq
- outputs/mpc_dynamic_preview_freq_120step -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_dynamic_preview_freq_120step
- outputs/mpc_dynamic_preview_q90_freq_120step -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_dynamic_preview_q90_freq_120step
- outputs/mpc_dynamic_preview_q90_min2000_freq_120step -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_dynamic_preview_q90_min2000_freq_120step
- outputs/mpc_dynamic_preview_q90_min2000_r100p5000_freq_120step -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_dynamic_preview_q90_min2000_r100p5000_freq_120step
- outputs/mpc_init_1500_3000_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_init_1500_3000_smoke
- outputs/mpc_lagged_predictive_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_lagged_predictive_smoke
- outputs/mpc_objective_ablation -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_objective_ablation
- outputs/mpc_param_cross_no_precool_preview200 -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_param_cross_no_precool_preview200
- outputs/mpc_param_cross_no_precool_preview50 -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_param_cross_no_precool_preview50
- outputs/mpc_param_cross_no_precool_preview50_ascii -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_param_cross_no_precool_preview50_ascii
- outputs/mpc_param_cross_no_precool_preview50_real_source -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_param_cross_no_precool_preview50_real_source
- outputs/mpc_param_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_param_smoke
- outputs/mpc_param_smoke_direct -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_param_smoke_direct
- outputs/mpc_pipe_delay_line_smoke_10step -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_pipe_delay_line_smoke_10step
- outputs/mpc_pipe_delay_smoke_10step -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_pipe_delay_smoke_10step
- outputs/mpc_preview_reserve_step_heat_80step -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_preview_reserve_step_heat_80step
- outputs/mpc_rate_limit_native_gekko_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_rate_limit_native_gekko_smoke
- outputs/mpc_rate_limit_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_rate_limit_smoke
- outputs/mpc_reg_tracking_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_reg_tracking_smoke
- outputs/mpc_reg_tracking_smoke_cvband -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_reg_tracking_smoke_cvband
- outputs/mpc_reg_tracking_smoke_cvmeas -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_reg_tracking_smoke_cvmeas
- outputs/mpc_reg_tracking_smoke_fixedsp_2 -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_reg_tracking_smoke_fixedsp_2
- outputs/mpc_reg_tracking_smoke_fixedsp_lowenergy -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_reg_tracking_smoke_fixedsp_lowenergy
- outputs/mpc_reg_tracking_smoke_kq050 -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_reg_tracking_smoke_kq050
- outputs/mpc_reg_tracking_smoke_qgen150 -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_reg_tracking_smoke_qgen150
- outputs/mpc_source_fallback_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_source_fallback_smoke
- outputs/mpc_supervised_jrev_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_supervised_jrev_smoke
- outputs/mpc_supervised_jrev_smoke2 -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_supervised_jrev_smoke2
- outputs/mpc_supervised_script_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_supervised_script_smoke
- outputs/mpc_switching_script_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_switching_script_smoke
- outputs/mpc_switching_single_v2_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_switching_single_v2_smoke
- outputs/mpc_switching_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_switching_smoke
- outputs/mpc_v4_allcases_200step_check -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_v4_allcases_200step_check
- outputs/mpc_v4_cv5e7_300step_check -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_v4_cv5e7_300step_check
- outputs/mpc_v4_cv5e7_original_equiv_200step_check -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_v4_cv5e7_original_equiv_200step_check
- outputs/mpc_v4_cv5e7_rc1000_rp10000_200step_check -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_v4_cv5e7_rc1000_rp10000_200step_check
- outputs/mpc_v4_cv5e7_rpump002_200step_check -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_v4_cv5e7_rpump002_200step_check
- outputs/mpc_v4_cv5e7_rpump10000_200step_check -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_v4_cv5e7_rpump10000_200step_check
- outputs/mpc_v4_cv5e7_rpump30000_freq_200step_check -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_v4_cv5e7_rpump30000_freq_200step_check
- outputs/mpc_v4_cv5e7_rpump30000_peak_200step_check -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_v4_cv5e7_rpump30000_peak_200step_check
- outputs/mpc_weight_tuning_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_weight_tuning_smoke
- outputs/mpc_weight_tuning_v2_100step_check -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_weight_tuning_v2_100step_check
- outputs/mpc_weight_tuning_v3_100step_check -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_weight_tuning_v3_100step_check
- outputs/mpc_weight_tuning_v4_100step_check -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_weight_tuning_v4_100step_check
- outputs/mpc_weight_tuning_v5_dynamic_target_100step_check -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\mpc_weight_tuning_v5_dynamic_target_100step_check
- outputs/off_active_min_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\off_active_min_smoke
- outputs/open_loop_lag_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\open_loop_lag_smoke
- outputs/parallel_runner_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\parallel_runner_smoke
- outputs/pid_param_search_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\pid_param_search_smoke
- outputs/pid_reg_bidir_35c_vd5_evap070_fanlow_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\pid_reg_bidir_35c_vd5_evap070_fanlow_smoke
- outputs/pid_reg_bidir_35c_vd5_evap070_fanlowest_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\pid_reg_bidir_35c_vd5_evap070_fanlowest_smoke
- outputs/pid_reg_bidir_35c_vd5_evap070_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\pid_reg_bidir_35c_vd5_evap070_smoke
- outputs/pid_reg_bidir_35c_vd5_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\pid_reg_bidir_35c_vd5_smoke
- outputs/pid_reg_bidir_35c_vd8_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\pid_reg_bidir_35c_vd8_smoke
- outputs/pid_reg_bidir_vd8_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\pid_reg_bidir_vd8_smoke
- outputs/prediction_diagnosis_foptd_observed_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\prediction_diagnosis_foptd_observed_smoke
- outputs/prediction_diagnosis_foptd_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\prediction_diagnosis_foptd_smoke
- outputs/prediction_diagnosis_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\prediction_diagnosis_smoke
- outputs/pycache_check -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\pycache_check
- outputs/rated_7_55_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\rated_7_55_smoke
- outputs/sat_table_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\sat_table_smoke
- outputs/smoke_run -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\smoke_run
- outputs/supervised_actual_gate_peak_bidir_smoke70 -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\supervised_actual_gate_peak_bidir_smoke70
- outputs/supervised_actual_gate_reg_bidir_smoke190 -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\supervised_actual_gate_reg_bidir_smoke190
- outputs/supervised_benefit_gate_peak_bidir_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\supervised_benefit_gate_peak_bidir_smoke
- outputs/supervised_benefit_gate_peak_bidir_smoke60 -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\supervised_benefit_gate_peak_bidir_smoke60
- outputs/supervised_benefit_gate_peak_bidir_smoke70 -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\supervised_benefit_gate_peak_bidir_smoke70
- outputs/supervised_benefit_gate_reg_bidir_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\supervised_benefit_gate_reg_bidir_smoke
- outputs/supervised_benefit_gate_reg_bidir_smoke60 -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\supervised_benefit_gate_reg_bidir_smoke60
- outputs/supervised_flow_supervisor_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\supervised_flow_supervisor_smoke
- outputs/supervised_integrated_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\supervised_integrated_smoke
- outputs/supervised_onoff_pid_bidir_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\supervised_onoff_pid_bidir_smoke
- outputs/supervised_runner_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\supervised_runner_smoke
- outputs/tmp_normalized_cost_check -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\tmp_normalized_cost_check
- outputs/v2_allcases_simple_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\v2_allcases_simple_smoke
- outputs/v4_direct_python_bootstrap_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\v4_direct_python_bootstrap_smoke
- outputs/v4_sankey_rates_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\v4_sankey_rates_smoke
- outputs/v4_sankey_rates_smoke_conda -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\v4_sankey_rates_smoke_conda
- outputs/vd15_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\vd15_smoke
- outputs/vd30_smoke -> C:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制\_archive\cleanup_20260629_104916\outputs\vd30_smoke

Kept required output directories:

- outputs/mpc_cv_wsphi_x2_x5_x10_60steps : present
- outputs/mpc_cv_wsphi_freq_x1_60steps : present
- outputs/mpc_cv_wsphi_sensitivity_60steps : present
- outputs/unified_initial_state_mpc_60steps : present
- outputs/current_evap_default_mpc_60steps : present
- outputs/evaporator_config_mpc_validation_60steps : present
- outputs/evaporator_candidate_validation_60steps : present
- outputs/evaporator_parameter_sensitivity : present
- outputs/pid_fixed_param_validation : present
- outputs/pid_pso_freq : present
- outputs/mpc_freq_1000floor_directdisp130_h45_temptrack400_full : present
- outputs/mpc_reduced_model_calibration : present
- outputs/supervised_buffer_sensitivity : present
- outputs/strategy_24case_figures : present
- outputs/latest_run_analysis : present

Review/not-touched output patterns:

- outputs/pid_pso_peak
- outputs/pid_two_stage_peak
- outputs/pid_two_stage_freq
- outputs/mpc_param_cross_no_precool_full
- outputs/mpc_freq_*_full
- outputs/*diagnostics*
- outputs/*sensitivity*
- outputs/empty btms_gekko_*

Failures: none.

No simulations, PSO jobs, git staging, or commits were run.


## 2026-07-14 Incremental Root Script Cleanup

- Scope: root scripts, compatibility wrappers, probe artifacts, and generated cache candidates only.
- Archive: `_archive/cleanup_20260714_094507/`
- Manifest: `PROJECT_CLEANUP_MANIFEST_20260714.md`
- Strategy: archive human-authored files; do not delete source code.
- Excluded: `outputs/`, `输出结果/`, `data/`, existing archives, IDE/agent directories, and external toolkits.
- Baseline correction: updated `test_final_controller_configured_pid.py` from stale frequency PID expectation `0.35` to the final configured value `2.0`.
- Baseline verification: 15 `unittest` tests passed after the correction.

### Script counts

| Category | Before | Archived | Remaining |
|---|---:|---:|---:|
| `run_*.py` | 41 | 29 | 12 |
| `plot_*.py` | 16 | 14 | 2 |
| `test_*.py` | 57 | 27 | 30 |
| Other compatibility/probe files | 9 | 6 | 3 |
| **Total** | **123** | **76** | **47** |

### Move ledger

| Original path | Classification | New path | Evidence |
|---|---|---|---|
| `AGC` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/compat/AGC` | Unreferenced incomplete code fragment without a file extension |
| `_patch_probe.txt` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/compat/_patch_probe.txt` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `merge_cv_band_scan_results.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/compat/merge_cv_band_scan_results.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `plot_delay_comparison_compact.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/plots/plot_delay_comparison_compact.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `plot_delay_step_response.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/plots/plot_delay_step_response.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `plot_delay_step_response_simple.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/plots/plot_delay_step_response_simple.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `plot_freq_pid_mpc_simple.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/plots/plot_freq_pid_mpc_simple.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `plot_mpc_calibration_dual_axis.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/plots/plot_mpc_calibration_dual_axis.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `plot_mpc_calibration_dual_axis_closed.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/plots/plot_mpc_calibration_dual_axis_closed.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `plot_mpc_calibration_dual_axis_fullband.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/plots/plot_mpc_calibration_dual_axis_fullband.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `plot_mpc_calibration_pareto.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/plots/plot_mpc_calibration_pareto.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `plot_mpc_calibration_scans.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/plots/plot_mpc_calibration_scans.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `plot_mpc_calibration_slide.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/plots/plot_mpc_calibration_slide.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `plot_mpc_weight_refinement_combined.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/plots/plot_mpc_weight_refinement_combined.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `plot_mpc_weight_scans.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/plots/plot_mpc_weight_scans.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `plot_peak_flow_existing_results.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/plots/plot_peak_flow_existing_results.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `plot_peak_pid_flow_chinese.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/plots/plot_peak_pid_flow_chinese.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_control_effectiveness_diagnostics.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_control_effectiveness_diagnostics.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_current_evap_default_mpc_60.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_current_evap_default_mpc_60.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_delay_comparison.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_delay_comparison.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_evaporator_candidate_validation.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_evaporator_candidate_validation.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_evaporator_candidate_validation_mpc_retry.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_evaporator_candidate_validation_mpc_retry.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_evaporator_config_mpc_short_validation.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_evaporator_config_mpc_short_validation.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_evaporator_parameter_sensitivity.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_evaporator_parameter_sensitivity.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_freq_pso_monitor_pycharm.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_freq_pso_monitor_pycharm.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_heat_exchange_mechanism_diagnostics.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_heat_exchange_mechanism_diagnostics.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_mpc_cv_wsphi_sensitivity_60.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_mpc_cv_wsphi_sensitivity_60.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_mpc_param_cross_full.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_mpc_param_cross_full.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_mpc_supervised.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_mpc_supervised.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_mpc_supervised_peak_tuned.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_mpc_supervised_peak_tuned.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_mpc_switching.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_mpc_switching.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_peak_flow_comparison.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_peak_flow_comparison.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_pid_fixed_param_validation.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_pid_fixed_param_validation.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_pid_pso_full_search_pycharm.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_pid_pso_full_search_pycharm.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_predictive_delta_t_buffer_sensitivity.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_predictive_delta_t_buffer_sensitivity.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_predictive_delta_t_buffer_sensitivity_compatible.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_predictive_delta_t_buffer_sensitivity_compatible.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_predictive_delta_t_buffer_sensitivity_user_temp.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_predictive_delta_t_buffer_sensitivity_user_temp.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_predictive_delta_t_buffer_sensitivity_windows.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_predictive_delta_t_buffer_sensitivity_windows.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_predictive_delta_t_threshold_sensitivity.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_predictive_delta_t_threshold_sensitivity.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_pump_mechanism_diagnostics.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_pump_mechanism_diagnostics.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_pump_weight_sensitivity.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_pump_weight_sensitivity.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_supervised_buffer_sensitivity.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_supervised_buffer_sensitivity.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_tank_capacity_diagnostics.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_tank_capacity_diagnostics.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_tank_capacity_diagnostics_continue.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_tank_capacity_diagnostics_continue.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_tank_initial_temperature_diagnostics.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_tank_initial_temperature_diagnostics.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_transport_delay_comparison.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/runs/run_transport_delay_comparison.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `run_unified_initial_state_mpc_60.py` | KEEP | Restored to project root after dependency check | Direct import from `run_mpc_sensitivity_60.py` |
| `run_local_freq_pso_scheduled.cmd` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/compat/run_local_freq_pso_scheduled.cmd` | Companion launcher for archived PID PSO runner |
| `test_delay_comparison.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_delay_comparison.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_delay_output_paths.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_delay_output_paths.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_merge_cv_band_scan_results.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_merge_cv_band_scan_results.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_mpc_cv_wsphi_sensitivity.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_mpc_cv_wsphi_sensitivity.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_mpc_w_energy_comp_sensitivity.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_mpc_w_energy_comp_sensitivity.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_mpc_weight_refinement_resumable.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_mpc_weight_refinement_resumable.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_mpc_weight_refinement_runner.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_mpc_weight_refinement_runner.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_peak_flow_comparison.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_peak_flow_comparison.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_pid_fixed_param_pycharm_defaults.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_pid_fixed_param_pycharm_defaults.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_pid_min_compressor_speed.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_pid_min_compressor_speed.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_pid_minimum_compressor_speed.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_pid_minimum_compressor_speed.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_pid_pso_search.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_pid_pso_search.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_plot_delay_comparison_compact.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_plot_delay_comparison_compact.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_plot_delay_step_response.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_plot_delay_step_response.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_plot_delay_step_response_simple.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_plot_delay_step_response_simple.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_plot_freq_pid_mpc_simple.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_plot_freq_pid_mpc_simple.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_plot_mpc_calibration_dual_axis.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_plot_mpc_calibration_dual_axis.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_plot_mpc_calibration_dual_axis_closed.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_plot_mpc_calibration_dual_axis_closed.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_plot_mpc_calibration_dual_axis_fullband.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_plot_mpc_calibration_dual_axis_fullband.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_plot_mpc_calibration_pareto.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_plot_mpc_calibration_pareto.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_plot_mpc_calibration_scans.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_plot_mpc_calibration_scans.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_plot_mpc_calibration_slide.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_plot_mpc_calibration_slide.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_plot_mpc_weight_combined_no_top_legend.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_plot_mpc_weight_combined_no_top_legend.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_plot_mpc_weight_refinement_combined.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_plot_mpc_weight_refinement_combined.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_plot_mpc_weight_scans.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_plot_mpc_weight_scans.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_predictive_delta_t_threshold_scan.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_predictive_delta_t_threshold_scan.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `test_transport_delay_comparison.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/tests/test_transport_delay_comparison.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `监督式综合判据.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/compat/监督式综合判据.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |
| `切换式单一判据.py` | ARCHIVE | `_archive/cleanup_20260714_094507/root_scripts/compat/切换式单一判据.py` | See per-file evidence in `PROJECT_CLEANUP_MANIFEST_20260714.md` |

### Generated directories

The generated directories were intentionally skipped because two BTMS Python test processes were still active:

- PID 48092, started 2026-07-11 20:19:37: `python -m unittest test_pid_local_refinement -v`
- PID 36696, started 2026-07-11 20:20:14: targeted `test_pid_local_refinement` tests

Skipped paths: root `__pycache__/`, `_gekko_tmp_*/`, and `tmp*/`. No process was terminated and no generated directory was recursively deleted.

### Git state

No files were staged, committed, pushed, or connected to a remote during this cleanup stage.

### Final verification

- Restored `run_unified_initial_state_mpc_60.py` after detecting its direct import from `run_mpc_sensitivity_60.py`.
- Archived the companion `run_local_freq_pso_scheduled.cmd` because it only launched the archived PID PSO runner.
- Archived the unreferenced incomplete `AGC` code fragment.
- Broken executable references to archived filenames: none.
- Root Python compilation: 61 files passed `py_compile`.
- Lightweight regression suite: 15 tests passed with `unittest`.
- Upload candidate size: approximately 0.62 MB; largest file approximately 136 KB.
- Credential-pattern scan: no matching files.
- GitHub target `https://github.com/yyycup/simulation-scripts.git`: reachable and empty before first push.
- `.gitignore` excludes result folders, data, archives, local agent state, caches, GEKKO temp folders, and external toolkits.
