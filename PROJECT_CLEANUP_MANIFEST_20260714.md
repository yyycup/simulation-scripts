# 2026-07-14 根目录脚本清理清单

## Classification Rules

- **KEEP**：核心流程、最终12组复现、candidate B、终端代价、最终汇报绘图或保护这些路径的测试。
- **ARCHIVE**：已完成的一次性扫描、诊断或绘图脚本族，当前没有外部活跃引用并已有保留入口替代。
- **REVIEW**：来源或替代关系不明确；本轮留在原处。

## Candidates

| Path | Class | Evidence | Action |
|---|---|---|---|
| `AGC` | ARCHIVE | 无扩展名的不完整旧代码片段，当前代码无引用 | 移动到时间戳归档 |
| `_patch_probe.txt` | ARCHIVE | 完整实验脚本族已结束且当前无活跃引用 | 移动到时间戳归档 |
| `merge_cv_band_scan_results.py` | ARCHIVE | 旧权重扫描/合并入口，通用run_mpc_sensitivity_60.py与最终2x3图已保留 | 移动到时间戳归档 |
| `mpc_sensitivity_compat.py` | KEEP | 当前通用MPC扫描入口或其参数回归 | 原位保留 |
| `plot_delay_comparison_compact.py` | ARCHIVE | 延迟对比阶段已结束，整族归档 | 移动到时间戳归档 |
| `plot_delay_step_response.py` | ARCHIVE | 延迟对比阶段已结束，整族归档 | 移动到时间戳归档 |
| `plot_delay_step_response_simple.py` | ARCHIVE | 延迟对比阶段已结束，整族归档 | 移动到时间戳归档 |
| `plot_final_mpc_report.py` | KEEP | 最终12组控制器比较或正式汇报路径 | 原位保留 |
| `plot_freq_pid_mpc_simple.py` | ARCHIVE | PID搜索或固定参数验证阶段已结束，最终参数及局部复现入口已保留 | 移动到时间戳归档 |
| `plot_mpc_calibration_dual_axis.py` | ARCHIVE | 标定绘图阶段产物，candidate B正式模型已保留 | 移动到时间戳归档 |
| `plot_mpc_calibration_dual_axis_closed.py` | ARCHIVE | 标定绘图阶段产物，candidate B正式模型已保留 | 移动到时间戳归档 |
| `plot_mpc_calibration_dual_axis_fullband.py` | ARCHIVE | 标定绘图阶段产物，candidate B正式模型已保留 | 移动到时间戳归档 |
| `plot_mpc_calibration_pareto.py` | ARCHIVE | 标定绘图阶段产物，candidate B正式模型已保留 | 移动到时间戳归档 |
| `plot_mpc_calibration_scans.py` | ARCHIVE | 标定绘图阶段产物，candidate B正式模型已保留 | 移动到时间戳归档 |
| `plot_mpc_calibration_slide.py` | ARCHIVE | 标定绘图阶段产物，candidate B正式模型已保留 | 移动到时间戳归档 |
| `plot_mpc_weight_combined_2x3.py` | KEEP | 最终权重粗细扫描2x3绘图 | 原位保留 |
| `plot_mpc_weight_refinement_combined.py` | ARCHIVE | 旧权重扫描/合并入口，通用run_mpc_sensitivity_60.py与最终2x3图已保留 | 移动到时间戳归档 |
| `plot_mpc_weight_scans.py` | ARCHIVE | 旧权重扫描/合并入口，通用run_mpc_sensitivity_60.py与最终2x3图已保留 | 移动到时间戳归档 |
| `plot_peak_flow_existing_results.py` | ARCHIVE | 早期流向对比绘图，已由最终12组报告替代 | 移动到时间戳归档 |
| `plot_peak_pid_flow_chinese.py` | ARCHIVE | 早期流向对比绘图，已由最终12组报告替代 | 移动到时间戳归档 |
| `predictive_delta_t_flow_controller.py` | KEEP | 当前预测式流向反转实现或回归 | 原位保留 |
| `predictive_delta_t_flow_mpc.py` | KEEP | 当前预测式流向反转实现或回归 | 原位保留 |
| `run_control_effectiveness_diagnostics.py` | ARCHIVE | 历史诊断或旧控制入口，当前主流程已替代 | 移动到时间戳归档 |
| `run_current_evap_default_mpc_60.py` | ARCHIVE | candidate B接入前后的阶段性验证，正式诊断/标定入口已保留 | 移动到时间戳归档 |
| `run_delay_comparison.py` | ARCHIVE | 延迟对比阶段已结束，整族归档 | 移动到时间戳归档 |
| `run_evaporator_candidate_validation.py` | ARCHIVE | candidate B接入前后的阶段性验证，正式诊断/标定入口已保留 | 移动到时间戳归档 |
| `run_evaporator_candidate_validation_mpc_retry.py` | ARCHIVE | candidate B接入前后的阶段性验证，正式诊断/标定入口已保留 | 移动到时间戳归档 |
| `run_evaporator_config_mpc_short_validation.py` | ARCHIVE | candidate B接入前后的阶段性验证，正式诊断/标定入口已保留 | 移动到时间戳归档 |
| `run_evaporator_mpc_capacity_diagnostic.py` | KEEP | candidate B蒸发器模型、标定或边界回归 | 原位保留 |
| `run_evaporator_parameter_sensitivity.py` | ARCHIVE | candidate B接入前后的阶段性验证，正式诊断/标定入口已保留 | 移动到时间戳归档 |
| `run_final_controller_comparison.py` | KEEP | 最终12组控制器比较或正式汇报路径 | 原位保留 |
| `run_final_controller_parallel.py` | KEEP | 最终12组控制器比较或正式汇报路径 | 原位保留 |
| `run_freq_pso_monitor_pycharm.py` | ARCHIVE | PID搜索或固定参数验证阶段已结束，最终参数及局部复现入口已保留 | 移动到时间戳归档 |
| `run_heat_exchange_mechanism_diagnostics.py` | ARCHIVE | 历史诊断或旧控制入口，当前主流程已替代 | 移动到时间戳归档 |
| `run_mpc_cv_wsphi_sensitivity_60.py` | ARCHIVE | 旧权重扫描/合并入口，通用run_mpc_sensitivity_60.py与最终2x3图已保留 | 移动到时间戳归档 |
| `run_mpc_evaporator_capacity_calibration.py` | KEEP | candidate B蒸发器模型、标定或边界回归 | 原位保留 |
| `run_mpc_param_cross_full.py` | ARCHIVE | 完整实验脚本族已结束且当前无活跃引用 | 移动到时间戳归档 |
| `run_mpc_sensitivity_60.py` | KEEP | 当前通用MPC扫描入口或其参数回归 | 原位保留 |
| `run_mpc_sensitivity_compatible_entry.py` | KEEP | 当前通用MPC扫描入口或其参数回归 | 原位保留 |
| `run_mpc_supervised.py` | ARCHIVE | 历史诊断或旧控制入口，当前主流程已替代 | 移动到时间戳归档 |
| `run_mpc_supervised_peak_tuned.py` | ARCHIVE | 历史诊断或旧控制入口，当前主流程已替代 | 移动到时间戳归档 |
| `run_mpc_switching.py` | ARCHIVE | 历史诊断或旧控制入口，当前主流程已替代 | 移动到时间戳归档 |
| `run_mpc_weight_refinement.py` | KEEP | 仍被当前核心链路使用或缺少明确替代 | 原位保留 |
| `run_mpc_weight_refinement_compatible.py` | KEEP | 仍被当前核心链路使用或缺少明确替代 | 原位保留 |
| `run_mpc_weight_refinement_resumable.py` | KEEP | 仍被当前核心链路使用或缺少明确替代 | 原位保留 |
| `run_mpc_weight_refinement_resumable_windows.py` | KEEP | 仍被当前核心链路使用或缺少明确替代 | 原位保留 |
| `run_peak_flow_comparison.py` | ARCHIVE | 早期流向对比绘图，已由最终12组报告替代 | 移动到时间戳归档 |
| `run_pid_fixed_param_validation.py` | ARCHIVE | PID搜索或固定参数验证阶段已结束，最终参数及局部复现入口已保留 | 移动到时间戳归档 |
| `run_pid_local_refinement.py` | KEEP | 最终PID参数来源和复现入口 | 原位保留 |
| `run_pid_pso_full_search_pycharm.py` | ARCHIVE | PID搜索或固定参数验证阶段已结束，最终参数及局部复现入口已保留 | 移动到时间戳归档 |
| `run_predictive_delta_t_buffer_sensitivity.py` | ARCHIVE | 反转阈值/窗口扫描已结束，控制器实现与策略回归保留 | 移动到时间戳归档 |
| `run_predictive_delta_t_buffer_sensitivity_compatible.py` | ARCHIVE | 反转阈值/窗口扫描已结束，控制器实现与策略回归保留 | 移动到时间戳归档 |
| `run_predictive_delta_t_buffer_sensitivity_user_temp.py` | ARCHIVE | 反转阈值/窗口扫描已结束，控制器实现与策略回归保留 | 移动到时间戳归档 |
| `run_predictive_delta_t_buffer_sensitivity_windows.py` | ARCHIVE | 反转阈值/窗口扫描已结束，控制器实现与策略回归保留 | 移动到时间戳归档 |
| `run_predictive_delta_t_threshold_sensitivity.py` | ARCHIVE | 反转阈值/窗口扫描已结束，控制器实现与策略回归保留 | 移动到时间戳归档 |
| `run_pump_mechanism_diagnostics.py` | ARCHIVE | 历史诊断或旧控制入口，当前主流程已替代 | 移动到时间戳归档 |
| `run_pump_weight_sensitivity.py` | ARCHIVE | 历史诊断或旧控制入口，当前主流程已替代 | 移动到时间戳归档 |
| `run_supervised_buffer_sensitivity.py` | ARCHIVE | 历史诊断或旧控制入口，当前主流程已替代 | 移动到时间戳归档 |
| `run_tank_capacity_diagnostics.py` | ARCHIVE | 历史诊断或旧控制入口，当前主流程已替代 | 移动到时间戳归档 |
| `run_tank_capacity_diagnostics_continue.py` | ARCHIVE | 历史诊断或旧控制入口，当前主流程已替代 | 移动到时间戳归档 |
| `run_tank_initial_temperature_diagnostics.py` | ARCHIVE | 历史诊断或旧控制入口，当前主流程已替代 | 移动到时间戳归档 |
| `run_transport_delay_comparison.py` | ARCHIVE | 延迟对比阶段已结束，整族归档 | 移动到时间戳归档 |
| `run_unified_initial_state_mpc_60.py` | KEEP | `run_mpc_sensitivity_60.py` 直接导入的统一初始状态基线 | 原位保留 |
| `run_local_freq_pso_scheduled.cmd` | ARCHIVE | 仅启动已归档的旧 `run_pid_pso_full_search_pycharm.py` | 移动到时间戳归档 |
| `test_btms_runtime.py` | KEEP | Windows BTMS运行时修复回归 | 原位保留 |
| `test_delay_comparison.py` | ARCHIVE | 延迟对比阶段已结束，整族归档 | 移动到时间戳归档 |
| `test_delay_output_paths.py` | ARCHIVE | 延迟对比阶段已结束，整族归档 | 移动到时间戳归档 |
| `test_evaporator_mpc_capacity_diagnostic.py` | KEEP | candidate B蒸发器模型、标定或边界回归 | 原位保留 |
| `test_final_comparison_flow_modes.py` | KEEP | 最终12组控制器比较或正式汇报路径 | 原位保留 |
| `test_final_controller_comparison.py` | KEEP | 最终12组控制器比较或正式汇报路径 | 原位保留 |
| `test_final_controller_configured_pid.py` | KEEP | 最终12组控制器比较或正式汇报路径 | 原位保留 |
| `test_final_controller_parallel.py` | KEEP | 最终12组控制器比较或正式汇报路径 | 原位保留 |
| `test_final_controller_parallel_pure.py` | KEEP | 最终12组控制器比较或正式汇报路径 | 原位保留 |
| `test_final_controller_parallel_system_tmp.py` | KEEP | 最终12组控制器比较或正式汇报路径 | 原位保留 |
| `test_flow_supervisor.py` | KEEP | 当前预测式流向反转实现或回归 | 原位保留 |
| `test_merge_cv_band_scan_results.py` | ARCHIVE | 旧权重扫描/合并入口，通用run_mpc_sensitivity_60.py与最终2x3图已保留 | 移动到时间戳归档 |
| `test_mpc_comp_dmax_terminal_cost.py` | KEEP | 当前通用MPC扫描入口或其参数回归 | 原位保留 |
| `test_mpc_cv_band_sequential_baseline.py` | KEEP | 当前通用MPC扫描入口或其参数回归 | 原位保留 |
| `test_mpc_cv_weight_sequential_baseline.py` | KEEP | 当前通用MPC扫描入口或其参数回归 | 原位保留 |
| `test_mpc_cv_wsphi_sensitivity.py` | ARCHIVE | 旧权重扫描/合并入口，通用run_mpc_sensitivity_60.py与最终2x3图已保留 | 移动到时间戳归档 |
| `test_mpc_evaporator_capacity_model.py` | KEEP | candidate B蒸发器模型、标定或边界回归 | 原位保留 |
| `test_mpc_evaporator_capacity_table.py` | KEEP | candidate B蒸发器模型、标定或边界回归 | 原位保留 |
| `test_mpc_params.py` | KEEP | 当前通用MPC扫描入口或其参数回归 | 原位保留 |
| `test_mpc_pump_dmax_sequential_baseline.py` | KEEP | 当前通用MPC扫描入口或其参数回归 | 原位保留 |
| `test_mpc_pump_weight_sequential_baseline.py` | KEEP | 当前通用MPC扫描入口或其参数回归 | 原位保留 |
| `test_mpc_reduced_model_calibration.py` | KEEP | 仍被当前核心链路使用或缺少明确替代 | 原位保留 |
| `test_mpc_sensitivity_60_rate_limits.py` | KEEP | 当前通用MPC扫描入口或其参数回归 | 原位保留 |
| `test_mpc_sensitivity_compat.py` | KEEP | 当前通用MPC扫描入口或其参数回归 | 原位保留 |
| `test_mpc_sensitivity_done_prefix.py` | KEEP | 当前通用MPC扫描入口或其参数回归 | 原位保留 |
| `test_mpc_sensitivity_final_weight_params.py` | KEEP | 当前通用MPC扫描入口或其参数回归 | 原位保留 |
| `test_mpc_w_energy_comp_sensitivity.py` | ARCHIVE | 完整实验脚本族已结束且当前无活跃引用 | 移动到时间戳归档 |
| `test_mpc_w_energy_comp_sequential_baseline.py` | KEEP | 当前通用MPC扫描入口或其参数回归 | 原位保留 |
| `test_mpc_weight_refinement_resumable.py` | ARCHIVE | 旧权重扫描/合并入口，通用run_mpc_sensitivity_60.py与最终2x3图已保留 | 移动到时间戳归档 |
| `test_mpc_weight_refinement_runner.py` | ARCHIVE | 旧权重扫描/合并入口，通用run_mpc_sensitivity_60.py与最终2x3图已保留 | 移动到时间戳归档 |
| `test_peak_flow_comparison.py` | ARCHIVE | 早期流向对比绘图，已由最终12组报告替代 | 移动到时间戳归档 |
| `test_pid_fixed_param_pycharm_defaults.py` | ARCHIVE | PID搜索或固定参数验证阶段已结束，最终参数及局部复现入口已保留 | 移动到时间戳归档 |
| `test_pid_local_refinement.py` | KEEP | 最终PID参数来源和复现入口 | 原位保留 |
| `test_pid_min_compressor_speed.py` | ARCHIVE | PID搜索或固定参数验证阶段已结束，最终参数及局部复现入口已保留 | 移动到时间戳归档 |
| `test_pid_minimum_compressor_speed.py` | ARCHIVE | PID搜索或固定参数验证阶段已结束，最终参数及局部复现入口已保留 | 移动到时间戳归档 |
| `test_pid_pso_search.py` | ARCHIVE | PID搜索或固定参数验证阶段已结束，最终参数及局部复现入口已保留 | 移动到时间戳归档 |
| `test_plot_delay_comparison_compact.py` | ARCHIVE | 延迟对比阶段已结束，整族归档 | 移动到时间戳归档 |
| `test_plot_delay_step_response.py` | ARCHIVE | 延迟对比阶段已结束，整族归档 | 移动到时间戳归档 |
| `test_plot_delay_step_response_simple.py` | ARCHIVE | 延迟对比阶段已结束，整族归档 | 移动到时间戳归档 |
| `test_plot_final_mpc_report.py` | KEEP | 最终12组控制器比较或正式汇报路径 | 原位保留 |
| `test_plot_freq_pid_mpc_simple.py` | ARCHIVE | PID搜索或固定参数验证阶段已结束，最终参数及局部复现入口已保留 | 移动到时间戳归档 |
| `test_plot_mpc_calibration_dual_axis.py` | ARCHIVE | 标定绘图阶段产物，candidate B正式模型已保留 | 移动到时间戳归档 |
| `test_plot_mpc_calibration_dual_axis_closed.py` | ARCHIVE | 标定绘图阶段产物，candidate B正式模型已保留 | 移动到时间戳归档 |
| `test_plot_mpc_calibration_dual_axis_fullband.py` | ARCHIVE | 标定绘图阶段产物，candidate B正式模型已保留 | 移动到时间戳归档 |
| `test_plot_mpc_calibration_pareto.py` | ARCHIVE | 标定绘图阶段产物，candidate B正式模型已保留 | 移动到时间戳归档 |
| `test_plot_mpc_calibration_scans.py` | ARCHIVE | 标定绘图阶段产物，candidate B正式模型已保留 | 移动到时间戳归档 |
| `test_plot_mpc_calibration_slide.py` | ARCHIVE | 标定绘图阶段产物，candidate B正式模型已保留 | 移动到时间戳归档 |
| `test_plot_mpc_weight_combined_2x3.py` | KEEP | 最终权重粗细扫描2x3绘图 | 原位保留 |
| `test_plot_mpc_weight_combined_no_top_legend.py` | ARCHIVE | 完整实验脚本族已结束且当前无活跃引用 | 移动到时间戳归档 |
| `test_plot_mpc_weight_refinement_combined.py` | ARCHIVE | 旧权重扫描/合并入口，通用run_mpc_sensitivity_60.py与最终2x3图已保留 | 移动到时间戳归档 |
| `test_plot_mpc_weight_scans.py` | ARCHIVE | 旧权重扫描/合并入口，通用run_mpc_sensitivity_60.py与最终2x3图已保留 | 移动到时间戳归档 |
| `test_predictive_delta_t_latch.py` | KEEP | 当前预测式流向反转实现或回归 | 原位保留 |
| `test_predictive_delta_t_policy.py` | KEEP | 当前预测式流向反转实现或回归 | 原位保留 |
| `test_predictive_delta_t_threshold_scan.py` | ARCHIVE | 反转阈值/窗口扫描已结束，控制器实现与策略回归保留 | 移动到时间戳归档 |
| `test_refrigeration_cycle_limits.py` | KEEP | candidate B蒸发器模型、标定或边界回归 | 原位保留 |
| `test_selected_predictive_delta_t_threshold.py` | KEEP | 当前预测式流向反转实现或回归 | 原位保留 |
| `test_transport_delay_comparison.py` | ARCHIVE | 延迟对比阶段已结束，整族归档 | 移动到时间戳归档 |
| `监督式综合判据.py` | ARCHIVE | 早期中文控制入口，当前统一控制策略已替代 | 移动到时间戳归档 |
| `切换式单一判据.py` | ARCHIVE | 早期中文控制入口，当前统一控制策略已替代 | 移动到时间戳归档 |

## Generated Directories

| Pattern | Process check | Resolved-path check | Action |
|---|---|---|---|
| `__pycache__/` | 清理前检查Python/GEKKO进程 | 必须位于项目根目录内 | 无占用时删除 |
| `_gekko_tmp_*/` | 清理前检查Python/GEKKO进程 | 必须是项目根目录直接子目录 | 无占用时删除 |
| `tmp*/` | 清理前检查Python/GEKKO进程 | 必须是项目根目录直接子目录 | 无占用时删除 |

## Fixed Boundaries

- 不处理 `outputs/`、`输出结果/`、`data/` 和已有 `_archive/` 内容。
- 不处理外部工具、IDE配置或代理配置目录。
- `nul` 是Windows保留设备名，本轮记录为跳过，不按普通文件移动。
- 人工编写的代码只归档，不直接删除。
