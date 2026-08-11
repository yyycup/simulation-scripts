# Physics-P Experiments

`run_p_mpc_operational.py` 是唯一正式 Physics-P 入口。此目录只保存辨识、校准、评估、绘图和调参实验。

所有命令必须从项目根目录执行。旧命令已退役，不提供根目录 wrapper。

## Identification

| 旧命令/文件 | 新模块 |
| --- | --- |
| `generate_predictor_identification_data.py` | `python -m experiments.physics_p.identification.generate_predictor_identification_data` |
| `fit_mpc_physics_predictor.py` | `python -m experiments.physics_p.identification.fit_mpc_physics_predictor` |
| `fit_mpc_lpv_predictor.py` | `python -m experiments.physics_p.identification.fit_mpc_lpv_predictor` |
| `evaluate_p_identification_rollouts.py` | `python -m experiments.physics_p.identification.evaluate_p_identification_rollouts` |
| `predictor_identification_data.py` | 库模块 `experiments.physics_p.identification.predictor_identification_data` |
| `mpc_lpv_predictor.py` | 库模块 `experiments.physics_p.identification.mpc_lpv_predictor` |
| `evaluate_mpc_predictors.py` | 库模块 `experiments.physics_p.identification.evaluate_mpc_predictors` |

## Calibration

| 旧命令 | 新命令 |
| --- | --- |
| `build_p_compressor_displacement_variant.py` | `python -m experiments.physics_p.calibration.build_p_compressor_displacement_variant` |
| `build_p_heat_generation_corrected_artifact.py` | `python -m experiments.physics_p.calibration.build_p_heat_generation_corrected_artifact` |
| `calibrate_p_actual_replay.py` | `python -m experiments.physics_p.calibration.calibrate_p_actual_replay` |
| `refine_p_thermal_capacities.py` | `python -m experiments.physics_p.calibration.refine_p_thermal_capacities` |

## Evaluation

| 旧命令/文件 | 新命令 |
| --- | --- |
| `evaluate_dual_p_shadow.py` | `python -m experiments.physics_p.evaluation.evaluate_dual_p_shadow` |
| `evaluate_p_frozen_mpc_plan.py` | `python -m experiments.physics_p.evaluation.evaluate_p_frozen_mpc_plan` |
| `evaluate_p_shadow_actual_replay.py` | `python -m experiments.physics_p.evaluation.evaluate_p_shadow_actual_replay` |
| `run_p_model_boundary_validation.py` | `python -m experiments.physics_p.evaluation.run_p_model_boundary_validation` |
| `compare_mpc_predictor_formal_results.py` | `python -m experiments.physics_p.evaluation.compare_mpc_predictor_formal_results` |
| `plot_p_peak_displacement_comparison.py` | `python -m experiments.physics_p.evaluation.plot_p_peak_displacement_comparison` |
| `figures/gen_fig_p_mpc_local_formal.py` | `python -m experiments.physics_p.evaluation.plot_p_mpc_local_formal_results` |

## Tuning and formal experiments

| 旧命令 | 新命令 |
| --- | --- |
| `run_p_mpc_short_comparison.py` | `python -m experiments.physics_p.tuning.run_p_mpc_short_comparison` |
| `run_p_mpc_controller_tuning.py` | `python -m experiments.physics_p.tuning.run_p_mpc_controller_tuning` |
| `run_p_mpc_local_formal.py` | `python -m experiments.physics_p.tuning.run_p_mpc_local_formal` |

每个 CLI 都可追加 `--help` 查看原有参数。历史 `docs/superpowers/` 记录保留迁移前命令，不代表当前支持接口。
