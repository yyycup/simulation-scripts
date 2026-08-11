# Project Map Min

## 1. 一句话
储能电池热管理仿真/控制：电池热、冷板冷却液、制冷循环、PID/MPC、调峰/调频。Candidate B 仍为默认 MPC 预测器；Physics-P 仅由专用入口显式启用。

## 2. 新对话默认读取
只读核心文件：`AGENTS.md`、`PROJECT_MAP_MIN.md`、`thermal_case_simulator.py`、`thermal_batch_config.py`、`thermal_control_strategies.py`、`mpc_flow_direction_strategies.py`、`thermal_loop.py`、`thermal_system.py`、`pid_param_search.py`；模型参数按需读取 `model_data/`。

## 3. 默认主入口
`thermal_case_simulator.py -> thermal_control_strategies.py -> mpc_flow_direction_strategies.py`；参数：`thermal_batch_config.py`。这条默认链继续选择 Candidate B。

## 4. Physics-P
唯一正式入口：`run_p_mpc_operational.py`。稳定模块：`p_mpc_run_support.py`、`mpc_physics_predictor.py`、`mpc_physics_shadow.py`、`mpc_predictor_selection.py`。正式资产：`model_data/physics_p_operational_v1.json`。实验：`experiments/physics_p/`；专属回归：`tests/physics_p/`。

## 5. 当前其他入口
最终控制器与论文结果复现优先读取：`run_final_controller_comparison.py`、`run_final_controller_parallel.py`、`run_mpc_sensitivity_60.py`、`plot_final_mpc_report.py`。Candidate B 蒸发器继续读取 `mpc_evaporator_capacity_model.py` 和对应标定/诊断入口。

## 6. 不建议读取
默认不扫描：`outputs/`、`输出结果/`、`data/`、`.git/`、`__pycache__/`、`tmp*/`、`_gekko_tmp_*/`、`.codex/`、`.agents/`、`simulink-agentic-toolkit*/`。历史 `_archive/` 已永久删除。

## 7. 新对话 Prompt
请先读 `AGENTS.md` 和 `PROJECT_MAP_MIN.md`。本项目是储能热管理 PID/MPC 仿真；只从核心链路或点名的 Physics-P 入口入手，不扫描 outputs/data。
