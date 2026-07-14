# Project Map Min

## 1. 一句话
储能电池热管理仿真/控制：电池热、冷板冷却液、制冷循环、PID/MPC、调峰/调频。历史清理批次已于2026-07-14在完成GitHub可运行基线后永久删除；当前运行代码、模型数据和测试均由Git管理。

## 2. 新对话默认读取
只读核心文件：`AGENTS.md`、`PROJECT_MAP_MIN.md`、`thermal_case_simulator.py`、`thermal_batch_config.py`、`thermal_control_strategies.py`、`mpc_flow_direction_strategies.py`、`thermal_loop.py`、`thermal_system.py`、`pid_param_search.py`；模型参数按需读取 `model_data/`。

## 3. 主入口
`thermal_case_simulator.py -> thermal_control_strategies.py -> mpc_flow_direction_strategies.py`；参数：`thermal_batch_config.py`。

## 4. 当前相关
最终控制器与论文结果复现优先读取：`run_final_controller_comparison.py`、`run_final_controller_parallel.py`、`run_mpc_sensitivity_60.py`、`plot_final_mpc_report.py`。candidate B蒸发器继续读取 `mpc_evaporator_capacity_model.py` 和对应标定/诊断入口。

## 5. 不建议读取
默认不扫描：`outputs/`、`输出结果/`、`data/`、`.git/`、`__pycache__/`、`tmp*/`、`_gekko_tmp_*/`、`.codex/`、`.agents/`、`simulink-agentic-toolkit*/`。`_archive/` 已永久删除，不再作为代码或历史入口。

## 6. 新对话 Prompt
请先读 `AGENTS.md` 和 `PROJECT_MAP_MIN.md`。本项目是储能热管理 PID/MPC 仿真；只从核心链路入手，不扫描 outputs/data/archive。
