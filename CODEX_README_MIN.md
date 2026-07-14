项目：储能电池热管理仿真/控制；电池热、冷板、制冷、PID/MPC、调峰/调频。
主流程：`thermal_case_simulator.py -> thermal_control_strategies.py -> mpc_flow_direction_strategies.py`；参数：`thermal_batch_config.py`。
当前：25 C 初始、MPC 60 步、WSPHI、蒸发器、PID/MPC、PSO-PID；读点名 `run_*.py`、`test_*.py`。
清理：旧根文件在 `_archive/cleanup_20260629_104632/root_files/`；旧输出在 `_archive/cleanup_20260629_104916/outputs/`；恢复时才读。
读取：`AGENTS.md` -> `PROJECT_MAP_MIN.md` -> 主流程/参数文件 -> 点名文件。
禁扫：`outputs/`、`_archive/`、`data/`、`.git/`、`__pycache__/`、`.pytest_cache/`、`.mypy_cache/`、GEKKO 目录、`simulink-agentic-toolkit*/`。
验证：`python -m py_compile thermal_case_simulator.py thermal_control_strategies.py mpc_flow_direction_strategies.py thermal_loop.py thermal_system.py thermal_batch_config.py pack.py age_model.py mpc_reduced_model_calibration.py pid_param_search.py`
