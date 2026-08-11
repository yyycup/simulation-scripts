项目：储能电池热管理仿真/控制；电池热、冷板、制冷、PID/MPC、调峰/调频。
默认主流程：`thermal_case_simulator.py -> thermal_control_strategies.py -> mpc_flow_direction_strategies.py`；默认预测器仍是 Candidate B。
Physics-P：唯一正式入口 `run_p_mpc_operational.py`；调峰 horizon 14、调频 horizon 12；入口显式使用 25 °C，普通 `simulate_case` 默认使用环境温度 35 °C。
P 稳定模块：`p_mpc_run_support.py`、`mpc_physics_predictor.py`、`mpc_physics_shadow.py`、`mpc_predictor_selection.py`。
P 实验：`experiments/physics_p/`，从项目根目录使用 `python -m experiments.physics_p...`；不再支持旧根命令。
P 测试：`python -m unittest discover -s tests/physics_p -t . -p "test_*.py"`。
P 正式资产：`model_data/physics_p_operational_v1.json`。
读取：`AGENTS.md` -> `PROJECT_MAP_MIN.md` -> 主流程/参数文件 -> 点名文件。
禁扫：`outputs/`、`data/`、`.git/`、`__pycache__/`、`.pytest_cache/`、`.mypy_cache/`、GEKKO 目录、`simulink-agentic-toolkit*/`。
