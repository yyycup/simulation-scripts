# 储能电池热管理仿真与控制

本项目用于储能电池热管理系统仿真与控制研究，包含电池热模型、冷板与冷却液动态、制冷循环、On-Off/PID/MPC控制、candidate B蒸发器预测模型、终端代价和预测式流向反转。

## 环境安装

```powershell
conda env create -f environment.yml
conda activate btms
```

## 核心结构

- `thermal_case_simulator.py`：完整工况仿真入口。
- `thermal_batch_config.py`：调峰、调频和控制参数。
- `thermal_control_strategies.py`：控制器创建入口。
- `mpc_flow_direction_strategies.py`：MPC与流向反转主体。
- `thermal_loop.py`：冷却回路、延迟和动态。
- `thermal_system.py`：泵、换热器和制冷循环。
- `model_data/`：运行所需的小型模型标定数据。
- `run_p_mpc_operational.py`：唯一正式 Physics-P 闭环入口。
- `p_mpc_run_support.py`：P 运行器共享场景、路径和汇总支持。
- `mpc_physics_predictor.py`、`mpc_physics_shadow.py`、`mpc_predictor_selection.py`：稳定 P 预测与选择契约。
- `experiments/physics_p/`：P 辨识、校准、评估和调参。
- `tests/physics_p/`：P 专属回归。

## Physics-P 显式运行入口

Candidate B 仍是仓库默认 MPC 预测器。Physics-P 只通过下面的唯一正式入口显式启用：

```powershell
python run_p_mpc_operational.py --help
```

正式参数资产位于 `model_data/physics_p_operational_v1.json`。辨识、校准、评估和调参脚本位于 `experiments/physics_p/`，必须从项目根目录用 `python -m ...` 调用；旧根目录脚本命令已退役。

Physics-P 专属回归：

```powershell
python -m unittest discover -s tests/physics_p -t . -p "test_*.py"
```

## 最终控制器比较

```powershell
python run_final_controller_comparison.py
```

服务器并行入口：

```powershell
python run_final_controller_parallel.py
```

## MPC参数扫描

```powershell
python run_mpc_sensitivity_60.py --help
```

## candidate B诊断

```powershell
python run_evaporator_mpc_capacity_diagnostic.py --help
```

## 最终绘图

```powershell
python plot_final_mpc_report.py
```

## 验证

```powershell
python -m unittest -q test_btms_runtime test_mpc_evaporator_capacity_model test_final_controller_comparison test_final_controller_configured_pid test_plot_final_mpc_report
```

## 注意事项

- Windows入口通过 `btms_runtime.py` 补充Conda环境的 `Library/bin`，以便加载本地DLL。
- `outputs/`、`输出结果/` 和 `data/` 是本地数据/结果目录，不上传Git。
- 长时间GEKKO扫描和完整对比建议在服务器运行。
- 正式论文/PPT结果仍以已确认的原12组完整仿真为准。
