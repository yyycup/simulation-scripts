# Single-Pack BTMS

## 在 PyCharm 中从这里开始

当前主控制器是 **Physics-P do-mpc NMPC**：完整 Plant 每 5 s 推进一步，
NMPC 每 15 s 重求解一次。预测模型为 14 状态离散 Physics-P，优化器为
do-mpc / CasADi / IPOPT；它不依赖 GEKKO、Fixed-QP 或 TD3。

直接运行以下文件：

- `runners/run_dompc_peak.py`：调峰，默认 1280 个 5 s 步；
- `runners/run_dompc_frequency.py`：调频，默认 720 个 5 s 步。

使用解释器：

```text
C:\Users\24776\miniforge3\envs\btms\python.exe
```

默认情况下，调峰 `Np=Nc=60`，调频 `Np=Nc=45`。不要传
`--control-horizon`，即可保持控制时域等于预测时域。

## 代码导航

| 想修改的内容 | 首先打开 |
| --- | --- |
| 完整物理 Plant / 电池 | `plant/pack.py` |
| 冷板、冷却液与制冷循环 | `plant/thermal_loop.py`、`plant/thermal_system.py` |
| 非线性预测模型 Physics-P | `predictor/physics_p.py`、`predictor/physics_p_casadi.py` |
| NMPC 权重、时域、转速和 DMAX | `controllers/dompc/config.py` |
| NMPC 方程与 IPOPT 目标函数 | `controllers/dompc/nmpc.py` |
| Plant 状态、未来电流与 Qgen 接入 | `controllers/dompc/adapter.py` |
| 每 5 s 闭环推进 | `simulation/case.py` |

`controllers/gekko/` 与 `controllers/fixed_qp/` 是保留的历史基线；当前
do-mpc NMPC 不会 import 它们。历史报告在 `docs/history/`，运行结果在
`outputs/`，两者都不属于运行时代码。
