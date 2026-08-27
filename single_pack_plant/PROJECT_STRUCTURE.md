# 工程结构

```text
single_pack_plant/
├─ plant/                 # 完整物理 Plant：电池、热回路、制冷循环
├─ predictor/             # Physics-P 及其 CasADi / 线性模型辅助实现
├─ controllers/
│  ├─ dompc/              # 当前主控制器：Physics-P do-mpc NMPC
│  ├─ gekko/              # 独立历史 GEKKO 基线
│  ├─ fixed_qp/           # 独立历史 State-Space QP 基线
│  ├─ factory.py          # 控制器创建入口
│  ├─ flow_logic.py       # 流向切换判据
│  └─ flow_supervisor.py  # GEKKO 流向切换控制器
├─ simulation/            # `simulate_case()` 与工况/公共配置
├─ runners/               # PyCharm 可直接运行的脚本
├─ analysis/              # 离线分析和绘图
├─ tests/                 # 仅保留核心回归测试
├─ docs/history/          # 历史报告和已移出的本地虚拟环境
├─ model_data/            # HPPC 和 Physics-P artifact
└─ outputs/               # 仿真输出，不纳入运行时代码
```

## 当前唯一推荐运行路径

```text
runners/run_dompc_peak.py
    -> simulation/case.py: simulate_case()
    -> controllers/factory.py: create_controller()
    -> controllers/dompc/adapter.py: command()
    -> controllers/dompc/nmpc.py: IPOPT solve
    -> plant/thermal_loop.py: Plant 推进 5 s
```

## 历史模块边界

- `controllers/dompc/` 是当前非线性 NMPC，实现与 GEKKO / Fixed-QP 独立。
- `controllers/gekko/` 只供旧 GEKKO 策略使用；其蒸发器容量辅助模型位于
  `controllers/gekko/evaporator_capacity.py`。
- `controllers/fixed_qp/` 仅保留作旧 QP 基线，不是当前运行入口。
- `pid_param_search.py` 是服务器 PID-PSO 的兼容 API，保留在根目录，避免
  破坏已有服务器搜索脚本。
