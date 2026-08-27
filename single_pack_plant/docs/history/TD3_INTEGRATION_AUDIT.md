# TD3 接入审计：单 Pack 层级控制

审计日期：2026-08-23  
审计范围：`single_pack_plant` 当前生产代码、当前 Physics-P 运行入口、模型数据文件，以及已有闭环结果中的运行耗时记录。  
本轮动作：只读审计；未实现 TD3，未修改 Plant、MPC、参数、测试或运行任务。

## 0. 先给结论

当前正式 Physics-P 闭环的层次是：

```text
完整物理 Plant
  = 52 个 PyBaMM 2RC-Thevenin 电芯
  + 13 节点冷板
  + 冷却液供回水延迟与 3 L 水箱
  + CoolProp R134a 制冷循环
  + 压缩机/水泵/风扇动态

MPC Predictor
  = Physics-P 降阶预测模型（不是 Plant）
```

当前完整 12 工况的 MPC 路径明确选择 `Physics-P + SinglePredictiveDeltaTMPC`。因此，第一版 TD3 应只在每次 MPC 求解前给现有控制器提供三个权重倍率，不应替换 Plant，也不应绕过 MPC 直接控制压缩机或水泵。

当前尚不存在通用 episode `reset()`、Gym 环境或在线权重 setter。最关键的接入障碍是：压缩机和水泵能耗权重在 GEKKO 模型构建时以 Python 浮点常量写入目标函数，创建控制器后仅修改 `MPCParams` 不会改变已经编译的优化问题。

## 1. 审计边界与当前真实入口

### 1.1 生产 Python 文件

当前目录除测试和生成输出外共有 17 个 Python 文件。与 TD3 接入直接相关的是：

- `run_full12_physics_p.py`：PyCharm 启动器，转交给已经准备好的 12 工况运行脚本。
- `thermal_case_simulator.py`：单工况闭环调度、状态记录和 CSV 输出。
- `thermal_control_strategies.py`：控制器工厂。
- `mpc_flow_direction_strategies.py`：连续 MPC 核心、目标函数、约束、求解与恢复。
- `predictive_delta_t_flow_controller.py`：当前双向工况使用的单一预测温差流向逻辑。
- `predictive_delta_t_flow_mpc.py`：预测温差触发门。
- `mpc_physics_predictor.py`：独立 Physics-P 状态和单步预测方程。
- `mpc_physics_shadow.py`：只读影子预测诊断。
- `mpc_predictor_selection.py`：Predictor 名称和 artifact 加载边界。
- `mpc_evaporator_capacity_model.py`：Candidate-B 容量模型；正式 Physics-P 路径不以它作为预测容量模型。
- `pack.py`：52 电芯电热 Plant。
- `thermal_loop.py`：冷板、水箱、执行器动态及制冷系统耦合。
- `thermal_system.py`：R134a 制冷循环、压缩机、水泵、风扇和换热器。
- `thermal_batch_config.py`：仿真、设备、MPC 和工况默认参数。
- `age_model.py`：只读式老化观测器，不参与 MPC 求解。
- `btms_runtime.py`、`__init__.py`：运行环境和包导出。
- `pid_param_search.py`：PID 参数搜索，与 TD3-MPC 在线控制无直接依赖。

### 1.2 当前 Physics-P 12 工况调用入口

当前根启动器 `run_full12_physics_p.py:20-22` 使用：

- 运行脚本：`outputs/full12_physicsP_20260823_114038/run_full12_physics_p.py`
- P artifact：`model_data/physics_p_operational_v1.json`

该运行脚本在 MPC 工况中传入：

- `mpc_flow_mode="single_predictive_delta_t"`
- `mpc_predictor="physics_p"`
- `mpc_predictor_artifact=<physics_p_operational_v1.json>`

因此本报告把这条路径视为“当前真实使用的 MPC”。`simulate_case()` 自身的默认 Predictor 仍是 Candidate-B；只有显式传入上述参数才会使用 P。

## 2. 当前 Plant 架构

### 2.1 Plant 不是单个类，而是两个连续 step 的组合

| 层 | 文件与位置 | 类/函数 | 主要输入 | 主要输出/更新 |
|---|---|---|---|---|
| 电池包 | `pack.py:15` | `BatteryPack` | 4×13 配置、HPPC 参数 | 52 个 PyBaMM Thevenin 仿真器；`temps`、`socs`、支路电流历史 |
| 电池单步 | `pack.py:176` | `BatteryPack.step(dt, T_plate, T_cabinet)` | `dt [s]`、13 个冷板节点温度 `[K]`、柜内温度 `[K]`；总电流从 `pack.total_current` 读取 | 原位更新 52 个电芯温度、SOC 和支路电流；无显式返回值 |
| 外部热回路 | `thermal_loop.py:366` | `simulate_thermal_loop_step(...)` | 当前水箱/冷板状态、压缩机/水泵指令、环境温度、流向、`dt`、执行器动态状态 | 返回下一时刻水箱/冷板/供回水状态、实际转速、制冷量、功率、循环热力学量 |
| 制冷循环 | `thermal_system.py:985` | `run_refrigeration_cycle(...)` | 实际压缩机/风扇转速、冷却液入口温度和质量流量、环境温度 | R134a 循环的 `Q_evap`、`Q_cond`、`W_comp`、COP、压力、焓、火用指标等 |
| 水泵 | `thermal_system.py:429` | `pump_model(N_pump_rpm)` | 水泵实际转速 `[rpm]` | 冷却液质量流量 `[kg/s]`、水泵功率 `[W]` |
| 完整闭环推进 | `thermal_case_simulator.py:559-576` | `simulate_case()` 循环体 | 控制器命令和当前全部状态 | 先推进外部热回路，再用新冷板温度推进电池包 |

时间层关系为：

```text
状态 x(k)
  -> MPC 计算 u_cmd(k)
  -> 热回路用 u_cmd(k) 得到热回路状态 x_loop(k+1)
  -> BatteryPack 用 T_plate(k+1) 得到电池状态 x_bat(k+1)
  -> 记录 k+1 状态
```

这个先后顺序是当前 Plant 的一部分，TD3 接入不得改变。

### 2.2 初始化和 reset

当前没有统一的 `Plant.reset()`，也没有 `BatteryPack.reset()`。

每次 `simulate_case()` 都通过重新构造对象完成 episode 初始化：

1. `thermal_loop.py:109` 的 `build_pack_config()` 生成电池配置；
2. `thermal_case_simulator.py:372-377` 新建 `BatteryPack`；
3. `thermal_case_simulator.py:384-397` 新建控制器；
4. `thermal_case_simulator.py:415-419` 初始化水箱和 13 个冷板节点；
5. `thermal_case_simulator.py:424-432` 初始化压缩机、水泵、风扇、制冷量和供回水延迟状态；
6. `thermal_case_simulator.py:378` 新建老化观测器。

后续 TD3 环境的 `reset()` 必须封装以上重建过程，不能只清空一个温度数组。52 个 PyBaMM `Simulation` 内部也带有时间推进状态，不能复用旧 episode 的对象。

### 2.3 单步时间、Plant 输入和命令传递

- 默认仿真/控制步长：`SIM_DT = 5.0 s`，定义于 `thermal_batch_config.py:32`。
- MPC 每个仿真步都重新求解，即下层控制周期也是 5 s。
- 调频 Physics-P 的 `MV_STEP_HOR=3` 只阻塞预测时域内的未来动作变化；外层仍每 5 s 重求解一次。

MPC 命令到 Plant 的实际传递链：

```text
SinglePredictiveDeltaTMPC.command()
  -> 返回 n_comp_cmd, n_pump_cmd [rpm]
  -> simulate_thermal_loop_step(N_comp_cmd, N_pump_cmd)
  -> first_order_lag 得到 N_comp_eff, N_pump_eff
  -> pump_model(N_pump_eff)
  -> run_refrigeration_cycle(N_comp_eff, N_fan_eff, ...)
  -> 冷板/供回水/水箱更新
  -> BatteryPack.step(dt, T_plate_next, T_cabinet)
```

Plant 每步外部输入包括：

- 电池包总电流 `pack.total_current [A]`；
- 压缩机转速指令 `N_comp_cmd [rpm]`；
- 水泵转速指令 `N_pump_cmd [rpm]`；
- 室外和柜内环境温度 `[K]`；
- 正向/反向流标志；
- 时间步长 `dt [s]`。

### 2.4 Plant 可取得状态和输出

| 需要的量 | 当前是否直接可取 | 当前来源 |
|---|---|---|
| 电池平均温度 | 是 | `pack.get_avg_temp()`；记录为 `Average temperature` |
| 电池最高温度 | 是 | `np.max(pack.temps) - 273.15`；记录为 `T_cell_max_C` |
| 电池最低温度 | 是 | `np.min(pack.temps) - 273.15` |
| 电池最大温差 | 是 | `max(pack.temps)-min(pack.temps)`；记录为 `Delta_T_cell_C` |
| 52 个电芯温度 | 是 | `pack.temps [K]` |
| 52 个电芯 SOC | 是 | `pack.socs` |
| 4 个并联支路电流 | 是 | `pack.branch_currents_history` 的本步末值 |
| 冷却液箱温度 | 是 | 循环变量 `t_tank_k` / `thermal_step["T_tank_K"]` |
| 供水温度 | 是 | `dynamic_state["T_pipe_supply_K"]` / `thermal_step["T_pipe_supply_K"]` |
| 回水温度 | 是 | `dynamic_state["T_pipe_return_K"]` / `thermal_step["T_pipe_return_K"]` |
| 蒸发器出口温度 | 是 | `thermal_step["T_evap_out_K"]` |
| 冷板入口/出口温度 | 是 | `T_plate_in_K`、`T_plate_out_K` |
| 13 个冷板节点温度 | 是 | `t_plate_k_array` / `T_plate_K_array` |
| 压缩机实际转速 | 是 | `dynamic_state["N_comp_eff"]` / `thermal_step["N_comp_eff"]` |
| 水泵实际转速 | 是 | `dynamic_state["N_pump_eff"]` / `thermal_step["N_pump_eff"]` |
| 风扇实际转速 | 是 | `dynamic_state["N_fan_eff"]` |
| 系统负载电流 | 是 | `pack.total_current`；记录为 `Total current` |
| 环境温度 | 循环内可取，但当前未写 CSV | `thermal_case_simulator.py:469-470` 的 `t_outdoor/t_cabinet` |
| 压缩机功率 | 是 | `thermal_step["W_comp_real"] [W]`；CSV 为 `Compressor power (kW)` |
| 水泵功率 | 是 | `thermal_step["W_pump_val"] [W]`；CSV 为 `Pump power (kW)` |
| 风扇功率 | 是 | `thermal_step["W_fan_real"] [W]` |
| 蒸发/冷凝热流 | 是 | `Q_evap_eff`、`Q_cond_eff` 和 `Q_dot_*` |
| COP、压力、焓、火用 | 是 | `thermal_step` 返回字典；`thermal_case_simulator.py:622-660` 记录 |
| SOH | 是，观测量 | `AgingModel280Ah.step()`；不反馈到 Plant 方程 |

`BatteryPack.step()` 本身没有汇总输出；完整 Plant 输出由 `thermal_step` 字典、`pack` 对象和循环局部变量共同组成。TD3 环境应建立一个只读的结构化 observation 快照，避免上层算法直接依赖散落的局部变量。

## 3. 当前 MPC 架构

### 3.1 实际类和单步求解接口

当前正式双向 Physics-P MPC 的类链为：

```text
thermal_control_strategies.create_controller()
  -> mpc_flow_direction_strategies.create_mpc_flow_controller()
  -> predictive_delta_t_flow_controller.SinglePredictiveDeltaTMPC
       -> BaseFlowMPCController
          -> forward_model: MPCControllerDual
          -> reverse_model: MPCControllerDual
```

主要接口：

- 上层控制接口：`SinglePredictiveDeltaTMPC.command()`，位于 `predictive_delta_t_flow_controller.py:70`。
- 实际连续优化接口：`MPCControllerDual.solve_step()`，位于 `mpc_flow_direction_strategies.py:1932`。

`solve_step()` 输入为：

- 当前步号 `i`；
- 电池平均温度测量 `t_batt_meas [K]`；
- 水箱温度测量 `t_cool_meas [K]`；
- 整段电池发热预测 `qgen_forecast [W]`；
- 环境温度 `T_amburrent [K]`；
- 上一步压缩机/水泵命令；
- 当前目标温度；
- 热回路观测状态：实际压缩机/水泵转速、`Q_evap_eff`、`Q_cond_eff`、供回水温度；
- 13 个冷板节点温度测量。

输出字典至少含：

- `n_comp`、`n_pump`：本步命令；
- 完整预测时域的电池/冷却液温度；
- 1、5、10、20 步和终端温度；
- 压缩机/水泵命令计划；
- 求解耗时、成功标志、错误和恢复信息；
- 目标函数诊断；
- P 输入域诊断和一阶温度预测 innovation。

### 3.2 时域、控制范围、DMAX 和求解器

| 参数 | 调峰 | 调频 | 定义位置 |
|---|---:|---:|---|
| 控制周期 | 5 s | 5 s | `thermal_batch_config.py:32` |
| 预测步数 `N` | 60 | 45 | `mpc_flow_direction_strategies.py:358,386` |
| 预测时长 | 300 s | 225 s | `N × dt` |
| 压缩机命令范围（正式 P） | 300–6000 rpm | 300–6000 rpm | `MPCControllerDual.__init__()`；300 rpm 表示 off |
| P 容量模型有效主动区 | 1000–6000 rpm | 1000–6000 rpm | P artifact；300–1000 使用启动桥接语义 |
| 水泵范围 | 1600–4800 rpm | 1600–4800 rpm | `thermal_batch_config.py:46-47` |
| 压缩机 DMAX | 6000 rpm/周期 | 6000 rpm/周期 | 最终场景参数；实际基本不限制 |
| 水泵 DMAX | 300 rpm/周期 | 600 rpm/周期 | 最终场景参数 |
| MV `DCOST` | 0.001 / 0.001 | 0.001 / 0.001 | 压缩机/水泵 |
| P 时域动作阻塞 | 1 步 | 3 步 | 调频未来动作每 15 s 一个块，但仍每 5 s 重求解 |

求解器配置位于 `mpc_flow_direction_strategies.py:1339-1372`：

- GEKKO `IMODE=6`；
- `NODES=2`；
- `SOLVER=3`，即 IPOPT；
- 调峰 P 的 `RTOL=OTOL=3e-6`；
- 调频 P 使用 `MV_STEP_HOR=3`。

### 3.3 当前真实目标函数

当前优化问题同时包含 GEKKO CV dead-band 目标和显式 `m.Minimize()` 目标。写成与代码一致的形式：

```text
J = J_CV_deadband
  + sum(k=0..N) [
        w_temp_obj * ((T_batt_corr(k)-T_ref)/1 C)^2
      + w_comp * P_comp_pred(k)/P_comp,max
      + w_pump * P_pump_pred(k)/P_pump,max
      + terminal_mask(k) * w_terminal
          * ((T_batt_corr(k)-T_ref)/1 C)^2
      + w_dcomp_quadratic * J_dcomp_quadratic(k)
    ]
  + J_MV_DCOST
```

其中：

- `J_CV_deadband` 由 `cv_temp` 的 `SPLO/SPHI/WSPLO/WSPHI` 生成；目标温度之外但位于 soft band 内时不使用同等超限惩罚。
- `J_MV_DCOST` 是 GEKKO 对两个 MV 的控制变化惩罚，`DCOST=0.001`。
- `T_batt_corr` 是 P 原始预测温度加可选的温度偏差路径；当前 `physics_p_temp_bias_gain=0`，所以不启用补偿。
- `P_comp_pred` 和 `P_pump_pred` 是 MPC 内部功率近似，不是 Plant 当步实际功率。

当前活动权重如下：

| 项 | 调峰 | 调频 | 当前是否活动 |
|---|---:|---:|---|
| CV 高温 `WSPHI` | 5e6 | 5e7 | 是 |
| CV 低温 `WSPLO` | 5e6 | 5e7 | 是 |
| soft band 半宽 | 0.30 °C | 0.45 °C | 是；即 24.70–25.30 °C / 24.55–25.45 °C |
| `w_temp_obj` | 0 | 0 | 否；不能把它误认为当前主温度权重 |
| `w_energy_comp` | 600 | 300 | 是 |
| `w_energy_pump` | 10000 | 15000 | 是 |
| `w_terminal_temp` | 1e6 | 5e5 | 是，仅终端点 |
| `w_dcomp`, `w_dpump` | 0.001, 0.001 | 0.001, 0.001 | 是，以 GEKKO `DCOST` 进入 |
| `w_dcomp_quadratic` | 0 | 0 | 否 |
| `w_delta_t` | 1.0 | 1.1 | 不进入连续 MPC 的 `j_total`；用于其他流向指标/诊断 |
| `w_bat_safety` | 0 | 0 | 否 |
| 冷量储备权重 | 常量存在 | 常量存在 | `MPC_COOLANT_RESERVE_ENABLED=False`，且当前 `j_total` 未加入这些项 |
| `w_term_peak` | 0 | 0 | 否；旧字段 |

注意：`j_spread` 和 `j_switch` 在当前连续优化模型中都被设为 0。当前温差对流向的影响发生在 `SinglePredictiveDeltaTMPC` 的外层预测温差门，不是压缩机/水泵连续目标中的温差项。

### 3.4 目标温度和软约束

- 标称目标温度：25.0 °C。
- 虽然代码保留热负荷预览和动态目标逻辑，但当前调峰/调频参数均为 `dynamic_target_min=dynamic_target_max=25.0`、`precool_max=warm_relief_max=0`，所以实际目标固定为 25.0 °C。
- `MPC_T_BAT_MAX_C=26.0`、margin 0.2 °C 对应的安全松弛变量已构造，但当前 `w_bat_safety=0`，不构成活动惩罚，也不是硬约束。
- P 冷却液预测必须留在 artifact 的 15–35 °C 验证域；超域会被判为预测域失败并触发恢复/回退。这是运行域检查，不是温度安全 soft band。

### 3.5 infeasible/recovery 机制

当前已有多层恢复机制，位于 `MPCControllerDual._solve_with_fixed_control_recovery()` 和 `solve_step()`：

1. 正常 IPOPT 求解；
2. 相同模型立即重试；
3. 切换 APOPT (`SOLVER=1`) 重试；
4. 从当前 Plant 观测状态重新播种 P 轨迹；
5. 固定当前控制量求可行轨迹后再恢复优化；
6. P 求解器实例耗尽恢复后可从当前观测状态重建 GEKKO 模型；
7. 最终失败返回安全回退命令和 `solved=False`，并记录完整错误/恢复原因。

正式 P 回退通常保持并裁剪上一步命令；若冷却液预计低于域下界、实测到达域下界或电池不高于目标，则压缩机回到 300 rpm off、水泵回到 1600 rpm。

## 4. Plant 与 MPC Predictor 的关系

### 4.1 Physics-P 的状态

独立可执行的 P 定义于 `mpc_physics_predictor.py`：

- `PhysicsPredictorState`：`mpc_physics_predictor.py:123`
- `initialize_physics_state()`：`mpc_physics_predictor.py:670`
- `step_physics_predictor()`：`mpc_physics_predictor.py:732`

它有 9 个标量动态状态和 4 个延迟历史缓冲区：

1. 压缩机实际转速；
2. 水泵实际转速；
3. 冷凝侧有效制冷量状态；
4. 蒸发侧有效制冷量状态；
5. 供水温度；
6. 等效冷板温度；
7. 回水温度；
8. 电池平均温度；
9. 水箱/等效冷却液温度；
10. 压缩机输入延迟历史；
11. 水泵流量延迟历史；
12. 供水延迟历史；
13. 回水延迟历史。

因此若按 dataclass 字段计数是 13；若按连续标量状态计数是 9，延迟缓冲长度由 `delay/dt` 决定。

MPC 内部不是直接循环调用这个 Python `step()`，而是在 `MPCControllerDual.__init__()` 中用同一 artifact 参数构建等价的 GEKKO 方程和延迟状态。

### 4.2 当前 P artifact 的主要参数

`model_data/physics_p_operational_v1.json` 标记为 `fit_status="validated"`。关键内容：

- 输入域：压缩机 1000–6000 rpm、水泵 1600–4800 rpm、冷却液 15–35 °C、环境 20–40 °C；
- 容量模型：`single_cubic_v1`，20 个系数，硬启动门 1000 rpm，容量上界 4800 W；
- 动态：`tau_comp=3.860668 s`、`tau_pump=4.238800 s`、泵流量延迟 5 s、`tau_cond=28.107799 s`、`tau_evap=45 s`、供水延迟 15 s、回水延迟 20 s；
- 热参数：`cp=3391 J/(kg·K)`、参考流量 `0.389906 kg/s`、电池等效热容 `218265.028 J/K`、冷却液热容 `10617.810 J/K`、电池-冷板导热 `521.860 W/K`、冷板时间常数 `1.509632 s`、环境导热 `7.723472 W/K`、冷板-流体有效度 `0.294197`、电池发热缩放 `0.74`。

artifact 没有填写 `calibration_source` 或 `validation_source` 顶层字段；其 `fit` 中保存了详细验证指标。不能仅从当前 JSON 断言每一个 P 参数直接来自某篇文献，它们表现为对当前 Plant 数据拟合/验证后的降阶参数。

### 4.3 与 Plant 相同和简化的部分

| 内容 | Plant | Physics-P |
|---|---|---|
| 电池电模型 | 52 个 2RC Thevenin、SOC/温度相关 HPPC 参数 | 不保留 2RC 电压和 52 个 SOC；用电流平方近似发热预览 |
| 电池热状态 | 52 个电芯温度 + 空间导热 | 1 个平均电池温度 |
| 冷板 | 13 个热节点、正反向流体温升 | 1 个等效冷板温度；空间温差另由经验网格外推用于流向门 |
| 制冷循环 | CoolProp R134a、压缩机效率图、蒸发器/冷凝器求解 | 拟合的制冷量和功率代数式 + 一阶滞后 |
| 水箱 | 3 L 物理水箱能量平衡 | 1 个等效冷却液热容状态 |
| 动态 | 执行器、制冷量、供回水延迟 | 同类结构，但时间常数是 artifact 标定值，不全部等于 Plant 默认值 |
| 流向 | 真正反转 13 节点流体通过顺序 | 平均温度预测方程不含 13 节点反转；外层经验温度网格决定是否切换 |

两者共享一部分物理量语义和数值，例如冷却液 `cp=3391`、5/15/20 s 类延迟结构和相同控制命令范围；但不共享一个统一参数对象。Plant 参数来自 `thermal_system.py`、`thermal_loop.py`、`pack.py` 配置；P 参数来自冻结的 JSON artifact。修改 Plant 不会自动更新 P。

### 4.4 每周期状态同步

每次 `solve_step()` 都同步：

- 电池平均温度；
- 水箱温度；
- 压缩机/水泵实际转速；
- `Q_evap_eff`、`Q_cond_eff`；
- 供水/回水温度；
- 冷板平均温度；
- 当前环境温度和未来电池发热窗口。

首次 P 求解还显式播种完整可行预测轨迹。后续周期保留 warm start，同时重新写入上述测量。这属于每 5 s 的状态校正，不等于让 P 复制 Plant 的 52 电芯和 13 冷板内部状态。

### 4.5 已有预测误差接口

当前已有一个有限版本的一步预测误差：

```text
innovation = max(0, T_plant(k) - T_pred_1(k-1))
```

实现于 `physics_p_temperature_bias_update()`，结果通过 `last_flow_info["physics_p_temp_bias_innovation_c"]` 写入 CSV。它是：

- 时间对齐的一步误差；
- 只保留 Plant 比预测更热的正向误差；
- 当前补偿增益为 0，innovation 仍可诊断，但不会改变 P 温度路径；
- 不是通用的有符号 `T_plant-T_pred` 接口。

`mpc_physics_shadow.py` 能输出 50/100/300 s 等预测，但当前只额外写入当步冷板实际温度，没有自动生成对应时刻的有符号误差。

后续若 TD3 需要有符号残差，最合适的位置是 `thermal_case_simulator.py:574` 的 `pack.step()` 之后：保存上一周期的 `t_batt_pred_1_c`，计算

```text
e_pred(k+1) = T_plant(k+1) - T_pred_1(k)
```

再放入下一次 TD3 observation。不能直接用同一时刻刚生成的 horizon 起点相减，否则只是状态同步误差而不是一步预测误差。

## 5. 当前闭环调用链

```text
run_full12_physics_p.py
  -> prepared runner.run_worker()
  -> thermal_case_simulator.simulate_case()
     -> 读取时间轴与调峰/RegD 电流
     -> BatteryPack(...)
     -> create_controller(...)
        -> create_mpc_flow_controller(...)
        -> SinglePredictiveDeltaTMPC(...)
     -> initialize_thermal_temperatures(...)
     -> initialize_refrigeration_dynamic_state(...)

     每个 5 s 周期：
       1. pack.total_current = current_profile[k]
       2. 读取 pack、水箱、冷板、执行器当前状态
       3. controller.command(...)
          -> MPCControllerDual.solve_step(...)
          -> 得到压缩机/水泵命令
          -> 必要时由预测温差门决定流向并对新方向再求解
       4. simulate_thermal_loop_step(...)
          -> 实际执行器、制冷循环、供回水、冷板、水箱推进
       5. pack.step(...)
          -> 52 个电芯推进
       6. controller.update_after_step(pack)
       7. 老化观测、功率/能量积分、诊断记录
       8. 下一周期

     -> 输出完整 CSV 和温度快照 CSV
```

TD3 上层的唯一正确插入点是步骤 2 和 3 之间：基于 `x(k)` 产生三个倍率，先写入 MPC 的运行时权重，再调用原有 `controller.command()`。TD3 不应插入步骤 4 和 5 之间。

## 6. 工况与 episode 接口

### 6.1 电流工况

`thermal_case_simulator.py:53` 的 `load_current_profile()`：

- 调峰：固定 `560 A`；
- 调频：读取 `AGC_DATA_FILE`，使用 `Seconds` 和 `RegD` 列，`I=RegD×1120 A`，再插值到仿真时间轴；
- 默认 RegD 文件：`thermal_batch_config.py:25-30`；
- `current_profile_override` 可直接传任意数组。

当前没有命名的 `Current step` 场景，但可通过 `current_profile_override` 表达 Constant、step 或任意自定义曲线。正式命名场景只有调峰和调频/RegD。

### 6.2 episode 长度

- `duration_s` 可限制任意持续时间；
- `max_steps` 可直接限制步数；
- `start_time_s` 可选取工况片段；
- 默认时间轴：调频 3600 s，调峰 6400 s；
- 300 s 对应 60 个 5 s 周期，600 s 对应 120 个周期。

### 6.3 环境和初始状态

当前默认：

- 电池初温固定 25 °C；
- 水箱和冷板初温默认 35 °C；
- 室外与柜内温度每步固定为 35 °C；
- 调峰初始 SOC 0.95，调频初始 SOC 0.55。

限制：

- `simulate_case(initial_thermal_temp_c=...)` 只改变水箱和冷板初温，不改变电池初温；
- 环境温度没有 `simulate_case()` 参数，循环中直接使用 `AMBIENT_TEMP_K`；
- 初始 SOC 由场景函数固定选择；
- 没有 episode 级随机负载、随机环境和随机初始状态接口。

因此 TD3 环境不能直接把 `simulate_case()` 当成标准 `reset/step` API；需要封装或抽取同一推进内核，并显式拥有工况、初始状态和随机数生成器。

## 7. 能耗计算接口

### 7.1 Plant 压缩机功率

`thermal_system.py:514-572`：

```text
m_dot_ref = rho_suction * eta_vol * V_disp * N_flow / 60
h2 = h1 + (h2s-h1)/eta_is
P_comp = m_dot_ref * (h2-h1) / eta_mech
```

- `rho`、焓和熵由 CoolProp R134a 取得；
- `eta_vol`、`eta_is` 来自转速-压比二维效率图；
- 机械效率 `eta_mech=0.9134570768`；
- 排量 `V_disp=5.525e-6 m³/rev`；
- 返回单位为 W。

### 7.2 Plant 水泵功率

`thermal_system.py:417-433`：

```text
Q_L_min = interp(N_pump, speed_points, flow_points)
P_pump = max(0, 0.0651*Q^2 - 1.1513*Q + 8.5959)
m_dot = Q/1000/60 * rho_cool
```

- 流量单位 L/min；
- 功率单位 W；
- 质量流量单位 kg/s。

### 7.3 单步和总能耗

`thermal_case_simulator.py:588-591`：

```text
P_total_kW = (P_comp_W + P_pump_W + P_fan_W) / 1000
E_total_kWh(k+1) = E_total_kWh(k) + P_total_kW * dt / 3600
```

CSV 中压缩机、水泵、风扇功率和总功率均为 kW，总能耗为 kWh。

TD3 reward 若使用真实能耗，应读取 Plant 的 `thermal_step` 功率或累计能耗差，而不是读取 MPC 目标里的归一化预测功率。

### 7.4 MPC 内部功率近似

Physics-P MPC 内部：

- 压缩机功率使用由详细制冷循环稳态网格拟合的转速-冷却液温度-环境温度表达式；
- 水泵功率使用转速三次多项式；
- 两者分别除以各自最大功率后进入目标函数。

这些表达式用于优化器预测和权衡，不等于 Plant 的 CoolProp 压缩机功率和流量二次水泵功率。固定 MPC 和 TD3-MPC 的最终能耗比较必须以 Plant CSV 为准。

## 8. 后续 TD3 可调 MPC 参数

### 8.1 三个动作适合，但温度项必须按真实代码定义

第一版可以使用：

```text
action = [alpha_T, alpha_comp, alpha_pump]
```

推荐映射：

```text
WSPLO_runtime = WSPLO_0 * alpha_T
WSPHI_runtime = WSPHI_0 * alpha_T
w_terminal_runtime = w_terminal_0 * alpha_T
w_comp_runtime = w_comp_0 * alpha_comp
w_pump_runtime = w_pump_0 * alpha_pump
```

这里 `alpha_T` 是一个动作，但同倍率缩放当前实际活动的温度惩罚族，保持 CV 高/低温和终端温度权重之间的基线比例。不建议让 TD3 调当前为 0 的 `w_temp_obj`，因为 `0×alpha_T` 永远为 0，也不代表当前温度控制强度。

如果实验定义坚持只缩放 dead-band WSP，也可以保持终端权重不动，但必须明确实验含义是“主 CV 权重调度”，不是整个温度目标调度；终端项可能掩盖动作效果。

### 8.2 当前在线可修改性

| 参数 | 当前定义 | 创建控制器后直接修改是否生效 | 原因 |
|---|---|---|---|
| `WSPLO/WSPHI` | `MPCParams`，每次 `solve_step()` 重新赋给 CV | 可以 | GEKKO CV 属性在求解前重写 |
| `w_energy_comp` | `MPCParams` | 不可以 | 构建 `j_total_expr` 时以 Python float 写入 |
| `w_energy_pump` | `MPCParams` | 不可以 | 同上 |
| `w_terminal_temp` | `MPCParams` | 不可以 | 构建终端目标时以 Python float 写入 |
| `w_temp_obj` | `MPCParams` | 不可以，且当前为 0 | 构建时固化 |

`MPCParams` 还是 `frozen=True` dataclass。现有 `physics_p_mpc_overrides` 仅在控制器创建前执行，并且允许 `w_energy_comp`，不允许 `w_energy_pump`；它不是在线调度接口。

### 8.3 最少控制器改造

后续实现时，`mpc_flow_direction_strategies.py` 至少需要：

1. 把压缩机、水泵和需要联动的终端温度权重改为 GEKKO `Param`，以 baseline 值初始化；
2. 在 `MPCControllerDual` 增加经范围/有限值检查的运行时权重 setter；
3. 在 `BaseFlowMPCController` 增加同名 setter，并同时更新 `forward_model` 和 `reverse_model`；
4. 每次 `solve_step()` 前更新两侧模型的 CV `WSPLO/WSPHI`；
5. 将实际倍率和实际权重写入 `last_flow_info`，以便复现实验；
6. 固定 MPC 统一使用 `[1,1,1]`，不要维护第二套目标函数。

不能每 5 s 重建整个 GEKKO 控制器来换权重：这会丢失 warm start，显著增加训练成本，并改变当前恢复行为。

### 8.4 本轮明确冻结的量

TD3 不得调节：

- 预测时域 60/45；
- 5 s 控制周期；
- 压缩机和水泵硬边界；
- DMAX 和 DCOST；
- soft band 宽度；
- 终端形式和目标温度；
- P artifact 参数；
- 流向切换阈值、buffer、hold 和正反向逻辑。

TD3 权重更新频率当前代码没有定义。后续需要单独确定，并在两个上层决策之间保持倍率不变；不能从现有代码臆测一个周期。

## 9. 后续 TD3 observation/state 候选

| 候选量 | 当前可直接取得 | 当前位置/方式 | 备注 |
|---|---|---|---|
| `Tavg` | 是 | `pack.get_avg_temp()` | 控制器当前已使用 |
| `Tmax` | 是 | `np.max(pack.temps)-273.15` | 当前已写 CSV |
| `DeltaTmax` | 是 | `np.ptp(pack.temps)` | 当前已写 CSV |
| `Tmin` | 是 | `np.min(pack.temps)-273.15` | 可帮助识别过冷 |
| `Ttank` | 是 | `t_tank_k` | MPC 当前已使用 |
| `Tplate_mean` | 是 | `np.mean(t_plate_k_array)` | P 状态同步已使用 |
| `Tsupply` | 是 | `dynamic_state["T_pipe_supply_K"]` | 当前已写 CSV |
| `Treturn` | 是 | `dynamic_state["T_pipe_return_K"]` | 当前已写 CSV |
| `Tref-Tavg` | 是，需计算 | controller target - `Tavg` | 与同时保留 `Tavg`、固定 `Tref` 时信息冗余 |
| 当前负载电流 | 是 | `pack.total_current` | 调频可正可负 |
| 环境温度 | 是 | `t_outdoor/t_cabinet` | 当前固定 35 °C，但未来随机化后有价值 |
| 压缩机实际转速 | 是 | `dynamic_state["N_comp_eff"]` | 应用 actuator lag 后的实际状态 |
| 水泵实际转速 | 是 | `dynamic_state["N_pump_eff"]` | 同上 |
| 上一步真实压缩机/水泵功率 | 是 | 上一步 `thermal_step` | observation 时需由环境持久化 |
| 平均 SOC | 是 | `np.mean(pack.socs)` | 长 episode 或跨 SOC 初始化时有用 |
| 当前流向 | 是 | `controller.direction` / `is_reversed` | 只作为状态，不让 TD3控制 |
| MPC solved/recovery 标志 | 是 | `controller.last_flow_info` | 可反映求解健康度 |
| 一步正向 prediction innovation | 是 | `physics_p_temp_bias_innovation_c` | 仅 `max(0, actual-pred)` |
| 有符号 `e_pred` | 否 | 需在下一步 Plant 更新后时间对齐计算 | 建议新增环境状态，不改 P 方程 |
| SOH | 是 | `AgingModel280Ah` | 当前短 episode 变化极小，不建议第一版必选 |

建议第一版 observation 保持小而可解释，例如：

```text
[Tavg 或 Tref-Tavg, Tmax, DeltaTmax,
 Ttank, Tsupply, Treturn,
 I_load, T_ambient,
 N_comp_eff, N_pump_eff,
 e_pred]
```

不要同时无理由保留 `Tavg` 和固定目标下的 `Tref-Tavg`；二者线性等价。归一化上下界、残差截断和历史堆叠当前代码没有依据，本报告不自行设定。

## 10. Plant 参数随机化候选

当前没有 episode 级参数随机化框架。以下只说明接口和现值，不建议任何随机范围。

| 参数族 | 当前值 | 定义位置 | reset 时修改难度 | 对冻结模型/验证的影响 |
|---|---|---|---|---|
| 单体热容 | 4747 J/K/电芯；Plant 总和 246844 J/K | `thermal_loop.build_pack_config()` | 低；构造 `BatteryPack` 前改配置 | 改变 Plant 热惯性；P 仍为 218265 J/K，形成模型失配；默认值不变时原验证可保持 |
| 电芯间导热 | 0.5 W/K | 同上 | 低 | 改变 4×13 温差；P 无对应空间状态 |
| 电池-冷板单体导热 | 10 W/K/电芯 | 同上 | 低 | 改变总换热和空间温度；P 的 521.860 W/K 不自动同步 |
| 空气换热 | edge 0.1 W/K；corner 字段 0.2 但当前方程实际读取 edge 字段并乘空间因子 | `build_pack_config()`、`pack.py:121-131` | 低 | 改变环境散热；需注意 `h_air_convection_corner` 当前未直接使用 |
| HPPC `R0/R1/R2/C1/C2` | SOC 0.05–0.95、288.15–313.15 K 查表；R 原始单位 mΩ，加载后乘 1e-3 转 Ω。R0 约 0.503–1.222 mΩ，R1 0.020–0.368 mΩ，R2 0.138–0.631 mΩ；C1 1032.2–39814.5 F，C2 44761.8–100252 F | `model_data/hppc_params.json`、`pack.py:208-230` | 中高；必须在内存中注入/缩放并重建 52 个 PyBaMM 模型，不能改共享 JSON | 会改变电流分配、SOC、2RC 和发热；直接影响电池验证，训练时应保持原文件冻结 |
| 冷板总热容 | 6000 J/K；每节点 461.538 J/K | `thermal_batch_config.py:111` | 中；当前由 simulator 局部计算，无参数对象 | P 等效冷板热容约 1995.995 J/K，随机化增加失配；影响冷板验证 |
| 冷板面积/换热 | `A_total=0.5 m²`、`h_nom=2000 W/(m²·K)`、`m_dot_nom=1.2 kg/s`、流量指数 0.8 | `thermal_system.py:67-71`、`thermal_loop.py:486-492` | 高；当前为模块全局量 | 影响冷板、流量、P 标定；全局修改不适合并行 episode |
| 冷却液 | `rho=1071 kg/m³`、`cp=3391 J/(kg·K)` | `thermal_system.py:37-38` | 高；模块全局且多个函数直接引用 | 同时影响水箱、流量、冷板和 P 共享语义 |
| 水箱 | 3 L，`C_tank=10895.283 J/K` | `thermal_system.py:39-41` | 高；全局 | P 冷却液热容为 10617.810 J/K；影响水箱验证 |
| 执行器/管路动态 | Plant：comp 5 s、pump 5 s、fan 3 s、evap 45 s、cond 75 s、supply 15 s、return 20 s | `thermal_loop.DEFAULT_REFRIGERATION_DYNAMICS` | 中低；`simulate_thermal_loop_step(dynamics=...)` 已支持传字典，但 `simulate_case()` 未暴露 | P 使用另一组标定时间常数；适合结构化注入，不应改默认字典 |
| 蒸发器/Chiller UA | `A=1 m²`、冷却液侧 2000、制冷剂侧 10000 W/(m²·K)、UA scale 0.70，再乘配置 `EVAP_UA_FACTOR=2.0`、flow exp 1.2、cap factor 1.5 | `thermal_system.py:81-88`、`thermal_batch_config.py:48-51` | 高；模块全局 | 直接改变 Physics-P 容量拟合基础和制冷验证 |
| 冷凝器 | `A=1 m²`、空气 1005 J/(kg·K)、空气流量 1.5 kg/s、两侧 h=800/5000 | `thermal_system.py:89-93` | 高 | 影响饱和温度、压比、功率和 P 功率拟合 |
| 压缩机效率/排量 | `eta_vol` 图 0.54–0.98、`eta_is` 图 0.37–0.70、`eta_mech=0.913457`、`Vdisp=5.525e-6 m³/rev` | `thermal_system.py:45-64` | 高；全局且制冷循环有缓存 | 改变制冷量和功率；必须清空缓存；会破坏 P 容量/功率一致性及验证 |
| 水泵性能 | 1600–4800 rpm 对应 11–37 L/min；功率系数 `(0.0651,-1.1513,8.5959)` | `thermal_system.py:110-117` | 高；全局 | 改变流量和真实功率；P 的参考流量及 MPC 泵功率多项式不会自动更新 |
| 风扇 | 最高 4000 rpm、额定 500 W；实际命令为四级 | `thermal_system.py:94-98`、`thermal_loop.py:355-363` | 高 | 影响冷凝器和总能耗；不在 TD3 动作中 |

重要实现约束：`thermal_system.run_refrigeration_cycle()` 带输入量化缓存，cache key 不包含上述模块参数。以后若 episode 改变制冷系统参数，必须使用实例化参数对象，或至少在 reset 时调用现有 `clear_refrigeration_cycle_cache()`；直接修改全局常量而不清缓存会得到跨 episode 的错误复用，并且无法安全并行训练。

推荐的随机化边界设计原则是：只随机 Plant，保持 Physics-P artifact 冻结，这样 TD3 学到的是对模型失配的适应；但 episode 必须监控 P 的输入域，不能让随机化在未定义条件下把 MPC 长期推到 artifact 验证域之外。具体随机范围尚无当前代码证据，本轮不设定。

### 10.1 参数来源在当前代码中的可追溯程度

- 水泵曲线的代码注释明确写为“digitized from literature curves”，但当前文件没有给出具体文献条目。
- 冷凝器风扇注释明确指向 `Energies 2020, 13, 6012`。
- 老化模型注释指向 `Lv, Z., et al., Materials 2025, 18, 1342`，但老化模型不参与 Plant 热状态反馈。
- HPPC JSON 当前没有文献元数据字段。
- Physics-P artifact 是验证过的拟合模型，但当前 JSON 的 `calibration_source/validation_source` 摘要字段为空。
- 其余换热、压缩机图和容量修正值在代码中是工程常量；在没有外部文档证据时，不能全部标为“直接来自文献”。

## 11. 当前计算性能和训练成本

### 11.1 已有真实闭环证据

本轮没有启动新的 benchmark，也没有干预正在运行的 12 工况任务。使用已经完成的两个调峰 Physics-P MPC CSV 和日志：

`C:/Users/24776/Desktop/科研/论文/小论文/仿真数据输出/优化后完整数据/full12_physicsP_pycharm_20260823_120831`

| 工况 | 1280 步完整墙钟时间 | 平均 MPC 求解 | 中位数 | P95 | 最大值 | solved rate | recovery rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 调峰单向 | 6753.4 s | 3.099 s | 2.719 s | 4.194 s | 90.271 s | 100% | 0.781% |
| 调峰双向 | 6582.4 s | 2.960 s | 2.528 s | 4.416 s | 32.415 s | 100% | 1.094% |

注意：这些工况与其他 worker 并行运行，所以是当前实际工作负载证据，不是隔离微基准。

### 11.2 Plant 单步和闭环时长

当前代码只单独记录 MPC 求解时间，没有在 `BatteryPack.step()` 或完整 Plant step 周围单独计时，因此不存在可信的纯 Plant 单步 benchmark。

已有同批次非 MPC 调峰结果提供代理范围：

- PID：约 1907–1918 s / 1280 步，即约 1.49–1.50 s/闭环步；
- On-off：约 2376–2382 s / 1280 步，即约 1.86 s/闭环步。

该范围包含控制、R134a 循环、52 个 PyBaMM 电芯、记录和并行资源竞争，不能等同于 `BatteryPack.step()` 的纯耗时。MPC 完整闭环平均约 5.14–5.28 s/仿真步，其中 GEKKO 求解均值约 2.96–3.10 s。

按上述完整 MPC 墙钟速度估算：

- 300 s episode（60 步）：约 309–317 s，即约 5.1–5.3 min；
- 600 s episode（120 步）：约 617–633 s，即约 10.3–10.6 min。

这只是从已完成完整工况线性折算，不是独立 300/600 s 实测。

### 11.3 TD3 训练瓶颈

串行估算：

- 1000 个 300 s episode：约 86–88 小时；
- 10000 个 300 s episode：约 36 天；
- 1000 个 600 s episode：约 7.1–7.3 天；
- 10000 个 600 s episode：约 71–73 天。

结论：当前 GEKKO MPC 是主要增量瓶颈，52 个 PyBaMM Plant 也不是轻量环境。直接做数千至数万个 full-Plant/full-MPC episode 会非常昂贵。第一版实现前至少需要：

1. 增加分段计时，分别测 Plant、制冷循环、MPC、日志；
2. 确认是否每个 TD3 决策都需要新的 MPC 权重，且在不改变 5 s 下层控制周期的前提下采用上层动作保持；
3. 保持 GEKKO 模型和 warm start 跨步复用；
4. 将并行训练设计为进程隔离，避免 GEKKO home、PyBaMM/CasADi 和制冷缓存共享；
5. 先做短 episode 和确定性回归，再决定是否需要代理训练环境。

不能因为训练慢就把 Physics-P 当真实 Plant；那会改变本项目的实验问题。若未来使用代理训练，仍必须回到完整 Plant 做最终评估，并明确区分训练环境和评价环境。

## 12. 最小 TD3 接入改造点

### 12.1 建议新增文件

第一版 `Fixed MPC vs TD3 adaptive-weight MPC` 最小可维护结构：

```text
td3_mpc_env.py
    episode reset/step、Plant 状态快照、TD3 observation、reward、done、随机种子

td3_weight_policy.py
    TD3 actor/critic、replay buffer、保存/加载；不包含 Plant 或 MPC 方程

train_td3_mpc.py
    训练入口、实验配置、checkpoint 和日志

evaluate_td3_mpc.py
    Fixed [1,1,1] 与 TD3 权重在相同工况/初值下的成对评价

tests/test_td3_mpc_weight_interface.py
tests/test_td3_mpc_env.py
```

若使用现成 RL 库，`td3_weight_policy.py` 可以省略，只保留环境、训练入口和评价入口；不要复制库内 TD3 实现。

### 12.2 最少修改的现有文件

必须修改：

- `mpc_flow_direction_strategies.py`
  - 把活动能耗权重和需要联动的终端温度权重变成运行时 GEKKO `Param`；
  - 增加三个倍率的 setter；
  - 同步 forward/reverse 两个模型；
  - 记录实际倍率/权重；
  - 保持目标函数形式、默认值、约束和 recovery 不变。

可能修改，但第一版可以避免：

- `thermal_control_strategies.py`：只有需要通过统一工厂创建 TD3 包装器时才加薄入口。
- `thermal_case_simulator.py`：只有希望复用原 CSV runner 做在线权重评估时才加可选 schedule hook；训练环境可直接复用现有 Plant 函数，避免改正式 runner。
- `__init__.py`：只有需要公开导出环境时才改。

### 12.3 必须保持不动的现有文件/资产

第一版权重自适应实验应冻结：

- `pack.py`；
- `thermal_loop.py`；
- `thermal_system.py`；
- `model_data/hppc_params.json`；
- `model_data/physics_p_operational_v1.json`；
- `mpc_physics_predictor.py`；
- `mpc_evaporator_capacity_model.py`；
- `predictive_delta_t_flow_controller.py`；
- `predictive_delta_t_flow_mpc.py`；
- `thermal_batch_config.py` 中的 horizon、dt、硬约束、DMAX、soft band 和流向参数；
- 当前完整物理 Plant 的时间推进顺序。

Plant 参数随机化属于后续独立阶段，不应与第一版在线权重接口同时实施。先证明 `[1,1,1]` 新接口与当前 Fixed MPC 数值完全一致，再加入 TD3。

## 13. 第一版实施前的验收门槛

1. 新的运行时权重接口在 `alpha=[1,1,1]` 时，与当前 MPC 单步命令、预测轨迹和目标诊断一致；
2. 固定动作环境与原 `simulate_case()` 在相同工况下的 Plant 温度和能耗一致；
3. forward/reverse 模型收到完全相同的倍率；
4. TD3 不能写 horizon、dt、DMAX、bounds、soft band 或流向状态；
5. 任何求解失败仍走当前 recovery，TD3 不覆盖 fallback 命令；
6. observation 使用 k 时刻状态，reward 使用 k→k+1 的 Plant 结果，时间层不可错位；
7. 每个 episode 重建 PyBaMM/热回路/MPC/老化状态；
8. Fixed 和 adaptive 评价使用同一电流、环境、初温、随机种子；
9. 日志必须保存三倍率、三组实际权重、Plant 功率、温度约束、求解成功和恢复状态；
10. 在开始长训练前，先报告纯 Plant、MPC 和完整 env step 的隔离耗时。

## 14. 简洁结论

现在开始实现第一版层级控制时，建议新增 TD3 环境、训练入口、成对评价入口及对应测试；现有代码只需在 `mpc_flow_direction_strategies.py` 增加运行时权重参数和统一 setter。完整 Plant、Physics-P artifact、流向逻辑、设备约束和时间推进顺序都应保持不动。

TD3 的动作不是压缩机/水泵转速，而是三个无量纲倍率。MPC 仍是唯一产生设备命令的下层控制器。固定 MPC 是同一代码路径下的 `alpha=[1,1,1]`，这样比较才不会混入第二套控制实现的差异。
