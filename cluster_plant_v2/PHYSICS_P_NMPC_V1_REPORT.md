# Physics-P 冻结预测器离散时间 NMPC（do-mpc）闭环验证报告

日期：2026-08-26
结果目录：`validation/results/physics_p_nmpc_20260826/`

## 1. 范围与结论

按规格实现了基于冻结 Physics-P 预测器的离散时间非线性 NMPC：

- **不是**线性 QP、**不是** GEKKO；do-mpc 5.1.1 + CasADi 3.7.2 + IPOPT 直接求解非线性 NLP；
- Physics-P 不重新辨识、不修改，仅作为 MPC 内部 5 s 一步预测模型；
- 完整集群热管理系统（含 PyBaMM 电池产热、冷板、制冷回路）是 Plant，do-mpc 不直接优化 Plant；
- 148 个既有回归测试全部通过；三工况 600 s 闭环对比固定基线（4000/3600 rpm）完成。

结论摘要：

| 工况 | 指标 | 固定基线 | Physics-P NMPC |
| --- | --- | --- | --- |
| S0 恒流 560 A | 最终电池均温 / 带外时间 | 25.15 °C / 0 s | 25.18 °C / 0 s |
| | 压缩机能耗 | 0.1483 kWh | **0.1351 kWh（−8.9%）** |
| S1 阶跃 280→560→1120 A | 平均带误差（±0.65 K 带宽） | 2.184 K | **2.146 K** |
| | 最终电池均温 | 30.59 °C | **30.51 °C** |
| | 压缩机能耗 | 0.1475 kWh | 0.2518 kWh（+71%，抗 1120 A 大阶跃） |
| T2 PJM RegD | 最终电池均温（目标 25 °C） | 24.44 °C（过冷） | **25.08 °C（更贴近目标）** |
| | 压缩机能耗 | 0.1483 kWh | **0.0890 kWh（−40%）** |

NMPC 在温度跟踪等效或更好的前提下，于恒流与波动工况显著省电；在大阶跃工况以更多压缩机能耗换取更小的带误差，符合目标函数中 `w_upper=5000` 的非对称超温惩罚语义。

## 2. 架构符合性对照

| 规格项 | 实现 |
| --- | --- |
| 模型形式 | `do_mpc.model.Model("discrete")`，`x(k+1) = [P(x[:14],u,d), n_comp_cmd, n_pump_cmd]` |
| 状态维度 | 16 = 14 物理（`t_bat,t_tank,t_plate,n_comp,n_pump,q_evap,z_pump_1,z_supply_1..3,z_return_1..4`）+ 2 增广（`n_comp_cmd_prev,n_pump_cmd_prev`） |
| 控制输入 | `u = [n_comp_cmd, n_pump_cmd]` |
| 时变扰动 TVP | `[current_a, t_amb_c, q_gen_w]` 沿时域逐点预览（冻结工件的 CasADi 函数将产热 `q_gen_w` 外置为第三扰动，`q_gen_w` 由电流预览经 4P 欧姆热公式生成；`T_ref` 与权重作为控制器参数冻结） |
| 预测域 | N=60 × 5 s = 300 s；每 15 s 重优化，中间两个 5 s 步保持指令 |
| 目标函数 | `e=(t_bat−25)`：阶段代价 `500e²+5000·max(e,0)²+5000·max(−e,0)²+0.1(n_c/6000)³+0.001(n_p/4800)³+0.01(Δn_c/6000)²+0.01(Δn_p/4800)²`；终端代价仅温度三项 |
| 硬约束 | `300≤n_comp_cmd≤6000`、`1600≤n_pump_cmd≤4800`；Δu 经增广状态以 `set_nl_cons` 双侧（成对单侧）硬约束 `±1200 / ±300` |
| 闭环接口 | Plant 测量 → 适配器映射 14 维（K→°C）→ IPOPT 求 300 s 序列 → 仅执行第一组指令 |
| 边界遵守 | 只用 `t_bat`；`T_cell_max`、ΔT 仅记录在时序输出中，未进入目标/约束 |

## 3. 实现文件

- `control/physics_p_nmpc_model.py`
  - 注入父目录 `sys.path`，导入 `single_pack_plant.predictor.physics_p_casadi.CasadiPhysicsP`（validated 工件 `physics_p_operational_v1.json`，满足 constant 时间常数 / direct 蒸发响应 / 0-1-3-4 步延迟前置检查）；
  - 16 维离散模型：`x = set_variable("_x","x",shape=(16,1))`，两个 `_u`，三个 `_tvp`；
  - 注意：CasADi 对稠密 SX 列向量的行切片会返回宽稀疏块，物理状态须用 `vertcat` 逐元素装配后传入冻结函数；
  - 自带冒烟检查：模型 rhs 与冻结函数逐步等价（差异 0.0）。
- `control/physics_p_nmpc_controller.py`
  - `PhysicsPNmpcParameters` 默认权重/边界完全按规格；
  - `mterm` 只含温度项（do-mpc 不允许终端代价含 u），Δu 惩罚在 `lterm` 中经增广状态表达；
  - Δu 硬约束成对写成 `u−u_prev ≤ dmax` 与 `u_prev−u ≤ dmax`（`set_nl_cons` 仅支持单侧上界）；
  - TVP 函数按 `t_now + k·5s` 取电流预览。
- `validation/validate_physics_p_nmpc.py`
  - Plant→Physics-P 适配器：电池容量加权均温、罐温、冷板代表温度、实际压缩机转速、`q_evap_applied`、供/回液队列（`queue_values` 最旧在前，与冻结预测器 `z_supply[0]`=当前供液、`z_return[0]`=即将入罐语义一致）全部 K→°C；`z_pump_1` 与 `n_pump` 取最近一次执行的泵指令；
  - 每 3 步调用一次 `make_step`，并在调用前把 `mpc._t0` 对齐到实际仿真时刻（do-mpc 内部时钟每次 `make_step` 只前进 `t_step`）；
  - 增广初值 = 上一次实际执行的指令，保证阶段 0 的 Δu 约束精确限定 15 s 实际动作；
  - 应用层仅保留与 NLP 相同界限的防御性裁剪。

## 4. 结果细节（600 s，dt=5 s）

### 4.1 S0 恒流 560 A

- 温度两者均守在 25±0.65 °C 带内；NMPC 稳态将压缩机压到约 2800–4200 rpm（基线固定 4000），压缩机能耗 −8.9%；
- 泵指令均值约 4755 rpm：冻结预测器中泵转速通过 `plate_fluid` 强化冷板换热，泵功率权重 `0.001` 很小，优化器用较高泵速换更低压缩机转速；
- 求解：40 次优化，平均 0.18 s/次。

### 4.2 S1 阶跃 280→560（25 s）→1120 A（50 s）

- 1120 A 下任何控制器都无法守住 25 °C（基线最终 30.59 °C，500 s 带外）；
- NMPC 将压缩机推至 5200–6000 rpm 区（多次触顶）并把泵降到约 1600–1800 rpm，平均带误差 2.146 K（−1.7%），最终温度低 0.08 K；代价是压缩机能耗 +71%；
- 瞬态最大温度 31.38 °C 略高于基线 31.04 °C：优化器在早期低电流段省电、阶跃后快速拉满，属于权重设定下的合理权衡；
- 求解：40 次优化，平均 4.95 s/次（压缩机饱和段 NLP 更难），满足 15 s 控制周期。

### 4.3 T2 PJM RegD

- 基线把系统过冷到 24.44 °C；NMPC 将最终温度控制在 25.08 °C，全程在带内，同时压缩机能耗 −40%；
- 压缩机指令在 1000–6000 rpm 间随负荷调整（累计动作量 34.9 k rpm，受 ±1200/15 s 限制），泵均值约 4250 rpm；
- 求解：40 次优化，平均 3.26 s/次。

## 5. 已知边界与遗留事项

1. **Plant 执行域下限**：完整 Plant 压缩机执行器拒绝 <1000 rpm 指令，而规格搜索域为 300–6000 rpm；NLP 内保持规格界限，应用层将指令裁剪到 ≥1000 rpm（T2 工况出现过裁剪）。
2. **模型-Plant 差异**：冻结预测器含压缩机/泵一阶滞后与固定时滞链，Plant 泵速即时生效、制冷回路由完整热力学求解；闭环表现证明差异在可接受范围，但长时程（>600 s）偏差需后续评估。
3. **未进入目标/约束的量**：`T_cell_max`、电池 ΔT 仅记录；后续可作软约束或加权项加入。
4. do-mpc 提示 `rterm` 未设置——规格的动作惩罚已经通过增广状态显式进入阶段代价，该告警为预期行为。

## 6. 复现

```powershell
Set-Location "c:\Users\24776\PycharmProjects\PythonProject\集成仿真多种控制"
# 模型冒烟（逐步等价性）
& "C:\Users\24776\miniforge3\envs\btms\python.exe" -m cluster_plant_v2.control.physics_p_nmpc_model
# 600 s 三工况闭环对比
& "C:\Users\24776\miniforge3\envs\btms\python.exe" -m cluster_plant_v2.validation.validate_physics_p_nmpc `
    --output-dir cluster_plant_v2/validation/results/physics_p_nmpc_20260826
```

输出：每工况时序 CSV、对比图（`*_comparison.png`）、汇总 `physics_p_nmpc_summary.csv/json`。
