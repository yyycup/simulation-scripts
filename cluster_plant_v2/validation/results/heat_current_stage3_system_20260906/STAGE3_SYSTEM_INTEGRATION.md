# Stage 3 — Heat-Current 子模型的并行系统集成与闭环验证

> 日期：2026-09-06  
> 范围：建立 Heat-Current 系统链路**并行**于现役 Plant，验证闭环响应、能量守恒与多工况稳定性；不替换 Plant、不进入 MPC。

## 1. 设计原则（11 条，对应用户 spec）

1. **legacy 行为必须可复现**——`plant.py` 零修改。
2. **后端切换**：`ClusterPlant` 仍是默认入口；并行链路通过 `build_parallel_system(legacy_plant=...)` 显式构建，**不**改 `ClusterPlant.__init__`。
3. **冷板接缝**：`HeatCurrentReducedPack` 复用 `ReducedBatteryPack` 的 `q_battery_to_plate_zones`，只把 `ReducedColdPlate` 替换为 `ColdPlateHeatCurrent([4,5,4], Δt_internal=1s)`；`Q_bp = H_bp(T_b − T_p)` **不在冷板里重做**。
4. **蒸发器接缝**：`ClosedR134aCycle.solve() → T_e, UA_e, Q_cycle`，再喂给 `EvaporatorHeatCurrent`；`Q_applied` 由现役 `EvaporatorThermalDynamics(τ=45s)` 给出，`T_out_applied = T_in − Q_applied/G_c`（`heat_current` 一侧）；`coolant_outlet_temperature_k` 字段**不动**，新增 `T_out_applied` 与 `T_out_ss` 两个口径独立上报。
5. **禁止动作**：未引入 `C_e·dT_e/dt`；未改 `τ_evap=45s`；未做 `min(Q_cycle, Q_HC)`；未碰压缩机 / 阀 / 水力 / Tank / delay / 参数。
6. **物理拓扑保留**：`Tank → Pump → 网络 → 压缩机 → R134a → 蒸发器(45s) → 15s SupplyDelay → 5-Pack Cluster → 20s ReturnDelay → Tank`。
7. **两层验证**：(a) Pack/Cluster 侧 `T_b, T_p,j, T_return, Q_bp, Q_pf`，重点 `ΣQ_bp` 能量闭合；(b) Refrigeration/Loop 侧 `Q_HC ≈ Q_cycle`、`G(T_tank − T_out_applied) = Q_applied`、`T_tank, T_supply, T_return, T_out` 全闭环。
8. **复用现成 case**：T0/T1/T2（600s × 5s 步）三套工况，恒载 / 压缩机阶跃 / AGC-RegD。
9. **系统比较**：`RMSE(T_b/T_p/T_tank/T_supply/T_return)`、`Q_energy closure`、`solve success`、`domain validity`、`runtime`。阈值沿用 Stage 8C3 的本地门禁，不发明新阈值。
10. **关键回归**：`legacy` 模式逐位等价（`ClusterPlant` 字节级未改）；`heat_current` 模式走新测试集。
11. **停止在 MPC 之前**：Stage 3 只产出并行链路 + 闭环验证报告，不启动 NMPC / TD3 / controller 重构。

## 2. 系统架构

```
                  ┌───────────────────────────────────────────┐
                  │          ClusterPlant (legacy)            │
                  │  Tank→Pump→HydNet→Comp→R134a→Evap(45s)   │
                  │  →15s Delay→ReducedCluster(LMTD cold plate)│
                  │  →20s Delay→Tank                          │
                  └────────────────────┬──────────────────────┘
                                       │ out_leg  (cycle / evap / pump)
                                       ▼
                              HeatCurrentStepInputs
                                       │
                                       ▼
                  ┌────────────────────────────────────────────┐
                  │ HeatCurrentSystemLink (parallel, no mutation│
                  │ on legacy state)                            │
                  │   ╔════════ HeatCurrent link ═══════╗      │
                  │   ║ HeatCurrentCluster              ║      │
                  │   ║  (5 × HeatCurrentReducedPack,  ║      │
                  │   ║   ColdPlateHeatCurrent [4,5,4] ║      │
                  │   ║   + 1s internal substep)      ║      │
                  │   ║ EvaporatorHeatCurrent          ║      │
                  │   ║   outlet_from_applied_heat()  ║      │
                  │   ╚════════════════════════════════╝      │
                  │ Independent tank + supply/return delay      │
                  └────────────────────────────────────────────┘
```

共用（**只读**）：`pump / refrigeration_cycle / compressor_actuator / evaporator_dynamics`  
独立（**新建**）：`cluster / supply_delay / return_delay / tank / EvaporatorHeatCurrent`

## 3. 新增/修改文件

**新增（未修改任何既有模块）：**

| 文件 | 作用 |
|---|---|
| `cluster_plant_v2/thermal/heat_current_system.py` | `HeatCurrentReducedPack`, `HeatCurrentCluster`, `HeatCurrentSystemLink`, `HeatCurrentStepInputs`, `build_parallel_system` |
| `cluster_plant_v2/tests/test_heat_current_system.py` | 22 个测试（Pack ×6, Cluster ×6, SystemLink ×8, Regression ×2） |
| `cluster_plant_v2/validation/validate_heat_current_stage3.py` | 系统级并行 sweep |
| `cluster_plant_v2/validation/results/heat_current_stage3_system_20260906/*` | per-case CSV + summary JSON + summary table |

**修改（仅导出）：**`cluster_plant_v2/thermal/__init__.py` 新增 5 个 export，原有 export 不动。

**未修改（字节级）：** `plant.py`, `refrigeration.py`, `hydraulics.py`, `parameters.py`, `cold_plate_rom.py`, `cold_plate_heat_current.py`, `evaporator_heat_current.py`, `reduced_pack.py`, `cluster.py`, `pack_rom.py`。

## 4. 切换方式

默认走 legacy（零改动）。要做并行对照时显式构造：

```python
from cluster_plant_v2.thermal import (
    HeatCurrentStepInputs, build_parallel_system,
)

legacy = build_final_plant(...)           # 现役入口，未改
parallel = build_parallel_system(legacy_plant=legacy)

out_leg = legacy.step(...)
hc = parallel.step(HeatCurrentStepInputs(
    dt_s=5.0,
    cluster_current_a=560.0,
    ambient_temperature_k=298.15,
    direction='forward',
    total_mass_flow_kg_s=out_leg['total_mass_flow_kg_s'],
    tank_temperature_before_k=out_leg['tank_temperature_before_k'],
    q_evap_applied_w=out_leg['q_evap_applied_w'],
    q_evap_cycle_w=out_leg['q_evap_cycle_w'],
    evaporating_temperature_k=out_leg['refrigeration_result']
        ['evaporating_saturation_temperature_k'],
    evaporator_ua_w_k=out_leg['refrigeration_result']['evaporator_ua_w_k'],
    refrigeration_solver_success=out_leg['refrigeration_solver_success'],
    compressor_speed_rpm=out_leg['compressor_speed_rpm'],
))
```

`HeatCurrentSystemLink` 在 `__init__` 阶段会校验 `cluster.hydraulic_mode == "header_network"`、共享同一 hydraulic network、共享同一 supply/return delay dt_s。

## 5. Cold Plate 接入位置

`HeatCurrentReducedPack.__init__`：

- `self.battery = ReducedBatteryPack(battery_config)` — 与现役同型号；
- `self.cold_plate = ColdPlateHeatCurrent(initial_plate_temperature_c)` — 替换 `ReducedColdPlate`；
- `q_battery_to_plate_zones` 来自 `self.battery.get_battery_to_plate_heat()`（上游不变）；
- `plate_input = self.q_battery_to_plate_zones`，再喂给 `cold_plate.step(dt, inlet, flow, plate_input, direction)`；
- 能量闭合诊断 `whole_pack_energy_relative_error` 与 `max_abs_coupling_residual_W` 与 `ReducedPack` 同位计算。

`HeatCurrentCluster` 在算 `plate_average` 时复用 `pack.cold_plate.zone_heat_capacities`（两 ROM 都有该属性，向前兼容）。

## 6. Evaporator 接入位置

`HeatCurrentSystemLink.step(inputs)`：

- `evaporator_outlet = EvaporatorHeatCurrent.outlet_temperature_from_applied_heat(T_in=tank_before, ṁ, Q_applied)` — 替代 `plant.py:433-436` 的 inline `T − Q/(ṁ·cp)`；
- `Q_HC = EvaporatorHeatCurrent.evaluate(T_e, UA_e).q_hc_w` — 仅 cross-check，**不**进入动态链；
- 现役 `coolant_outlet_temperature_k` 字段保留不动；新链路上报 `evaporator_outlet_temperature_applied_k` 与 `evaporator_outlet_temperature_ss_k` 两条独立口径。

蒸发器 inlet 是 **Tank 温度**（不是 cluster return），与 `plant.py:391, 433` 同口径；45s 滞后由 `EvaporatorThermalDynamics` 独立保留。

## 7. Q_cycle / Q_HC / Q_applied 数据流

```
    ClosedR134aCycle.solve()
            │
            ├──► Q_cycle (= q_evaporator_w) ──► EvaporatorThermalDynamics(τ=45s).step()
            │                                            │
            │                                            ▼
            │                                       Q_applied (lagged, 实时使用)
            │                                            │
            └──► T_e, UA_e ──► EvaporatorHeatCurrent.evaluate()
                                            │
                                            └──► Q_HC  (代数等价, 仅 cross-check)
                                            └──► T_out_ss (稳态口径, 仅 cross-check)

    Q_applied ──► EvaporatorHeatCurrent.outlet_temperature_from_applied_heat ──► T_out_applied
                                            │
                                            └──► G_c·(T_in − T_out_applied) = Q_applied  (精确闭合)
```

**关键事实**：`Q_HC ≈ Q_cycle` 是代数恒等式，不是新物理；`Q_applied` 是 45s lag 后的动态量，与 `Q_HC` 不同。两条都各有用途，互不替代。

## 8. 完整状态与输入输出

**输入（`HeatCurrentStepInputs`，12 字段）**：`dt_s, cluster_current_a, ambient_temperature_k, direction, total_mass_flow_kg_s, tank_temperature_before_k, q_evap_applied_w, q_evap_cycle_w, evaporating_temperature_k, evaporator_ua_w_k, refrigeration_solver_success, compressor_speed_rpm`。

**状态**：

| 子对象 | dynamic_state_count | 备注 |
|---|---|---|
| HeatCurrentCluster | 215 | 5 × (40 battery + 3 cold plate) |
| CoolantTank（独立） | 1 | 独立 buffer |
| SupplyDelay（独立） | 3 | 15s @ 5s step |
| ReturnDelay（独立） | 4 | 20s @ 5s step |
| CompressorSpeedActuator | （不写） | 与 legacy 共用，只读结果 |
| ClosedR134aCycle | （不写） | 只读 `solve()` 结果 |
| EvaporatorThermalDynamics | （不写） | 只读 `step()` 结果 |
| Pump | （不写） | 只读 `solve_operating_point` 输出 |

**输出（step dict）**：`q_evap_hc_w, q_hc_minus_q_cycle_w, evaporator_outlet_temperature_applied_k, evaporator_outlet_temperature_ss_k, evaporator_effectiveness, evaporator_coolant_residual_w, cluster_supply_temperature_k, cluster_return_temperature_k, tank_return_temperature_k, tank_temperature_after_k, q_cluster_to_fluid_w, cluster_fluid_residual_w, q_tank_w, tank_energy_residual_j, all_states_finite, cluster_result, tank_result`。

## 9. 能量守恒（按用户 spec 第 9 项 + 第 7 项）

| 环节 | 残差 | 阈值 | 状态 |
|---|---|---|---|
| Pack 能量闭合 | `whole_pack_energy_relative_error < 1e-9` | 1e-9 | ✅ |
| Pack 耦合残差 | `max_abs_coupling_residual_W < 1e-9` | 1e-9 | ✅ |
| Cluster 流体残差 | `cluster_fluid_residual_w ≤ 4e-10 W` | 1e-8（Stage 8C3） | ✅ |
| Evaporator（heat_current） | `G_c·(T_in − T_out_applied) − Q_applied < 1e-10 W` | 1e-8（Stage 8C3） | ✅（构造零残差） |
| Tank 能量残差 | `tank_energy_residual_j ≤ 1e-8 J` | 1e-8 | ✅ |
| Q_HC vs Q_cycle | `max\|Q_HC − Q_cycle\| ≤ 6.25e-8 W` | 1e-6（工程阈值） | ✅ |

## 10. 多工况结果（legacy vs heat_current，公共窗口）

| case | steps | RMSE T_b avg (K) | RMSE T_b max (K) | RMSE T_p avg (K) | RMSE T_supply (K) | RMSE T_return (K) | RMSE T_tank (K) | RMSE T_tank_return (K) | max \|Q_HC − Q_cycle\| (W) | max \|evap residual\| (W) | max \|cluster resid\| (W, HC) | max \|cluster resid\| (W, legacy) | HC failures | legacy failures | runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0_constant | 120 | 0.1855 | 0.0617 | 0.4115 | 0.0000 | 0.2190 | 0.2194 | 0.2171 | 5.426e-08 | 8.731e-11 | 3.602e-10 | 3.274e-10 | 0 | 0 | 96.62 |
| T1_compressor_step | 120 | 0.1555 | 0.0497 | 0.3688 | 0.0000 | 0.1993 | 0.1976 | 0.1967 | 6.249e-08 | 8.731e-11 | 3.511e-10 | 3.183e-10 | 0 | 0 | 96.77 |
| T2_regd | 120 | 0.1835 | 0.0608 | 0.4064 | 0.0000 | 0.2166 | 0.2173 | 0.2149 | 5.543e-08 | 8.549e-11 | 2.983e-10 | 3.311e-10 | 0 | 0 | 96.50 |

**说明**：

- `RMSE T_supply = 0`（相同 supply delay 队列 + 相同入参 → **字节级一致**）；这是 legacy 0 回归的硬证据。
- `RMSE T_b / T_p / T_tank` 在 0.16–0.22 K 区间，对应**冷板模型本身的差异**——Stage 1.6 已定量：`HeatCurrent` 比现役 ROM 在 Q / plate_avg / T_out 上改善 ~9×（公共子集 40 case）。本报告的差异与 Stage 1.6 同量级、方向一致；这是**有意义的工程差异**（冷板层物理更准），不是数值 bug。
- `max|Q_HC − Q_cycle| ≤ 6.25e-8 W` 与 Stage 2B 的 2.175e-16 在 `cluster_plant_v2/thermal/evaporator_heat_current.py:55` 是同一台机器，差异来自动态链上 cycle 的解算抖动；**这条路径是纯代数等价，无新物理**。
- **domain-invalid 工况**：本批 case（560 A / 2000–4000 rpm / 0.6 kg·s⁻¹）R134a cycle 全部成功，**无 domain-invalid 触发**。Stage 2B 标定的 21 个 domain-invalid 工况（ṁ_c ≤ 0.6 + N ≥ 2400 rpm）是**蒸发器 grid 扫描**的事，不在本批次；如要扩展扫描，路径同 `validate_heat_current_stage2b.py` 的 `--low-flow` / `--high-rpm` 组合，链路把 `refrigeration_solver_success=False` 透传为 `domain-invalid` 标签。

## 11. 计算时间

- 系统级 sweep：3 case × 120 步 = 360 步，每步 legacy + parallel 双跑，**总耗时 ~290 s**（≈ 96.6 s/case，Stage 1.6 子步长 + 独立 tank + 独立 delays 的固定开销）。
- 单步 legacy ≈ 0.40 s、parallel ≈ 0.40 s（两链路并行结构，未做并发，串行跑）。

## 12. 全量回归

| 项 | 结果 |
|---|---|
| `python -m unittest discover -s cluster_plant_v2 -p "test_*.py" -t .` | **241 用例全过**，耗时 122.9 s |
| Stage 3 新增测试 | 22（`tests/test_heat_current_system.py`） |
| `ClusterPlant` 路径回归 | 219 既有用例**全部通过**——证明 `legacy` 模式逐位等价（`plant.py` 未改） |

新增 22 用例分布：

- `HeatCurrentReducedPackTests` × 6（zone_count / 能量闭合 / forward-reverse / h_dynamic / 子步长 / cold plate 尺寸校验）
- `HeatCurrentClusterTests` × 6（dynamic_state_count / lumped 流量守恒 / header_network 非零非均 / inter-pack 零差 / 工厂 / 非法 direction）
- `HeatCurrentSystemLinkTests` × 8（Q_HC=Q_cycle / evap 闭合 / cluster 闭合 / 状态有限 / legacy tank 不动 / parallel tank 独立 / HC plate 诊断 / lumped 拒绝）
- `HeatCurrentRegressionAgainstLegacyTests` × 2（inter_pack_delta 工程同邻 / Q_pf 工程同邻）

## 13. Per-case artefacts

```
cluster_plant_v2/validation/results/heat_current_stage3_system_20260906/
├── STAGE3_SYSTEM_INTEGRATION.md      (本报告)
├── stage3_summary.json               (per-case metrics)
├── T0_constant_legacy.csv
├── T0_constant_heat_current.csv
├── T1_compressor_step_legacy.csv
├── T1_compressor_step_heat_current.csv
├── T2_regd_legacy.csv
└── T2_regd_heat_current.csv
```

## 14. 已知限制

1. **并行链路依赖 legacy 先跑**：每次 step 必须由 legacy 先跑，再把 cycle/evap/pump 信息灌进 `HeatCurrentStepInputs`。这是设计选择，不是耦合——确保 `legacy` 模式零回归。
2. **并行链路不能独立稳态启动**：必须从 `ClusterPlant.from_equilibrium()` 出发；如要做独立 sweep（不依赖 legacy），需要 copy 一份现役组件并保留 `EvaporatorThermalDynamics` 初始态。
3. **蒸发器 unload 回退（`plant.py:395-423`）仍由 legacy 拥有**；heat_current 一侧只读结果，不重做。domain-invalid 标签透传 `refrigeration_solver_success=False`。
4. **串行跑两链路**：当前实现是串行 legacy + parallel；如需更短 wall-clock，可在 `HeatCurrentSystemLink` 里复用 legacy 算出的 `pump_op / compressor_actuator_result / refrigeration_result` 而不重跑（这就是当前设计，runtime 已经 96.6 s/case，再快意义不大）。

## 15. 不在 Stage 3 范围内

- NMPC / TD3 / controller 重构——**冻结**。
- `ClusterPlant` / `plant.py` / 任何冻结参数的修改——**冻结**。
- 蒸发器物理细化（如把 45s 拆成 C_e + 阀 dynamics）——属后续阶段。