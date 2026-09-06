# Stage 4.5 — Independent Heat-Current Plant Assembly

> **状态**：✅ **独立 HC Plant 已建立成功。可以进入 Stage 5。**
>
> 任务书规定"先汇报 Independent Heat-Current Plant 是否建立成功，再开始 Stage 5"。
> 本报告即该汇报。

---

## 1. 任务来源

Stage 5 的"整个系统独立验证"目标要求 HC 这一套自己独立递推，但 `HeatCurrentSystemLink`
（Stage 3）属于 **shadow comparison**：它从 legacy `ClusterPlant.step()` 接收
`tank_temperature_before_k` / `q_evap_applied_w` / `q_evap_cycle_w` / `T_e` /
`UA_e` / `refrigeration_solver_success` / `compressor_speed_used_rpm` 等9 个内部量
作为 inputs，再灌给 HC cluster + evap outlet 公式 + tank/delay 副本。

因此选 **方案 B**：不动 frozen 模块，新建 `HeatCurrentPlant`，自己持有完整 9 个实例，
按现役物理顺序独立推进。Stage 3 的 `HeatCurrentSystemLink` 保留作为 shadow tool。

## 2. 交付清单

| 类型 | 路径 | 状态 |
|---|---|---|
| 核心模块 | `cluster_plant_v2/thermal/heat_current_plant.py` | 新增（约 510 行） |
| 独立性测试 | `cluster_plant_v2/tests/test_heat_current_plant_independence.py` | 新增 5 用例 |
| 全量回归 | **256 / 256 通过**（基线 251 + Stage 4.5 新增 5） | |
| Frozen 模块 | **`plant.py` / `refrigeration.py` / `hydraulics.py` / `pack_rom.py` / `parameters.py` 均未修改** | ✔ |

## 3. 架构

```
HeatCurrentPlant
├── cluster                (HeatCurrentCluster — 5 个 HeatCurrentReducedPack)
├── hydraulic_network       (共享 — 几何阻力数据，运行时不变)
├── pump                    (CoolantPump — dynamic_state_count=0, 安全共享)
├── tank                    (fresh CoolantTank — 自己的温度态)
├── refrigeration_cycle     (ClosedR134aCycle — dynamic_state_count=0, 安全共享)
├── compressor_actuator     (fresh — 自己的转速态)
├── evaporator_dynamics     (fresh — 自己的 Q_applied + buffer energy 态)
├── supply_delay            (fresh CoolantTransportDelay — 自己的队列)
├── return_delay            (fresh CoolantTransportDelay — 自己的队列)
└── evaporator_heat_current (EvaporatorHeatCurrent — 零状态)
```

**关于"共享 vs fresh"的判断标准**：

| frozen 类 | dynamic_state_count | 共享 | 理由 |
|---|---|---|---|
| `CoolantPump` | 0 | ✔ 共享 | `solve_operating_point` 是纯函数 |
| `ClosedR134aCycle` | 0 | ✔ 共享 | `solve` 是纯函数 |
| `ParallelHeaderHydraulicNetwork` | n/a | ✔ 共享 | 仅承载几何阻力数据，运行时不变 |
| `CoolantTank` | 1 | ✘ fresh | `temperature_k` 是 mutable |
| `CompressorSpeedActuator` | 1 | ✘ fresh | `speed_rpm` 是 mutable |
| `EvaporatorThermalDynamics` | 1 | ✘ fresh | `q_evap_applied_w` + `evaporator_buffer_energy_j` 是 mutable |
| `CoolantTransportDelay` | steps | ✘ fresh | `_queue` 是 mutable |

`build_independent_hc_plant(legacy_plant)` 把 fresh 实例的**初始状态**从 legacy 当前
值拷贝过来；从 step 2 开始两个系统**完全独立递推**。

## 4. `HeatCurrentPlant.step()` — 9 步物理链

```text
1. operating_point = self.pump.solve_operating_point(pump_speed_rpm, network)
2. actuator_result = self.compressor_actuator.step(dt, compressor_command_rpm)
3. refrigeration_result = self.refrigeration_cycle.solve(
       compressor_speed_used_rpm, fan_rpm,
       tank_temperature_before, total_mass_flow, ambient_temperature_k
   )
   - low-pressure protection: 0.9→0.2 geometric re-try（与 plant.py:402-423 完全一致）
4. evaporator_result = self.evaporator_dynamics.step(dt, q_evap_cycle_w)
   - q_evap_applied = evaporator_result["q_evap_applied_w"]
   - evaporator_outlet = T_tank_before - q_evap_applied / (ṁ · cp)
5. cluster_supply = self.supply_delay.step(evaporator_outlet)
6. cluster_result = self.cluster.step(
       dt, cluster_current_a, cluster_supply, total_mass_flow, ambient, direction
   )
   - cluster_return = cluster_result["return_temperature_k"]
7. tank_return = self.return_delay.step(cluster_return)
8. tank_result = self.tank.step(dt, tank_return, total_mass_flow)
9. diagnostics — 9 个 Q + 4 个 T + 2 个 queue + compressor 状态
```

**与 `ClusterPlant.step()` 的关系**：物理顺序**逐行对齐**，物理公式**完全复用** frozen 模块。
**没有新增** `min(Q_cycle, Q_HC)` / `C_e · dT_e/dt` / 新泵参数 / 新 HTC 标定。

## 5. 独立性测试（任务书 6 条对账）

| 任务书条款 | 测试 | 结果 |
|---|---|---|
| ① 分别构造 Legacy 和 HeatCurrentPlant | `test_hc_initial_state_seeded_from_legacy` | ✔ 初始状态一致 |
| ② 给完全相同初始条件 | 同上 | ✔ |
| ③ 只 step HeatCurrentPlant，Legacy 状态 bitwise 不变 | `test_hc_step_does_not_mutate_legacy_state` | ✔ 8 个动态字段全部 bit-equal |
| ④ 只 step Legacy，HeatCurrentPlant 状态不变 | `test_legacy_step_does_not_mutate_hc_state` | ✔ 7 个字段全部 array_equal |
| ⑤ 连续多步后 HC 的 tank/delay/cluster/compressor/q_evap_applied 来自自身上一时刻 | `test_hc_step_uses_own_previous_state` | ✔ 两套独立 HC + 同一 legacy 起点 + 不同 legacy 轨迹 → 两个 HC bit-identical |
| ⑥ 全量 regression 通过 | 256/256 | ✔ |

## 6. 与 Stage 3 边界的对比

| 维度 | Stage 3 `HeatCurrentSystemLink` | Stage 4.5 `HeatCurrentPlant` |
|---|---|---|
| 上游量从 legacy 喂 | ✔（9 个 inputs） | ✘ 完全独立 |
| 物理链完整性 | 1-2 + 5-8（缺 3-4） | 1-9 完整 |
| 适用范围 | shadow comparison | 独立系统验证 |
| 继续维护 | ✔ | ✔ |

两个对象**可以共存**：Stage 3 适合"在相同上游边界条件下做局部系统对照验证"，Stage 4.5
适合"在相同外部输入条件下做独立全系统动态验证"。两者证据互补。

## 7. 状态

- ✅ 核心模块已落：`thermal/heat_current_plant.py`
- ✅ 独立性测试已落：`tests/test_heat_current_plant_independence.py`，5 用例全过
- ✅ 全量回归 256 / 256 通过
- ✅ Frozen 模块未动一字
- ⏸ 提交 + tag `heat-current-independent-v1`（紧随本报告后）

## 8. 是否可以进入 Stage 5

**可以。**

独立 HC Plant 已具备 Stage 5 任务书要求的两个前提：

1. ✔ 完整 9 步物理链（pump → comp → cycle → evap_dynamics → HC_evap → supply_delay →
   HC_cluster → return_delay → tank），全部由 HC plant 自身状态驱动
2. ✔ 外部输入接口 `HeatCurrentPlantInputs` 与 `ClusterPlantInputs` 字段完全对齐
   （`cluster_current_a`, `compressor_command_rpm`, `pump_rpm`, `fan_rpm`,
   `ambient_temperature_k`, `flow_direction`），允许两个系统在**完全相同外部输入**下
   做独立递推对照

Stage 5 任务书的 6 / 7 / 8 / 9 节指标（系统温度 RMSE、累计能量、空间梯度、动态因果）
全部可以基于 `HeatCurrentPlant.step()` 输出计算，无需再访问 legacy。

---

**报告作者**：Codex / Claude 会话
**报告时间**：2026-09-06
**依据**：`thermal/heat_current_plant.py`（10 个 frozen 模块 import + 4 个 fresh 实例 + 9 步 step）、
`tests/test_heat_current_plant_independence.py`（5 用例 35 s）