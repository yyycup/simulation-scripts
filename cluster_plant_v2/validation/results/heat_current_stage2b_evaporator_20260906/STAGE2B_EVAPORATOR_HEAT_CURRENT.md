# Stage 2B — Evaporator Heat-Current 独立接口

日期：2026-09-06 · 分支：`heat-current` · 前置：Stage 2 只读审计
模块：`cluster_plant_v2/thermal/evaporator_heat_current.py`
测试：`cluster_plant_v2/tests/test_evaporator_heat_current.py`（22 例）
扫描：`cluster_plant_v2/validation/validate_evaporator_heat_current_stage2b.py`

---

## 0. 一句话结论

**Stage 2B 不是新模型，是一次边界切分。** 现役蒸发器本来就在用 ε-NTU，
热量流形式 `Q = (T_in − T_e)/R` 与它是**代数恒等式**（实测相对偏差 2.2e-16，
即双精度机器精度）。真正的收益是把这条关系从制冷循环里**剥离**成一个
无状态、可单测、可单独替换的单元，并把三个长期混用的 `Q` 显式分开。

---

## 1. 现役蒸发器的物理链

```
                    ┌─────────────── ClosedR134aCycle.solve() ───────────────┐
  N_comp ──┐        │  未知量 (T_e, T_cond, ṁ_r)，least_squares 三残差       │
           │        │    r1 = ṁ_r − ṁ_comp(N_comp, ρ_suction(T_e))   → 0    │
  T_c,in ──┼──►     │    r2 = Q_cycle(T_e, ṁ_r) − Q_NTU(T_e, UA_e)   → 0 ★  │
  ṁ_c   ───┤       │    r3 = Q_cond,refrig − Q_cond,NTU              → 0    │
  T_amb ───┘       └────────────────────────────────────────────────────────┘
                                        │
                        T_e (解出)      │      UA_e (解出，依赖 ṁ_r)
                                        ▼
                        ε = 1 − exp(−UA_e / G_c)
                        Q_NTU = ε · G_c · (T_c,in − T_e)
                                        │
                                        ▼
                        EvaporatorThermalDynamics  τ = 45 s
                                        │
                                        ▼
                              Q_applied  ──►  Plant 用这个
```

**关键事实：耦合在残差方程里，不是 `min()`。**

现役代码里没有任何 `min(Q_cycle, Q_NTU)`。求解器强制 `Q_cycle − Q_NTU = 0`，
`min()` 被**隐式满足**。自洽机制是双向的：

| 若 `Q_NTU < Q_cycle` | 后果 |
|---|---|
| `T_e` 下降 | ΔT = `T_c,in − T_e` 增大 → `Q_NTU` ↑ |
| 吸气密度下降 | `ṁ_r` 减小 → `Q_cycle` ↓ |

两侧相向而行，自动收敛。实测 259 个工况 `|Q_HC − Q_cycle|/Q_cycle ≤ 1.26e-10`，
远紧于求解器自身 5e-3 的关门阈值。

---

## 2. 45 s 到底是什么

**结论：经验性综合惯性，不是纯换热惯性。**

| 候选来源 | 判定 | 依据 |
|---|---|---|
| 纯金属热容 `C_e` | ❌ | 纯金属 `C_e ≈ 2 kJ/K` → `τ = C/UA ≈ 0.3 s`，比 45 s 小两个数量级 |
| 循环本身 | ❌ | `ClosedR134aCycle.dynamic_state_count = 0`，准静态 |
| 压缩机惯性 | ❌ | 已单独建模，`τ = 5 s` |
| 水箱 | ❌ | `C = 10.90 kJ/K` → `τ = 2.68 s @1.2 kg/s`，也远小于 45 s |
| **阀+回路综合惯性** | ✅ | 理想 TXV 无独立模型，热力阀感温包滞后典型 20–60 s，被吸收进 45 s |

**⇒ 现在加 `C_e·dT_e/dt` 就是对同一份惯性重复计数。** 这是本次明确推迟的项。
要拆，必须重标定，且要与 `test_evaporator_dynamics.py` 已钉死的
"1τ/3τ/5τ 精确响应"和"精确离散与步长无关"两条性质同时成立——不是小改动。

---

## 3. 新模块：状态 / 输入 / 输出

`EvaporatorHeatCurrent`，`dynamic_state_count = 0`（**零状态**）。

| 成员 | 公式 | 说明 |
|---|---|---|
| `capacity_rate(ṁ_c)` | `G_c = ṁ_c · c_p` | 输出 `G_c` |
| `resistance(G_c, UA_e)` | `R_e,c = 1 / (G_c · (−expm1(−UA_e/G_c)))` | 输出 `R_e,c` |
| `evaluate(...)` | 见下 | 输出全部四项 |
| `outlet_temperature_from_applied_heat(...)` | `T_out,applied = T_in − Q_applied/G_c` | 动态口径 |

`evaluate()` 输入（**全部由上层传入，模块自己不求解**）：

| 参数 | 来源 |
|---|---|
| `coolant_inlet_temperature_k` | **水箱温度**（不是 Cluster 回水） |
| `coolant_mass_flow_kg_s` | 水泵工况 |
| `evaporating_temperature_k` | `solve()` 解出的 `T_e` |
| `evaporator_ua_w_k` | `solve()` 解出的 `UA_e` |

`evaluate()` 输出：
`coolant_capacity_rate_w_k`、`evaporator_effectiveness`、
`evaporator_resistance_k_w`、`q_hc_w`、`coolant_outlet_temperature_ss_k`。

### 三个 Q 的分离（本次最重要的语义澄清）

| 符号 | 含义 | 谁用 |
|---|---|---|
| `Q_HC` | 换热器**稳态能力** | 本模块 |
| `Q_cycle` | 循环**稳态制冷量**；被求解器强制 = `Q_HC` | 诊断 |
| `Q_applied` | 经 45 s 滞后后**实际作用**在冷却液上的热 | **Plant 实际使用** |

对应两个出口温度，本模块**显式分开**：

```
T_out_ss      = T_in − Q_HC      / G_c     ← 稳态能力口径
T_out_applied = T_in − Q_applied / G_c     ← Plant 实际口径
```

冷启动 5 s 后两者相差可达 **15.93 K**——这个数就是 45 s 惯性的真实分量，
也说明把两个口径混为一谈会造成多大的误判。

---

## 4. `UA_e` 与 `T_e` 的来源与可信度

### `UA_e` — 串联热阻合成

```python
h_c = max(50, 2.0 × 0.70 × 2000  × (ṁ_c/1.2)^1.2)     # 冷却液侧
h_r = max(500, 2.0 × 0.70 × 10000 × (ṁ_r/0.02)^0.8)   # 制冷剂侧
UA_e = 1 / (1/(h_c·A) + 1/(h_r·A)),  A = 5.5 m²
```

| 参数 | 值 | 属性 |
|---|---|---|
| `EVAPORATOR_AREA_M2` | 5.5 | fiat（工程假设） |
| `EVAPORATOR_UA_FACTOR` | 2.0 | fiat |
| `EVAPORATOR_UA_SCALE` | 0.70 | fiat（污垢/非理想折减） |
| `h_c` 名义值 | 2000 W/(m²·K) | fiat |
| `h_r` 名义值 | 10000 W/(m²·K) | fiat |
| 流量指数 | 1.2（水侧）/ 0.8（制冷剂侧） | derivable（Dittus-Boelter 类关联式） |

### 精度不敏感（与冷板形成鲜明对比）

实测 ε = **0.938 … 0.974**（NTU ≈ 2.8–3.6），已深度饱和。
审计已量化：把 `ṁ_c` 和 `UA_e` **同时翻倍**，Q 只涨 **+7%**。

> **⇒ 蒸发器是"能力受限"，不是"面积受限"。** 冷板那边 5.49× 的 HTC 口径分裂
> 造成 16% 误差；这里 UA_e 的精度几乎不影响结果。**不要在 UA_e 上花标定预算。**

### `T_e` — 被求解的因变量

不是常数，不是输入。边界 `[280.15 K, T_c,in − 0.5]`，实测落在 **9.25 … 17.75 ℃**。

---

## 5. 耦合位置：热量流模型 ↔ R134a 循环

**唯一耦合点在 `Q_cycle − Q_NTU = 0` 这条残差上。**

新模块**故意不持有**这个耦合：它不调用 `solve()`，不持有制冷剂状态，
不知道压缩机、不知道膨胀阀。`T_e` 和 `UA_e` 由上层喂进来。

这样切分的好处：

- 循环求解器与换热器关系可以**分别**替换/标定/测试
- 没有反向依赖（AST 层面断言：模块不 import `refrigeration`，不出现 `.solve`）
- 后续若要把 ε-NTU 换成真正的分布参数模型，接口不变

---

## 6. 是否存在动态重复计数？

**不存在，并且已用零状态保证不会引入。**

| 惯性来源 | 状态数 | 时间常数 |
|---|---|---|
| `CompressorSpeedActuator` | 1 | 5 s |
| `EvaporatorThermalDynamics` | 1 | 45 s |
| `CoolantTank` | 1 | 2.68 s @1.2 kg/s |
| **`EvaporatorHeatCurrent`（新）** | **0** | — |

新模块 `dynamic_state_count = 0`，测试断言实例上不存在 `q_evap_applied_w`
之类的状态属性。**它只是纯函数式的代数映射，不可能重复计数。**

---

## 7. 与现役的偏差（259 工况扫描）

网格：`N_comp` ∈ {1200…4800} × `ṁ_c` ∈ {0.3, 0.6, 0.9, 1.2} ×
`T_c,in` ∈ {288.15…303.15} × `T_amb` ∈ {298.15, 308.15, 318.15}

工况统计：**尝试 280**，**成功 259**，**守卫跳过 56**，**闭合失败 21**（见 §8.2）。

| 指标 | 结果 | 判读 |
|---|---|---|
| **max \|Q_HC − Q_NTU\| / Q_NTU** | **2.175e-16** | ✅ **机器精度，恒等式成立** |
| max \|Q_HC − Q_cycle\| / Q_cycle | 1.263e-10 | ✅ 求解器闭合残差（非恒等式） |
| max \|G_c(T_in − T_out_ss) − Q_HC\| | 1.155e-10 W | ✅ 能量闭合 |
| max \|G_c(T_in − T_out_ap) − Q_ap\| | 1.149e-10 W | ✅ **用户要求验证的第二条恒等式** |
| ε 范围 | 0.9378 … 0.9741 | 深度饱和 |
| max \|T_out_ss − T_out_applied\| | 15.9304 K | 冷启动 5 s 后；45 s 惯性的分量 |

**逐点检查：CSV 首行 `err_q_hc_vs_ntu_rel = 0.0`**——直接 `evaluate_operating_point`
的工况下是**逐位相同**，连 1 ulp 都不差。

---

## 8. 本次发现的两个现役限制（均未修改）

### 8.1 求解器边界重叠 → 抛异常

条件：**冷却液入口温度接近或高于环境温度**时。
求解器边界 `T_cond ≥ T_amb + 0.5` 与 `T_e ≤ T_c,in − 0.5` 区间重叠，
迭代会踩进 `T_cond ≤ T_e` 非法区，抛
`ValueError: condensing temperature must exceed evaporating temperature`。

冷水机组总是向更热的环境排热，真实工况 `T_amb > T_c,in`，不会触发。
扫描改用 `T_amb ≥ T_c,in + 5 K` 守卫过滤（跳过 56 例）。

### 8.2 高转速 + 低流量 → 循环无法闭合（21 / 280 例）

失败模式不是异常，是 `solver_success = False`
（*"numerical solver converged but physical closure gates failed"*）。

**全部集中在 ṁ_c ≤ 0.6 kg/s 且 N_comp ≥ 2400 rpm。** 定点探针：

| ṁ_c [kg/s] | N [rpm] | T_e [K] | Q_NTU [W] | Q_cycle [W] | 残差 | 关门 |
|---|---|---|---|---|---|---|
| 0.3 | 1200 | 283.638 | 4304.1 | 4304.1 | 0.000% | ✓ |
| 0.3 | 2400 | **280.150** ↓ | 7649.6 | 7904.6 | 3.225% | ✗ |
| 0.3 | 3600 | **280.150** ↓ | 7654.2 | 9968.6 | 23.217% | ✗ |
| 0.3 | 4800 | **280.150** ↓ | 7657.1 | 11771.6 | **34.953%** | ✗ |
| 1.2 | 4800 | 283.318 | 19127.1 | 19127.1 | 0.000% | ✓ |

**机理：`T_e` 撞到下限 280.15 K 被钉死 → `Q_NTU` 饱和在
`G_c·(T_in − T_e,min) ≈ 7.65 kW`；而 `Q_cycle` 随转速线性增长到 11.8 kW。**
残差方程 `Q_cycle − Q_NTU = 0` 无解，物理上门限必然失败。

工程解读：压缩机排量 72 cc/rev，4800 rpm 时泵出的制冷剂流量远超
0.3 kg/s 冷却液能带走的热量。真机会表现为蒸发温度持续下降 →
**低压保护或蒸发器冻结停机**。模型里 `T_e` 被下限挡住，于是退化成"闭合失败"。

> 这是**现役模型的包络边界**，物理上对应真实机组的保护动作，
> 与新模块无关（新模块不参与求解）。按要求**未改 `refrigeration.py`**。
> 明细见 `evaporator_heat_current_stage2b_failures.csv`。

---

## 9. 硬性遵守的边界

| 约束 | 状态 |
|---|---|
| 不增加 `C_e` | ✅ `dynamic_state_count = 0` |
| 不改 45 s | ✅ 未触碰 `EvaporatorThermalDynamics` |
| 不改 `refrigeration.py` 原字段 | ✅ 零改动，含 `coolant_outlet_temperature_k` |
| 不接 Plant | ✅ 未触碰 `plant.py` |
| 模块内不调用 `solve()` | ✅ AST 层断言 |
| 保持现役制冷循环行为 | ✅ 回归 219/219 通过 |

---

## 10. 测试与回归

```
新建：cluster_plant_v2/tests/test_evaporator_heat_current.py   22 例
全量：219 tests in 61.7s  →  OK   （基线 197 + 新增 22）
```

测试覆盖：网格恒等式、求解边界恒等式、`R = 1/(G·ε)` 代数一致性、
ε 解析一致、两种出口口径能量闭合、与 Plant 公式逐位一致、
UA→0/UA→∞ 物理极限、NTU→0 的 `expm1` 数值良态、零状态、
AST 层反向依赖检查、输出键契约、全部入参守卫。

---

## 11. 已知遗留（有意推迟，非遗漏）

1. **§4.1 口径不一致**：`refrigeration.py:432` 的 `coolant_outlet_temperature_k`
   用 `Q_cycle` 算，而 `plant.py:433` 实际用 `Q_applied`。动态过程中两者不等。
   新模块已提供正确口径，但**未修改现役字段**（用户指示）。
2. **45 s 物理拆分**为 `C_e` + 阀动态 —— 需重标定，且要同时满足已钉死的两条性质。
3. **Plant 接入** —— 按用户意图，等蒸发器与冷板都就绪后统一集成，避免两轮回归。

---

## 12. 复现命令

```bash
# 从仓库根 集成仿真多种控制/ 执行
"C:/Users/24776/miniforge3/envs/btms/python.exe" -m unittest discover \
    -s cluster_plant_v2 -p "test_*.py" -t .

"C:/Users/24776/miniforge3/envs/btms/python.exe" -m \
    cluster_plant_v2.validation.validate_evaporator_heat_current_stage2b
```

产物：`evaporator_heat_current_stage2b_grid.csv`（259 行 × 20 列）、
`evaporator_heat_current_stage2b_summary.txt`。
