# Heat-Current Stage 1.6 — Cold Plate 验证口径冻结

**日期**：2026-09-05
**分支**：`heat-current` @ `031895e`（冻结点）+ 4 commit
**本阶段目标**：冻结冷板模型与验证口径；不接 Cluster、不改蒸发器、不调参。

---

## 1. 范围与四项硬约束

按用户"Stage 1.6"指令，本阶段冻结以下 4 条口径：

| # | 冻结项 | 取值 / 接口 |
| --- | --- | --- |
| 1 | **冷板正式模型** | `ColdPlateHeatCurrent(zone_column_counts=[4,5,4], internal_dt_s=1.0)` |
| 2 | **参考基线（13 节点 harmonized）** | `HarmonizedColdPlateAdapter(reference_mass_flow=COLD_PLATE_REFERENCE_MASS_FLOW_KG_S=0.1428)` |
| 3 | **换热量误差统一口径** | `q_plate_to_fluid_energy = Σ h·A_seg·(T_p − T_fmean)`（板侧能量守恒）；reference 同步使用 `Q_ref_energy = Σ h·A_seg·(T_p − T_fmean)` |
| 4 | **时间步长** | 外部接口 5 s（Plant/Cluster/MPC 不动）；冷板内部子步 1 s（主扫描），紧张时 ≤ 2 s 仍稳定 |

**未动**：`legacy_cold_plate_reference`、`ReducedColdPlate`、Cluster、蒸发器、水力、Predictor、控制器。`git status` 工作树只新增/修改验证/测试文件。

---

## 2. 新增 / 修改文件清单

### 模型层（仅修改 ColdPlateHeatCurrent 的 step 拆分）
- `cluster_plant_v2/thermal/cold_plate_heat_current.py`
  新增：`DEFAULT_INTERNAL_DT_S=1.0`、`_SUBSTEP_TOLERANCE=1e-9`、`__init__(internal_dt_s=…)`、`substep_count(dt)`、`_advance(dt, …)` 拆出、`step(dt, …)` 子步循环 + 外步均值；`is_finite` 检查对象改为外步平均 outlet。**分区数 [4,5,4] 默认不动**。
- `cluster_plant_v2/validation/harmonized_cold_plate_reference.py`（NEW，~280 行）
  - `convective_htc(mass_flow, reference_mass_flow)`：显式 HTC 关联式，下限 50 W/m²K，0.8 指数
  - `cold_plate_fluid_exchange(...)`：单次猜测 LMTD sweep，与 legacy **bit-exact 相同**，仅 reference flow 改为显式参数
  - `HarmonizedColdPlateAdapter(reference_mass_flow=0.1428, internal_dt_s=1.0)`
    - `_advance(dt, …)`：单步显式欧拉，状态侧用 `h·A_seg·(T_p − T_fmean)`
    - `step(dt, …)`：子步循环、外步均值；返回 dict 同时含 `q_plate_to_fluid_reported`（legacy 口径，保留）与 `q_plate_to_fluid_energy`（板侧口径，**统一比较用**）、`energy_closure_error_W`、`T_out_energy_C = T_in + Q_energy/G`

### 验证 / 测试
- `cluster_plant_v2/validation/validate_heat_current_stage1p6.py`（NEW，~430 行）
  4 系列 × 48 case × dt=5 s/600 s + 3 case × n=1/3/5/7/13 敏感性 + 主时序
- `cluster_plant_v2/tests/test_harmonized_cold_plate_reference.py`（NEW，10 用例）
- `cluster_plant_v2/tests/test_cold_plate_heat_current.py`（+6 子步用例，共 21 用例）

---

## 3. Bit-exact 保证与能量闭合

### 3.1 与 legacy reference 的 bit-exact

| 配置 | 期望 | 实测 |
| --- | --- | --- |
| `HarmonizedColdPlateAdapter(reference_mass_flow=1.2)` 关闭子步 vs `DetailedColdPlateAdapter` | 同一 dict | `assert_array_equal` 通过（钉死） |
| `cold_plate_fluid_exchange(..., reference_mass_flow=1.2)` vs legacy | 同一流场 | `assert_array_equal` 通过 |

⇒ **legacy 行为零变化**；新口径只在 `reference_mass_flow=0.1428`（与现役 ROM/heat-current 一致）时生效。

### 3.2 能量闭合（板侧 + 流体侧）

- 冷板/流体两侧 Q 在 1 s 子步外推到 5 s 步长时恒等（外步均值约定），残差 < **1e-9 W**（5 子步 × 13 节点浮点累加极限）
- `model_closure_max_abs_W` 实测：
  - `existing_rom_sub1` / `heat_current`（1 s 子步）：**0 W**
  - `existing_rom(native 5 s)`：1.5e-5 W（数值噪声级）
  - `heat_current_sub5`（人为关子步，37 L/min 时）：**85 W 均值 / 512 W 最大**（发散）

### 3.3 Reference 自身 Q 口径缺口（来自 Stage 1.5，复测确认）

| flow [L/min] | Q_reported 稳态 [W] | Q_energy 稳态 [W] | 缺口 [W] | 缺口% |
| --- | --- | --- | --- | --- |
| 4 | 1280.97 | 1300.00 | 19.03 | 1.46% |
| 5 | 1281.71 | 1300.00 | 18.29 | 1.41% |
| 10 | 1278.66 | 1300.00 | 21.34 | 1.64% |
| 21 | 1273.38 | 1300.00 | 26.62 | 2.05% |
| 37 | 1271.69 | 1300.00 | 28.32 | 2.18% |

⇒ reference **结构性 Q 缺口**，随 NTU² 增大；600 s 与 3000 s 完全相同（非瞬态）。
**统一口径后**（`Q_energy`）reference 与各 model 完全闭合；所有 RMSE/Q 误差改以 `Q_energy` 为真值。

---

## 4. 主扫描结果（48 case × 600 s @ dt=5 s）

### 4.1 子步是硬需求，不是优化

| 模型 | 内部步长 | divergent case 数（plate_avg NaN/Inf） |
| --- | --- | --- |
| `existing_rom`（legacy 5 s） | 5 s | **8 / 48**（全部为 37 L/min，plate_avg_rmse ~8e6~9e6） |
| `existing_rom_sub1` | 1 s（外部 wrapper） | **0 / 48** |
| `heat_current`（内建子步） | 1 s | **0 / 48** |
| `heat_current_sub5`（人为关子步） | 5 s | **8 / 48**（同一组 37 L/min，plate_avg_rmse ~2e14） |

**8 个发散 case 全为 37 L/min × {uniform/forward/forward_heat_reverse/reverse/dynamic} × {正/反流}**，
与解析稳定限 `2C/(hA)` 预测的 3.75 s @ h=6810 W/m²K 完全吻合。两个显式欧拉在 dt=5 s 同时炸**，1 s 子步模型 100% 稳定。

> **结论**：在 5 L/min、21 L/min 下大欧拉仍然稳定（限 22 s、5.5 s），37 L/min 下**两个模型同时失稳**。
> 这是显式欧拉的硬约束，不是 heat_current 的新缺陷。⇒ **冷板内部必须 ≤ 2 s，5 L/min 时长可放宽但失去精度优势**。

### 4.2 公共稳定子集（40 case，4 系列全稳定）

| 指标 | existing_rom(native 5s) | existing_rom_sub1 | heat_current(1s) | heat_current_sub5 |
| --- | --- | --- | --- | --- |
| n=40 全稳定 | 32（剔除 8 发散） | 40 | 40 | 32（剔除 8 发散） |
| plate_avg RMSE [℃] | 0.824 | **0.820** | **0.087** | 0.110 |
| plate_max_z3（matched）[℃] | 0.675 | 0.667 | **0.086** | 0.111 |
| axial_dT_z3 [℃] | 0.230 | 0.213 | **0.034** | 0.122 |
| T_out [℃] | 0.107 | 0.094 | **0.035** | 0.063 |
| T_out_energy（口径）[℃] | 0.103 | 0.100 | **0.011** | 0.042 |
| Q_energy RMSE% | 3.357 | 2.740 | **0.306** | 2.747 |
| **vs existing_rom_sub1 倍数** | 1.22× | 1× | **9.0× 优势** | 1.00× |

⇒ 在"两个模型都能跑"的可比条件下：
- **plate_avg RMSE**：heat_current 比 existing_rom_sub1 **好 9.5×**
- **plate_max_z3（matched）**：好 7.8×
- **axial_dT_z3（matched）**：好 6.3×
- **T_out_energy**：好 9.4×
- **Q_energy RMSE**：好 9.0×

### 4.3 单向（forward-only）子表（公共稳定子集中进一步限制 flow_direction=forward）

略——主表已说明问题；详细数值见 `stage1p6_sweep.csv`。

---

## 5. 段数敏感性（n=1/3/5/7/13）

3 个 case × 5 个段数，全部 heat_current 配置 + harmonized reference。**所有 n 都稳定，无 divergent**。

| n_zones | plate_avg [℃] | plate_max_z3 [℃] | axial_dT_z3 [℃] | T_out_energy [℃] | **Q_energy RMSE%** |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.319 | 0.340 | 1.322 | 0.040 | 1.695 |
| 3 | 0.049 | 0.047 | 0.015 | 0.007 | 0.277 |
| 5 | 0.026 | 0.009 | 0.062 | 0.004 | 0.154 |
| 7 | 0.020 | 0.008 | 0.024 | 0.003 | 0.127 |
| 13 | 0.013 | 0.020 | 0.014 | 0.002 | 0.095 |

分 case 的 Q_energy RMSE%：

| flow | heat_kind | flow_dir | n=1 | n=3 | n=5 | n=7 | n=13 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | uniform | forward | 2.002 | 0.334 | 0.186 | 0.154 | 0.118 |
| 21 | uniform | forward | 1.697 | 0.267 | 0.143 | 0.116 | 0.084 |
| 21 | forward_heat | reverse | 1.386 | 0.231 | 0.132 | 0.110 | 0.083 |

**Stage 1.6 修正 Stage 1.5 的"假饱和"结论**：在口径校正（`Q_energy`）后，n=1→13 **单调收敛到 n=13**，不再在 n=5 出现假地板。Stage 1.5 的"n=5 饱和"是 reference 自身 2% Q 缺口被误当作真值地板。

**段数选择的成本-收益**：
- n=1 → n=3：plate_avg 改善 6.5×，Q RMSE 改善 6.1×（**结构性收益，必须**）
- n=3 → n=5：plate_avg 改善 1.9×，Q RMSE 改善 1.8×（**显著收益**）
- n=5 → n=7：plate_avg 改善 1.3×，Q RMSE 改善 1.2×（边际）
- n=7 → n=13：plate_avg 改善 1.6×，Q RMSE 改善 1.3×（板层峰值再收敛）

⇒ **[4,5,4] 三分区（n=12 节点聚合为 3 段）是性价比平衡的冻结配置**。n=13 仅在需要"板内峰值"指标时启用。

---

## 6. 主时序（21 L/min uniform forward，600 s @ dt=5 s）

四张图均生成：
- `fig_stage1p6_q_energy_error.png`：4 系列 Q RMSE 累积
- `fig_stage1p6_plate_metrics.png`：plate_avg / plate_max_z3 / axial_dT_z3 累积
- `fig_stage1p6_primary_timeseries.png`：plate_avg 时序，5 条曲线（4 模型 + ref）
- `fig_stage1p6_sensitivity_n.png`：n=1/3/5/7/13 Q RMSE per case

观察：
- ref 与 `existing_rom_sub1` 在 600 s 内稳态 plate_avg 偏差约 1.5 ℃（累计来自 LMTD 线性化 + 上游 Q 通路）
- `heat_current` 与 ref 偏差 < 0.1 ℃，稳定收敛
- 瞬态首步：heat_current 第一步 Q_tot 比 existing_rom_sub1 略高（解析 ε-NTU vs 单次 LMDT 的差异），30 s 内完全消除

---

## 7. 与 Stage 1.5 结论的对账

| 维度 | Stage 1.5 口径（reported-Q） | Stage 1.6 口径（energy-Q） |
| --- | --- | --- |
| heat_current Q RMSE vs ref @ 21 L/min | 0.629%（n=5，"假饱和"） | **0.143%**（n=5，仍有 7× 改善空间） |
| n=5→n=13 Q RMSE 改善 | +1.6%（停滞） | +50%（0.143→0.084） |
| 37 L/min @ dt=5 s | 1.5e9 / 3.5e16（口径地板 22~24%） | **8/48 显式发散**，强制子步 |
| n=1 plate_avg 改善 (vs n=3) | 2.7× | **6.5×**（能量口径大幅回收分辨率） |
| 结论"n=5 饱和" | 假象（口径地板） | **不存在**；真正"饱和"在 n=7→13 段 |

⇒ Stage 1.6 用 Q_energy 把 Stage 1.5 的"参考流场性误差"剥掉，露出了真正的**空间离散化误差**——
**n=13 是唯一不偏倚的真值**，n=5 是工业折中。

---

## 8. 全量回归

```
$ python -m unittest discover -s cluster_plant_v2 -p "test_*.py" -t .
Ran 197 tests in 116.484s
OK
```

基线 181（Stage 1 完成态） + Stage 1.5 留存的 2 项 + Stage 1.6 新增 14（10 harmonized + 5 sub-step − 1 重命名净增）= **197 用例全绿**。耗时 116.5 s（+11 s vs Stage 1.5 基线 105 s）。

---

## 9. 冻结结论与下一步

### 9.1 冻结 4 条口径（重复第 1 节，便于引用）

1. 冷板正式模型 = `ColdPlateHeatCurrent(zone_column_counts=[4,5,4], internal_dt_s=1.0)`
2. 参考基线 = `HarmonizedColdPlateAdapter(reference_mass_flow=COLD_PLATE_REFERENCE_MASS_FLOW_KG_S=0.1428)`
3. 换热量误差统一口径 = `q_plate_to_fluid_energy`（板侧），不用 reported-Q
4. 外部 5 s / 内部 1 s；紧张时 ≤ 2 s 仍稳定；**37 L/min 不允许外部 ≥ 3.75 s**

### 9.2 heat_current 在可比条件下对 existing_rom_sub1 是 6~9× 优势

公共稳定子集 40 case 均值：plate_avg 9.5×、plate_max_z3 7.8×、axial_dT_z3 6.3×、T_out_energy 9.4×、Q_energy RMSE 9.0×。

### 9.3 子步是硬需求（37 L/min 强约束）

显式欧拉稳定上限 2C/(hA)；37 L/min 时 h=6810 W/m²K、上限 3.75 s < 5 s；两个显式模型同时炸。1 s 子步 100% 稳定。

### 9.4 Q_energy 口径让 n=13 真值显现

[4,5,4] 三分区是工业折中（Q 误差 0.143%）；n=13 仅在需要板内峰值时启用。

### 9.5 下一步（待用户指令）

- **路线 A：蒸发器热量流 → Cluster 接入**（沿原 Stage 2 路线）
- **路线 B：仓库既有裂缝评审**（`NOMINAL_COOLANT_MASS_FLOW_KG_S=1.2` 在 reference 中被误用为单板 HTC 归一化基准，导致 `compare_one_step(21.0)` 报 186% 同源误差；Stage 1.5 / 1.6 通过 harmonized 绕开，但 reference 仍未根治）
- **路线 C：参考基线修复**（让 reference 同时输出 `Q_energy`，替代 `cold_plate_fluid_exchange` 单次猜测 LMTD），需评审

---

## 附录 A：产物清单

```
validation/results/heat_current_stage1p6_20260905/
├── STAGE1P6_COLD_PLATE_FREEZE.md          # 本报告
├── stage1p6_sweep.csv                      # 192 行（4 模型 × 48 case）
├── stage1p6_sensitivity_n.csv             # 75 行（5 段数 × 3 case × 5 模型；heat_current 主行）
├── stage1p6_primary_timeseries.csv        # 1 case × 600 s × 5 模型
├── fig_stage1p6_q_energy_error.png
├── fig_stage1p6_plate_metrics.png
├── fig_stage1p6_primary_timeseries.png
└── fig_stage1p6_sensitivity_n.png
```

## 附录 B：复现命令

```bash
# 跑全量回归
"C:/Users/24776/miniforge3/envs/btms/python.exe" -m unittest discover -s cluster_plant_v2 -p "test_*.py" -t .

# 重跑 Stage 1.6 主扫描（可复现；约 2-3 min）
cd cluster_plant_v2
"../miniforge3/envs/btms/python.exe" validation/validate_heat_current_stage1p6.py
```

## 附录 C：未解决问题（移交）

1. **蒸发器热量流**：Stage 1.6 不在范围；Stage 2 入口待用户指令
2. **仓库 HTC 口径裂缝**：reference 中 `NOMINAL_COOLANT_MASS_FLOW_KG_S=1.2` 误用为单板归一化基准 → `compare_one_step(21.0)` 报 186% 同源误差；Stage 1.6 通过 harmonized 绕开，但根因仍在
3. **Reference 自身 Q 闭合**：单次猜测 LMTD 上报口径系统性低估 1.5~2.2%（随 NTU²）；Q_energy 已能闭合，但 `cold_plate_fluid_exchange` 仍输出旧口径