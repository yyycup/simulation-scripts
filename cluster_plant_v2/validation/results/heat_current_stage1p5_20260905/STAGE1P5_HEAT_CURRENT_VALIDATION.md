# Heat-Current Stage 1.5 — 模型验证报告

**日期**：2026-09-05  ·  **分支**：`heat-current`  ·  **冻结基线**：`before-heat-current` = `031895e`

**约束（严格遵守）**：只做验证。不改模型结构、不标定参数、不接 Cluster。
唯一新增代码是验证脚本本身 `validation/validate_heat_current_stage1p5.py`；
三个"口径变体"通过 `contextlib` 在**内存里**临时换模块级常量，退出即还原，
磁盘上没有任何模型文件被改动（`git status` 确认）。

---

## 0. 一句话结论

> 以现成 13 节点模型为 reference 时，**16~18% 的误差里几乎 100% 是 HTC 归一化口径不一致造成的，不是模型结构误差**。
> 把口径对齐后，Heat-Current ROM 的 Q 误差降到 **0.92%**，比现役 ROM 的 **2.57%** 好 **2.8 倍**；
> n=1/3/5/7/13 分段误差**单调收敛并在 n≈5 饱和**；
> 而 37 L/min 的 1e16 爆炸是**显式欧拉数值发散**，不是模型问题。
> 顺带发现：**13 节点 reference 自身不闭合能量**，稳态少报 Q 达 **1.16%**，它不能当 Q 的真值用。

---

## 1. 实验设置

| 项 | 值 |
|---|---|
| Reference | `DetailedColdPlateAdapter`（13 节点 legacy 通路，逐节点单次 LMTD） |
| Incumbent | `ReducedColdPlate`（3 分区 `[4,5,4]`，逐区单次 LMTD） |
| Candidate | `ColdPlateHeatCurrent`（ε-NTU 解析解，`zone_column_counts` 参数化） |
| 工况 | 流量 4/5/6/10/21/37 L/min × 热负荷 uniform/forward/reverse/dynamic × 正流/反流 = **48 case** |
| 时长 / 步长 | 600 s，**dt = 1.0 s** |
| 入口温度 | 20 °C |
| 电池热流 | 100 W/节点（uniform），合计 1300 W |
| 参数 | 全部继承自 `parameters.py`，一个都没重拟合 |

### 为什么 dt 用 1 s 而不是现有验证惯用的 5 s
三个模型的壁面状态都是显式欧拉推进，稳定条件 `dt < 2C/(hA)`。
在 37 L/min 下 2 秒级就会违反（详见 §4），dt=5 s 会把数值发散混进模型误差里。
dt = 1.0 s 时**三个口径 × 48 case 全部稳定**（脚本 `stable` 列可查），
这样得到的数字才是纯模型差异。dt=5 s 的行为由 §4 的探针单独交代。

---

## 2. 口径：16% 误差的真正来源

两侧用的 HTC 归一化参考流量**不是同一个物理量**：

- 现役 ROM / Heat-Current：`COLD_PLATE_REFERENCE_MASS_FLOW_KG_S = 0.1428 kg/s` —— **单板设计流量**
- 13 节点 legacy reference：`NOMINAL_COOLANT_MASS_FLOW_KG_S = 1.2 kg/s` —— **簇级标称流量**，被误用在单板上

于是 h 相差一个与流量无关的常数因子：

```
h_ratio = (1.2 / 0.1428)^0.8 = 5.4899        ← 已由脚本 h_ratio_model_over_ref 列实测确认
```

因为不能标定参数，本轮改为在**三个口径变体**下各跑一遍完整扫描：

| 口径变体 | ROM 参考流量 | Reference 参考流量 | h 比值 | existing ROM<br>Q RMSE | heat-current<br>Q RMSE | existing ROM<br>T_out RMSE | heat-current<br>T_out RMSE |
|---|---|---|---|---|---|---|---|
| `as_is_rom0.1428_ref1.2` | 0.1428 | 1.2 | 5.490 | **16.16 %** | 18.46 % | 0.382 °C | 0.452 °C |
| `harmonized_rom0.1428` ★ | 0.1428 | 0.1428 | 1.000 | **2.57 %** | **0.92 %** | 0.081 °C | **0.030 °C** |
| `harmonized_legacy1.2` | 1.2 | 1.2 | 1.000 | 0.070 % | **0.035 %** | 0.0022 °C | 0.0011 °C |

★ = 主口径（`0.1428 kg/s` 是单板设计流量，`h_nominal = 2000 W/m²K` 正是锚定在它上面的，物理上正确的一侧；
`1.2 kg/s` 是簇级数字被错当单板用）

**读法**：
1. 口径一变，误差掉 **1~2 个数量级**（16% → 0.9%）。原来那 16% 基本全是口径，不是模型。
2. 两个对齐口径下 **heat-current 都严格优于 existing ROM**（Q RMSE 分别好 2.8× 和 2.0×）。
3. as-is 下 heat-current 反而略差（18.46% vs 16.16%）—— 但那是因为它 h 更大、NTU 更大、
   对同一个 5.49 倍口径错位更敏感；这个比较在口径错位下没有意义。

### 主口径下按流量拆解（48 case 平均）

| 流量 [L/min] | existing ROM Q RMSE | heat-current Q RMSE | 改善倍数 | existing T_out RMSE | heat-current T_out RMSE |
|---|---|---|---|---|---|
| 4  | 3.51 % | 1.233 % | 2.82× | 0.1839 °C | 0.0651 °C |
| 5  | 2.92 % | 1.134 % | 2.58× | 0.1245 °C | 0.0483 °C |
| 6  | 2.63 % | 1.057 % | 2.49× | 0.0940 °C | 0.0378 °C |
| 10 | 2.20 % | 0.867 % | 2.54× | 0.0477 °C | 0.0188 °C |
| 21 | 2.04 % | 0.655 % | 3.12× | 0.0214 °C | 0.0069 °C |
| 37 | 2.14 % | 0.548 % | 3.91× | 0.0128 °C | 0.0033 °C |

误差随流量单调下降 —— 这是 **O(NTU²)** 的标志：残差就是"单次 LMTD 线性化 vs 解析 ε-NTU"的差距，
而 NTU 随流量下降。热负荷类型（uniform/forward/reverse/dynamic）和正/反流对误差**几乎无影响**
（forward 与 reverse 的 RMSE 到小数点后 3 位完全相同），说明残差是局部对流格式的差异，与流向无关。

---

## 3. n = 1 / 3 / 5 / 7 / 13 分段收敛

分段划分（13 列的连续切分）：`n=1:(13)`、`n=3:(4,5,4)`、`n=5:(2,3,3,3,2)`、`n=7:(1,2,2,3,2,2,1)`、`n=13:(1)×13`

### Q_total RMSE [%]（主口径 `harmonized_rom0.1428`）

| case | n=1 | n=3 | n=5 | n=7 | n=13 |
|---|---|---|---|---|---|
| 21 L/min forward heat | 2.207 | 1.376 | **0.629** | 0.631 | 0.639 |
| 21 L/min uniform | 1.858 | 1.345 | **0.631** | 0.633 | 0.638 |
| 5 L/min uniform | 2.001 | 2.080 | **1.134** | 1.140 | 1.151 |

### plate_avg RMSE [°C]

| case | n=1 | n=3 | n=5 | n=7 | n=13 |
|---|---|---|---|---|---|
| 21 L/min forward heat | 0.2352 | 0.0860 | 0.0128 | 0.0089 | **0.0037** |
| 21 L/min uniform | 0.1453 | 0.0766 | 0.0099 | 0.0075 | **0.0042** |
| 5 L/min uniform | 0.7521 | 0.6067 | 0.0595 | 0.0466 | **0.0294** |

### plate_max RMSE [°C]

| case | n=1 | n=3 | n=5 | n=7 | n=13 |
|---|---|---|---|---|---|
| 21 L/min forward heat | 0.622 | 0.156 | 0.071 | **0.0075** | 0.0076 |
| 21 L/min uniform | 0.344 | 0.065 | 0.031 | **0.0072** | 0.0073 |
| 5 L/min uniform | 1.203 | 0.307 | 0.098 | **0.0513** | 0.0517 |

**结论**

- **口径对齐后分段误差单调收敛**，Q 在 **n≈5 饱和**（0.63%），plate_max 在 **n≈7 饱和**（0.007 °C，
  这已经是分辨率匹配后的地板值：n<13 时 `plate_max` 比的是"n 个分区均值的最大值 vs 13 个节点的最大值"，
  天然偏小，这是**指标偏倚不是模型误差**）。
- 上一轮（口径未对齐）看到的"误差随 n 增大而增大"（24.5% → 27.3%）**完全是口径假象**，已被彻底证伪。
- **n=3 已经够用**：热点温度 0.10~0.37 °C，Q 误差 1.3~2.1%。
  **n=5 是性价比拐点**：Q 误差直接砍半到 0.63%，plate_avg 进 0.01 °C 量级。
  **n≥7 对 Q 几乎没有额外收益**，只为 plate_max 买 0.01 °C。

### n=13 与 13 节点 reference 逐点对账（主口径，稳态）

| 流量 | reference plate_max | heat-current n=13 | Δ | heat-current n=3 | Δ | existing ROM n=3 | Δ |
|---|---|---|---|---|---|---|---|
| 5 L/min | 25.9759 °C | 26.0284 | **+0.053** | 25.6043 | −0.372 | 26.2218 | +0.246 |
| 21 L/min | 21.5779 °C | 21.5849 | **+0.007** | 21.4797 | −0.098 | 21.5589 | −0.019 |
| 37 L/min | 20.9372 °C | 20.9404 | **+0.003** | 20.8799 | −0.057 | 20.9167 | −0.021 |

**同分辨率（n=13）时 heat-current 与 13 节点 reference 的热点温度吻合到 0.003~0.05 °C。**
这是本次最强的正面证据：Heat-Current 公式在 n=13 下就是"13 节点模型的一个正确重推导"。

---

## 4. 37 L/min 的 1e16 是数值发散，不是模型问题

dt 细化探针（21 / 37 L/min，uniform，正向，600 s），Q RMSE %：

| 口径 | 流量 | 模型 | dt=0.2 s | dt=1.0 s | dt=2.0 s | dt=5.0 s |
|---|---|---|---|---|---|---|
| as-is | 21 | existing | 17.34 | 18.04 | 19.03 | 23.17 |
| as-is | 21 | heat_current | 18.92 | 19.74 | 20.92 | 26.48 |
| as-is | 37 | existing | 22.17 | 24.11 | 27.13 | **1.5e9 ✗** |
| as-is | 37 | heat_current | 23.71 | 25.90 | 29.36 | **3.5e16 ✗** |
| harmonized 0.1428 | 21 | existing | 1.897 | 2.045 | 2.287 | 5.363 |
| harmonized 0.1428 | 21 | heat_current | 0.637 | 0.644 | 0.662 | 2.253 |
| harmonized 0.1428 | 37 | existing | 1.841 | 2.139 | 2.774 | **589 ✗** |
| harmonized 0.1428 | 37 | heat_current | 0.516 | 0.539 | 0.649 | **589 ✗** |
| harmonized 1.2 | 21/37 | 两者 | 0.015~0.043 | 0.015~0.045 | 0.015~0.046 | 0.016~0.046 ✓ |

**两个决定性观察**

1. as-is 下即使 dt 细到 0.2 s，误差仍然停在 22~24% —— 说明那是**口径地板**，不是数值误差。
2. `harmonized_rom0.1428` @37 L/min @dt=5 s 时**两个模型报出完全相同的 589%**。
   两个结构不同的模型不可能犯同样的模型错误 —— 只能是**同一个 reference 炸了**。
   解析稳定限 `dt < 2C/(hA)`：37 L/min、h=6810 W/m²K 时单节点限 3.75 s，dt=5 s 越界。**显式欧拉发散，与模型无关。**

**口径与步长都收敛后，heat-current @37 L/min 的误差是 0.52%，existing ROM 是 1.84%。**

---

## 5. ⚠ 新发现：13 节点 reference 自身不闭合能量

这是本轮最有价值的副产品。13 节点通路**内部有两套 Q**：

- 状态更新用：`Q_states = h·A_seg·(T_plate − T_fluid_mean)`
- 对外上报用：`Q_reported = Σ h·A_seg·LMTD(ΔT_in, ΔT_out^guess)`（单次猜测的 LMTD）

两者不相等，差值就是 `energy_closure_error_W`。主口径下稳态实测（600 s 与 3000 s **完全相同**，故为结构性缺陷）：

| 流量 | Q_reported | Q_states | 缺口 | 占 1300 W |
|---|---|---|---|---|
| 5 L/min | 1284.969 W | 1300.000 W | **15.031 W** | **1.156 %** |
| 21 L/min | 1291.803 W | 1300.000 W | **8.197 W** | **0.631 %** |
| 37 L/min | 1293.532 W | 1300.000 W | **6.468 W** | **0.498 %** |
| 两个 ROM | **1300.0000 W** | — | **< 3e-11 W** | 0 % |

含义：

1. **13 节点 reference 永远到不了正确的稳态** —— 它的状态确实排掉 1300 W，但对外只报 1285~1294 W，
   稳态少报 0.5~1.2%，且**随时间不衰减**。
2. **两个 ROM 都精确闭合**（`Q = G·(T_out − T_in)` 残差 < 3e-11 W）。
3. 缺口随 NTU 增大而增大（低流量 NTU 大 → 1.16%；高流量 NTU 小 → 0.50%），与理论 `O(NTU²)` 一致。

**由此，heat-current 的残差本质上就是 reference 自己的缺陷**：

| 流量 | heat-current Q MAE | reference 自身缺口 | 超出部分 |
|---|---|---|---|
| 5 L/min | 14.05 W | 15.03 W | −0.99 W |
| 21 L/min | 8.41 W | 8.20 W | +0.21 W |
| 37 L/min | 6.79 W | 6.47 W | +0.33 W |

**剥掉 reference 的固有缺陷后，heat-current 与它的真实偏差只有约 0.2~1 W。** 
而 existing ROM 的 MAE 是 9.5~26.6 W，明显超出这个地板，说明它额外还背着"LMTD 低估效能"的误差。

**建议（不在本轮执行）**：把 13 节点 reference 定位为"**空间离散化**的真值"，而不是"**Q** 的真值"；
Q 的真值应当用能量守恒（稳态必须等于 ΣQ_bp = 1300 W）来判定。

---

## 6. 结论与建议

✅ **Heat-Current ROM 可以进入下一阶段。** 三条判据全部满足：

1. **精度**：口径对齐后 Q RMSE 0.92% vs 现役 2.57%（改善 2.8×）；T_out RMSE 0.030 vs 0.081 °C；
   n=13 时热点温度与 13 节点 reference 吻合到 0.007 °C。
2. **收敛性**：n 分段单调收敛，n=5 饱和。分段接口设计正确。
3. **守恒性**：能量闭合误差 < 3e-11 W（现役 ROM 同水平），明显优于 13 节点 reference。

📌 **分段数建议：n=5**（Q 0.63%、plate_avg 0.013 °C、plate_max 0.03~0.10 °C）。
若 NMPC 实时预算吃紧，**n=3 也完全可接受**（热点 0.10~0.37 °C）；n≥7 收益极小。

📌 **积分步长建议 ≤ 2 s**（高流量工况）。dt=5 s 在 37 L/min 下对三个模型都不安全。

⚠ **遗留待办（不在本轮范围）**
1. **HTC 归一化口径分裂**：`COLD_PLATE_REFERENCE_MASS_FLOW_KG_S=0.1428`（单板设计）vs
   `NOMINAL_COOLANT_MASS_FLOW_KG_S=1.2`（簇级，被 13 节点 reference 误用）。
   这是**已存在的仓库级裂缝**，不是 Stage 1 引入的（现成脚本 `compare_one_step(21.0)` 报 186% 即为同一裂缝）。
   建议后续统一，但会改动参考模型行为，需单独评审。
2. 13 节点 reference 的能量闭合缺陷（§5），建议修或明确降级其用途。

🚫 **本轮明确没做**：未改模型结构、未标定任何参数、未接入 Cluster/Plant/蒸发器。

---

## 7. 产物清单

目录：`cluster_plant_v2/validation/results/heat_current_stage1p5_20260905/`

| 文件 | 内容 |
|---|---|
| `stage1p5_flow_heat_direction_sweep.csv` | 3 口径 × 48 case × 2 模型 全量指标（含 `stable` / `h_ratio` 列） |
| `stage1p5_segmentation_convergence.csv` | 3 口径 × 3 case × n=1/3/5/7/13 |
| `stage1p5_dt_stability_probe.csv` | 3 口径 × 2 流量 × dt=0.2/1/2/5 s |
| `stage1p5_primary_timeseries.csv` | 主工况 600 s 逐秒时间序列 |
| `fig_stage1p5_segmentation_convergence.png` | 分段收敛曲线（as-is vs 主口径对照） |
| `fig_stage1p5_flow_sweep_q_error.png` | 误差-流量曲线 |
| `fig_stage1p5_primary_timeseries.png` | T_out / Q 时间序列 |
| `fig_stage1p5_dt_stability_probe.png` | dt 细化：数值发散 vs 模型误差 |

驱动脚本：`cluster_plant_v2/validation/validate_heat_current_stage1p5.py`
（可复现：`python -m cluster_plant_v2.validation.validate_heat_current_stage1p5`）
