# 对话交接文档 · cluster_plant_v2 冷板管网拓扑研究

生成时间：2026-09-04
覆盖时段：2026-09-01 ~ 2026-09-04
用途：新开对话时把此文件作为上下文喂回，或粘贴文末的"开场白模板"

---

## 0. 三句话版本

1. 给 5-Pack 电池簇液冷管网做了**同程式 vs 异程式**的对照实验（新建 `DirectReturnHeaderNetwork` 子类实现异程式），跑完 1800 s 稳态。
2. 结论：异程式流量不均衡度 **13.78%**（同程 4.91%），Pack 间温差 **2.7×**，电池最高温 +0.10~0.11 K，**但泵功与 COP 差异 < 0.05%**——能耗上完全打平。
3. 顺带挖出同程式那 4.9% 残余的**公式级根因**：它保证的是"几何段数相等"（各 6 段）而非"水力阻力相等"（Σ R·ṁ²），二次阻力律让镜像路径的和呈拱形 [80,95,100,95,80]。

---

## 1. 项目背景

**cluster_plant_v2** = 五 Pack 电池簇储能热管理系统的高保真仿真工厂 + NMPC 验证平台。
双模型架构：高保真 Plant（CoolProp 真实 R134a 物性 + 5×13 电池 ROM）当"真实现实"，冻结的 16 状态 Physics-P 降阶模型当"控制器脑内世界"。

- 运行环境：`C:\Users\24776\miniforge3\envs\btms\python.exe`
- 跑测试（在**父目录** `集成仿真多种控制` 下）：
  `"C:/Users/24776/miniforge3/envs/btms/python.exe" -m unittest discover -s cluster_plant_v2 -p "test_*.py" -t .`
  基线 155 个用例全通过（新增后为 156）。

---

## 2. 本轮做的事：管网拓扑与流向对照实验

### 2.1 实验设计

| 变量 | 取值 |
|---|---|
| E1 冷板内流向 | 正向 1→2→3 / 反向 3→2→1 |
| E2 管网拓扑 | 同程式 reverse-return / 异程式 direct-return |
| 放电电流 | 560 A / 1120 A |
| 压缩机/风机/泵转速 | 4000 / 1200 / 3600 rpm（恒定）|
| 总管流量 | 各拓扑自配 `build_engineering_pump`，两者同为 **0.89964 kg/s** |

**关键隔离手段**：每种拓扑各配一台泵，使总管流量完全一致 → 拓扑成为唯一自变量。
注意：`ClusterPlant.__init__` 硬禁止 `hydraulic_mode == "header_network"`，所以异程式只能靠新建网络子类实现，不能靠现成开关。

### 2.2 异程式的实现方式

`DirectReturnHeaderNetwork(ParallelHeaderHydraulicNetwork)`，只重写 `_quantities()` 两行：

```python
# 父类（同程式）
return_flows        = np.cumsum(flows)
return_path_delta_p = np.cumsum(return_delta_p[::-1])[::-1]   # 后缀和

# 子类（异程式）
return_flows        = np.cumsum(flows[::-1])[::-1]
return_path_delta_p = np.cumsum(return_delta_p)               # 前缀和
```

阻力系数从 `build_medium_header_network()` 原样克隆，**只改拓扑**。

### 2.3 ★ 1800 s 稳态结果（2026-09-03 16:07 跑完，已落盘）

| case_id | flow_spread % | batt_max °C | inter_pack_dT K | cluster_dT K |
|---|---|---|---|---|
| E1_560A_forward（同程正向）| 4.908 | 16.484 | 0.1889 | 2.677 |
| E1_560A_reverse（同程反向）| 4.908 | 16.631 | 0.1886 | 2.918 |
| E2_560A_direct（异程）| 13.784 | 16.584 | **0.5115** | 2.943 |
| E1_1120A_forward | 4.908 | 31.888 | 0.2124 | 2.947 |
| E1_1120A_reverse | 4.908 | 31.921 | 0.2122 | 3.007 |
| E2_1120A_direct | 13.784 | 32.002 | **0.5742** | 3.243 |

**E2 拓扑（同程正向 → 异程正向）**
- 流量极差：4.908% → 13.784%（**2.81×**）
- Pack 间温差：**2.71×**（560 A）/ **2.70×**（1120 A）
- 电池最高温：+0.100 K（560 A）/ +0.114 K（1120 A）

**E1 流向（正向 → 反向）**
- 簇内温差：+0.241 K（560 A）/ +0.061 K（1120 A）
- 电池最高温：+0.147 K / +0.034 K
- Pack 间温差**几乎不变**（0.1889 → 0.1886）——反转让五个 Pack 同步升温，不加剧不均

**能耗**：泵功、电网侧功率、电网侧 COP 差异均 **< 0.05%**。异程式没省泵功，纯白送温差。

**比值全瞬态恒定**：120/300/600/1200/1800 s 的异程/同程温差比 ≈ 2.75/2.74/2.73/2.71/2.70。
流量分配是每步代数求解的**准静态量**，不均衡度是结构性常量；热瞬态只等比放大绝对值。

---

## 3. 机理推导（公式级，本轮最有价值的产出）

### 3.1 水力模型

二次阻力律 `Δp = R·ṁ²`。记号：N=5，f_i 支路流量，u_k/v_k 供/回液段流量。

$$u_k=\sum_{j=k+1}^{N}f_j \quad(\text{两种拓扑相同，}=[5,4,3,2,1]\bar f\ \text{于均流时})$$

$$v_k^{RR}=\sum_{j=1}^{k+1}f_j \qquad v_k^{DR}=u_k$$

$$P_i^{RR}=\sum_{k=0}^{i}\Delta p_{s,k}+\Delta p_{b,i}+\sum_{k=i}^{N-1}\Delta p_{r,k}
\quad\Rightarrow\quad \text{段数}= (i+1)+(N-i)=\mathbf{N+1=6}$$

$$P_i^{DR}=\sum_{k=0}^{i}\Delta p_{s,k}+\Delta p_{b,i}+\sum_{k=0}^{i}\Delta p_{r,k}
\quad\Rightarrow\quad \text{段数}= \mathbf{2(i+1)=2,4,6,8,10}$$

约束（求解器 least_squares）：`P_1=…=P_5=P*` 且 `Σf_i=Q`。

### 3.2 ★ 同程式残余 4.9% 的根因（推翻了"段数相等就该均流"）

镜像的供液路径与回液路径，其**和并非常数**，而是对称拱形。均流时（段流量 = [5,4,3,2,1]f̄ 与 [1,2,3,4,5]f̄）：

$$h_i \;\propto\; \sum_{m=N-i}^{N} m^2 \;+\; \sum_{m=i+1}^{N} m^2 \;=\; [80,\,95,\,100,\,95,\,80]$$

模型实测（均流 f̄=0.17993）：母管 Δp = `[32514, 38610.4, 40642.6, 38610.4, 32514]` Pa，
比值 **32514/40642.6 = 0.800**、**38610.4/40642.6 = 0.950** —— 与 80/100、95/100 **精确吻合**。

定性原因：平方律把小流量段"折价"了。支路 1 走过 [5, 1,2,3,4,5]f̄（含 1f̄、2f̄ 这类贡献极小的段），支路 3 走过 [5,4,3, 3,4,5]f̄ 全是中大流量段 → 支路 3 的 Σṁ² 最大。

等压降约束反推流量偏差（Δp ∝ f² ⇒ 流量偏差是压降偏差的一半）：

$$\frac{\delta f}{\bar f}\approx\frac12\cdot\frac{\Delta h_{max}-\Delta h_{min}}{\Delta p_{branch}}
=\frac12\cdot\frac{40642.6-32514}{81285.1}=\mathbf{5.0\%}\quad(\text{实测 }4.908\%)$$

**结论：同程式是一阶自平衡、二阶残余。** 它保证"几何段数相等"（各 6 段）而非"水力阻力相等"（Σ R·ṁ²）。残余正比于母管/支路阻力比（MEDIUM = 0.005）。

### 3.3 ★ 节点压力对比（两种拓扑的真正区别）

以回液出口为 0 基准（单位 kPa）：

| | 支路 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| **同程** | 供液取压点 | 107.60 | 101.17 | 97.55 | 95.90 | 95.47 |
| **同程** | 回流汇入点 | 22.28 | 21.86 | 20.21 | 16.58 | 10.16 |
| **同程** | **支路压差** | **85.31** | **79.32** | **77.33** | **79.32** | **85.31** |
| **异程** | 供液取压点 | 107.00 | 100.79 | 97.40 | 95.91 | 95.54 |
| **异程** | 回流汇入点 | **10.16** | **16.37** | **19.76** | **21.25** | **21.62** |
| **异程** | **支路压差** | **96.84** | **84.42** | **77.64** | **74.67** | **73.93** |

- **同程式**：供液压力与回液背压**同向下降** → 压差保住 → 流量接近均分
- **异程式**：供液压力下降、回液背压上升，**反向夹击**远端 → 支路压差两端差 **31%**

**重要澄清**：两种拓扑的"五路路径总压降"都**严格相等**（同程 117 758.79 Pa / 异程 117 162.02 Pa）。这是**基尔霍夫定律**，并联网络的普适约束，**不是同程式的特权**。

流量比 = 压差比的平方根（已验证）：
- 异程 `f₁/f₅ = √(96840.7/73926.1) = 1.1446` ↔ 实测 `0.19639/0.17159 = 1.1446` ✓
- 同程 `f₁/f₃ = √(85313.8/77334.8) = 1.0503` ↔ 实测 `0.18433/0.17550 = 1.0503` ✓

### 3.4 流向为什么"正向优于反向"

`pack_reference.py`：`column_factor = 1.0 − 0.4·col/12`，列 0 最热。
冷板 ROM 按 `ZONE_COLUMN_COUNTS = [4,5,4]` 切三个串联区，**Zone 0 = 列 0–3 即最热区**。
正向 = 冷却液入口正对最热端（传热温差最大）；反向 = 先被最凉区预热，到最热区时驱动力已衰减。

### 3.5 温差的两层含义（易混淆，务必区分）

| 温差来源 | 同程式 | 异程式 | 谁决定 |
|---|---|---|---|
| **Pack 间不均**（管路导致）| 0.19 K | 0.51 K | **管网拓扑** |
| **整簇温度跨度**（冷板内列向梯度 + 回路沿程升温）| ~2.7 K | ~2.9 K | 冷板与电池本体，与拓扑无关 |

同程式让五个 Pack **彼此一致**，但每个 Pack **内部**该有多宽还得多宽。

### 3.6 流量→温度灵敏度（经验拟合，见 `analyze_flow_temperature_sensitivity.py`）

- 560 A：**0.0445 K / 每 1% 流量极差**
- 1120 A：**0.0508 K / 每 1% 流量极差**
- 注：这是对 600 s 与 1800 s 全部工况反解的经验系数，非解析推导，预测误差约 3%（脚本自述）。

---

## 4. 已交付文件清单

| 路径（相对 `cluster_plant_v2/`）| 说明 |
|---|---|
| `validation/scan_piping_topology_and_flow_direction.py` | **主扫描脚本**，含 `DirectReturnHeaderNetwork` |
| `validation/results/piping_topology_scan_20260901/PIPING_TOPOLOGY_COMPARISON_20260901.md` | 报告（⚠️ 目前仍是 **600 s** 数据，待回填 1800 s）|
| `validation/results/piping_topology_scan_20260901/piping_topology_scan.csv` | **1800 s 稳态数据，24 行** |
| `validation/results/piping_topology_scan_20260901/piping_topology_scan_600s_backup.csv` | 600 s 备份 |
| `validation/results/piping_topology_scan_20260901/scan_1800s.log` | 1800 s 运行日志（含最终汇总表）|
| `validation/results/piping_topology_scan_20260901/direct_return_piping_diagram.svg` | 异程式管网示意图（精简版）|
| `validation/results/piping_topology_scan_20260901/analyze_flow_temperature_sensitivity.py` | 反解流量→温度灵敏度 |
| `validation/results/piping_topology_scan_20260901/sweep_header_resistance_ratio.py` | 扫母管阻力比 → 预测不均衡度（✅ 已修 L50 字段名并跑通）|
| `validation/results/piping_topology_scan_20260901/header_ratio_sweep.csv` | **母管阻力比扫参结果**（✅ 2026-09-04 已落盘，9 个比值 × 两拓扑）|

---

## 5. 待办与遗留（按优先级）

1. ⬜ **回填 1800 s 数据到报告** —— `PIPING_TOPOLOGY_COMPARISON_20260901.md` 里的表格还停在 600 s
2. ✅ **`validation/scan_cop_real_anchor_gap.py` 未完成编辑已修复**（2026-09-04）：补 `from cluster_plant_v2.hydraulics import CoolantPump, ParallelHeaderHydraulicNetwork`，删掉死导入 `build_engineering_pump` / `build_medium_header_network`。语法/导入已验证（ast.parse 通过、`CoolantPump`/`ParallelHeaderHydraulicNetwork` 可正常 import）。⚠️ 尚未端到端运行——若要跑，记得从父目录用 `-m cluster_plant_v2.validation.scan_cop_real_anchor_gap` 方式（踩坑：`python 文件.py` 会 `ModuleNotFoundError`）
3. ✅ **`sweep_header_resistance_ratio.py` 已修并跑通**（2026-09-04）。L50 键名 `pack_mass_flows_kg_s` → `pack_mass_flows`。结果落盘 `validation/results/piping_topology_scan_20260901/header_ratio_sweep.csv`。**核心结论**：
   - 同程残余在母管阻力比 0.001 下 = **0.9963%**（预测温差 560A 0.044 K / 1120A 0.051 K）——"压到约 1%"的预测**精确命中**
   - 但 0.001 已使同程残余 ≈ 异程的 1/3（异程 2.946%），且 0.0002 时同程本底 0.200%（非线性本底，母管无关）——**继续降母管阻力不再划算**
   - 异程要追平同程现役 4.908%，母管阻力比得降到 0.0002（**25 倍**）——工程上不可行
   - 工程落点：**同程 + 母管阻力比 0.001** 是最优组合（残余 0.996% vs 现役 4.908%，且比异程任何方案都稳）
4. ⬜ 用 `electrical_power_w`（电网侧口径）重跑 `validate_physics_p_nmpc.py`，把 NMPC 节能率改述为电网侧口径（原 RegD "−40%" 按轴功率算，计入固定风机后约 −29~−32%）
5. ⬜ 清理第一代控制器 `cluster_domp_model.py` / `cluster_domp_mpc.py` / `chiller_surrogates.py`
6. ⬜ 显性化 `control/physics_p_nmpc_model.py` 对 `single_pack_plant` 的 `sys.path` 依赖
7. ⬜ 补低压比效率样本数据（`COMPRESSOR_PRESSURE_RATIO_AXIS = [4,5,6.4,8,10]`，实际运行 1.90~3.44 全被 clip 到 4.0 列）
8. ⬜ **身份 bootstrap 未定**：`~/.workbuddy/BOOTSTRAP.md` 仍在，已问过两次未答（我的名字/称呼、用户称呼、所在城市）

---

## 6. 踩过的坑（新对话别重蹈）

- **并发数上限 3**：`ProcessPoolExecutor` 每个 worker 自带 CoolProp 状态，`max_workers=8` 跑 1800 s 扫描**中途静默死亡**（已复现一次）。降到 3 后成功。
- **实测耗时**：6 工况 × 1800 步，3 并发约 **9 分钟**落盘（我按每步 0.7 s 估成 42 min，严重高估；实测远快）。
- **运行目录**：必须从 `集成仿真多种控制`（**父目录**）跑，否则 `ModuleNotFoundError: No module named 'cluster_plant_v2'`。
- **输出必须重定向到文件**：不重定向时日志会被管道缓冲吞掉，看起来像"没输出"。
- **字段名**：网络 solve 返回的是 `pack_mass_flows` / `supply_segment_flows` / `supply_segment_delta_p`（**没有** `pack_mass_flows_kg_s`，也没有现成的 `supply_path_delta_p`，路径和要自己 `cumsum`）。
- **并行编辑风险**：仓库可能存在其他会话/IDE 并行编辑 `parameters.py`，改前先重读，防止常量双重定义（Python 静默取后者）。
- **画图偏好**：用户明确嫌"太乱"。示意图只保留必要元素（母管段流量、支路流量、方向箭头），**不要**加端口圆点、双立管文字标注、中段三角箭头、多行小字说明。

---

## 7. 新对话开场白模板（可直接粘贴）

```
继续 cluster_plant_v2 的冷板管网拓扑研究。完整上下文见
cluster_plant_v2/CONVERSATION_HANDOVER_20260904.md，请先读它。

当前状态：1800 s 稳态扫描已跑完（数据在
validation/results/piping_topology_scan_20260901/piping_topology_scan.csv），
核心结论是异程式流量不均衡 13.78% vs 同程式 4.91%，Pack 间温差 2.7×，
但能耗两者打平（差异 < 0.05%）。

下一步我想做：______
（候选：① 回填 1800 s 数据到报告 ② 跑 sweep_header_resistance_ratio.py 验证
降低母管阻力比的效果 ③ 清理 scan_cop_real_anchor_gap.py 的未完成编辑
④ 用 electrical_power_w 重述 NMPC 节能率）
```
