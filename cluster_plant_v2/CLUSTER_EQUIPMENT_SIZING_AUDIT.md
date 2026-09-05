# Stage 8D：Cluster Equipment Sizing Audit

> 状态：只读设备尺度审计与单因素 screening；未修改冻结 Cluster Plant、Cluster-P V1、旧 Physics-P、MPC、Battery-ROM 或 Cold-Plate-ROM 方程。  
> 计算基线：600 s，步长 5 s，初始 25 °C，环境 35 °C，风机 1200 rpm，压缩机 4000 rpm，泵 3600 rpm（约 25.2 L/min），5 个 Pack。  
> 重要边界：S1 的 3/5/8 kW 是 `screening-only surrogate`，仅用于判断容量尺度；它不是最终制冷循环模型，不能写回冻结 Plant。

## 1. 当前 5-Pack Cluster 能量规模

- 每 Pack：4P13S，52 × 280 Ah LFP。
- 按 3.2 V/cell：单 Pack 约 46.592 kWh。
- 5-Pack Cluster：约 232.96 kWh。

因此，公开 200–250 kWh 液冷储能柜可用于数量级交叉检查，但不能因其电气拓扑不同而机械套用参数。

## 2. 当前设备参数表

| 项目 | 当前值 | 当前物理含义/能力 |
|---|---:|---|
| 制冷循环 | R134a；4000 rpm 时约 1.650 kW；6000 rpm 时约 2.349 kW | 冻结循环在 Tin=25 °C、Tamb=35 °C、25.2 L/min、fan=1200 rpm 的能力 |
| Coolant tank | 3 L | 方程按全部参与混合和蓄热的热水箱处理；热容 10.90 kJ/K |
| Pump | 4000 rpm≈28 L/min；3600 rpm≈25.2 L/min | 冻结转速范围 1600–4800 rpm 对应约 11.2–33.6 L/min |
| 每 Pack 支路流量 | 平均约 5.04 L/min | 五支路存在小幅不均匀：约 4.92–5.16 L/min |
| Cold-plate reference mass flow | 1.2 kg/s / Pack | 等价约 67.23 L/min / Pack，进入 `h=2000(mdot/1.2)^0.8` |
| 支路名义压降 | 20 kPa | Stage 7B 工程假设；不是实测硬件数据 |
| 当前网络工作压降 | 约 29.44 kPa @ 25.2 L/min | header + branches 的模型求解值 |

## 3. 参数来源：inherited / new assumption

| 参数 | 来源判定 | 审计说明 |
|---|---|---|
| 3 L tank | inherited | 原单 Pack 同样使用 3 L；扩展为 5 Pack 后未按有效回路液量重定标 |
| Pump 4000 rpm→28 L/min | inherited | 原单 Pack 的泵参考点被保留；当前通过五支路分流得到约 5 L/min/Pack |
| Cold-plate 1.2 kg/s | inherited, unverified engineering parameter | 代码单位明确是 kg/s，但当前仓库没有可追溯来源；也没有证据证明它属于 per-Pack、whole-system 或其他参考条件 |
| 20 kPa branch drop | new engineering assumption | 用于构建簇级支路阻力，不是继承测量值 |
| R134a component sizing | inherited/partly inherited | 循环结构已扩展到簇级接口，但容量仍处于约 1.6–2.35 kW 量级 |

原单 Pack 4000 rpm 时也是约 28 L/min，因此其 3 L tank 的体积周转时间约 6.43 s；当前 Cluster 在 25.2 L/min 时约 7.14 s。也就是说，**tank 周转尺度几乎原样继承**，没有随 Pack 数增加。

## 4. 公开工程参考数据及来源

| 来源、文档与日期 | 系统/设备 | 公开参数 | 本审计用途与可信度 |
|---|---|---|---|
| Xinrex，232 kWh Liquid Cooling Energy Storage Cabinet，网页 | 232.96 kWh、280 Ah、液冷 | 5 kW cooling，3.2 kW input，50% ethylene glycol | 同量级系统官方页面；较高 |
| SWD，232 kWh / 105 kW Energy Storage System datasheet，2024-11-21 | 232.96 kWh，5 × 46.59 kWh Pack，280 Ah | 5 kW cooling @ W18/L45，50% ethylene glycol | 容量和 Pack 数高度相近；较高 |
| Jilipow，JL-BC-5-768280-L Specification，2024-04-15 | 215 kWh，280 Ah，5 Packs | 5.6 kW @ L45/W18，2.34 kW input，40 L/min @ 160 kPa，50% EG，R134a 1.8 kg；含 expansion tank | 最接近的完整系统规格；较高；未给 tank 容积 |
| ZetaCube C&I ESS User Manual V1.3，2025-06-17 | 261 kWh，314 Ah，5 × 52S | rated 30 L/min，max 50 L/min，50% EG，R410A | 流量数量级参考；中等，不宜映射制冷剂/拓扑 |
| GoodWe BAT-C 208/261 kWh，官方产品页 | 208.9/261.2 kWh | liquid cooling | 证明同级产品类别；参数不完整 |
| KSTAR 3/5/8 kW Liquid Cooling Unit，官方规格 | 独立液冷机 | 3/5/8 kW；30/40/50 L/min @ 90 kPa；R134a；50% EG；输入 1.1/1.9/2.9 kW | 候选容量和泵流量边界；较高，但不是整柜配套证明 |
| GS Energy 5 kW Liquid Cooling System，官方页 | 独立液冷机 | 5 kW @ W18/A45，2.6 kW input，37 L/min，R134a | 5 kW 设备尺度佐证；较高 |
| Kansa 5.2 kW 液冷机，供应商页 | 独立液冷机 | 14.8 L/min，35 L tank，370 W pump | tank 辅助参考；低，系统边界不可直接迁移 |
| T/CES 液冷系统技术标准征求意见稿 | 储能液冷系统 | 要求缓冲设备吸收冷却液体积变化，可为高位水箱、膨胀罐或其他容器 | 支持区分 expansion reservoir 和 thermal buffer；中等 |
| Journal of Energy Storage cold-plate study，2026 | 特定 LFP 冷板几何 | 其最优点约 3 L/min，h≈276.33 W/(m²·K)，Δp≈1912 Pa | 只证明冷板结果强依赖几何；不能直接移植 |

来源链接：

- [Xinrex 232.96 kWh cabinet](https://xinrex.com.cn/product-_19/832.html)
- [SWD 232.96 kWh datasheet](https://assets.zyrosite.com/YX4lNJX8jkSOP443/5--swd-232kwh---105kw-energy-storage-system-swd232105lfp-ynqbloxow7fkxopp-Aq2WyJqov6I1rDKz.pdf)
- [Jilipow JL-BC-5 specification](https://v6-file.globalso.com/upload/p/502/file/2024-11/jl-bc-5-768280-l-specification.pdf)
- [ZetaCube user manual](https://cdn.enfsolar.com/z/pp/2025/9/lwa1gbt79be5g8/zetacube-c-i-energy-storage-system-user-manual-v1-3.pdf)
- [GoodWe BAT-C](https://en.goodwe.com/bat-c-208-261kwh)
- [KSTAR liquid cooling unit](https://www.kstar.com/cn/index.php/product/info/164.html)
- [GS Energy 5 kW liquid cooling system](https://www.gs-ess.com/product/detail?id=41)
- [Kansa 5.2 kW unit](https://www.kansa.cn/list_12/33.html)
- [T/CES technical-standard draft](https://www.ces.org.cn/res/ces/2305/eb6f0a4c218a256d03cf677fb37237a9.pdf)
- [Cold-plate paper](https://www.sciencedirect.com/science/article/pii/S2352152X26010947)

公开证据的共同结论是：约 215–233 kWh、约 0.5P 的 280 Ah 液冷系统常见约 5–5.6 kW 制冷和约 37–40 L/min 流量。它支持数量级判断，不证明本 Plant 必须复制任何一台商业设备。

## 5. Cluster Qgen vs Qevap capability map

### 5.1 Pack 模型实际发热扫描

每个点均使用冻结 Pack 模型运行 600 s；冷却液入口保持 25 °C、支路约 5 L/min、环境 35 °C。正式结果没有用固定 I² 比例替代。

| Icluster (A) | Pack 首步 Qgen (W) | Pack 600 s 平均 Qgen (W) | Cluster 平均 Qgen (W) | 含空气换热后的平均冷却负荷 (W) |
|---:|---:|---:|---:|---:|
| 0 | 0.0 | 0.0 | 0.0 | 348.8 |
| 280 | 206.7 | 252.6 | 1,263.2 | 1,602.4 |
| 560 | 827.7 | 999.0 | 4,995.2 | 5,305.3 |
| 840 | 1,864.6 | 2,131.6 | 10,658.1 | 10,922.5 |
| 1120 | 3,318.9 | 3,614.1 | 18,070.4 | 18,275.1 |

空气项在此工况不是散热，而是 35 °C 环境向较冷 Pack 的净热输入，因此所需制冷量略大于 Qgen。

### 5.2 冻结制冷循环能力

| Compressor rpm | Qevap (W) | Pcomp shaft (W) | model shaft COP |
|---:|---:|---:|---:|
| 2000 | 820.7 | 81.8 | 10.03 |
| 3000 | 1,260.3 | 136.1 | 9.26 |
| 4000 | 1,650.5 | 206.8 | 7.98 |
| 5000 | 2,025.7 | 298.5 | 6.79 |
| 6000 | 2,348.9 | 385.0 | 6.10 |

| Icluster (A) | Qevap/Qload @ 4000 rpm | Qevap/Qload @ 6000 rpm | 长期热平衡判断 |
|---:|---:|---:|---|
| 0 | 4.73 | 6.74 | 有余量 |
| 280 | 1.03 | 1.47 | 4000 rpm 仅刚好；6000 rpm 有余量 |
| 560 | 0.31 | 0.44 | 不可长期平衡 |
| 840 | 0.15 | 0.22 | 不可长期平衡 |
| 1120 | 0.09 | 0.13 | 不可长期平衡 |

在本扫描条件下，能力交点位于 280–560 A 之间；**Qevap,max < Qload** 对 560 A 及以上成立。

## 6. Chiller sizing analysis

### 6.1 S1 单因素 screening

| I (A) | 候选能力 (kW) | Ttank final (°C) | Tsupply final (°C) | Battery Tavg final (°C) | Battery Tmax (°C) | last-60s Qevap (kW) | last-60s Qload (kW) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 560 | native 1.65 | 23.82 | 22.73 | 26.853 | 26.949 | 1.586 | 5.217 |
| 560 | 3 | 21.65 | 19.84 | 26.429 | 26.575 | 2.706 | 5.313 |
| 560 | 5 | 18.80 | 16.07 | 25.857 | 26.070 | 4.145 | 5.443 |
| 560 | 8 | 15.20 | 11.31 | 25.100 | 25.401 | 5.938 | 5.606 |
| 1120 | native 1.65 | 28.20 | 26.82 | 32.765 | 32.880 | 1.768 | 17.228 |
| 1120 | 3 | 25.88 | 23.70 | 32.344 | 32.508 | 3.014 | 17.333 |
| 1120 | 5 | 22.85 | 19.66 | 31.779 | 32.009 | 4.609 | 17.473 |
| 1120 | 8 | 19.03 | 14.57 | 31.045 | 31.359 | 6.591 | 17.657 |

600 s 内的电芯/冷却液热容会掩盖长期容量不足。560 A 下 5 kW 接近但仍小于末段负荷，5.5–6 kW 是最低连续能力的合理名义区间，8 kW 提供工况和老化余量。1120 A 连续负荷即使用 8 kW 也不能长期平衡。

### 6.2 最终模型应采用的 physical scaling 方法

**方法 A：compressor displacement scaling。** 以当前 4000 rpm、约 5.525 cc/rev 为筛选锚点，5 kW 的一阶容量倍率约 3.03（约 16.7 cc/rev），8 kW 约 4.85（约 26.8 cc/rev）。实际修改会提高 refrigerant mass flow、Qevap、Qcond 和 Pcomp；蒸发/冷凝饱和温度会变化，HX 可能成为瓶颈，COP 不能假设不变。

**方法 B：compressor + evaporator/condenser combined sizing。** 同时重选排量、蒸发器/冷凝器面积或 UA、冷凝风量/风机。制冷剂流量和换热端能力共同扩大，COP 更可能保持在可解释区间，但必须重新求解循环闭合和能量守恒。

**方法 C：hardware-map calibration。** 选定 5–8 kW 实际液冷机后，用其 W18/L45 等额定点和多工况表标定能力、功率、流量及压头。当前模型在代表点的 shaft COP 约 6.1–10，而公开整机额定 COP 大约 1.9–2.6（边界也更严苛），因此不能仅放大 Qevap 而保留原 Pcomp/COP。

## 7. Tank sizing analysis

| Vtank (L) | Ctank (kJ/K) | turnover @25.2 L/min (s) | max |dTtank|/5s (°C) | 560 A Tavg final (°C) | 1120 A Tavg final (°C) |
|---:|---:|---:|---:|---:|---:|
| 3 | 10.90 | 7.14 | 0.503 | 26.853 | 32.765 |
| 5 | 18.16 | 11.90 | 0.302 | 26.859 | 32.758 |
| 10 | 36.32 | 23.81 | 0.151 | 26.878 | 32.740 |
| 15 | 54.48 | 35.71 | 0.101 | 26.896 | 32.727 |
| 30 | 108.95 | 71.43 | 0.050 | 26.931 | 32.706 |

3→30 L 对 600 s 电芯终温影响小（560 A 约 +0.08 °C；1120 A 约 -0.06 °C），但把水箱每步温变降低约 10 倍，并显著延长液路状态时间尺度。当前 3 L 只有约 1.43 个仿真步的周转时间，作为完整混合热容时偏激进。

语义判断：

- 若 3 L 是 **expansion reservoir**，它不应默认全部作为主循环的理想混合热容；此时问题首先是状态方程语义，而非单纯改成 15 L。
- 若该状态代表 **active loop inventory / thermal buffer tank**，10–15 L 是更稳健的 sensitivity 区间，15 L 可作为待验证 nominal。

因此 tank 暂不直接 RESCALE；先查 BOM、管路有效液量、膨胀罐连接方式和是否存在独立 buffer。

## 8. Pump/flow sizing analysis

| Pump rpm | Total flow (L/min) | Network Δp (kPa) | Pump power (W) | Branch min–max (L/min) |
|---:|---:|---:|---:|---:|
| 1600 | 11.2 | 5.82 | 1.75 | 2.19–2.30 |
| 2857 | 20.0 | 18.54 | 9.98 | 3.90–4.10 |
| 3600 | 25.2 | 29.44 | 19.97 | 4.92–5.16 |
| 4000 | 28.0 | 36.35 | 27.40 | 5.46–5.74 |
| 4286 | 30.0 | 41.72 | 33.70 | 5.85–6.15 |
| 4800 | 33.6 | 52.34 | 47.34 | 6.56–6.89 |

25→30 L/min（仍在冻结泵范围内）的 600 s 收益：560 A 的 Tavg/Tmax 分别约 -0.015/-0.018 °C，1120 A 约 -0.047/-0.048 °C；泵功率增加约 72.8%。25→40 L/min 是 affinity-law 外推，Tavg 仅下降约 0.037 °C（560 A）和 0.128 °C（1120 A），功率约变为 4.1 倍。

50 L/min 超出冻结泵范围，并使 3 L tank 的体积周转约 3.6 s，小于 5 s 仿真步：560 A 在第 120 步前因冷却液低于循环审计边界而停止；1120 A 出现每步约 34 °C 的水箱数值振荡。两者都不能作为物理性能证据。

结论：当前设计确实形成约 5 L/min/Pack，但该选择尚无硬件泵图或冷板压降曲线背书。在修正/确认 cold-plate correlation 前，提高总流量不是主导改进。若选择公开 5–6 kW 液冷机，35–40 L/min 可能是新的系统候选，但必须连同泵曲线、总压头和管网一起重建，不能把现泵直接外推。

## 9. Cold-plate 1.2 kg/s audit

代码中 1.2 的单位明确为 kg/s，并在每个 Pack 的冷板换热系数中使用：

`h = 2000 * (mdot_pack / 1.2)^0.8`

有效壁面到流体 UA 为 `h × 0.5 m²`。当前仓库没有可信来源或原始历史证明 1.2 是 per-Pack 参考点。1.2 kg/s 等价 67.23 L/min，甚至高于原单 Pack 泵的 28 L/min 参考流量和约 37 L/min 上界，因此它应标为 **unverified engineering parameter**；现有证据不能确认它是 whole system、误写单位或其他参考工况。

| Per-Pack flow (L/min) | mdot (kg/s) | mdot/1.2 | h (W/m²K) | h/h_nom | UA (W/K) |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.05355 | 0.0446 | 166.2 | 8.31% | 83.1 |
| 5 | 0.08925 | 0.0744 | 250.1 | 12.51% | 125.1 |
| 8 | 0.14280 | 0.1190 | 364.3 | 18.22% | 182.2 |
| 10 | 0.17850 | 0.1488 | 435.5 | 21.78% | 217.8 |

在当前约 5.04 L/min/Pack 下，每 Pack UA≈125.9 W/K，五 Pack 合计≈629 W/K；原单 Pack 在 28 L/min 下约 496 W/K。即 Cluster 热负荷扩大约 5 倍后，按当前公式得到的总冷板 UA 仅约为原单 Pack 的 1.27 倍。

S4 将 reference flow 仅在运行时改为相当于 3/5/8/10 L/min 的质量流量并在每例后恢复。reference=5 L/min 时，当前实际流量附近 h≈2000 W/(m²K)、UA≈1006 W/K/Pack；相对当前 reference，600 s 的 Tavg 下降约 0.110 °C（560 A）和 0.147 °C（1120 A）。reference=8 L/min 在被测范围给出更低 Tmax；reference=3 L/min 因换热过强和串联温差效应反而扩大局部 Tmax。故方向明确但最优值不能由本轮短时单因素扫描定案。

## 10. 20 kPa branch pressure drop assessment

20 kPa 是 Stage 7B 新增的工程假设，而非原单 Pack 参数，也不是供应商/实测曲线。当前网络在 25.2 L/min 总流量下求得约 29.44 kPa 总压降。公开液冷机的 90–160 kPa 多为整机可用供液压头，不能等同为单支路冷板压降。

结论：保留 20 kPa 作为当前实验基线，但标记 `UNCERTAIN / NEED SOURCE`。后续需要冷板几何/CFD 或供应商 Δp–flow 曲线、header 尺寸和阀件局部阻力；建议 10–40 kPa branch sensitivity，而不是现在改 nominal。

## 11. S1–S4 sensitivity results

| 实验 | 唯一改变项 | 主结论 |
|---|---|---|
| S1 | Chiller screening capacity：native/3/5/8 kW | 560 A 需要约 5.3 kW 最低持续能力；8 kW 才有明确短时余量；1120 A 即 8 kW 仍不足 |
| S2 | Tank：3/5/10/15/30 L | 主要改变液路状态速率，短时电芯终温不敏感；3 L 的数值/控制时间尺度很快 |
| S3 | Total flow：20/25/30/35/40/50 L/min | 冻结泵可实现到 33.6 L/min；25→30 温度收益很小而泵功率明显增加；40/50 为外推，50 无效 |
| S4 | Cold-plate reference：3/5/8/10/current 67.23 L/min equivalent | 当前 1.2 kg/s 显著压低 h/UA；应重定标并重新验证，但不能只凭短时扫描定唯一值 |

完整结果位于 `validation/results/stage8d_equipment_sizing_audit/`，其中 `s1.csv`–`s4.csv` 是汇总表，`*_timeseries.csv` 是轨迹，`qgen_scan.csv` 和 `cycle_capability.csv` 是能力图基础数据。

## 12. Actuator capacity limitation analysis

- 560 A：600 s 模型平均所需冷却约 5.31 kW；冻结循环最大约 2.35 kW。最低持续 Qevap 应不低于约 5.3 kW，工程 nominal 建议 5.5–6 kW，并保留到约 8 kW 的最大/瞬态余量。
- 1120 A：平均所需约 18.28 kW；当前最大能力仅为其约 13%。若 1120 A 是持续设计点，需要约 18–20 kW 的持续 Qevap 或明确限制电流占空比/持续时间。5–8 kW 不能解决长期热平衡。

因此这首先是 **actuator capacity limitation**，不是 MPC tuning problem。MPC 能分配有限能力、处理预冷和约束，但不能在 `Qcool,max < Qload` 时创造稳态热平衡。

## 13. KEEP / RESCALE / UNCERTAIN 分类

| 项目 | 分类 | 理由 |
|---|---|---|
| Chiller | **RESCALE** | 当前 2.35 kW 最大能力低于 560 A 的约 5.3 kW 负荷；模型和同级公开设备均支持 5–6 kW 数量级 |
| Tank | **UNCERTAIN / NEED SOURCE** | 3 L 作为完整混合热容偏小，但它若实际代表膨胀罐，则不能简单按 thermal buffer 扩容 |
| Pump flow | **UNCERTAIN / NEED SOURCE** | 当前约 5 L/min/Pack 可运行且短时增流收益小；没有泵图、冷板压降和选定 chiller 的流量要求 |
| Cold-plate reference flow | **RESCALE** | 1.2 kg/s/Pack 无可追溯来源，并导致当前 h 仅 nominal 的约 12.5%；明显存在尺度/参考条件错配 |
| Branch pressure drop | **UNCERTAIN / NEED SOURCE** | 20 kPa 是新工程假设，尚无几何、CFD、供应商或实测证据 |

本轮没有把任何一项设备参数判为无条件 KEEP；对于三项 UNCERTAIN 参数，只建议在新证据出现前**临时保持当前基线用于对照**。

## 14. 推荐 nominal parameter set

| 项目 | Current | Candidate range | Recommended nominal | 理由/置信度 |
|---|---:|---:|---:|---|
| Chiller continuous Qevap（560 A 设计点） | 1.65 kW @4000；2.35 kW max | 3/5/8 kW screening | 5.5–6 kW continuous，约 8 kW max/headroom | 模型负荷与相近商业系统一致；高。需物理重选 compressor+HX |
| Chiller（若 1120 A 持续） | 同上 | 8–20 kW | 18–20 kW continuous，或限制 duty | 模型热平衡约束；高，但需先确认工况定义 |
| Effective mixed tank volume | 3 L | 5–30 L | 15 L，仅当状态确实代表 active mixed inventory | sensitivity 支持时间尺度；语义来源不足，中低 |
| Total flow | 25.2 L/min | 20–40 L/min | 暂以 30 L/min 做模型内候选；选 5–6 kW 硬件后重点验证 35–40 L/min | 温度收益小且泵功率增长快；中低 |
| Cold-plate reference flow | 1.2 kg/s = 67.23 L/min/Pack | 3–10 L/min/Pack | 5 L/min/Pack equivalent = 0.08925 kg/s；保留 5–8 L/min sensitivity | 当前工作点应接近相关式 reference；方向中等、来源中低 |
| Branch nominal Δp | 20 kPa | 10–40 kPa | 暂保留 20 kPa 作为对照 | 没有来源；低 |

该表是**下一阶段重开 Plant 的设计输入建议**，不是本轮写入值。

## 15. 推荐 sensitivity range

- Chiller：5、5.5、6、8 kW；另设 18、20 kW 仅用于 1120 A 连续需求审查。
- Tank：5、10、15、30 L；同时建立“expansion-only”和“active mixed volume”两种语义。
- Total flow：25、30、35、40 L/min；需新的泵图时不再沿用当前 affinity 外推。
- Cold-plate reference：5、6.5、8 L/min/Pack equivalent，并用冷板 Δp/h 或 UA 数据约束。
- Branch Δp：10、20、30、40 kPa，在固定 header 和新泵图下审查分流、功率和均匀性。

## 16. 如果修改参数，需要重新验证哪些 Stage

1. 先冻结设计工况：560 A/约 0.5P 是否持续；1120 A 是否仅脉冲。
2. 获取设备 BOM/datasheet：chiller map、pump curve、tank/loop inventory、coolant、cold-plate Δp/UA。
3. 物理调整 compressor displacement、evaporator/condenser UA/area、fan 和功率模型；禁止纯 Qevap factor 成为正式 Plant。
4. 修正 cold-plate reference/correlation；再确定 tank effective volume、pump map、branch resistance。
5. 重跑 Plant 的组件测试、五支路质量守恒/不均匀性、制冷循环质量与能量闭合、600 s 双负荷、流向切换和项目隔离回归。
6. 对旧冻结基线给出数值差异说明；改尺度后不应再要求 Plant 输出与旧基线 `max_abs_difference=0`，而应冻结新基线。
7. 重新生成 Cluster-P 数据、标定并通过 Cluster-P gates。
8. 最后恢复 Stage 9B MPC。

## 17. 对 Cluster-P V1 的影响

本轮不修改也不重新标定 Cluster-P V1。若 chiller、cold-plate、tank 或 pump 任一物理参数正式改变，当前 Cluster-P 的状态转移和 actuator effectiveness 将不再代表新 Plant，必须在 Plant 重新验证后重新生成数据和标定。原 Cluster-P V1 应保留为旧 Plant 基线，而不是静默覆盖。

## 18. 对 Stage 9B MPC 的影响

Stage 9B 继续暂停。当前容量不足会让 MPC 权重调节被误解为控制器性能问题；先解决设计点和 actuator envelope，再确定约束、功率代价、预冷策略和层级控制。尤其不能用 MPC 权重掩盖 1120 A 下约 10–16 kW 的持续制冷缺口。

## 19. 最终是否建议修改冻结 Plant

**建议未来在独立、获批的 Stage 8D2 中重开 Plant；本轮不修改。**

重开理由：chiller 能力不足和 cold-plate reference-flow 尺度错配已有相互独立的模型证据；继续围绕当前 Plant 标定 MPC 的返工风险高。Tank、pump 和 branch Δp 仍缺来源，不能在同一次修改中凭猜测定值。

## 20. 若获批修改，只列计划

1. 明确 560/1120 A 的持续时间和允许温度边界。
2. 确认 5.5–6 kW nominal / 8 kW max 的 chiller 硬件或性能图；重新选 compressor、HX 和 fan，不做输出倍率写回。
3. 用 cold-plate 几何/测试/供应商曲线确认 reference flow 与 UA；以 5–8 L/min/Pack 为初始验证窗。
4. 区分 expansion reservoir 与 active loop thermal volume，再选择 tank 方程和有效容积。
5. 联合选定 30–40 L/min 范围内的泵曲线、总压头及 branch/header 阻力。
6. 建立新 Plant 基线、逐级回归，然后才重建 Cluster-P V1 successor 并恢复 MPC。

本 Stage 8D 到此停止，不执行上述修改。

