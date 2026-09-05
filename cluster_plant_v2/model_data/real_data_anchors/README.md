# 真实数据锚点（Real Data Anchors）

抓取日期：2026-09-01（制冷/压缩机/整机锚点）；2026-09-04（管网拓扑与流量分配锚点，见文末专节）。
用途：为 `refrigeration.py` 与 `hydraulics.py` 中的工程假设参数提供可溯源的现实参照与标定锚点。
本目录不修改任何冻结模型参数；所有数值仅作为对标与后续标定输入。

## 目录文件

| 文件 | 内容 | 条数 |
|---|---|---|
| `ess_chiller_rated_points.csv` | 储能液冷整机额定参数（能力/输入功率/流量/压头/充注量） | 5 |
| `r134a_compressor_catalog_points.csv` | R134a 压缩机样本点（排量/制冷量/输入功率/COP/转速） | 12 |
| `system_level_measured_points.csv` | 系统级实测点（巴士空调 17.5 kW 实测、EV 热泵变速曲线、直冷台架） | 10 |
| `thermal_property_anchors.csv` | 物理热特性参数对照：冷却液/电芯比热/质量/导热系数等，模型值 ↔ 真实值逐项比对 | 13 |
| `jilipow_jl-bc-5-768280-l_specification.pdf` | Jilipow 液冷电池柜规格书原件（22 页，公开 PDF 存档） | - |
| `jilipow_extracted_text.txt` | 上述 PDF 的文字提取（pypdf），液冷单元参数见 Table 8 | - |

## 三层数据与可信度

1. **整机额定点（高可信）**：官方产品页/规格书。结论与本仓库审计一致——215~233 kWh 液冷系统配 5~5.6 kW 制冷、37~40 L/min、输入 2.3~3.2 kW，
   **整机 COP ≈ 1.9~2.4**。
2. **压缩机样本点（中可信）**：经销商/厂商页面额定值。R134a 变频压缩机样本 COP 集中在 **2.8~3.6**（压缩机级、ARI 类工况）。
   关键样本：Highly BTH420SDPC9EQA（41.8 cc、5680 W、COP 3.4，电池热管理专用）、Highly VTE848（13 kW R134a 储能热管理）、
   Boyard JVSB150Z24（15 cc、2100 W、COP 3.20 @3600 rpm、工况 A: Tevap 7.2 °C / Tcond 54.4 °C，可用 CoolProp 反算容积/等熵效率）。
3. **系统实测点（中可信，个别低）**：巴士空调实测 **17.5 kW @2000 rpm，输入 6.35 kW，COP 2.78**（与本模型 RegD 峰值 17.4 kW 同量级，
   但工况为车用空调低蒸发温度，非本系统高蒸发温度工况）；三花 EV 热泵变速曲线（容量/功率随转速单调、COP 随转速下降）。

## 与当前模型的差距（对标口径）

当前 8D4 冻结配置（72 cc、模型值）：4000 rpm shaft COP ≈ 4.60、6000 rpm ≈ 3.39（仅含机械效率 0.913，不含电机/变频损耗）。

- 整机级真实 COP 为 1.9~2.4 → 模型若对外报能耗，至少需补电机效率（0.85~0.92）与变频损耗（0.94~0.97），整机 COP 会降至 2.6~3.8，仍偏乐观。
- 压缩机样本 COP 2.8~3.6（ARI 低蒸发温度工况）；本系统蒸发温度高（≈20 °C），真实 COP 应更高，模型方向正确但缺电机损耗项。
- 5.6 kW 整机配 40 L/min @ 160 kPa，与当前冻结泵域（1600~4800 rpm ≈ 11~34 L/min）和 20 kPa 支路假设形成对照。

## 已完成的锚点反算验证（2026-09-01）

用 CoolProp 对 Boyard JVSB150Z24 工况 A 点（15 cc/rev、3600 rpm、Tevap 7.2 °C、Tcond 54.4 °C、吸气 35 °C、液体 46.1 °C）反算：

- 实际质量流量 = Q/(h1−h4) = 12.91 g/s；理论排量流量 = ρ·V·n = 14.55 g/s → **η_vol = 0.887**
- 压比 ≈ 5.1；等熵压缩功率 421 W vs 电输入 656 W → **η_is × η_motor = 0.641**（若电机效率 0.85~0.92，则 η_is ≈ 0.53~0.60）

与模型 5×5 效率图在 3000~4000 rpm、压比 5 处的取值（η_vol 0.91~0.94、η_is 0.62~0.63）方向一致，说明该图的量级并非凭空，
但等熵通道未含电机损耗——这正是"模型 COP 偏高"的量化来源。

## 后续标定建议（按优先级）

1. 用 Boyard JVSB150Z24 工况 A 点（排量/转速已知）经 CoolProp 反算 η_vol 与 η_is，
   与 `refrigeration.py` 中 5×5 效率图在相近压比处对照，必要时整体下修。
2. 在压缩机模型中新增电机效率 × 变频损耗项，使整机轴端功率与 2.3~3.2 kW 级真实输入对齐。
3. 用巴士空调 17.5 kW / 6.35 kW 实测点校核 17 kW 级能力段的功率量级（注意工况换算：蒸发温度差异）。
4. 效率图标定完成后重生成 `chiller_surrogates.json` 与 Physics-P 工件，再走逐级回归。
5. **（2026-09-04 新增）** 用 ToneCooling TC-1P104S 规格点（24.6 kPa @ 10 L/min，50% EGW）重校
   `BRANCH_DESIGN_DELTA_P_PA = 20 kPa`——注意本项目冷板为 13 列三温区串联结构（流道更长），
   可按 1.5~3.3 倍系数区间论证（见下文 A 节换算）。
6. **（2026-09-04 新增）** 拿到真实柜体母管管径/段长/弯头清单后，用 Darcy-Weisbach 精确化
   `MEDIUM_HEADER_TO_BRANCH_RESISTANCE_RATIO`——当前几何推算 ≈ 0.008（B 节），模型值 0.005 在合理区间内。

## 管网拓扑与流量分配锚点（2026-09-04 新增，含完整出处）

抓取方式：WebSearch 三线并查（学术文献 / 厂商 datasheet / 中文工程文献）。
可信度分级沿用上文三层：**A = 同行评审论文，B = 厂商规格书/CFD 页，C = 行业通稿/专利解读**。
**科研引用资格（2026-09-04 核实，详见 F 节）**：A 级可进参考文献；B 级仅可按 manufacturer-datasheet
网页引用并标注访问日期；C 级不可引用、仅限内部对标。
所有数值已内联，不依赖链接存活；URL 均为 2026-09-04 检索当日可访问地址。

### A. 支路（冷板）压降锚点 —— 对标 `BRANCH_DESIGN_DELTA_P_PA = 20 kPa`

| 来源 | 数据 | 可信度 |
|---|---|---|
| ToneCooling TC-1P104S BESS 冷板规格书（2180×790×42 mm，1P104S，50% EGW） | **24.6 kPa @ 10 L/min**（规格 <25 kPa，最大 30 kPa） | B |
| ToneCooling CFD 文章（同板型仿真页） | 8 / 10 / 12 L/min → 19.6 / 27.5 / 37.7 kPa（近似平方律） | B |
| VoltCoffer BESS 数值研究（底部冷板，⚠️ 非评审自述仿真博客，2026-09-04 由 B 降 C） | 5 / 7.5 / 10 / 12.5 / 15 L/min → 3.2 / 5.1 / 9.5 / 14.8 / 21.5 kPa | C |
| VoltCoffer 储能模组设计与仿真（同上，降 C） | 6 / 8 / 10 / 12 / 14 L/min → 8 / 12 / 15 / 20 / 26 kPa | C |
| MDPI Batteries 9(11):538（21700 圆柱小冷板） | 2 → 6 L/min → 1.09 → 8.08 kPa（小尺度下限参考） | A |

换算与对比：ToneCooling 10 L/min = 1.667e-4 m³/s × 1071 kg/m³ = 0.1785 kg/s →
**R ≈ 7.7×10⁵ Pa/(kg/s)²**。本项目支路设计点 20 kPa @ 0.08925 kg/s → R = 2.51×10⁶，
**约为真实产品的 3.3 倍**——本项目冷板 13 列三温区串联（流道更长），偏高有物理合理性，量级同档。

### B. 母管/支路阻力比 —— `MEDIUM_HEADER_TO_BRANCH_RESISTANCE_RATIO = 0.005` 的双重背书

1. **Darcy-Weisbach 几何推算**（本会话 2026-09-04 推算，量级验证而非精确标定）：
   假设母管 DN25（内径 25 mm，A = 4.91×10⁻⁴ m²）、摩擦系数 f = 0.03、段长 0.25 m（Pack 间距）、
   ρ = 1071 kg/m³（50% EGW）：
   `R_seg = f·(L/D)·ρ/(2A²) ≈ 6.7×10⁸ Pa/(m³/s)²` → 近端段（过全流量 0.6 kg/s）Δp ≈ **210 Pa**；
   支路 ≈ 软管（DN10 × 1.5 m，~6.5 kPa）+ 冷板 20 kPa ≈ **27 kPa** → **比值 ≈ 0.008**。
   结论：模型值 0.005 落在几何合理区间 **0.005~0.01** 内。⚠️ 管径/段长/弯头均未实测，仅量级背书。
2. **流量不均匀度间接对标**：见 C 节——本项目异程 13.78% 落在真实"未优化"区间，
   说明母管阻力比设定未明显失真。

### C. 流量不均匀度工程锚点

| 来源 | 系统 | 数据 | 可信度 |
|---|---|---|---|
| 于海洋, 姚昕烨, 郭玉恒, 侯飞, 潘海平. 基于STAR-CCM+的储能预制舱液冷管路结构设计优化[J]. **电池工业, 2026, 30(4): 460–471**（✅ 正式版已出版；该刊不注册 DOI，按卷期页引用）。**★ 原文已精读（2026-09-04，详见 `YU2026_pipeline_analysis.md`）：① 原方案判定为异程式逐级分流布局（"先后从 1 簇流至 6 簇、Pack 1 流入 Pack 8"，簇内流量单调递减 11.03→9.05 L/min）；② 反算真实母管/支路阻力系数比 ≈0.0013~0.004（⚠️ 修正早期压降比口径错误，曾误报 0.07/0.008——母管段过 7~8 倍流量，压降比须折算流量差），与本仓库 0.005 同量级→模型不均匀度预测偏保守；真实异程极差 19.8%（N=8 累积）vs 模型 13.78%（N=5），不矛盾；③ 支路阻力 R≈7.4e5 与 ToneCooling 7.7e5 双重吻合；④ 流量均衡后整舱温差仅降 0.36 ℃——"温差两层含义"活例证。** 引用用途限定：量级/成本锚点 + 拓扑判定（原文有据），不可引 ±8.5% 作"均流达标线"（实为单边改善，负偏差未动） | 5 MWh 储能预制舱 | Pack 间流量偏差 **+16.4%（未优化）→ +6.4%（节流后）**、负偏差维持 −8.5%；电芯最大温差 3.74 → 3.39 ℃；理论泵功 +20%（+140 W，19 个节流阀） | A* |
| 马志伟. 储能预制舱液冷管路流量均匀性优化[J]. **电池工业**, 网络首发 2026-05-28（录用定稿；✅ 元数据已补全） | 实际储能工程项目 | 各支路加限流孔精准调控阻力后**不均匀偏差 ±5% 以内（工程验收线）**；冷板等效多孔介质方法 | A* |
| Xian Y., Zhang Z., Bai X., Tao H., Li Y., Yang L., Bian X. (Nanjing Tech Univ.). Study on uniform distribution of liquid cooling pipeline in container battery energy storage system[J]. **J. Energy Storage**, 2025, DOI 10.1016/j.est.2025.115395（Elsevier SCI；孔板 orifice-plate 均流热设计，与限流孔主题直接对口；⚠️ 完整数据待取原文） | 集装箱 BESS | 支路孔板均流的热设计方法框架 | A |
| Siddiqui, O. K., Al-Zahrani, M., Al-Sarkhi, A., & Zubair, S. M. (2020). *Arabian Journal for Science and Engineering*, 45(7), 6005–6020, DOI 10.1007/s13369-020-04691-4（Springer，PIV 实测+数值，**引文已核实完整**） | 10 通道矩形歧管 | U 型宽歧管归一化速度 1.34~0.52、窄歧管 2.82~0.18；增大流量改善 U 型、恶化 Z 型 | A |
| J. Energy Storage《Optimization of an immersion cooling 46.5 kW/46.5 kWh battery module using flow resistance network shortcut method》（Elsevier，2024；⚠️ DOI/卷期从原文页补全） | 46.5 kWh 浸没式模块，32 L/min | **Z-flow RMSE 0.11 vs U-flow 0.30（比值 2.7×）**；RCT 优化后 0.04 / 0.068；温差均匀性 +16.45% / +56.16% | A |
| x-techcon：Archer eVTOL 电池包冷却专利解读 | 六电池包机翼冷却回路 | U 型近进出口 pack 流量大、中间小；工程解法 = 每支路定制节流器（modified U-flow） | C |
| 搜狐《大容量液冷储能柜…全链路仿真》（与 CNKI 第一条同案例的行业通稿） | 418 kWh 工商业柜 / 1P72S | 8 L/min/包、900 W/包 @ 0.5C；整柜电芯温差工程目标 ≤ 3 ℃ | C |

出处 URL：
- CNKI-1（预制舱节流阀）: https://publish.cnki.net/journal/portal/dcgy/client/paper/266b9a1e0947cc69f219a0f478ba2d22
- CNKI-2（限流孔 ±5%）: https://publish.cnki.net/journal/portal/dcgy/client/paper/6c5d52f751af3dd2152597197edc1402
- Springer PIV: https://www.springerprofessional.de/en/flow-distribution-in-u-and-z-type-manifolds-experimental-and-num/18098550
- J. Energy Storage（46.5 kWh）: https://www.sciencedirect.com/science/article/pii/s2352152x24039690
- Archer eVTOL 解读: https://m.x-techcon.com/article/148956.html
- 搜狐通稿: https://www.sohu.com/a/1058552044_122931178
- ToneCooling 规格书: https://tonecooling.com/?p=10396/
- ToneCooling CFD 页: https://tonecooling.com/?p=7360/
- VoltCoffer-1: https://www.voltcoffer.com?p=16641/
- VoltCoffer-2: https://www.voltcoffer.com/?p=23594/
- MDPI Batteries: https://www.mdpi.com/2313-0105/9/11/538/xml

### D. 命名对照（文献 ↔ 本项目，易踩坑）

| 文献命名 | 进出口位置 | 对应本项目 |
|---|---|---|
| **Z 型（Z-flow）** | 异侧 | **同程式 reverse-return** |
| **U 型（U-flow）** | 同侧 | **异程式 direct-return** |

方向性互证：文献 U 型"近进出口支路流量大" ↔ 本项目异程式"支路 1 流量最大（供/回反向夹击远端）"。

### E. 对本项目的对标结论（2026-09-04）

- **异程 13.78%**（5 pack）落在真实"未优化 direct-return"工程区间（±16.4% 为 40+ pack 整舱，量级吻合）；
- **同程 4.908% 已优于 ±5% 工程验收线**（CNKI-2）——不加任何节流即达标；
- JES 论文 Z/U 不均匀度比 **2.7×** 与本项目 **2.70×** 同量级（几何不同，纯量级巧合，仅作方向性参照）；
- 母管阻力比 0.005 从"无依据假设"升级为**"几何推算（≈0.008）+ 工程对标双重背书"**，但仍是待实测假设。

### F. 科研引用资格速查（2026-09-04 核实）

**✅ 可直接进参考文献（同行评审）**
1. Siddiqui et al. (2020)，*Arabian J. Sci. Eng.* 45(7): 6005–6020，DOI 10.1007/s13369-020-04691-4
   ——引文已核实完整（KFUPM 机构库 + Semantic Scholar 双确认，peer-reviewed）。
2. J. Energy Storage（Elsevier，SCIE）2024 年 46.5 kWh 模块文——DOI/卷期需从原文页补全
   （PII: s2352152x24039690）。
3. MDPI *Batteries* 9(11): 538——SCIE 开放获取；按 MDPI 规则 DOI 应为 10.3390/batteries9110538，
   引用前核对。⚠️ 部分单位对 MDPI 期刊认可度有争议，视本校政策。
4. 补充（核实中意外获得，正式引文齐全）：刘周斌, 朱涛, 姜巍, 等. 储能锂离子电池包冷却系统的
   数值模拟与结构优化[J]. 中国电力, 2023, 56(10): 202–210.（STAR-CCM+ 储能包冷却，
   含仿真-实验 ≤5% 误差验证；主题为浸没式 vs 间接液冷，与流量分配主线部分相关）

**✅ 已补全元数据（2026-09-04 检索；于海洋篇正式版已出版）**
- 于海洋, 姚昕烨, 郭玉恒, 等. 基于STAR-CCM+的储能预制舱液冷管路结构设计优化[J].
  电池工业, **2026, 30(4): 460–471**（正式版 2026-08-25 刊出；英文题名 Optimization of the
  liquid cooling pipeline structure for a prefabricated energy storage cabin based on STAR-CCM+）。
- 马志伟. 储能预制舱液冷管路流量均匀性优化[J]. 电池工业, 网络首发 2026-05-28（录用定稿，
  正式卷期出版后替换）。
- **DOI 情况**：《电池工业》（ISSN 1008-7923, CN 32-1448/TM）官方目录页与 CNKI 页面均无 DOI
  字段，该刊未注册 DOI——**引用按 GB/T 7714 卷期页格式即可，无 DOI 属正常**，勿编造。
- 该刊为知网收录普通学术期刊（非北大核心/CSCD）；若研究需核心期刊背书，
  用下条 Xian 2025 替代或并列。
- **核心对标数据 ±16.4% / ±8.5% / ±5% 的正式引用已可落地**（引文如上）。

**✅ 核心期刊替代首选（SCI）**
- Xian Y. et al. Study on uniform distribution of liquid cooling pipeline in container
  battery energy storage system[J]. J. Energy Storage, 2025, DOI 10.1016/j.est.2025.115395
  （南京工业大学；孔板均流，与限流孔 ±5% 主题直接对口；DOI 经 NSTL 记录确认）。

**⚠️ 仅可按厂商数据引用（不可进正文数据表）**
- ToneCooling TC-1P104S 规格书：网页引用格式（ToneCooling, "TC-1P104S BESS Liquid Cold Plate",
  URL, accessed 2026-09-04），只能作量级佐证，审稿人可能挑战。

**❌ 不可引用（仅内部对标）**
- VoltCoffer（自述仿真博客，已降 C 级）、搜狐通稿、x-techcon Archer 解读。
- 升级路径：Archer 专利解读 → 检索 USPTO/WIPO 原始专利号，**专利文献可正式引用**。

**关于 B 节 Darcy-Weisbach 推算**：属本文自行计算（非文献），论文中按"本文推导"呈现；
DN25 母管/段长 0.25 m 等几何假设若写进论文，需引用管路标准或柜体规格作支撑。

补充出处链接（核实用）：
- Siddiqui 2020 机构记录: https://pure.kfupm.edu.sa/en/publications/flow-distribution-in-u-and-z-type-manifolds-experimental-and-nume/
- 《中国电力》2023 引文来源: https://news.bjx.com.cn/html/20231123/1345343.shtml
- 《电池工业》网络首发列表（于海洋等 2025 出处）: https://dcgy.cbpt.cnki.net/portal/journal/portal/client/shoufa_list?pageNum=7
- 《电池工业》网络首发列表（马志伟 2026 出处）: https://dcgy.cbpt.cnki.net/portal/journal/portal/client/shoufa_list?pageNum=3
- 于海洋等原文详情页（HTML/PDF 下载入口）: https://publish.cnki.net/journal/portal/dcgy/client/paper/266b9a1e0947cc69f219a0f478ba2d22
- 马志伟原文详情页（HTML/PDF 下载入口）: https://publish.cnki.net/journal/portal/dcgy/client/paper/6c5d52f751af3dd2152597197edc1402
- Xian 2025 NSTL 记录: https://zgxa.nstl.gov.cn/paper_detail.html?id=8aa85e6a0ee6057c4e3f93460a52a8a2

---

## 注意事项

- `system_level_measured_points.csv` 中标注 `low-medium` 的行来自扫描件/预览页，使用前需复核原始文档。
- book118 源文档为付费预览，仅预览可见部分已转录；原件需自行获取。
- 所有 URL 均为抓取当日可访问地址，可能失效；CSV 内已内联数值，不依赖链接存活。
