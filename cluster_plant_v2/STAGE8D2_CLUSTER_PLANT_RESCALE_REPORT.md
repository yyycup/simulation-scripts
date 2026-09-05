# Stage 8D2：Cluster Plant Equipment Rescaling Report

## 结论

建议将本轮结果冻结为 **Cluster Plant V2.1 / Cluster-Sized Baseline**，适用边界为 Stage 8D/8D2 定义的 600 s 工况族。只修改了 Chiller 的物理尺度参数和 Cold Plate reference flow；Tank、Pump、Branch/Header、时间常数、延迟、Battery/Pack topology 均保持不变。

Cluster-P V1 与 MPC 没有修改。旧 Predictor 与新 Plant 的轨迹一致性门禁按预期失败，因此下一步必须是 Stage 9A2，而不是 MPC 调参。

## 1. 修改边界和顺序

严格分两步执行：

1. Step A：扫描并固化 Chiller 物理参数，独立验证制冷循环与 Plant 动态。
2. Step B：在 Step A 基础上扫描 5/6.5/8 L/min/Pack Cold Plate reference flow，再单独固化冷板参数。

未修改项目：

- Tank = 3 L。
- Pump 模型、1600–4800 rpm 范围及当前 3600 rpm 验证点。
- Branch Δp = 20 kPa、Header 参数。
- Battery-ROM、Pack topology。
- Compressor tau = 5 s、Evaporator tau = 45 s。
- Supply delay = 15 s、Return delay = 20 s。
- Cluster-P V1、MPC。

## 2. Chiller sizing scan

扫描轴：

- Compressor displacement：14/16/18/20/22 cc/rev。
- Evaporator area：2.0/2.5/3.0/3.5/4.0 m²。
- Condenser area：2.0/2.5/3.0/3.5/4.0 m²。
- Nominal condenser-air mass flow：2.0/2.5/3.0/3.5 kg/s。

共 500 个物理组合，每个组合分别求解 4000/6000 rpm；103 个组合同时进入 5.5–6.0 kW 和 7.5–8.5 kW 初始目标窗，并通过质量流量、蒸发器、冷凝器和循环能量闭合。正式模型没有使用 `Qevap *= factor`。

初选 20 cc/rev、4/4 m²、3.5 kg/s 得到 5.675/7.914 kW。3600 s 探索表明，随 SOC 降低和发热上升，最后 600 s Qload 达到约 5.97 kW，该初选存在约 0.59 kW 热缺口。因此回到 A 候选集合，选择目标窗上沿且冷凝温度有余量的最终参数。

## 3. Chiller 最终参数

| 参数 | 旧 Plant | V2.1 最终值 | 比例/说明 |
|---|---:|---:|---|
| Compressor displacement | 5.525 cc/rev | **22.0 cc/rev** | 3.98× |
| Evaporator area | 1.0 m² | **2.5 m²** | 2.5× |
| Condenser area | 1.0 m² | **4.0 m²** | 4× |
| Nominal condenser-air mass flow | 1.5 kg/s | **3.5 kg/s** | 2.33×；1200 rpm 时模型空气流量为 1.05 kg/s |
| Compressor efficiency maps | 原图 | 不变 | 未用效率图补容量 |
| Mechanical efficiency | 原值 | 不变 | 0.913457... |

这是一组 compressor + evaporator + condenser + fan-side 联合扩容：排量决定制冷剂流量基础，两侧 HX 和空气侧能力避免仅提高排量后受到换热器/冷凝温度约束。

## 4. 4000/6000 rpm 制冷能力

条件：Tin=25 °C、Tamb=35 °C、总流量约 25.2 L/min、fan=1200 rpm。

| 指标 | 4000 rpm | 6000 rpm |
|---|---:|---:|
| Qevap | **5,985.55 W** | **8,239.60 W** |
| Qcond | 6,792.41 W | 9,718.11 W |
| Refrigerant compression power Wref | 806.85 W | 1,478.51 W |
| Shaft power Wshaft | 883.30 W | 1,618.59 W |
| Shaft COP | 6.776 | 5.091 |
| Refrigerant mass flow | 0.037690 kg/s | 0.054335 kg/s |
| Tevap | 19.727 °C | 17.772 °C |
| Tcond | 44.571 °C | 48.627 °C |
| Evaporator UA | 2,079.23 W/K | 2,098.27 W/K |
| Condenser UA | 1,178.04 W/K | 1,188.74 W/K |

两个点均通过：

- mass-flow relative residual < 5e-15；
- evaporator/condenser relative residual < 3e-14；
- cycle energy relative residual < 1e-16。

6000 rpm 的 Tcond 仍低于模型 55 °C 上界约 6.4 °C，未使用贴边候选。

## 5. Cold Plate 候选与最终 reference flow

保持不变：

- `h_nominal = 2000 W/(m²·K)`；
- exponent = 0.8；
- area = 0.5 m²/Pack；
- plate heat capacity = 6000 J/K/Pack。

扫描候选：

| Reference flow | Reference mass flow | 当前约 5.04 L/min/Pack 下 h | 当前工作点 UA/Pack |
|---:|---:|---:|---:|
| 5 L/min | 0.08925 kg/s | 约 2013 W/(m²·K) | 约 1006 W/K |
| 6.5 L/min | 0.116025 kg/s | 约 1632 W/(m²·K) | 约 816 W/K |
| **8 L/min** | **0.1428 kg/s** | **约 1382 W/(m²·K)** | **约 691 W/K** |

最终选择 **8 L/min/Pack = 0.1428 kg/s/Pack**。

原因不是单点最低温度：在 20→25.2→30 L/min 总流量扫描中，8 L/min reference 给出最平滑、单调的温度响应；5 和 6.5 L/min 候选在 30 L/min 下出现 Tmax 或跨 Pack 温差反向放大。8 L/min reference 保留了流量变化对 h/UA 的可控敏感性，同时不再像旧 1.2 kg/s reference 那样把当前 h 压到 nominal 的约 12.5%。

实现时拆开了一个旧隐式耦合：原 `NOMINAL_COOLANT_MASS_FLOW_KG_S=1.2` 同时被蒸发器和冷板引用。V2.1 新增冷板专用 `COLD_PLATE_REFERENCE_MASS_FLOW_KG_S=0.1428`；蒸发器原 1.2 kg/s reference 保持不变，避免 Step B 再次改变已冻结的 Step A Chiller。

## 6. T0–T4 和流向验证

正式场景均为 dt=5 s、Tank=3 L、Pump=3600 rpm、Branch Δp=20 kPa、Tamb=35 °C；每例 600 s。

| Case | Tavg final (°C) | Tmax max (°C) | Cluster ΔT max (K) | Inter-Pack ΔT max (K) | Tail Qload (kW) | Tail Qevap (kW) | Deficit (kW) | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| T0 constant, 560 A/4000 rpm | 25.151 | 25.583 | 1.186 | 0.073 | 5.566 | 5.599 | -0.033 | PASS |
| T1 compressor 2000→4000 rpm | 25.444 | 25.830 | 1.052 | 0.065 | 5.523 | 5.004 | +0.518 | PASS；阶跃期间有预期瞬态缺口 |
| T2 RegD | 24.436 | 25.376 | 1.188 | 0.074 | 3.991 | 5.585 | -1.594 | PASS |
| T3 560 A thermal-balance | 25.151 | 25.583 | 1.186 | 0.073 | **5.566** | **5.599** | **-0.033** | PASS；容量侧可平衡 |
| T4 1120 A/6000 rpm stress | 30.311 | 30.887 | 1.566 | 0.100 | **19.043** | **7.917** | **+11.127** | PASS；稳定运行但不可热平衡 |

T3 的末段 Qevap 比 Qload 高约 32.6 W，说明在本轮 600 s 设计工况下 actuator capacity 已允许热平衡。Tavg 仍有约 +0.54 °C/h 的短窗斜率，说明热容状态尚未达到严格稳态；本结论是“容量可平衡”，不是“所有热状态已完全收敛”。

额外 3600 s、560 A 探索不是正式冻结 gate：随 SOC 降低，末段 Qload 上升到约 6.05 kW，最终候选 Qevap 约 5.59 kW，仍有约 0.46 kW 缺口。这表明 V2.1 不应被描述为覆盖完整 0.5C 深放电全过程；若该工况成为正式需求，应重新定义容量设计点或使用更高转速/功率调度。

流向门禁：

- Forward：T0/T3 120/120 steps，PASS。
- Reverse：120/120 steps，PASS。
- Forward→Reverse @300 s：120/120 steps，PASS。

所有方向的制冷求解、质量守恒、局部热守恒和状态有限性均通过。

## 7. 1120 A stress 结论

T4 使用最大 6000 rpm：

- Tail Qload≈19.04 kW。
- Tail Qevap≈7.92 kW。
- Thermal deficit≈11.13 kW。
- Tail Wshaft≈1.61 kW，COP≈4.93。

因此 1120 A 仍是明确的 stress/短时工况。V2.1 没有按 1120 A 持续热平衡设计；这一缺口不能由 Cluster-P 或 MPC 权重消除。

## 8. 新旧 Plant 差异

同为 560 A、4000 rpm、3600 rpm pump、600 s：

| 指标 | 旧 Plant | V2.1 | 差异 |
|---|---:|---:|---:|
| Battery Tavg final | 26.853 °C | 25.151 °C | -1.702 °C |
| Battery Tmax | 26.949 °C | 25.583 °C | -1.366 °C |
| Tail Qevap | 1.586 kW | 5.599 kW | +4.013 kW |
| Compressor shaft power | 约 0.210 kW | 约 0.890 kW | +0.680 kW |
| Cold-plate h at working flow | 约 252 W/(m²·K) | 约 1382 W/(m²·K) | 5.49× |
| Cold-plate UA/Pack | 约 126 W/K | 约 691 W/K | 5.49× |

旧结果和 Stage 8D 数据均保留；V2.1 是新的物理基线，不要求与旧 Plant `max_abs_difference=0`。

## 9. 测试结果

| 验证层级 | 结果 |
|---|---|
| Step A 直接 refrigeration/dynamics/Plant tests | 22 tests，PASS |
| R134a speed × flow matrix | 18/18 cases，PASS |
| Step B cold-plate/ReducedPack/Cluster/final/isolation targeted | 37 tests，PASS |
| 全部 Plant、validation 与 project isolation tests | **152 tests，PASS** |
| T0–T4 + forward/reverse/switch | **7/7 cases，PASS** |
| Cluster-P/MPC control tests | 18 tests：17 PASS，1 expected FAIL |

Control 侧唯一失败为旧 offline Predictor/MPC trajectory 与新 Plant 的逐点比较：100 个比较量中 19 个超出旧容差，最大差异主要来自制冷能力已由约 1.65 kW 改为约 5.99 kW。这不是 V2.1 Plant failure，也不应在本轮通过修改 MPC 测试或权重隐藏。

## 10. 是否建议冻结新 Plant

**建议冻结。** 名称：`Cluster Plant V2.1 / Cluster-Sized Baseline`。

冻结理由：

- Chiller 由物理设备参数扩容，而非输出倍率。
- 4000/6000 rpm 同时达到 5.986/8.240 kW 目标尺度。
- Cold Plate reference 已从无法追溯的 1.2 kg/s/Pack 拆分并重定标为 0.1428 kg/s/Pack。
- 560 A、600 s 工况具备容量侧热平衡；1120 A 缺口被明确保留为 stress 特性。
- Plant、validation、isolation 和全部流向门禁通过。

冻结边界：该结论不声称 4000 rpm 可以覆盖 560 A 完整深放电全过程；额外 3600 s 结果已记录为后续调度/设计包络问题。

## 11. Stage 9A2 需要重标定的内容

保持现有 20-state Cluster-P V1 状态定义和接口作为起点，但必须：

1. 用 V2.1 重新生成 forward、reverse 和切换工况的 identification/validation/test 数据。
2. 重新识别 compressor→Qevap、冷却液温度、Battery Tavg/Tmax/ΔT 的动态增益。
3. 重新覆盖 4000 rpm 约 6 kW、6000 rpm 约 8.2 kW 的 actuator envelope。
4. 重新识别 cold-plate reference 改变后的流量敏感性和 Pack 串联温差。
5. 保留 5/45 s actuator/evaporator tau 和 15/20 s delays，并重新验证 Predictor 是否正确恢复这些时间位置。
6. 单独包含 1120 A stress 数据，但不要让其不可平衡特性污染 560 A nominal 标定目标。
7. 生成新的 Predictor artifact/version；保留旧 Cluster-P V1 与旧 Plant 对应关系，不静默覆盖。
8. Predictor gates 通过后才恢复 Stage 9B MPC，不在此之前调 MPC 权重。

## 12. 结果文件

- Chiller physical scan：`validation/results/stage8d2_cluster_plant_rescale/chiller_sizing_scan.csv`
- Step A refrigeration matrix：`validation/results/stage8d2_cluster_plant_rescale/step_a_refrigeration/`
- Cold Plate scan：`validation/results/stage8d2_cluster_plant_rescale/cold_plate_reference_scan.csv`
- Cold Plate formula table：`validation/results/stage8d2_cluster_plant_rescale/cold_plate_reference_formula.csv`
- V2.1 case summaries：`validation/results/stage8d2_cluster_plant_rescale/v21_validation/v21_summary.csv`
- V2.1 full timeseries：`validation/results/stage8d2_cluster_plant_rescale/v21_validation/*_timeseries.csv`

Stage 8D2 到此停止。未 commit，未 push；未进入 Stage 9A2 或 MPC。
