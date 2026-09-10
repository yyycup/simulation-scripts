# HC 固定管容、质量守恒输运

日期：2026-09-08。本轮按 `karpathy-guidelines` 实施已确认的管路升级方案。

## 2026-09-09 默认延迟定版

依据于海洋等（2026）的 5 MWh 预制舱管路数据，单 Pack 设计流量为
10 L/min，二级母管为 Φ25、三级支管为 Φ14。缩放到当前 5-Pack Plant 后，
论文流量锚点为 50 L/min，与当前 3600 rpm 下约 50.4 L/min 一致。因此保留
参考流量，将默认供液/回液延迟由 15/20 s 定为 5/5 s；在 1071 kg/m³ 下，
对应两侧各 4.4982 kg、4.2 L，总等效管容 8.4 L。

公开整机数据进一步给出直接的充液量约束：Billion Fusio 215 kWh、5 模组液冷柜
标称约 11 L 冷却液；SolaX TRENE 261 kWh 液冷柜的维护规程给出一次加注约
9-10 L。当前模型显式水箱为 3 L，冷板没有独立冷却液蓄积状态，因此剩余
6-8 L 由供回液输运状态等效表示。在 50.4 L/min 下，对应总输运时间
7.14-9.52 s；按 5 s Plant 步长取总计 10 s，并在缺少供回不对称证据时分为
供液 5 s、回液 5 s。最终等效管容为两侧各 4.2 L，加水箱共 11.4 L。

据此，5/5 s 定为当前 5-Pack Plant 的工程名义默认值。它有论文流量/管径和
同规模整机充液量双重支持，但仍不是特定实物的几何或阶跃实测标定值。若以后
获得实际内径、管长、部件持液量或温度阶跃数据，应以实测重新计算并替换。

公开来源：

- Billion Fusio 215/258/344 kWh 产品参数：
  https://www.billion.com/ESS/Fusio-215-258-344kWh
- SolaX TRENE 261 kWh 官方维护手册：
  https://id.solaxpower.com/uploads/file/trene-p125B261l-e-maintenance-manual-en.pdf

下文 15/20 s、29.4 L 及九工况结果属于 2026-09-08 旧默认的历史验证记录，
未由参数修改追溯覆盖。

更新后的定向验证为 27/27 通过，覆盖默认 Plant、延迟门、短时三工况、固定管容
质量守恒和 HC 能量账本。3600 rpm 工作点下，两侧固定质量均为 4.4982 kg，
总等效管容为 8.4 L；do-mpc 延迟链由 3+4 状态同步缩为 1+1 状态，7 状态模型
完成构建及 Plant 状态映射。

5/5 s 定版九工况已在独立目录
`validation/results/heat_current_mass_transport_20260909_delay5s_final/` 完成：9/9 工况
通过，每种后端每工况 600 s / 120 步。独立核验重新读取 36 个 CSV，HC 原始系统
能量残差峰值为 2.5851136342680547e-08 W，供回液质量最大变化均为 0.0 kg，
电池平均温度相对 legacy 的 RMSE 范围为 0.0453-0.0755 K。完整核验见
`full_validation_audit.json`。冻结的 Physics-P 14 状态预测器仍保留原 3+4 延迟状态，
不属于本次 Plant/HC 定版范围，相关闭环结论需在预测器重标定后重新验证。

## 模型与适用边界

新增 `thermal/coolant_mass_transport.py` 中的 `CoolantMassTransport`。管路绝热、无壁体蓄热、无轴向扩散；以从出口到入口排列的流体段保存 `(质量, 温度)`。冷却液密度、比热固定，每个外部步内入口温度和流量保持不变。

每步输入质量 `dm = mass_flow_kg_s * dt_s`，按出口方向排出同样质量。可排出部分流体段，也可在一步内排出多个流体段；当 `dm` 大于整个管内容量时，计入该步内直接穿过管路的新入口流体。相邻相同温度流体段可合并，不改变质量和能量。

```text
M_pipe = sum(m_parcel)                         恒定
E_pipe = cp * sum(m_parcel * T_parcel)          J
T_out_avg = sum(m_removed * T_removed) / dm     K
E_after - E_before = dm * cp * (T_in - T_out_avg)
```

返回的是整个外部步的质量加权平均出口温度，供下游冷板或水箱使用。零流量时保存全部状态，返回出口端温度作为诊断值；整个 HC 回路仍沿用泵必须维持正流量的约束。

在恒流下，传播时间是 `M_pipe / mass_flow`；在变流量下，前沿到达取决于累计输送质量达到管内质量，不能用一个固定时间队列表示，也不能简单用当前瞬时流量推算全部历史延迟。

本轮没有新增压降、管壁散热或冷却液物性拟合。电池产热、冷板换热、R134a 循环、泵、水箱及执行器物理方程沿用原实现。流向标志仍只改变冷板遍历方向，未将供回水管路整体倒置。

## 管容与初始化

正式九工况采用同一名义工况：泵速 3600 rpm，求得参考流量约 `0.8996400000000264 kg/s`。

| 管路 | 参考延迟 | 固定质量 | 按 1071 kg/m³ 换算的等效体积 |
|---|---:|---:|---:|
| 供水 | 15 s | 13.4946 kg | 12.6 L |
| 回水 | 20 s | 17.9928 kg | 16.8 L |

这些质量在低流量、高流量、泵速阶跃工况之间保持一致。它们是现有名义流量及延迟对应的等效管容，尚未由真实管路长度、内径和支路容积标定。

`from_fixed_delay()` 将初始 FIFO 温度历史映射为等质量流体段，复制从出口到入口的温度顺序。它是在指定参考流量下定义初始空间温度分布，不能从旧时间队列重建任意变流量历史的真实质量分布。

## 接入与兼容

通过以下参数显式启用新管路：

```python
from cluster_plant_v2.thermal.heat_current_plant import build_independent_hc_plant

hc = build_independent_hc_plant(
    legacy_plant=legacy,
    transport_reference_mass_flow_kg_s=reference_flow,
)
```

省略该参数时继续使用旧固定时间 FIFO。原 `ClusterPlant` 和 `HeatCurrentSystemLink` 保留旧模式。新管路也可直接构造并传入 `HeatCurrentPlant`，其质量和比热应与所建回路一致。

能量账本读取新管路实际 `stored_energy_j`，不再用温度队列乘等质量系数估算。对新管路，`transport_flow_mismatch_w` 为零；`residual_system_w` 保持原始残差，不减去任何补偿项。

`queue_values` 现在对新管路表示流体段温度，须与 `parcel_masses_kg` 配对理解，不能直接作等权平均。流体段数量可变化，HC 的状态计数改为实时读取；新组件报告质量及温度存储坐标数量，不声称是固定阶数 ROM。

## 验证记录

- 新增 9 个用例先因缺失组件而失败，实现后通过；覆盖部分排出、超过管容、零流量、参考 FIFO 一致性、累计质量前沿、时间步细分、随机变流量守恒及 HC 原始残差。
- 定向回归 **111/111 通过**，耗时 98.733 s；包括旧固定时间延迟与既有热流组件测试。
- 120 s / 24 步泵速阶跃冒烟完成，新 HC 原始系统残差峰值 `1.621128831175156e-08 W`。CSV 位于新结果目录的 `smoke/`。
- 60 s / 12 步参考流量对照包含电流、压缩机转速和流向变化，旧 FIFO 与新质量输运的 7 个温度、热流输出比较：`max_abs_difference = 0.0`。见 `reference_flow_comparison.json`。变流量时允许出现物理响应差异。
- 九工况各 600 s 的完整运行及独立 CSV 核验均已完成，**9/9 通过**。HC 共 1080 步，两种后端合计 2160 步；运行退出码与独立核验退出码均为 0。

## 完整运行

结果目录：`validation/results/heat_current_mass_transport_20260908/full/`。

验收条件：每工况 120 个时间步、全部状态有限、全部制冷循环求解成功、HC 原始系统残差小于 `1e-6 W`；管内质量在各步和九个工况之间保持一致。先前“扣除流量差额后闭合”的判断不用于这次验收。

独立核验重新读取全部 **36 个 CSV**，逐步核对时间轴、有限值、质量恒定、管路实际储能差分与账本的一致性，并由能量收支各项重新计算原始系统残差。结果见 `full_validation_audit.json`：

| 工况 | HC 原始系统残差最大绝对值 W | 管内质量最大变化 kg | 结果 |
|---|---:|---:|---|
| V1 恒定输入 | 2.4316e-08 | 0.0 | 通过 |
| V2 电流阶跃 | 2.4820e-08 | 0.0 | 通过 |
| V3 压缩机阶跃 | 2.3991e-08 | 0.0 | 通过 |
| V4 泵速阶跃 | 1.8252e-08 | 0.0 | 通过 |
| V5 低流量 | 2.2793e-08 | 0.0 | 通过 |
| V6 高流量 | 2.5337e-08 | 0.0 | 通过 |
| V7 反向流动 | 2.4068e-08 | 0.0 | 通过 |
| V8 流向切换 | 1.6880e-08 | 0.0 | 通过 |
| V9 RegD | 2.0760e-08 | 0.0 | 通过 |

全工况原始系统残差峰值为 **2.53370444625034e-08 W**；管内质量变化峰值为 **0.0 kg**。新增模型的 `transport_flow_mismatch_w` 在全部步中为零，不参与扣减。

对旧 `ClusterPlant` 的电池平均温度 RMSE 为约 0.023–0.141 K。这个对照同时包含冷板换热律及管路模型差异，不等于新管路对真实设备的误差，也没有用调参强制消除该差异。

验证命令（从包含 `cluster_plant_v2` 的父目录运行）：

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -B -u -m cluster_plant_v2.validation.validate_heat_current_stage5 --conservative-transport --no-figures --output-dir cluster_plant_v2/validation/results/heat_current_mass_transport_20260908/full
```

本次首个完整工况由串行入口保存；之后切换到结果目录内的 `run_remaining_cases.py`，最多三个进程并行调用同一个 `_run_one()`。已完成 CSV 复用，未完成工况完整重算。日志分别保存在 `full_stdout.log` 和 `parallel_stdout.log`，完整退出状态记录在 `parallel_exit_code.txt`。

旧结果目录未覆盖。此验证只检验模型数值守恒、运行稳定性和温度响应对照，不替代真实设备或管路参数验证。
