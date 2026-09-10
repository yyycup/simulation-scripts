# 热量流模型：储能账本与独立初始化修复

日期：2026-09-08。范围：热流模型装配、能量账本及其验证调用。

本轮修复了管路储能缺少质量因子、独立 Plant 初态复制不完整、初始化提前推进状态、蒸发器接口未被调用四个问题。没有改动电池产热、冷板换热、泵、水箱、R134a 循环、执行器或固定时间延迟的物理方程。

## 1. 管路储能及能量边界

旧账本使用 `cp * sum(T_queue)` 作为焦耳，缺少节点质量。现在初始化必须显式传入 `transport_reference_mass_flow_kg_s`，每个温度队列节点对应固定等效质量：

```text
m_node = m_dot_ref * dt_delay                 [kg]
M_pipe = m_dot_ref * tau_delay                [kg]
E_pipe = m_node * cp * sum(T_queue)           [J]
```

参考流量保存在能量快照中，各步保持不变。运行流量变化时不会重算质量，因此不会凭空产生 `cp*T*delta_mass` 储能。这里定义的是与参考流量、固定延迟一致的等效管内容量，尚不是实测管路容积标定。

本账本的边界是电池、冷板、冷却液管路和水箱，蒸发器冷却液侧的 `Q_evap_applied` 是输出热流：

```text
R_system = Q_gen - Q_air - Q_evap_applied
           - d(E_battery + E_plate + E_supply + E_return + E_tank)/dt
```

它不等同于包含压缩机轴功、冷凝器和制冷剂缓冲能量在内的总系统账本。电池到冷板、冷板到冷却液的内部热流在上述总式中抵消。冷板内冷却液温度是代数输出，没有额外的动态储能状态。

在运行流量等于参考流量时，FIFO 的队列变化与焓流差一致，回路可在数值精度内闭合。

在流量变化时，原来的固定时间 FIFO 与固定管内容量不再物理一致。其可独立计算的差额为：

```text
Q_mismatch = (m_dot - m_dot_ref) * cp
             * [(T_evap_out - T_supply) + (T_cluster_return - T_tank_return)]
```

新增 `transport_flow_mismatch_w` 单独报告这个量；`residual_system_w` 和历史名称 `residual_loop_implicit_transport_w` 保留原始残差，不减掉差额制造“闭合”。恒流的通过结论不能外推成变流量下已经严格守恒。

若下一阶段要求固定物理管容下的变流量严格守恒，需要修改输运模型，例如按累计输送质量推进流体，或使用守恒的有限体积离散。这会改变延迟及温度响应，本轮没有实施。

## 2. 独立 Plant 初态与接口

- `build_independent_hc_plant()` 深复制每个 Pack 的电池对象，保留配置、SOC、极化和温度等状态；冷板传递对应区域的壁温。
- 默认冷板保留来源的区域热容和面积；自定义工厂保留自己的参数，但区域划分必须匹配。
- 供水、回水复制完整温度队列，蒸发器缓冲能量也被保留。
- 泵、制冷循环和水力网络也复制为独立实例，避免后续参数编辑影响另一个 Plant。
- Pack 数默认从来源推断；显式指定时必须与来源一致。
- `from_equilibrium()` 在电池簇副本上计算初始回水，不再提前推进正式状态。
- `step()` 和初始化通过 `EvaporatorHeatCurrent.outlet_temperature_from_applied_heat()` 计算出口，仍使用滞后的 `Q_evap_applied`。
- 冷板瞬时热流诊断由 HC 换热律在首步重新计算，不承诺复制原 LMTD 模型的瞬时诊断值。

## 3. 本轮验证结果

定向回归：**98/98 通过**，其中新增 7 个修复用例。首次复现观察到 4 个行为断言失败，另有 3 个测试暴露缺失的参考流量接口；修复后全部通过。CSV 临时目录测试曾因 Windows 沙箱权限失败，随后通过权限审查在沙箱外原样重跑，完整 98 项通过，耗时 51.711 s。

修改前保存了一条 60 s / 12 步 HC 轨迹，包含电流、泵速、压缩机转速和流向变化。修改后逐项比较 7 个温度、热流输出，**max_abs_difference = 0.0**。该结论限定于此默认配置轨迹；任意运行时刻初态复制修复会有意改变此前错误的重置行为。

三组冒烟验证分别在 legacy、HC 上各运行 120 s / 24 步，所有状态有限、制冷循环求解成功。原始逐步 CSV 已重新读取校验，关键结果如下：

| 工况 | 后端 | 原始系统残差最大绝对值 W | FIFO 流量差额最大绝对值 W | 扣除独立端口推导差额后的剩余误差 W |
|---|---|---:|---:|---:|
| 恒流恒输入 | legacy | 1.8384e-08 | 0 | 1.8384e-08 |
| 恒流恒输入 | HC | 1.6164e-08 | 0 | 1.6164e-08 |
| 恒流，电流/压缩机/流向变化 | legacy | 1.4755e-08 | 0 | 1.4755e-08 |
| 恒流，电流/压缩机/流向变化 | HC | 2.5209e-08 | 0 | 2.5209e-08 |
| 泵速 3600→4500 rpm | legacy | 1569.6091 | 1569.6091 | 1.4755e-08 |
| 泵速 3600→4500 rpm | HC | 1439.0980 | 1439.0980 | 2.0177e-08 |

最后一列用于定位差额来源，不作为变流量能量守恒通过指标。通过条件是恒流原始残差小于 1e-6 W；变流量时原始残差与独立端口公式的差小于 1e-6 W。

在同一组恒流 HC 状态上重算旧账本，旧系统残差峰值为 7733.4531 W；修复账本为 1.6164e-08 W。证据说明本工况的大残差来自储能账本的质量遗漏，不能解释成“未暴露的冷板冷却液储能”。

Stage 4、Stage 5 现有验证入口及报告生成各完成 20 s / 4 步接口冒烟。**本轮没有重新运行九工况各 600 s 的完整 Stage 5，也没有运行整个仓库的全部测试。** 旧 Stage 4/5 结果保留；其管路储能解释应以上述修正为准。

## 4. 文件与复现

- 修改：`thermal/heat_current_energy_balance.py`、`thermal/heat_current_stage5_ledger.py`、`thermal/heat_current_plant.py`。
- 同步调用及报告说明：`validation/validate_heat_current_stage4.py`、`validation/validate_heat_current_stage5.py`。
- 测试：`tests/test_heat_current_energy_balance.py`、`tests/test_heat_current_closure_repairs.py`。
- 新证据目录：`validation/results/heat_current_closure_20260908/`，包含前后轨迹、6 个逐步 CSV、`verification_summary.json` 和 `verify_repairs.py`。
- 账本 API 变化：`initial_energy_snapshot()` 新增必填参考流量；本目录内非历史调用均已同步。历史结果、外部临时脚本若调用旧签名，需要显式给出参考流量。CSV 增加 `transport_flow_mismatch_w` 列。

从包含 `cluster_plant_v2` 的工作区父目录运行：

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -B -m cluster_plant_v2.validation.results.heat_current_closure_20260908.verify_repairs
```

该脚本使用既有独立性测试的标准工况构造器，仅作为本轮证据复现入口。它读取并保留修改前的 `before_trajectory.json`，重建本轮其余证据；不会覆盖旧 Stage 4/5 结果。
