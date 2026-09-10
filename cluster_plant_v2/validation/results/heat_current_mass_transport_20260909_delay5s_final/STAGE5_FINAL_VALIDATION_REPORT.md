# Stage 5 v2 最终验证报告——质量守恒输运与 5/5 s 延迟

**日期**：2026-09-10
**分支**：`heat-current`
**版本标签**：`heat-current-final-v2`
**验证入口**：`validation/validate_heat_current_stage5.py`
**结果目录**：`validation/results/heat_current_mass_transport_20260909_delay5s_final/`

## 1. 定版结论

热流 Plant 的供液和回液延迟定为 **5 s + 5 s**。热流后端采用固定管内容积、
按累计输送质量推进流体前沿的 `CoolantMassTransport`。九个 600 s 工况全部完成，
质量和能量守恒达到数值精度。

旧版报告中的 kW 级循环残差不是未建模的冷板冷却液储能。旧账本按
`cp * sum(T_queue)` 计算 FIFO 队列能量，遗漏了每个队列节点的流体质量；修正为
`m_node * cp * sum(T_queue)` 后，参考流量下的循环残差回到数值精度。旧版解释和
15/20 s 结果因此只保留作历史记录。

## 2. 延迟数据的依据

于海洋等人的储能预制舱液冷管路研究给出单 Pack **10 L/min**、五个 Pack 并联，
对应总流量约 **50 L/min**；当前模型参考流量为 **50.4 L/min**，两者一致。
公开的 215–261 kWh 液冷储能柜资料给出整机冷却液充液量约 **9–11 L**。扣除模型中
3 L 水箱后，水箱外管路和换热部件的等效液量约 **6–8 L**，在 50.4 L/min 下对应
总输运时间 **7.14–9.52 s**。模型时间步长为 5 s，因此采用对称的供液 5 s、回液
5 s，总计 10 s。

这个数值是文献和公开设备数据约束下的工程标称值，不是当前设备的实测辨识结果。
按参考流量计算，两段管路各含 **4.4982 kg** 冷却液，管路总体积约 **8.4 L**；加上
3 L 水箱，模型显式表示的循环液量约 **11.4 L**。

## 3. 实现范围

- `thermal/coolant_mass_transport.py` 保存流体段的实际质量和温度，并以进出质量相等的
  方式推进，固定总库存不随瞬时流量变化。
- `thermal/heat_current_stage5_ledger.py` 给固定时间 FIFO 的储能补上参考节点质量；
  `initial_energy_snapshot()` 要求调用方显式传入参考质量流量，避免隐藏质量假设。
- `parameters.py` 将供液和回液延迟定为 5 s；`ClusterPlant`、`HeatCurrentPlant` 和
  `control/cluster_domp_model.py` 因而采用 1 + 1 个 5 s 延迟状态。
- 初始化、深复制和蒸发器路由的闭合修复一并纳入本版本。

## 4. 验证证据

本次运行在本地最终目录保存九个工况、两个后端的逐步结果及汇总，共重新读取并审计
**36 个 CSV**。仓库按 `.gitignore` 约定不跟踪 CSV；本标签保存每工况 JSON 汇总、
总汇总和独立审计结果。

| 检查项 | 结果 |
|---|---:|
| 完整工况 | 9/9 |
| 每个后端仿真长度 | 600 s / 120 步 |
| legacy / heat-current 有限状态与求解器成功 | 全部通过 |
| heat-current 最大 `abs(R_system)` | `2.5851e-08 W` |
| heat-current 最大管内质量变化 | `0.0 kg` |
| `T_b_avg` legacy 对 heat-current RMSE | `0.0453–0.0755 K` |
| 当前全量单元测试 | 283/283 通过 |

V4 泵阶跃中，legacy 固定时间 FIFO 的 `R_loop` 和 FIFO 流量失配均约为
**2.259 kW**；质量输运后端同工况的 `R_loop` 为 `6.024e-10 W`。这说明固定时间
FIFO 在变流量下不能同时保持固定延迟和固定管存质量，而质量输运实现保持固定库存并
闭合原始能量账本。

完整数值见 `STAGE5_FINAL_VALIDATION.md`、`stage5_summary.json` 和
`full_validation_audit.json`。本版本没有把账本修复前后的 kW 残差与输运模型差异混成
单一性能对比，因为两者回答的是不同问题。

## 5. 验证命令

```powershell
C:\Users\24776\miniforge3\envs\btms\python.exe -B -m unittest discover `
  -s cluster_plant_v2 -p 'test_*.py' -q

C:\Users\24776\miniforge3\envs\btms\python.exe `
  cluster_plant_v2\validation\validate_heat_current_stage5.py `
  --output-dir cluster_plant_v2\validation\results\heat_current_mass_transport_20260909_delay5s_final
```

本次复核的全量测试输出为 `Ran 283 tests in 143.749s`、`OK`。九工况结果由保存的
CSV、JSON 和独立审计文件复核，未在本次文档整理时重复运行约半小时的仿真。

## 6. 适用边界

冻结的 Physics-P 14 状态预测器及其簇级 16 状态扩展仍采用 15/20 s、3 + 4 个延迟
状态。它们与当前 5/5 s Plant 配对后的闭环性能结论需要在重新识别和重标定后重验；
这不影响本报告中的 Plant 开环、质量守恒和能量闭合结论。技术边界见仓库根目录
`PREDICTOR_PLANT_DELAY_BOUNDARY_20260910.md`。

`initial_energy_snapshot()` 的新增参考质量流量参数是有意的接口收紧。仓库内调用点已
同步，未同步的外部脚本需要补充该参数。

## 7. 数据来源

1. 于海洋等，《基于 STAR-CCM+ 的储能预制舱液冷管路结构设计优化》，2026。
2. Billion Electric，Fusio 215/258/344 kWh 储能系统产品资料：
   <https://www.billion.com/ESS/Fusio-215-258-344kWh>
3. SolaX Power，TRENE-P125B261L-E Maintenance Manual：
   <https://id.solaxpower.com/uploads/file/trene-p125B261l-e-maintenance-manual-en.pdf>
