# P模型短时蒸发器响应精修结果

日期：2026-07-23  
状态：离线辨识轨迹与已执行MPC指令影子回放通过，尚未提升为正式MPC默认预测器

## 结论

5秒制冷量大误差的主因不是20项稳态容量公式，而是P模型的动态结构与详细制冷对象不一致。详细对象中的有效蒸发器制冷量直接以稳态制冷量为目标，按45秒一阶惯性响应；旧P模型却额外串联了20秒输入延迟、冷凝器惯性和蒸发器惯性。

本轮仅修正三项：

- `evap_response_model`由兼容旧产物的`cascaded`改为`direct`；
- `evap_input_delay_s`由20秒改为0秒；
- `tau_evap_s`固定为详细对象使用的45秒。

其余容量、动态和热网络参数全部冻结。详细制冷循环、L模型、MPC权重、DMAX、PSO和Candidate B正式配置均未修改。

## 选择保护

新增独立的短时响应精修入口。候选模型只在验证集综合动态MAE严格低于基准时才会被选择；否则返回原产物。此次验证指标由0.30152降至0.22377，改善25.79%，因此候选被接受。

旧产物没有`evap_response_model`字段时仍按`cascaded`解释，保持向后兼容。

## 辨识测试轨迹结果

| 预测长度 | 旧串联P制冷量MAE | 新直接45秒P制冷量MAE |
|---|---:|---:|
| 5秒 | 366.32 W | 6.34 W |
| 50秒 | 139.22 W | 33.84 W |
| 100秒 | 87.97 W | 40.51 W |
| 300秒 | 74.64 W | 38.52 W |

新P在5秒主动制冷样本上的MAPE为0.328%。单步计算时间中位数约139.9微秒，未增加计算复杂度。300秒电池温度MAE由约0.0334°C变为0.0415°C，存在约0.0081°C的小幅退化，说明下一阶段若继续改进，应处理热网络长时偏差，而不是再次改动蒸发器短时惯性。

## 已执行MPC指令影子回放

使用既有90步调峰和调频闭环轨迹，每个预测起点以测量状态重置，并回放其后实际执行的压缩机和泵指令。合并结果如下：

| 预测长度 | 旧串联P制冷量MAE | 新直接45秒P制冷量MAE |
|---|---:|---:|
| 5秒 | 429.42 W | 21.34 W |
| 50秒 | 229.77 W | 105.79 W |
| 100秒 | 182.51 W | 122.30 W |
| 300秒 | 311.53 W | 201.31 W |

短时根因修正可迁移到MPC轨迹，但300秒制冷量主动样本MAPE仍为18.25%，长时热状态和工况分布误差尚未完全解决。因此当前产物继续作为影子候选，不替换Candidate B。

## 产物

- 正式候选：`outputs/mpc_predictor_low_speed_retrain_v1/short_response_refinement_v1/artifacts/physics_p_cubic20_direct45_selected.json`
- 辨识测试轨迹报告：`outputs/mpc_predictor_low_speed_retrain_v1/short_response_refinement_v1/reports/formal_selected_rollouts/`
- 新P影子回放报告：`outputs/mpc_predictor_low_speed_retrain_v1/short_response_refinement_v1/reports/actual_command_shadow_replay/`
- 旧P影子回放基准：`outputs/mpc_predictor_low_speed_retrain_v1/short_response_refinement_v1/reports/actual_command_shadow_replay_cascaded_baseline/`

## 当前决策

- 5秒蒸发器动态结构问题视为已解决。
- 保留新直接45秒响应P作为下一阶段研究候选。
- Candidate B继续作为正式MPC默认预测模型。
- 不复制新P到`model_data/`，不让P接管闭环控制。
- 后续优先校正50至300秒热网络偏差，再讨论P闭环替换。
