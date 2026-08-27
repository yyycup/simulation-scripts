# TD3 Phase 1C：MPC 目标函数重新标定报告

## 1. 结论

本轮已经把显式温度项与压缩机能耗项调整到彼此可比较的数量级，但没有得到可稳定调度的二维权重动作。

```text
alpha_T: unusable
alpha_comp: unusable
recommended TD3 action space: 暂无，保持 Fixed MPC
recommended alpha ranges:
  alpha_T = 1.0（固定）
  alpha_comp = 1.0（固定）
  alpha_pump = 1.0（固定）
```

因此当前不能开始 `Fixed MPC vs TD3 adaptive-weight MPC` 训练。主要原因不是权重没有注入，而是：

1. RegD 下两个倍率对控制结果的响应很弱且不单调；
2. Constant 下基准 `[1.0, 1.0]` 同时优于两个耦合角点，不能形成可信的温度—能耗 Pareto 方向；
3. 放大显式目标后求解时间和 recovery 明显恶化，不满足 5 s 控制周期，更不适合大量 TD3 episode。

## 2. 本轮边界

保持不变：

- Plant；
- Physics-P Predictor 及其 artifact；
- horizon；
- `dt=5 s`；
- DMAX/DCOST；
- soft band 与 WSPLO/WSPHI 基础值；
- flow logic；
- actuator limits；
-基础水泵能耗权重。

本轮没有实现 TD3，没有训练强化学习模型，也没有把实验权重写入正式 peak/frequency 默认配置。

## 3. 实现的运行时接口

在 `mpc_flow_direction_strategies.py` 中增加二维显式目标倍率接口：

```python
set_runtime_objective_multipliers(alpha_temp, alpha_comp)
```

该接口满足：

- `w_T = w_T0 * alpha_T`；
- `w_comp = w_comp0 * alpha_comp`；
- `alpha_pump = 1.0`；
- WSPLO/WSPHI 不随 `alpha_T` 改变；
- terminal、pump、DMAX/DCOST、约束和流向逻辑不随倍率改变；
- 无需重建 GEKKO 模型即可在下一次求解前更新权重。

`thermal_case_simulator.py` 增加可选输入：

```python
mpc_runtime_objective_multipliers=(alpha_T, alpha_comp)
```

旧的 Phase 1B 三倍率接口仍作为兼容入口保留，但与新接口互斥。

## 4. 当前目标函数及诊断

当前求解器中的相关目标结构为：

```text
J = J_CV
  + w_T * J_T
  + w_terminal * J_terminal
  + w_comp * J_comp
  + w_pump * J_pump
  + J_other
```

其中显式温度项和压缩机项已经是无量纲形式：

```text
J_T = ((T - T_ref) / T_scale)^2
T_scale = 1.0 °C

J_comp = P_comp / P_comp,max
```

本轮新增并输出每个预测时域的实际贡献：

- `objective_cv_contribution`；
- `objective_temperature_contribution`；
- `objective_terminal_contribution`；
- `objective_compressor_contribution`；
- `objective_pump_contribution`；
- `objective_other_contribution`；
- 各项占总目标比例；
- `J_T/J_comp` 比值；
- 目标项闭合误差。

## 5. 600 s RegD 标定

标定搜索结果如下：

| w_T0 | w_comp0 | CV 占比 | 温度项占比 | 压缩机项占比 | J_T/J_comp | Tmax (°C) | 压缩机能耗 (kWh) | 最大求解时间 (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 300 | 99.9612% | 0 | 0.0007% | 0 | 26.069114 | 0.032208 | 24.43 |
| 600 | 300 | 99.9605% | 0.0008% | 0.0007% | 1.118 | 26.069114 | 0.032100 | 22.57 |
| 60,000 | 30,000 | 99.8206% | 0.0797% | 0.0589% | 1.353 | 26.069114 | 0.032350 | 22.80 |
| **600,000** | **300,000** | **98.6610%** | **0.7777%** | **0.5191%** | **1.498** | **26.070949** | **0.030734** | **18.54** |
| 6,000,000 | 3,000,000 | 84.0663% | 11.8126% | 4.0357% | 2.927 | 26.062852 | 0.026062 | 63.16 |

选取实验基准：

```text
w_T0 = 600000
w_comp0 = 300000
```

选择理由：

- 显式温度项与压缩机项比值为 `1.498`，彼此处于同一数量级；
- 两个显式项合计约占总目标 `1.297%`，不再相差 4–5 个数量级；
- 相对 reference，`Tmax` 仅增加约 `0.001835 °C`；
- 120/120 次求解返回 solved；
- 更高一档权重虽然改变了控制，但 `J_T/J_comp=2.927`，最大单次求解时间增至 `63.16 s`，不作为基准。

该选择只用于 Phase 1C 实验，不代表应该部署为正式默认权重。

## 6. RegD 二维倍率结果

| alpha_T | alpha_comp | Tmax (°C) | 压缩机能耗 (kWh) | 总能耗 (kWh) | 平均压缩机转速 (rpm) | solved | recovery | 最大求解时间 (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.7 | 0.7 | 26.070958 | 0.030481 | 0.036639 | 3302.5 | 100% | 5 | 21.09 |
| 0.7 | 1.0 | 26.070957 | 0.030481 | 0.036639 | 3302.5 | 100% | 5 | 20.71 |
| 0.7 | 1.3 | 26.070956 | 0.030481 | 0.036706 | 3302.5 | 100% | 6 | 20.76 |
| 1.0 | 0.7 | 26.066782 | 0.030457 | 0.036902 | 3297.5 | 100% | 7 | 71.57 |
| 1.0 | 1.0 | 26.070949 | 0.030734 | 0.037021 | 3324.4 | 100% | 5 | 18.54 |
| 1.0 | 1.3 | 26.070239 | 0.030478 | 0.036900 | 3302.5 | 100% | 4 | 19.46 |
| 1.3 | 0.7 | 26.070942 | 0.030481 | 0.036713 | 3302.5 | 100% | 6 | 23.77 |
| 1.3 | 1.0 | 26.070941 | 0.031212 | 0.037443 | 3378.2 | 100% | 7 | 39.88 |
| 1.3 | 1.3 | 26.070940 | 0.030481 | 0.036776 | 3302.5 | 100% | 7 | 23.72 |

### 固定 alpha_T=1.0 检查 alpha_comp

```text
alpha_comp:       0.7        1.0        1.3
compressor kWh:   0.030457   0.030734   0.030478
Tmax °C:          26.066782  26.070949  26.070239
```

压缩机能耗和温度均不单调，因此 `alpha_comp` 不可用。

### 固定 alpha_comp=1.0 检查 alpha_T

```text
alpha_T:          0.7        1.0        1.3
compressor kWh:   0.030481   0.030734   0.031212
Tmax °C:          26.070957  26.070949  26.070941
```

能耗方向基本符合预期，但 `alpha_T=0.7→1.3` 造成的 `Tmax` 总变化只有约 `0.000016 °C`，不足以构成有意义的温控动作，因此 `alpha_T` 也不可用。

## 7. Constant 交叉检查

| alpha_T | alpha_comp | Tmax (°C) | 压缩机能耗 (kWh) | 总能耗 (kWh) | 平均压缩机转速 (rpm) | solved | recovery | 最大求解时间 (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1.0** | **1.0** | **26.000944** | **0.029312** | **0.035808** | **3188.4** | 99.17% | 41 | 120.17 |
| 1.3 | 0.7 | 26.036914 | 0.032510 | 0.040276 | 3520.8 | 99.17% | 17 | 118.79 |
| 0.7 | 1.3 | 26.047322 | 0.032060 | 0.039060 | 3474.1 | 100% | 21 | 92.41 |

最重要的发现是 `[1.0,1.0]` 基准同时取得最低 `Tmax` 和最低压缩机/总能耗。两个耦合角点都比基准更热、更耗能，所以不能把角点之间的差异解释为稳定的温度—能耗权衡。

Constant 中大量 recovery 和 90–120 s 的最大求解时间也说明，控制差异受到求解路径和恢复逻辑显著影响。一维耦合 `beta` 目前同样不能作为 TD3 动作。

## 8. 数值等价性与测试

### 默认行为等价

比较第一次与第二次独立运行的 600 s `reference_t0_c300`：

```text
rows = 120
compared numeric columns excluding solve time = 155
max_abs_difference = 0
nonzero_columns = 0
```

求解耗时列因墙钟时间不同而不参与物理数值比较。

### 自动测试

```text
python -B -m unittest \
  single_pack_plant.tests.test_isolated_runtime \
  single_pack_plant.tests.test_runtime_mpc_weight_interface \
  single_pack_plant.tests.test_phase1b_weight_diagnostics \
  single_pack_plant.tests.test_phase1c_objective_rescaling -v

Ran 16 tests
OK
```

测试覆盖：

- 单 Pack 运行时隔离；
- 默认倍率与改动前冻结解等价；
- 二维倍率只改变显式温度/压缩机权重；
- WSP、terminal、pump 等冻结项保持不变；
- 正反向两个 MPC 模型同步更新；
- 目标贡献与求解器总目标闭合；
- 非法倍率在任何模型修改前被拒绝；
- Phase 1C 标定筛选和汇总计算。

## 9. 输出文件

权威结果目录：

```text
outputs/phase1c_objective_rescaling_20260824_v2/
```

主要文件：

- `calibration_summary.csv`；
- `alpha_grid_summary.csv`；
- `constant_cross_check_summary.csv`；
- `experiment_manifest.json`；
- `runs/mpc/*.csv`：每个 600 s Case 的完整时序结果。

三个 Constant CSV 已在主进程被交互中断前全部写完；汇总文件随后从这些现有 CSV 重建，没有重跑仿真。重建时逐个核对了运行时模式、`alpha_T`、`alpha_comp`、固定 `alpha_pump=1` 以及实际 `w_T/w_comp`。

## 10. 后续建议

当前应保持 Fixed MPC，不进入 TD3 训练。下一阶段若继续，应先解决 MPC 优化问题本身：

1. 找出 Constant 下大量 recovery 的具体触发步骤和恢复动作；
2. 区分权重不敏感究竟来自 CV 主导、预测时域内温度控制敏感度太低，还是 recovery 覆盖了正常最优解；
3. 对代表状态做离线候选动作目标面检查，验证改变权重是否真的会改变最优控制排序；
4. 在任何 TD3 接入前恢复求解可靠性，并满足 5 s 控制周期约束；
5. 只有在 Constant、Current step、RegD 中都出现稳定、单调且大于数值噪声的温度—能耗权衡后，才重新定义 TD3 action space。

本轮到此停止，不实施 TD3，也不继续扩大权重搜索。
