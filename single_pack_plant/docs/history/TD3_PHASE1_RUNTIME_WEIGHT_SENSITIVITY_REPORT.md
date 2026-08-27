# TD3 Phase 1：MPC 运行时权重接口与敏感性报告

## 1. 结论先行

本阶段已经完成运行时三权重倍率接口、`[1, 1, 1]` 严格数值等价性验证，以及 Constant、Current step、RegD 三类工况的 30 组、每组 150 s 闭环敏感性实验。没有实现 TD3、RL 环境或训练代码。

主要结论：

- `[1, 1, 1]` 与修改前 Fixed MPC 的单步结果严格一致，观测到的 `max_abs_difference = 0`；
- `alpha_T` 在本轮 150 s 工况中为弱敏感，增大后没有呈现稳定的“温度下降、能耗上升”单调关系；
- `alpha_comp` 对实际压缩机动作几乎是死区：安全试验范围内压缩机平均转速没有变化，Current step 与 RegD 的完整结果为零差异；
- `alpha_pump` 对水泵动作、水泵能耗及供回水温度为中等敏感，但对电池温度和总能耗仍然较弱；
- `1.6` 不是适合第一版 TD3 的上界：`alpha_T=1.6` 在 Constant 出现一次未求解，`alpha_pump=1.6` 在 Constant 和 Current step 各出现一次未求解；
- 技术接口已经具备，但当前证据不足以直接冻结“三动作 TD3 环境”。建议先把三者均限制在已验证无未求解的 `[0.7, 1.3]`，再做更长工况确认；尤其应重新评估是否保留 `alpha_comp`。

## 2. 修改范围

### 2.1 修改的现有文件

- `mpc_flow_direction_strategies.py`
  - 把压缩机能耗、水泵能耗和终端温度权重改为 GEKKO `Param`；
  - 增加模型级和双流向控制器级 `set_runtime_weight_multipliers()`；
  - 增加运行时权重诊断；
  - Physics-P solver rebuild 后恢复当前倍率，避免回到默认值。
- `thermal_case_simulator.py`
  - 增加默认关闭的 `mpc_runtime_weight_multipliers=None` 参数；
  - 仅在显式传入时调用 MPC setter；
  - 把运行时倍率和实际权重写入闭环 CSV。

### 2.2 新增文件

- `tests/test_runtime_mpc_weight_interface.py`
- `run_runtime_mpc_weight_sensitivity.py`
- `TD3_PHASE1_RUNTIME_WEIGHT_SENSITIVITY_REPORT.md`

### 2.3 明确未修改

以下冻结文件保持不变：

- `pack.py`
- `thermal_loop.py`
- `thermal_system.py`
- `model_data/hppc_params.json`
- `model_data/physics_p_operational_v1.json`
- `mpc_physics_predictor.py`

同时没有修改 flow 切换逻辑、预测时域、5 s 控制周期、DMAX、DCOST、soft band、设备上下限和 Plant 时间推进顺序。五个在修改前记录过 SHA256 的冻结文件在完成后逐一复核一致；`hppc_params.json` 本轮未被写入，最终 SHA256 为 `B646258BAA3B7CBF0F7142E5E60226E70ED772C689AB655EECF92448888EF186`。

## 3. 运行时权重接口

### 3.1 权重映射

接口保持原目标结构，只把三个原先固化在模型表达式中的 Python float 替换成可更新的 GEKKO `Param`：

```text
runtime_WSPLO          = WSPLO_0          * alpha_temp
runtime_WSPHI          = WSPHI_0          * alpha_temp
runtime_w_terminal     = w_terminal_0     * alpha_temp
runtime_w_energy_comp  = w_energy_comp_0  * alpha_comp
runtime_w_energy_pump  = w_energy_pump_0  * alpha_pump
```

`w_temp_obj` 没有进入倍率调度，仍保持正式配置中的 `0.0`。

原优化结构可写为：

```text
J = w_temp_obj * J_track
  + runtime_w_energy_comp * J_comp
  + runtime_w_energy_pump * J_pump
  + terminal_mask * runtime_w_terminal * J_terminal
  + 原有可选二次动作变化项
```

GEKKO CV 的 `WSPLO/WSPHI` 仍由原有 CV 机制使用，只是每次求解前取运行时值。

### 3.2 基线权重

| 工况 | WSPLO/WSPHI | terminal | compressor energy | pump energy |
|---|---:|---:|---:|---:|
| Peak | 5,000,000 | 1,000,000 | 600 | 10,000 |
| Frequency | 50,000,000 | 500,000 | 300 | 15,000 |

### 3.3 Setter 行为

```python
controller.set_runtime_weight_multipliers(
    alpha_temp,
    alpha_comp,
    alpha_pump,
)
```

实现保证：

- 三个输入先统一转为 float，并检查有限且严格大于 0；
- 在更新任何子模型前完成全部校验，避免部分更新；
- forward 和 reverse 使用同一倍率；
- 仅更新 GEKKO `Param` 和 CV 权重，不重建模型；
- 保留原 GEKKO 对象和 warm start；
- 不改变 recovery/fallback 的触发和控制动作；
- 如果原有 Physics-P recovery 重建 solver，重建后自动恢复当前倍率；
- 默认倍率为 `[1.0, 1.0, 1.0]`。

每次求解结果和 `last_flow_info` 记录：

```text
alpha_temp
alpha_comp
alpha_pump
runtime_WSPLO
runtime_WSPHI
runtime_w_terminal
runtime_w_energy_comp
runtime_w_energy_pump
```

闭环 CSV 对应增加 `MPC_Alpha_*` 和 `MPC_Runtime_*` 字段。

## 4. `[1, 1, 1]` 严格等价性

### 4.1 固定验证条件

- Predictor：`physics_p`；
- artifact：`model_data/physics_p_operational_v1.json`；
- `dt = 5 s`；
- 为单元回归缩短到 `N = 8`，修改前、修改后完全相同；
- 初始电池、冷却液和冷板温度均为 25 °C；
- 电流预览为恒定 560 A；
- forward 和 reverse 分别独立求解。

### 4.2 比较结果

下列量在修改前后逐项一致：

- `n_comp = 5999.9991097 rpm`；
- `n_pump = 2700.0000317 rpm`；
- 9 点电池预测温度轨迹；
- 9 点冷却液预测温度轨迹；
- 8 点压缩机控制计划；
- 8 点水泵控制计划；
- `J_track`、`J_comp`、`J_pump`、`J_terminal`；
- 各加权目标项和 `J_total`；
- `solved=True`；
- `solve_recovery_used=False`。

正向和反向的全部上述数值也一致。实际比较得到：

```text
max_abs_difference = 0
command_difference = 0
prediction_trajectory_difference = 0
objective_diagnostic_difference = 0
recovery_change = 0
```

等价性测试使用修改前冻结的数值作为 golden regression，而不是把两个新控制器互相比较。

## 5. 自动化测试和烟测

修改前基线：

```text
7 tests passed
```

修改后完整 `single_pack_plant/tests`：

```text
10 tests passed
Ran 10 tests in 2.099 s
```

新增测试覆盖：

- `[1,1,1]` 对修改前 golden 数值的 forward/reverse 等价性；
- 非有限、0、负数和非数值输入拒绝；
- 校验失败时 forward/reverse 均不发生部分更新；
- 双模型倍率完全同步；
- GEKKO 模型对象 ID 不变，即 setter 不重建模型；
- `w_temp_obj` 保持不变。

另外执行了一步非默认权重闭环烟测：

```text
alpha = [1.3, 0.7, 1.6]
runtime_WSPLO/WSPHI = 6,500,000
runtime_w_energy_comp = 420
runtime_w_energy_pump = 16,000
MPC_Solved = True
```

当前 `single_pack_plant` 文件夹没有独立的 validation Python 脚本；因此本报告不把不存在的额外 validation 套件声称为通过。30 组真实 Plant 闭环实验同时提供了运行路径验证。

## 6. 敏感性实验设计

### 6.1 固定条件

- 每组时长：150 s，30 步；
- `dt = 5 s`；
- 初始电池温度：原配置 25 °C；
- 初始冷却系统温度和环境温度：原配置 35 °C；
- Predictor：Physics-P；
- artifact：正式 validated artifact；
- flow：双向，`single_predictive_delta_t`；
- Peak 工况保持生产 `N=60`；RegD 保持生产 `N=45`；
- 所有同工况案例使用相同初态、电流、环境温度、artifact、horizon 和 flow logic；
- 没有随机搜索，没有同时改变两个倍率。

工况定义：

- Constant：560 A；
- Current step：前 75 s 为 280 A，后 75 s 为 840 A；
- RegD：读取项目现有 `PJM_RegD_MaxLoad_2h.csv`，按项目原逻辑乘以 1120 A 并插值。

### 6.2 数据完整性

- 30 个主结果 CSV；
- 每个 CSV 30 行；
- `sensitivity_summary.csv` 30 行；
- `sensitivity_relative_to_baseline.csv` 30 行；
- 每个 CSV 的实际 `MPC_Alpha_*` 与请求倍率逐行严格一致。

完整数据位于：

```text
outputs/td3_phase1_runtime_weight_sensitivity_20260823_164832/
```

主要文件：

- `sensitivity_summary.csv`：30 组全部绝对指标；
- `sensitivity_relative_to_baseline.csv`：相对各工况 `[1,1,1]` 的差值和百分比；
- `experiment_manifest.json`：固定条件和输入定义；
- `runs/mpc/*.csv`：逐步完整闭环结果。

## 7. 敏感性结果

### 7.1 三个基线

| 工况 | Tavg max (°C) | Tmax max (°C) | DeltaT max (°C) | Tavg RMSE (°C) | Ecomp (kWh) | Epump (kWh) | Efan (kWh) | Etotal (kWh) | solved | recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Constant | 25.907699 | 25.969563 | 0.108390 | 0.654729 | 0.0138891 | 0.000199284 | 0.00184414 | 0.0159325 | 100% | 1 |
| Current step | 26.019139 | 26.080620 | 0.107628 | 0.618522 | 0.0139155 | 0.000186004 | 0.00184414 | 0.0159456 | 100% | 0 |
| RegD | 25.938283 | 26.000448 | 0.108861 | 0.584713 | 0.0139229 | 0.000184865 | 0.00184414 | 0.0159519 | 100% | 3 |

### 7.2 推荐安全候选区间 `[0.7,1.3]` 内的最大变化

表中温度为相对基线的最大绝对差，能耗为最大绝对相对变化。

| 工况 | 动作 | abs ΔTavg max | abs ΔTmax max | abs ΔDeltaT max | abs ΔRMSE | abs ΔEcomp | abs ΔEpump | abs ΔEtotal | abs Δ平均泵速 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Constant | alpha_T | 0.02184 °C | 0.02266 °C | 0.00101 °C | 0.01723 °C | 0.378% | 6.962% | 0.242% | 86.4 rpm |
| Constant | alpha_comp | 0.05173 °C | 0.05306 °C | 0.00220 °C | 0.04199 °C | 0.798% | 15.074% | 0.507% | 143.4 rpm |
| Constant | alpha_pump | 0.02184 °C | 0.02266 °C | 0.00101 °C | 0.01723 °C | 0.378% | 6.962% | 0.242% | 86.4 rpm |
| Current step | alpha_T | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Current step | alpha_comp | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Current step | alpha_pump | 0.00799 °C | 0.00883 °C | 0.00175 °C | 0.00223 °C | 0.175% | 7.135% | 0.069% | 86.5 rpm |
| RegD | alpha_T | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| RegD | alpha_comp | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| RegD | alpha_pump | 0.00073 °C | 0.00093 °C | 0.00062 °C | 0.00032 °C | 0.088% | 3.906% | 0.032% | 56.5 rpm |

该区间的所有 18 个非基线案例均为 100% solved。注意 Constant 中 `alpha_comp` 引起的主要变化不是压缩机转速变化，而是非线性求解分支和水泵轨迹变化。

### 7.3 `1.6` 的求解可靠性

| 工况/动作 | solved rate | recovery | infeasible/solution-not-found | 最大求解时间 |
|---|---:|---:|---:|---:|
| Constant / alpha_T=1.6 | 96.67% | 3 | 1 | 120.25 s |
| Constant / alpha_comp=1.6 | 100% | 2 | 0 | 108.74 s |
| Constant / alpha_pump=1.6 | 96.67% | 3 | 1 | 120.24 s |
| Current step / alpha_T=1.6 | 100% | 1 | 0 | 24.53 s |
| Current step / alpha_comp=1.6 | 100% | 0 | 0 | 6.48 s |
| Current step / alpha_pump=1.6 | 96.67% | 3 | 1 | 120.26 s |
| RegD 三个 1.6 案例 | 100% | 3 | 0 | 22.9–23.1 s |

三个失败步均出现 `Solution Not Found`，在常规重试、备用 solver、Physics-P observed-state reseed、fixed-control recovery 和 solver rebuild 后仍未解决。它们不是脚本崩溃，而是原有 fallback 返回的未求解控制步。

## 8. 三个动作是否真的有用

### 8.1 `alpha_T`：weak

实际结果不支持“增大温度权重就稳定降低温度并增加能耗”的简单关系：

- Current step 中 0.7、1.0、1.3 三条完整轨迹完全相同；
- RegD 中 0.7、1.0、1.3、1.6 四条完整轨迹完全相同；
- Constant 中变化非单调：1.3 的 Tavg max 比基线高约 0.0218 °C，1.6 虽略低，但包含一个未求解步；
- 压缩机平均转速在所有正常比较中固定为 5850 rpm，说明短工况内压缩机基本处于上限饱和轨迹；
- 同时缩放 CV 上下带权重和终端权重，并不能解除 DMAX、饱和、warm-start 局部解和 recovery 带来的分段行为。

因此本轮把 `alpha_T` 判为 **weak**。它在更长工况或离开饱和区后可能变得有效，但本轮没有证据将其评为 medium/strong。

### 8.2 `alpha_comp`：weak，当前接近死动作

增大压缩机能耗权重没有稳定抑制实际压缩机动作：

- Current step 的 0.7、1.0、1.3、1.6 全部逐步结果完全相同；
- RegD 的四个倍率全部逐步结果完全相同；
- Constant 中实际压缩机平均转速仍全部为 5850 rpm；
- Constant 的压缩机能耗变化最大约 0.798%，但方向不单调，主要伴随水泵轨迹和 recovery 分支变化；
- `alpha_comp=1.6` 在 Constant 虽没有未求解，但最大单步求解时间达到 108.74 s，且没有获得明确的压缩机控制收益。

因此 `alpha_comp` 判为 **weak**。在当前 150 s、35 °C 初始冷却系统条件下，它是三个候选中最接近“死动作”的一个。

### 8.3 `alpha_pump`：对泵/冷却液 medium，对电池结果 weak

水泵权重确实能影响水泵和冷却液通道：

- `[0.7,1.3]` 内平均泵速最大变化：Constant 86.4 rpm、Current step 86.5 rpm、RegD 56.5 rpm；
- 水泵能耗最大相对变化：6.96%、7.14%、3.91%；
- 相对基线的逐步最大供水温度差分别达到约 1.17、1.09、0.86 °C；
- 回水温度逐步最大差分别约 0.87、0.81、0.59 °C；
- 冷板进出口温度跟随供回水通道变化。

但电池侧和总能耗侧仍弱：

- `[0.7,1.3]` 内 Tavg max 最大变化只有 0.02184、0.00799、0.00073 °C；
- DeltaT max 最大变化只有 0.00101、0.00175、0.00062 °C；
- 总能耗最大变化只有 0.242%、0.069%、0.032%。

`alpha_pump=1.6` 在 Constant 和 Current step 出现未求解，而且 Current step 的较大温度/能耗差主要由失败步和 fallback 引起，不能当作有益的平滑敏感性。

因此综合判为 **medium（直接泵与冷却液通道）/ weak（电池热指标和总能耗）**。它不是完全死动作，但也没有证据支持扩大范围。

## 9. 后续 TD3 动作范围建议

第一版试验范围建议：

```text
alpha_T    in [0.7, 1.3]
alpha_comp in [0.7, 1.3]
alpha_pump in [0.7, 1.3]
```

依据：

- 本轮该闭区间内 18 个非基线案例均 100% solved；
- 1.6 已在 `alpha_T` 和 `alpha_pump` 上触发未求解；
- `alpha_comp=1.6` 没有带来可识别的直接控制收益，却显著增加最大求解时间；
- 不能依据本轮结果外推到理论性的 `[0.5, 2.0]`。

这不是最终动作范围，只是后续验证的安全候选范围。对动作维度的建议：

- 暂时保留 `alpha_T`，但在更长、非饱和工况中复核；
- `alpha_comp` 应列为优先删除候选。如果更长工况仍然不改变压缩机动作，应从 TD3 action 中删除，或改为调节另一个经验证可控的 MPC 参数；
- 保留 `alpha_pump` 做下一轮验证，因为它对泵和冷却液状态有真实响应，但 reward 中不能只看总能耗，否则这一维的学习信号可能很弱。

## 10. 是否已经满足建立 `td3_mpc_env.py` 的条件

### 技术接口：满足

- 三倍率可以在每次 MPC solve 前在线更新；
- 不重建 GEKKO 模型；
- forward/reverse 同步；
- 默认 Fixed MPC 数值等价；
- 诊断和闭环 CSV 已能记录实际倍率；
- 输入非法时能在进入模型前失败。

### 三动作实验依据：尚不满足

当前不建议立即把三个动作冻结成 TD3 action space，原因：

1. `alpha_comp` 在 Current step 和 RegD 完全无效，Constant 也不改变实际压缩机转速；
2. `alpha_T` 在两个工况中完全无效，没有证明期望的温度—能耗权衡；
3. `alpha_pump` 主要改变泵和冷却液状态，对电池温度及总能耗的反馈较弱；
4. 1.6 存在明确求解可靠性问题；
5. 150 s 内压缩机长期饱和，可能掩盖能耗权重的真实作用；
6. 平均 MPC 求解时间约为 2.4–9.8 s/步，个别步超过 120 s，直接做数千 episode 会成为严重训练瓶颈；
7. TD3 环境还需要先定义 MPC 未求解时的 reward、termination 和 action fallback 语义。

## 11. 建立 TD3 环境前的最小剩余工作

不修改 Plant、Predictor 或控制约束的前提下，建议先完成：

1. 用 `[0.7,1.3]` 在至少 600 s 的 Constant、Current step、RegD 上复核三动作，观察是否离开初始饱和段；
2. 增加一个仍位于 Physics-P validated domain、但压缩机不长期饱和的代表工况；
3. 根据长工况决定是否删除 `alpha_comp`；
4. 明确 TD3 遇到 MPC unsolved/recovery 时的 reward 和 episode 处理规则；
5. 评估训练用 MPC 加速或 episode 并行方案，但不得以改变模型、horizon 或约束换取表面速度。

完成这些确认后，才适合新增 `td3_mpc_env.py`。本阶段到此停止，不实现 TD3 环境、网络或训练算法。
