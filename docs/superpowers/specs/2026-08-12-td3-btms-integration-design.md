# BTMS TD3 独立接入设计

## 目标

在不改变现有 PID、MPC、热模型和制冷模型行为的前提下，为 BTMS 增加一个可训练的 Gymnasium 环境，以及独立的 TD3 训练和评估入口。

首个里程碑只验证环境和训练链路能够运行：完成调峰、调频短烟测，以及一次极短的 TD3 保存、加载和评估。正式长训练、超参数搜索和控制器效果对比不在本轮范围内。

## 设计原则

- TD3 与现有控制器隔离，不加入 `thermal_control_strategies.create_controller()`。
- 直接复用现有低层物理接口，不重构已验证的 `thermal_case_simulator.simulate_case()`。
- 调峰和调频共用一个环境类，通过 `scene` 参数区分，并分别保存模型。
- 首版保持状态、动作、奖励和命令行入口最小化，不增加预测器、模仿学习或流向控制。
- 训练权重和运行结果写入已忽略的 `outputs/td3/`，不纳入 Git。

## 文件边界

新增以下正式文件：

- `td3_btms/__init__.py`：导出公开环境类。
- `td3_btms/env.py`：实现环境、动作映射、观测构造和奖励计算。
- `run_td3_training.py`：TD3 训练入口。
- `run_td3_evaluation.py`：确定性评估入口。
- `tests/td3/__init__.py`：测试包标识。
- `tests/td3/test_td3_btms_env.py`：环境与短训练回归测试。

本轮不修改以下现有模块的控制行为：

- `thermal_case_simulator.py`
- `thermal_control_strategies.py`
- `thermal_loop.py`
- `thermal_system.py`
- `pack.py`
- MPC 和 Physics-P 相关模块

## 环境架构

`BTMSTd3Env` 继承 `gymnasium.Env`，构造参数至少包括：

- `scene`：只能是 `peak` 或 `freq`。
- `current_profile`：可选的一维电流数组，用于测试和烟测注入。
- `agc_data_file`：调频正式运行的数据文件路径。
- `max_steps`：可选的回合长度上限。

随机种子只通过 Gymnasium 标准的 `reset(seed=...)` 接口传入，避免构造参数和重置参数产生两套状态。

`reset()` 执行以下工作：

1. 按场景创建电池包配置和 `BatteryPack`；电芯初温沿用 `INITIAL_TEMP_C=25°C`。
2. 按现有普通仿真边界将水箱、冷板和制冷动态状态初始化为环境温度 35°C。
3. 构造或读取电流曲线。
4. 将压缩机初值设为关闭转速，将水泵初值设为最低转速。
5. 返回初始观测和包含场景、步号的 `info`。

`step(action)` 的数据流为：

1. 校验并裁剪二维动作。
2. 将动作映射为压缩机和水泵命令。
3. 设置当前步电池总电流。
4. 以固定正向流动调用 `simulate_thermal_loop_step()`。
5. 用更新后的冷板温度调用 `BatteryPack.step()`。
6. 计算观测、奖励、终止状态和诊断信息。

环境不调用 MPC、Candidate-B 或 Physics-P 预测器。

## 观测空间

观测是长度为 9 的 `float32` 连续向量：

1. 当前总电流。
2. 电池平均 SOC。
3. 电池平均温度。
4. 电池最高温度。
5. 电芯最大温差。
6. 水箱冷却液温度。
7. 冷板平均温度。
8. 压缩机实际转速。
9. 水泵实际转速。

各量使用固定物理尺度转换为量级接近 1 的数值，不使用运行时均值方差，也不依赖 Stable-Baselines3 的 `VecNormalize`。观测空间允许有限范围外的归一化数值，以避免裁剪掩盖物理异常；非有限值会终止回合。

SOC 必须保留。现有电池模型的电气参数和发热响应随 SOC 变化，而且调峰与调频初始 SOC 不同。仅使用电池平均 SOC，不暴露每个电芯的 SOC。

## 动作空间

动作空间为 `Box(low=-1, high=1, shape=(2,), dtype=float32)`：

- `action[0]` 线性映射到压缩机 `300–6000 rpm`。
- `action[1]` 线性映射到水泵 `1600–4800 rpm`。

动作在映射前裁剪至 `[-1, 1]`。流向固定为 `is_reversed=False`。压缩机的关闭与低速启动行为继续由现有物理模型处理，不增加离散启停动作。

## 奖励函数

奖励为以下归一化代价之和的负值：

```text
reward = -(
    temperature_tracking_cost
    + 0.05 * power_cost
    + 4.0 * delta_temperature_violation_cost
    + 4.0 * high_temperature_violation_cost
)
```

其中：

- `temperature_tracking_cost`：电池平均温度相对 25°C 目标的平方误差，以 2°C 为归一化尺度。
- `power_cost`：压缩机、水泵和风扇总功率，以 5 kW 为归一化尺度。
- `delta_temperature_violation_cost`：仅对超过 0.5°C 的最大电芯温差施加平方惩罚，以 0.5°C 为归一化尺度。
- `high_temperature_violation_cost`：仅对电芯最高温度超过 27°C 的部分施加平方惩罚，以 2°C 为归一化尺度。

首版不加入 MPC 模仿奖励、动作变化率惩罚、流向奖励或多阶段奖励塑形。上述权重作为模块常量集中定义，后续调整必须单独验证。

## 回合结束规则

- 电流曲线结束或达到 `max_steps`：`truncated=True`。
- 任一关键状态或奖励出现 `NaN/Inf`：`terminated=True`，并返回明确的终止原因。
- 电芯最高温度达到 45°C：`terminated=True`，并在当步奖励上增加 `-100` 的固定异常惩罚。
- 普通温度偏差和温差超限不提前结束，只通过奖励反映，使智能体有机会学习恢复。

## 场景与输入数据

- 调峰默认使用现有 560 A 恒定电流定义。
- 调频正式运行使用现有 AGC 解析规则，并允许通过 `--agc-data-file` 显式指定文件。
- AGC 文件不存在、缺少所需列、曲线为空或包含非有限值时，运行立即失败并报告具体原因。
- 单元测试和 `--smoke` 使用代码内构造的短确定性电流数组，不依赖用户桌面文件。

## 训练入口

`run_td3_training.py` 提供：

- `--scene peak|freq`
- `--total-timesteps`
- `--seed`
- `--output-dir`
- `--agc-data-file`
- `--smoke`

正式模式要求显式提供正整数 `--total-timesteps`。`--smoke` 使用短电流曲线和很少的学习步数，不启动长训练。

训练使用 Stable-Baselines3 `TD3("MlpPolicy", ...)`，并为两维动作配置零均值、标准差 0.1 的正态动作噪声。模型保存在：

```text
outputs/td3/<scene>/<run_name>/model.zip
```

同目录保存 `metadata.json`，至少记录场景、随机种子、学习步数、时间步长、动作边界、奖励权重、输入数据路径和输入数据 SHA-256。调峰与调频不得复用同一个模型文件。

## 评估入口

`run_td3_evaluation.py` 确定性加载指定模型，运行一个回合并输出：

- `trajectory.csv`：逐步电流、SOC、温度、温差、转速、功率、奖励和终止信息。
- `summary.json`：场景、步数、平均温度绝对误差、最高温度、最大电芯温差、平均系统功率、总能耗和累计奖励。

评估入口不负责生成正式论文图，也不自动与 PID/MPC 比较。

## 错误处理

- 非法场景、非正时间步、空电流曲线和错误动作形状立即抛出 `ValueError`。
- 文件相关错误包含实际解析路径。
- 环境不吞掉底层物理模型异常，也不静默切换为简化模型。
- `info` 返回实际命令、实际转速、总功率和结束原因，便于定位训练异常。

## 验证与成功标准

1. `gymnasium.utils.env_checker.check_env` 通过。
2. `reset()` 和 `step()` 返回符合 Gymnasium API 的形状、类型和标志。
3. 两个动作端点精确映射到既有压缩机、水泵边界。
4. 固定种子和固定电流输入下，重复重置得到一致初始状态。
5. 调峰和调频各完成短环境烟测，所有关键状态与奖励为有限值。
6. TD3 完成一次极短学习，模型能够保存、重新加载并执行确定性评估。
7. 新增测试通过，现有控制器选择和核心物理模块没有行为修改。

本里程碑不以控制效果优于 PID/MPC、收敛到最优策略或完成全长度正式训练作为验收条件。

## 非目标

- 不训练正式调峰或调频模型。
- 不搜索网络结构、学习率、奖励权重或探索噪声。
- 不控制流向、风扇或其他执行器。
- 不接入现有 PID/MPC 工厂。
- 不修改 Physics-P、Candidate-B、MPC 权重或热系统参数。
- 不提交模型权重、训练日志或仿真结果。
