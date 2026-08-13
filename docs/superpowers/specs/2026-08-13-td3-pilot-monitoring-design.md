# TD3 500 步试训练与监控设计

## 目标

在现有 BTMS TD3 环境、奖励函数和训练算法不变的前提下，为训练入口增加轻量监控与检查点，并分别完成调峰、调频 500 步试训练。

本轮用于判断训练链路是否稳定、奖励是否有改善趋势、执行器是否长期饱和以及温度安全约束是否正常。它不用于宣称 TD3 已经收敛或优于 PID/MPC。

## 范围

本轮只修改：

- `run_td3_training.py`
- `tests/td3/test_td3_btms_env.py`

不修改：

- `td3_btms/env.py`
- `run_td3_evaluation.py`
- PID、MPC、Physics-P 和电池热管理物理模型
- 奖励权重、TD3 网络、学习率、探索噪声和执行器边界

## 方案选择

采用自定义 Stable-Baselines3 Callback。Callback 在训练进程内读取环境每步返回的 `info`，因此能够记录奖励以外的温度、功率和执行器状态。

不采用以下方案：

- 只保留 SB3 `Monitor` 日志：无法完整记录 BTMS 物理量。
- 仅在训练结束后评估：无法观察训练过程中是否发散或动作饱和。
- TensorBoard：本轮用户选择 CSV、PNG 和检查点，不增加另一套日志依赖。

## 训练记录

每个环境步记录一行，写入 `training_history.csv`。字段固定为：

- `training_step`
- `episode_index`
- `episode_step`
- `reward`
- `current_a`
- `mean_soc`
- `mean_temp_c`
- `max_temp_c`
- `delta_temp_c`
- `tank_temp_c`
- `plate_mean_temp_c`
- `total_power_w`
- `n_comp_cmd_rpm`
- `n_pump_cmd_rpm`
- `n_comp_eff_rpm`
- `n_pump_eff_rpm`
- `terminated`
- `truncated`

Callback 只接受单环境训练。若 Stable-Baselines3 传入多个并行环境，立即抛出明确错误，避免日志行与环境状态错配。

训练正常结束或发生异常时，已经收集到的记录均写出 CSV。异常仍继续向上传播，不被日志层吞掉。

## 检查点

每 100 个训练步保存一次模型：

```text
checkpoints/checkpoint_000100_steps.zip
checkpoints/checkpoint_000200_steps.zip
checkpoints/checkpoint_000300_steps.zip
checkpoints/checkpoint_000400_steps.zip
checkpoints/checkpoint_000500_steps.zip
```

最终模型仍保存为 `model.zip`。检查点只用于保留阶段状态，本轮不增加断点续训功能。

## 训练曲线

训练结束后生成 `training_curve.png`，使用无界面的 Matplotlib `Agg` 后端。图片包含四组共享训练步横轴的曲线：

1. 单步奖励和 50 步滚动平均奖励。
2. 电池平均温度、最高温度和最大温差，并标出 25°C 目标和 0.5°C 温差阈值。
3. 系统总功率，单位 kW。
4. 压缩机、水泵指令转速及实际转速。

当记录少于 50 步时，滚动平均使用已有样本计算；CSV 为空时绘图函数报错，不生成误导性空图。

## 命令行接口

训练入口增加：

- `--checkpoint-interval`：非负整数，正式训练默认 100；传入 0 表示关闭检查点。
- `--no-training-plot`：只保留 CSV 和检查点，不生成 PNG。

原有 `--smoke` 行为保持不变。烟测默认关闭阶段检查点，仍可快速验证保存和加载链路。

## 500 步试训练

使用固定随机种子 7，分别运行：

```powershell
D:\conda_envs\btms_td3\python.exe run_td3_training.py `
  --scene peak `
  --total-timesteps 500 `
  --seed 7 `
  --checkpoint-interval 100 `
  --output-dir outputs/td3/pilot_peak_500
```

```powershell
D:\conda_envs\btms_td3\python.exe run_td3_training.py `
  --scene freq `
  --total-timesteps 500 `
  --seed 7 `
  --checkpoint-interval 100 `
  --output-dir outputs/td3/pilot_freq_500
```

调频试训练前必须解析并检查 `thermal_batch_config.AGC_DATA_FILE`；若需要替换数据源，则显式增加 `--agc-data-file`。输出目录必须事先不存在，训练入口继续拒绝覆盖旧结果。

## 训练后评估

每个 500 步模型完成一次确定性 100 步评估，分别写入：

- `outputs/td3/pilot_peak_500_eval100/`
- `outputs/td3/pilot_freq_500_eval100/`

本轮不自动运行 1280 步调峰或 720 步调频完整评估。只有 100 步评估的状态、奖励和动作均有限且安全后，才决定是否投入完整评估和更长训练。

## 验收与判读

代码验收：

1. Callback 能从单环境 `infos` 生成固定字段记录。
2. 100 步边界生成正确命名的检查点。
3. CSV 行数等于实际训练步数。
4. PNG 存在且非空。
5. 原有 16 项 TD3 测试和相邻回归继续通过。

试训练验收：

1. 调峰、调频训练均完成 500 步，无 `NaN/Inf`。
2. 每个训练目录包含最终模型、元数据、500 行 CSV、PNG 和 5 个检查点。
3. 每个 100 步评估目录包含 100 行轨迹 CSV 和汇总 JSON。
4. 评估过程中没有非有限状态或 45°C 安全终止。
5. 报告奖励前 100 步与后 100 步均值、最大温度、最大温差、平均功率，以及压缩机和水泵指令上下限占用比例。

500 步奖励未改善不视为代码失败。它表示当前默认 TD3 超参数或奖励设计需要后续诊断，不能据此启动正式长训练。

## 非目标

- 不调节奖励权重或 TD3 超参数。
- 不实现断点续训、早停、并行环境或 TensorBoard。
- 不运行完整工况或正式长训练。
- 不提交模型、CSV、PNG 或检查点到 Git。
- 不与 PID/MPC 做性能优劣结论。
