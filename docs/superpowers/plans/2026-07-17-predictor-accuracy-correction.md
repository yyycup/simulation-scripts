# MPC双预测模型精度修正实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## 2026-07-18：P模型统一容量公式决策

用户确认优先级为：物理可解释、计算简单、便于后续嵌入，并接受验证集 MAPE 约 4%。因此本阶段不采用双物理容量分支，也不采用神经网络或容量查表。

采用 `single_enhanced_v1` 单一低阶容量公式：

- 2000 rpm 为明确物理启停边界，低于 2000 rpm 的容量严格为 0；
- 主动区最多使用 14 个具名物理项，包括水泵比、冷却液温度、环境温度、压缩机工作点、二次项及有限交互项；
- 拟合仅使用 train，模型复杂度和正则化仅由 validation 选择，test 仅在模型冻结后评估一次；
- 保留旧六系数 `legacy_six_term` 工件读取和原平滑门行为；
- 不修改 P 动态热结构、L、MPC、Candidate B 或完整制冷循环模型。

冻结后的验证集主动区结果：MAPE 4.050%、RMSE 85.805 W、最大相对误差 15.874%。独立测试集主动区结果：MAPE 4.403%、RMSE 104.741 W、最大相对误差 17.784%，低于 2000 rpm 的 MAE 为 0 W。结果工件与报告均保存到 `outputs/mpc_predictor_accuracy_correction_v2/`，不覆盖旧六系数基线。

## 2026-07-18：P模型冷板能量平衡修正

将冷板由“仅向供液温度滞后”修正为集中热容能量平衡：同时接收电池热流并向冷却液放热；等效冷板热容由现有 `plate_tau_s` 和参考流体热容流率推导，不增加自由参数。动态辨识目标同时改为每一步都包含供液、冷板、回液、电池和冷却液温度误差。冻结测试结果相对旧动态工件：冷板 MAE 4.894→2.559°C，电池 0.150→0.098°C，冷却液 0.555→0.389°C，供液 0.796→0.570°C，回液 0.570→0.460°C，蒸发制冷量 158.4→104.3 W。冷板仍为当前主要误差源，暂不接入 MPC。

`q_cond_w` 经代码追踪确认仅为 P 内部一级制冷滞后状态，并非真实冷凝器热量；因此从动态观测目标中移除 `q_cond_eff_w`，工件显式记录 `q_cond_role=internal_refrigeration_lag_not_condenser_prediction`。重新冻结后的测试 MAE：蒸发量 118.0 W、供液 0.569°C、回液 0.479°C、冷却液 0.403°C、电池 0.102°C、冷板 2.547°C。除冷板外均满足本轮建议门槛；冷板剩余偏差来自单节点结构，不再通过放宽参数边界补偿。

## 2026-07-18：P模型冷板换热有效度修正

在冷板—冷却液热导上增加唯一的新物理参数 `plate_fluid_effectiveness`，范围为 `(0, 1]`，旧工件缺失该字段时按 `1.0` 运行。单独拟合该参数虽可修正冷板，但会显著破坏其他温度状态，因此未采用；随后仅联合拟合 `battery_plate_conductance_w_k`、`plate_tau_s` 和 `plate_fluid_effectiveness`，其他容量、制冷动态和热参数全部冻结，且只用验证集选择候选。最终参数分别为 `295.604 W/K`、`2.758 s` 和 `0.4300`，验证集综合加权 MAE 从 `0.5443` 降到 `0.2759`。独立测试集冷板 MAE 从 `2.547°C` 降到 `0.194°C`，回液、电池和冷却液也改善；供液 MAE 从 `0.569°C` 增至 `0.901°C`，仍低于 `1°C`。结果保存在 `outputs/mpc_predictor_accuracy_correction_v2/`，尚未接入 MPC 或复制到 `model_data/`。

## 2026-07-18：P模型供液物性修正

分时间尺度和激励类型诊断确认供液系统性低估来自有效热容流率，而非管路延迟：植物模型的冷却液比热为已知常数 `3391 J/(kg·K)`，旧动态拟合却将其降至 `2981.237 J/(kg·K)`以补偿当时尚未修正的冷板结构。现将该物性恢复为 `3391 J/(kg·K)`并从动态自由参数向量中移除，后续不得再次拟合；专用供液修正流程只改变这一项，其余容量、制冷动态和冷板参数全部冻结。验证集综合加权 MAE 从 `0.2759` 降到 `0.2465`。独立测试集供液 MAE 从 `0.901°C` 降到 `0.415°C`，冷板 `0.227°C`、回液 `0.341°C`、电池 `0.107°C`、冷却液 `0.298°C`、蒸发量 `117.1 W`。50/100/300秒测试点均较修正前改善，但最激烈的反向联合阶跃在50秒仍有 `1.919°C` 瞬态误差，因此当前仍为离线候选，未接入MPC。

**Goal:** 修正模型L的验证语义，建立B0/P/L同口径原始单位评估，并使用完整稳态与动态数据在既有物理结构内重新辨识模型P，使误差结论可比较、可验收、可回退。

**Architecture:** 先修复验证元数据，保证训练回退不会被误报为验证结果；随后扩展离线评估器，对三个独立模型使用相同初值、相同测试轨迹和相同50/100/300秒窗口。模型P先重新拟合稳态容量，再重新拟合已批准的动态与热参数；本计划不增加新物理系数、不融合P/L、不改变Candidate B运行时默认值。

**Tech Stack:** Python 3.11、NumPy、Pandas、SciPy `least_squares`、`unittest`、JSON/CSV工件、现有BTMS完整制冷循环数据生成器。

---

## 当前基线与范围

当前 `dynamic_smoke.csv` 每个验证/测试轨迹只有19个可评分步，原始单位自由滚动MAE为：

| 变量 | 验证MAE | 测试MAE |
|---|---:|---:|
| `t_batt_c` | 0.0388 °C | 0.0314 °C |
| `t_cool_c` | 0.3602 °C | 0.1369 °C |
| `t_supply_c` | 0.7400 °C | 0.3808 °C |
| `t_plate_c` | 1.0617 °C | 1.2490 °C |
| `t_return_c` | 1.0693 °C | 0.8093 °C |
| `q_evap_eff_w` | 410.3 W | 486.0 W |
| `q_cond_eff_w` | 115.2 W | 122.6 W |

本计划保留以下边界：

- Candidate B保持正式默认模型；
- P与L独立实现、独立评估，不生成融合模型；
- 不改植物模型、完整制冷循环或MPC权重；
- 不把临时工件复制到`model_data/`；
- 不覆盖`outputs/mpc_predictor_identification_v1/`已有结果；
- P结构内只重新辨识已批准的容量系数、时间常数、延迟、热容、换热系数和冷板时间常数；
- 若完整数据重拟合仍未达标，本计划停止并输出残差证据，不擅自增加新方程或参数。

## 文件职责

- Modify `fit_mpc_lpv_predictor.py` — 修正验证来源、方向覆盖和训练回退语义。
- Modify `test_mpc_lpv_predictor.py` — 固定方向训练覆盖和无有效验证滚动的行为。
- Modify `evaluate_mpc_predictors.py` — 统一B0/P/L自由滚动、逐步重置和原始单位指标。
- Modify `test_evaluate_mpc_predictors.py` — 固定模型行、变量单位、时间窗口、场景分组和验收规则。
- Read/run `generate_predictor_identification_data.py` — 生成新版本完整稳态/动态数据，不修改代码。
- Read/run `fit_mpc_physics_predictor.py` — 分两阶段拟合P容量和P动态，不修改代码，除非测试暴露现有契约缺陷。
- Read/run `fit_mpc_lpv_predictor.py` — 使用完整动态数据重新拟合L。
- Generated only `outputs/mpc_predictor_accuracy_correction_v1/` — 新数据、临时工件和报告；不进入Git。

---

### Task 1: 修复模型L的验证来源语义

**Files:**
- Modify: `fit_mpc_lpv_predictor.py:222-349`
- Modify: `test_mpc_lpv_predictor.py:190-220`

- [ ] **Step 1: 写入“验证方向没有独立训练覆盖”的失败测试**

在`LpvContractTest`中加入：

```python
def test_direction_fitted_from_global_training_is_not_marked_validated(self):
    frame = synthetic_frame()
    frame = frame.loc[
        ~((frame["split"] == "train") & (frame["flow_direction"] == -1))
    ].copy()

    artifact = fit_lpv_artifact(frame)
    reverse = artifact["fit"]["validation_by_direction"]["-1"]

    self.assertFalse(reverse["independently_fitted"])
    self.assertTrue(reverse["fallback_fitted"])
    self.assertFalse(reverse["validated"])
    self.assertEqual(artifact["fit"]["fit_status"], "partially_validated")
```

- [ ] **Step 2: 写入“验证轨迹不可滚动时不得冒充验证选型”的失败测试**

```python
def test_short_validation_uses_explicit_training_only_selection(self):
    frame = synthetic_frame()
    validation = frame[frame["split"] == "validation"].groupby(
        "scenario_id", sort=True
    ).head(1)
    frame = pd.concat(
        [frame[frame["split"] != "validation"], validation], ignore_index=True
    )

    artifact = fit_lpv_artifact(frame)
    fit = artifact["fit"]

    self.assertEqual(fit["selection_metric_source"], "training")
    self.assertEqual(fit["selection_method"], "training_multistep_weighted_temperature_mae")
    self.assertEqual(fit["fit_status"], "train_only_unvalidated")
    self.assertIsNone(fit["base_validation_mae_c"])
    self.assertIsNone(fit["pump_validation_mae_c"])
    self.assertFalse(fit["pump_schedule_selected"])
```

- [ ] **Step 3: 运行RED测试**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q `
  test_mpc_lpv_predictor.LpvContractTest.test_direction_fitted_from_global_training_is_not_marked_validated `
  test_mpc_lpv_predictor.LpvContractTest.test_short_validation_uses_explicit_training_only_selection
```

Expected: 第一项错误地得到`validated=True`；第二项错误地报告validation selection或使用训练指标填充validation字段。

- [ ] **Step 4: 实现统一的验证可用性和方向覆盖判定**

在`fit_mpc_lpv_predictor.py`加入：

```python
def _has_usable_validation(frame):
    return any(
        len(group) > max(ORDERS)
        for _, group in frame.groupby("scenario_id", sort=True)
    )


def _direction_validation_record(selected, train, validation, direction):
    sources = selected["fit"]["sample_sources"][str(direction)]
    independently_fitted = all(
        item["sample_source"] != "global_train"
        for item in sources.values()
    )
    usable_scenarios = sum(
        len(group) > selected["order"]
        for _, group in validation.loc[
            validation["flow_direction"] == direction
        ].groupby("scenario_id", sort=True)
    )
    return {
        "training_scenario_count": int(
            train.loc[
                train["flow_direction"] == direction, "scenario_id"
            ].nunique()
        ),
        "usable_validation_scenario_count": int(usable_scenarios),
        "independently_fitted": bool(independently_fitted),
        "fallback_fitted": not independently_fitted,
        "validated": bool(independently_fitted and usable_scenarios),
    }
```

在`fit_lpv_artifact()`中先确定统一指标来源，禁止候选之间混用训练和验证指标：

```python
validation_usable = _has_usable_validation(validation)
selection_frame = validation if validation_usable else train
selection_metric_source = "validation" if validation_usable else "training"

base, base_mae, base_search = _best_candidate(
    train, selection_frame, dt_s, False
)
pump, pump_mae, pump_search = _best_candidate(
    train, selection_frame, dt_s, True
)
choose_pump = (
    selection_metric_source == "validation"
    and pump_schedule_improves(base_mae, pump_mae)
)
```

只在`selection_metric_source == "validation"`时写入`*_validation_mae_c`。训练回退时保留`selection_mae_c`，并将`fit_status`设为`train_only_unvalidated`。

- [ ] **Step 5: 运行Task 1测试和完整LPV测试**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q test_mpc_lpv_predictor
```

Expected: 全部通过；合成完整验证数据仍为`validated`，smoke只有正向有效验证时仍为`partially_validated`。

- [ ] **Step 6: 提交Task 1**

```powershell
git add -- fit_mpc_lpv_predictor.py test_mpc_lpv_predictor.py
git commit -m "fix: 修正LPV验证来源语义"
```

---

### Task 2: 建立原始单位、分变量、分时间尺度指标

**Files:**
- Modify: `evaluate_mpc_predictors.py`
- Modify: `test_evaluate_mpc_predictors.py`

- [ ] **Step 1: 写入原始单位指标失败测试**

```python
def test_variable_metrics_keep_physical_units_and_horizons(self):
    predictions = pd.DataFrame({
        "model": ["p"] * 6,
        "scenario_id": ["case"] * 6,
        "scene": ["peak"] * 6,
        "flow_direction": [1] * 6,
        "evaluation_mode": ["free_rollout"] * 6,
        "horizon_s": [50, 50, 100, 100, 300, 300],
        "variable": ["t_plate_c", "q_evap_eff_w"] * 3,
        "unit": ["degC", "W"] * 3,
        "actual": [30.0, 1000.0, 30.0, 1000.0, 30.0, 1000.0],
        "predicted": [31.0, 1100.0, 32.0, 1200.0, 33.0, 1300.0],
    })

    result = grouped_variable_metrics(predictions)

    self.assertEqual(set(result["horizon_s"]), {50, 100, 300})
    self.assertEqual(set(result["unit"]), {"degC", "W"})
    self.assertEqual(set(result["variable"]), {"t_plate_c", "q_evap_eff_w"})
    self.assertIn("p95_abs", result.columns)
```

- [ ] **Step 2: 运行RED测试**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q `
  test_evaluate_mpc_predictors.PredictorReportTest.test_variable_metrics_keep_physical_units_and_horizons
```

Expected: `grouped_variable_metrics`尚不存在。

- [ ] **Step 3: 实现固定变量契约和分组指标**

在`evaluate_mpc_predictors.py`加入：

```python
VARIABLE_SPECS = {
    "t_batt_c": ("degC", 0.2),
    "t_cool_c": ("degC", 0.5),
    "t_plate_c": ("degC", 1.0),
    "t_supply_c": ("degC", 1.0),
    "t_return_c": ("degC", 1.0),
    "q_evap_eff_w": ("W", 500.0),
    "q_cond_eff_w": ("W", 500.0),
    "n_comp_eff_rpm": ("rpm", 500.0),
    "n_pump_eff_rpm": ("rpm", 500.0),
}
HORIZONS_S = (50, 100, 300)


def grouped_variable_metrics(predictions):
    keys = [
        "model", "scene", "flow_direction", "evaluation_mode",
        "horizon_s", "variable", "unit",
    ]
    rows = []
    for group_key, group in predictions.groupby(keys, sort=True, dropna=False):
        metrics = error_metrics(group["actual"], group["predicted"])
        rows.append(dict(zip(keys, group_key)) | metrics | {"sample_count": len(group)})
    return pd.DataFrame(rows).sort_values(keys, kind="stable").reset_index(drop=True)
```

物理单位指标与归一化综合指标分列保存，禁止用无单位的综合MAE替换`degC/W/rpm`指标。

- [ ] **Step 4: 增加低速/主动制冷量与温度综合指标测试**

测试必须断言：

```python
self.assertIn("low_speed_mae_w", capacity_row)
self.assertIn("active_mape_percent", capacity_row)
self.assertIn("weighted_temperature_mae", temperature_row)
self.assertEqual(set(report["evaluation_mode"]), {"free_rollout", "one_step_reset"})
```

- [ ] **Step 5: 运行评估器单元测试并提交**

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q test_evaluate_mpc_predictors
git add -- evaluate_mpc_predictors.py test_evaluate_mpc_predictors.py
git commit -m "feat: 添加预测模型原始单位指标"
```

Expected: 所有评估器测试通过。

---

### Task 3: 实现B0/P/L同轨迹离线比较

**Files:**
- Modify: `evaluate_mpc_predictors.py`
- Modify: `test_evaluate_mpc_predictors.py`

- [ ] **Step 1: 写入模型无关报告失败测试**

```python
def test_report_has_three_independent_models_without_fusion(self):
    report = build_comparison_report(self.steady, self.dynamic, self.p_path, self.l_path)
    self.assertEqual(set(report["temperature_metrics"]["model"]), {"b0", "p", "l"})
    self.assertNotIn("h", set(report["temperature_metrics"]["model"]))
    self.assertEqual(set(report["temperature_metrics"]["horizon_s"]), {50, 100, 300})
    self.assertEqual(
        set(zip(report["temperature_metrics"]["scene"], report["temperature_metrics"]["flow_direction"])),
        {("peak", 1), ("peak", -1), ("freq", 1), ("freq", -1)},
    )
```

- [ ] **Step 2: 实现相同初值的自由滚动和逐步重置**

为每个完整测试场景执行：

```python
for model_name, predictor in predictors.items():
    predictor.reset(first_row)
    for index in range(1, len(scenario)):
        free_prediction = predictor.step(inputs_from(scenario.iloc[index]))
        capture_if_horizon(free_prediction, scenario.iloc[index])

        reset_predictor = predictors_for_reset[model_name]
        reset_predictor.reset(scenario.iloc[index - 1])
        one_step_prediction = reset_predictor.step(inputs_from(scenario.iloc[index]))
        capture_one_step(one_step_prediction, scenario.iloc[index])
```

将该循环提取为只读函数`rollout_model_predictions(model_name, dynamic_test, artifact)`。
函数返回DataFrame，固定列为`model/scenario_id/scene/flow_direction/evaluation_mode/`
`horizon_s/variable/unit/actual/predicted`；输入DataFrame先复制，函数内不得修改调用方数据。

B0必须使用Candidate B容量工件与当前正式动态/热参数走同一离线物理推进代码；P使用显式P工件；L使用显式L工件。自由滚动期间禁止用未来植物状态替换预测状态。

- [ ] **Step 3: 写出固定报告文件**

CLI必须生成：

```text
capacity_metrics.csv
temperature_metrics.csv
stability_metrics.csv
acceptance_summary.csv
comparison_metadata.json
```

其中`temperature_metrics.csv`至少包含：

```text
model,scene,flow_direction,evaluation_mode,horizon_s,variable,unit,
mae,rmse,mean_bias,p95_abs,sample_count
```

- [ ] **Step 4: 实现既定验收规则，不自动切换默认模型**

`acceptance_summary.csv`必须计算：

- 主动制冷容量MAPE `<=2.5%`；
- 主动制冷容量RMSE `<=100 W`；
- 主动制冷容量最大相对误差 `<=12%`；
- 1000–1800 rpm制冷量 `<=25 W`；
- 动态主动制冷量一步MAPE `<=5%`；
- P/L相对B0的50/100/300秒加权温度综合MAE至少改善20%；
- 任一时间尺度MAE不得比B0恶化超过10%；
- 任一NaN、发散或不稳定均判失败。

报告只写`passes_accuracy_gate`，不得修改运行时配置或`model_data/`。

- [ ] **Step 5: 运行Task 3测试并提交**

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q test_evaluate_mpc_predictors
git add -- evaluate_mpc_predictors.py test_evaluate_mpc_predictors.py
git commit -m "feat: 添加B0与双预测模型统一评估"
```

---

### Task 4: 生成新的完整辨识数据集

**Files:**
- Read/run: `generate_predictor_identification_data.py`
- Generated: `outputs/mpc_predictor_accuracy_correction_v1/data/steady_full.csv`
- Generated: `outputs/mpc_predictor_accuracy_correction_v1/data/dynamic_full.csv`

- [ ] **Step 1: 运行数据生成与schema回归**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q `
  test_predictor_identification_data test_generate_predictor_identification_data
```

Expected: 全部通过。

- [ ] **Step 2: 生成新版本完整数据，不覆盖v1**

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' generate_predictor_identification_data.py `
  --dataset all `
  --mode full `
  --seed 20260714 `
  --output-root outputs/mpc_predictor_accuracy_correction_v1/data
```

Expected: 稳态网格轴为12档压缩机转速、5档泵速、冷却液温度`15/17.5/20/25/30/35 °C`和环境温度`20/25/30/35/40 °C`，共`12×5×6×5=1800`行，确定性split为train/validation/test=`1080/360/360`；动态数据包含15个场景，train/validation/test均覆盖三种激励类型和正反流向。

- [ ] **Step 3: 验证数据边界和哈希**

运行只读检查，断言：

```python
assert set(dynamic["split"]) == {"train", "validation", "test"}
assert set(dynamic["flow_direction"]) == {-1, 1}
assert set(
    zip(dynamic["split"], dynamic["excitation_kind"])
) == {
    (split, excitation)
    for split in ("train", "validation", "test")
    for excitation in ("compressor-only", "pump-only", "combined")
}
assert steady.select_dtypes(include="number").apply(np.isfinite).all().all()
assert dynamic.select_dtypes(include="number").apply(np.isfinite).all().all()
```

Expected: 全部成立；记录配置哈希、源代码哈希和场景清单。

---

### Task 5: 先重新拟合并验收P的稳态蒸发器容量

**Files:**
- Read/run: `fit_mpc_physics_predictor.py`
- Generated: `outputs/mpc_predictor_accuracy_correction_v1/artifacts/physics_p_capacity_full.json`
- Generated: `outputs/mpc_predictor_accuracy_correction_v1/reports/p_capacity/`

- [ ] **Step 1: 使用完整稳态训练/验证分割拟合容量**

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' fit_mpc_physics_predictor.py `
  --steady-csv outputs/mpc_predictor_accuracy_correction_v1/data/steady_full.csv `
  --output outputs/mpc_predictor_accuracy_correction_v1/artifacts/physics_p_capacity_full.json
```

Expected: `fit.fit_status=validated`，候选只使用train拟合并由validation选择；test未被读取。

- [ ] **Step 2: 运行稳态容量测试集评估**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
@'
import json
from pathlib import Path
import pandas as pd
from evaluate_mpc_predictors import capacity_metrics
from mpc_physics_predictor import evaluate_physics_capacity, load_physics_artifact

artifact = load_physics_artifact(
    "outputs/mpc_predictor_accuracy_correction_v1/artifacts/physics_p_capacity_full.json",
    require_validated=True,
)
steady = pd.read_csv(
    "outputs/mpc_predictor_accuracy_correction_v1/data/steady_full.csv"
)
test = steady.loc[steady["split"] == "test"].copy()
test["q_pred_w"] = [
    evaluate_physics_capacity(
        row.n_comp_eff_rpm,
        row.n_pump_eff_rpm,
        row.t_cool_c,
        row.t_ambient_c,
        artifact=artifact,
    )
    for row in test.itertuples(index=False)
]
result = capacity_metrics(
    test[["n_comp_cmd_rpm", "q_evap_ss_w", "q_pred_w"]]
)
result["max_predicted_capacity_1000_to_1800_rpm_w"] = float(
    test.loc[test["n_comp_cmd_rpm"] <= 1800, "q_pred_w"].max()
)
output = Path(
    "outputs/mpc_predictor_accuracy_correction_v1/reports/p_capacity/capacity_metrics.json"
)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True))
'@ | & 'C:\Users\24776\miniforge3\envs\btms\python.exe' -
```

- [ ] **Step 3: 应用容量停止门**

只有同时满足以下条件才进入动态拟合：

```text
active_mape_percent <= 2.5
rmse <= 100
active_max_relative_error_percent <= 12
max_predicted_capacity_1000_to_1800_rpm_w <= 25
all_predictions_finite == true
all_predictions_nonnegative == true
```

Expected: 通过则继续Task 6；不通过则保留报告并停止，不调整误差尺度或容量上限掩盖问题。

---

### Task 6: 在既有P结构内重新辨识动态和热参数

**Files:**
- Read/run: `fit_mpc_physics_predictor.py`
- Read/run: `mpc_physics_predictor.py`
- Test: `test_mpc_physics_predictor.py`
- Generated: `outputs/mpc_predictor_accuracy_correction_v1/artifacts/physics_p_dynamic_full.json`

- [ ] **Step 1: 运行P模型完整回归**

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q test_mpc_physics_predictor
```

Expected: 全部通过；低速、单调性、参数消费、训练/验证/测试隔离不回退。

- [ ] **Step 2: 使用完整稳态与动态数据联合拟合**

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' fit_mpc_physics_predictor.py `
  --steady-csv outputs/mpc_predictor_accuracy_correction_v1/data/steady_full.csv `
  --dynamic-csv outputs/mpc_predictor_accuracy_correction_v1/data/dynamic_full.csv `
  --output outputs/mpc_predictor_accuracy_correction_v1/artifacts/physics_p_dynamic_full.json
```

Expected: 只重新辨识现有参数：执行器时间常数、输入/流量延迟、冷凝/蒸发时间常数、供回液延迟、热容、换热系数、冷板时间常数和环境换热；不增加新参数。

- [ ] **Step 3: 验证常数与调度时间常数选择**

检查工件：

```python
fit = artifact["fit"]["dynamic_fit"]
assert fit["selection_source"] == "validation_only"
if fit["selected_time_constant_model"] == "scheduled":
    assert fit["scheduled_improvement"] >= 0.10
```

Expected: 调度候选失败时明确回退constant；改善不足10%时不启用调度。

- [ ] **Step 4: 运行P的独立测试集原始单位评估**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
@'
from pathlib import Path
import pandas as pd
from evaluate_mpc_predictors import grouped_variable_metrics, rollout_model_predictions
from mpc_physics_predictor import load_physics_artifact

artifact = load_physics_artifact(
    "outputs/mpc_predictor_accuracy_correction_v1/artifacts/physics_p_dynamic_full.json",
    require_validated=True,
)
dynamic = pd.read_csv(
    "outputs/mpc_predictor_accuracy_correction_v1/data/dynamic_full.csv"
)
test = dynamic.loc[dynamic["split"] == "test"].copy()
predictions = rollout_model_predictions("p", test, artifact)
metrics = grouped_variable_metrics(predictions)
output = Path(
    "outputs/mpc_predictor_accuracy_correction_v1/reports/p_dynamic/temperature_metrics.csv"
)
output.parent.mkdir(parents=True, exist_ok=True)
metrics.to_csv(output, index=False, encoding="utf-8")
print(metrics.to_string(index=False))
'@ | & 'C:\Users\24776\miniforge3\envs\btms\python.exe' -
```

Expected: 单独报告`q_evap_eff_w`、`t_supply_c`、`t_plate_c`、`t_return_c`、`t_batt_c`、`t_cool_c`的MAE/RMSE/bias/P95，禁止只报告归一化综合MAE。

- [ ] **Step 5: 应用P动态停止门**

必须满足：

```text
active_one_step_q_evap_mape_percent <= 5
free_rollout_all_finite == true
all_reported_raw_unit_metrics_finite == true
```

Expected: 通过则进入Task 7和最终B0比较；不通过则输出每变量、每场景、每流向和50/100/300秒残差，停止本计划，不修改模型结构。相对B0的20%改善和10%恶化上限只在Task 8的同轨迹报告中判定。

---

### Task 7: 使用完整数据重新拟合L并验证修正后的语义

**Files:**
- Read/run: `fit_mpc_lpv_predictor.py`
- Test: `test_mpc_lpv_predictor.py`
- Generated: `outputs/mpc_predictor_accuracy_correction_v1/artifacts/lpv_l_full.json`

- [ ] **Step 1: 运行L回归并拟合完整数据**

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q test_mpc_lpv_predictor
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' fit_mpc_lpv_predictor.py `
  --dynamic-csv outputs/mpc_predictor_accuracy_correction_v1/data/dynamic_full.csv `
  --output outputs/mpc_predictor_accuracy_correction_v1/artifacts/lpv_l_full.json
```

Expected: 正反方向均有独立训练和有效验证时`fit_status=validated`；任一方向使用`global_train`时该方向`validated=false`。

- [ ] **Step 2: 检查稳定性和泵调度门槛**

```python
assert artifact["fit"]["max_spectral_radius"] < 0.995
if artifact["fit"]["pump_schedule_selected"]:
    assert artifact["fit"]["pump_improvement_fraction"] >= 0.05
```

Expected: 60步滚动有限；不稳定候选不进入验证排名。

---

### Task 8: 生成最终B0/P/L同口径报告并决定是否继续

**Files:**
- Read/run: `evaluate_mpc_predictors.py`
- Generated: `outputs/mpc_predictor_accuracy_correction_v1/reports/final_offline/`

- [ ] **Step 1: 在同一独立测试轨迹上运行三模型比较**

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' evaluate_mpc_predictors.py `
  --steady-csv outputs/mpc_predictor_accuracy_correction_v1/data/steady_full.csv `
  --dynamic-csv outputs/mpc_predictor_accuracy_correction_v1/data/dynamic_full.csv `
  --physics-artifact outputs/mpc_predictor_accuracy_correction_v1/artifacts/physics_p_dynamic_full.json `
  --lpv-artifact outputs/mpc_predictor_accuracy_correction_v1/artifacts/lpv_l_full.json `
  --output-root outputs/mpc_predictor_accuracy_correction_v1/reports/final_offline
```

Expected: B0、P、L各自拥有50/100/300秒、调峰/调频、正向/反向、自由滚动/逐步重置指标；没有融合模型行。

- [ ] **Step 2: 核对报告完整性**

```powershell
$root = 'outputs/mpc_predictor_accuracy_correction_v1/reports/final_offline'
Get-Item "$root/capacity_metrics.csv", "$root/temperature_metrics.csv", `
  "$root/stability_metrics.csv", "$root/acceptance_summary.csv", `
  "$root/comparison_metadata.json"
```

Expected: 五个文件存在且非空；元数据记录数据哈希、工件哈希、场景清单和评估代码版本。

- [ ] **Step 3: 执行最终停止门**

```text
任一模型出现NaN/发散/稳定性失败 -> 不进入MPC集成
任一模型任一时间尺度比B0恶化超过10% -> 不进入MPC集成
相对B0综合温度MAE改善不足20% -> 不进入MPC集成
容量或动态制冷量门槛失败 -> P不进入MPC集成
```

Expected: 只形成离线候选结论，不复制工件、不切换默认值、不开始正式闭环。

- [ ] **Step 4: 运行最终相关回归**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q `
  test_predictor_identification_data `
  test_generate_predictor_identification_data `
  test_mpc_physics_predictor `
  test_mpc_lpv_predictor `
  test_evaluate_mpc_predictors
```

Expected: 零失败。

- [ ] **Step 5: 提交评估代码并提供人工决策材料**

```powershell
git status --short
git log -8 --oneline
```

报告必须列出：

- B0/P/L每变量原始单位误差；
- 50/100/300秒自由滚动和逐步重置结果；
- 每个未通过阈值；
- P是否仍被`q_evap`、冷板或回液误差限制；
- 两个临时工件的精确路径；
- Candidate B仍为默认值。

未经用户新的明确确认，不进入运行时集成、不复制到`model_data/`、不修改默认预测器。

---

## 自检结果

- 规格覆盖：包含L验证语义、原始单位指标、完整数据、P容量、P动态、L完整拟合、B0/P/L统一比较和停止门。
- 范围控制：没有加入P/L融合、没有加入新物理参数、没有修改MPC或植物模型。
- 数据边界：所有拟合只使用train，选择只使用有效validation，最终指标只使用test。
- 输出边界：所有新数据和工件进入`outputs/mpc_predictor_accuracy_correction_v1/`，不覆盖旧结果、不进入Git。
- 回退原则：未达标时保留Candidate B并停止，不通过修改尺度、阈值或容量上限掩盖误差。
