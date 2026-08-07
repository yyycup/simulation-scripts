# Physics-P压缩机排量恢复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 先以独立实验验证Physics-P和详细物理对象同时恢复原始1.00倍压缩机排量，验证合格后将1.00倍基础工件恢复为运行脚本默认值。

**Architecture:** 复用现有`--artifact`入口和排量同步上下文，第一阶段不改变默认配置，只用基础工件触发`scale=1.0`的无缩放路径。调峰与调频完整工况均合格后，通过一项回归测试驱动默认工件路径变更；0.90倍工件、缩放辅助函数和历史结果全部保留。

**Tech Stack:** Python 3.12、pandas、NumPy、GEKKO、unittest、matplotlib、PowerShell。

---

### Task 1: 固定1.00倍工件与验证输入

**Files:**
- Inspect: `outputs/mpc_predictor_low_speed_retrain_v1/thermal_bias_correction_v3/physics_p_heat_generation_corrected.json`
- Inspect: `run_p_mpc_operational.py:35-79`
- Inspect: `test_p_mpc_operational.py:1-80`

- [ ] **Step 1: 验证基础工件存在且未声明缩小排量**

Run:

```powershell
@'
import json
from pathlib import Path

p = Path(r"outputs/mpc_predictor_low_speed_retrain_v1/thermal_bias_correction_v3/physics_p_heat_generation_corrected.json")
d = json.loads(p.read_text(encoding="utf-8"))
assert p.is_file()
assert float(d.get("thermal", {}).get("compressor_displacement_scale", 1.0)) == 1.0
print("artifact_scale=1.0")
'@ | & 'C:\Users\24776\miniforge3\envs\btms\python.exe' -
```

Expected: `artifact_scale=1.0`。

- [ ] **Step 2: 验证现有同步入口对1.00倍为无操作**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest `
  test_p_mpc_operational.PhysicsPMpcOperationalTest.test_displacement_variant_scales_plant_only_inside_context
```

Expected: `Ran 1 test`和`OK`；上下文退出后`thermal_system.V_disp_m3_per_rev`恢复原值。

- [ ] **Step 3: 记录当前0.90倍完整基线**

Run:

```powershell
Get-Content 'outputs/p_mpc_operational_v1/temperature_bias_compensation_v1/peak_h14_bias2_full1280/operational_summary.csv'
Get-Content 'outputs/p_mpc_operational_v1/frequency_bias_validation_v1/freq_gain0_full720/operational_summary.csv'
```

Expected: 调峰1280步、调频720步两个汇总均可读取；后续只与这两份基线比较。

### Task 2: 运行1.00倍调峰和调频冒烟验证

**Files:**
- Read: `run_p_mpc_operational.py`
- Create: `outputs/p_mpc_operational_v1/displacement_restore_1p00_v1/peak_smoke60/`
- Create: `outputs/p_mpc_operational_v1/displacement_restore_1p00_v1/freq_smoke60/`

- [ ] **Step 1: 运行调峰60步冒烟**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' `
  run_p_mpc_operational.py `
  --scenes peak `
  --steps 60 `
  --artifact outputs/mpc_predictor_low_speed_retrain_v1/thermal_bias_correction_v3/physics_p_heat_generation_corrected.json `
  --output-root outputs/p_mpc_operational_v1/displacement_restore_1p00_v1/peak_smoke60 `
  --force
```

Expected: 日志显示`displacement_scale=1`、`temp_bias_gain=2`，最终`qualified=True`。

- [ ] **Step 2: 运行调频60步冒烟**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' `
  run_p_mpc_operational.py `
  --scenes freq `
  --steps 60 `
  --artifact outputs/mpc_predictor_low_speed_retrain_v1/thermal_bias_correction_v3/physics_p_heat_generation_corrected.json `
  --output-root outputs/p_mpc_operational_v1/displacement_restore_1p00_v1/freq_smoke60 `
  --force
```

Expected: 日志显示`displacement_scale=1`、`temp_bias_gain=0`，最终`qualified=True`。

- [ ] **Step 3: 检查冒烟结果完整性**

Run:

```powershell
@'
from pathlib import Path
import pandas as pd

root = Path(r"outputs/p_mpc_operational_v1/displacement_restore_1p00_v1")
for scene in ("peak_smoke60", "freq_smoke60"):
    summary = pd.read_csv(root / scene / "operational_summary.csv").iloc[0]
    assert int(summary["steps"]) == 60
    assert float(summary["compressor_displacement_scale"]) == 1.0
    assert float(summary["solve_success_rate"]) == 1.0
    assert float(summary["prediction_domain_valid_rate"]) == 1.0
    assert bool(summary["qualified"])
    print(scene, "PASS")
'@ | & 'C:\Users\24776\miniforge3\envs\btms\python.exe' -
```

Expected: `peak_smoke60 PASS`和`freq_smoke60 PASS`。任一断言失败时停止，不运行完整工况。

### Task 3: 运行1.00倍完整工况并作出晋升判断

**Files:**
- Create: `outputs/p_mpc_operational_v1/displacement_restore_1p00_v1/peak_full1280/`
- Create: `outputs/p_mpc_operational_v1/displacement_restore_1p00_v1/freq_full720/`
- Read: `outputs/p_mpc_operational_v1/temperature_bias_compensation_v1/peak_h14_bias2_full1280/operational_summary.csv`
- Read: `outputs/p_mpc_operational_v1/frequency_bias_validation_v1/freq_gain0_full720/operational_summary.csv`

- [ ] **Step 1: 运行调峰1280步完整工况**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' `
  run_p_mpc_operational.py `
  --scenes peak `
  --steps 1280 `
  --artifact outputs/mpc_predictor_low_speed_retrain_v1/thermal_bias_correction_v3/physics_p_heat_generation_corrected.json `
  --output-root outputs/p_mpc_operational_v1/displacement_restore_1p00_v1/peak_full1280 `
  --force
```

Expected: `qualified=True`，结果为1280行，求解成功率和预测域有效率均为100%。

- [ ] **Step 2: 运行调频720步完整工况**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' `
  run_p_mpc_operational.py `
  --scenes freq `
  --steps 720 `
  --artifact outputs/mpc_predictor_low_speed_retrain_v1/thermal_bias_correction_v3/physics_p_heat_generation_corrected.json `
  --output-root outputs/p_mpc_operational_v1/displacement_restore_1p00_v1/freq_full720 `
  --force
```

Expected: `qualified=True`，结果为720行，求解成功率和预测域有效率均为100%。

- [ ] **Step 3: 生成1.00相对0.90的统一指标表**

Run:

```powershell
@'
from pathlib import Path
import pandas as pd

rows = []
cases = {
    "peak": (
        Path(r"outputs/p_mpc_operational_v1/temperature_bias_compensation_v1/peak_h14_bias2_full1280/operational_summary.csv"),
        Path(r"outputs/p_mpc_operational_v1/displacement_restore_1p00_v1/peak_full1280/operational_summary.csv"),
    ),
    "freq": (
        Path(r"outputs/p_mpc_operational_v1/frequency_bias_validation_v1/freq_gain0_full720/operational_summary.csv"),
        Path(r"outputs/p_mpc_operational_v1/displacement_restore_1p00_v1/freq_full720/operational_summary.csv"),
    ),
}
metrics = [
    "temperature_mae_c", "temperature_max_c", "temperature_final_c",
    "energy_kwh", "solve_time_mean_s", "solve_time_p95_s",
    "solve_success_rate", "prediction_domain_valid_rate",
]
for scene, (base_path, new_path) in cases.items():
    base = pd.read_csv(base_path).iloc[0]
    new = pd.read_csv(new_path).iloc[0]
    row = {"scene": scene, "baseline_scale": 0.9, "new_scale": 1.0}
    for metric in metrics:
        row[f"baseline_{metric}"] = base[metric]
        row[f"new_{metric}"] = new[metric]
        row[f"delta_{metric}"] = float(new[metric]) - float(base[metric])
    rows.append(row)
out = Path(r"outputs/p_mpc_operational_v1/displacement_restore_1p00_v1/displacement_restore_summary.csv")
pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
print(out)
'@ | & 'C:\Users\24776\miniforge3\envs\btms\python.exe' -
```

Expected: 生成两行汇总，分别为`peak`和`freq`。

- [ ] **Step 4: 执行晋升断言**

Run:

```powershell
@'
import pandas as pd

d = pd.read_csv(r"outputs/p_mpc_operational_v1/displacement_restore_1p00_v1/displacement_restore_summary.csv")
peak = d.loc[d.scene == "peak"].iloc[0]
freq = d.loc[d.scene == "freq"].iloc[0]
assert peak.new_solve_success_rate == 1.0
assert freq.new_solve_success_rate == 1.0
assert peak.new_prediction_domain_valid_rate == 1.0
assert freq.new_prediction_domain_valid_rate == 1.0
assert freq.new_temperature_final_c < freq.baseline_temperature_final_c
assert freq.new_temperature_max_c < freq.baseline_temperature_max_c
assert freq.new_temperature_mae_c <= freq.baseline_temperature_mae_c
assert peak.new_temperature_mae_c <= peak.baseline_temperature_mae_c + 0.01
assert peak.new_solve_time_mean_s <= peak.baseline_solve_time_mean_s * 1.20
assert freq.new_solve_time_mean_s <= freq.baseline_solve_time_mean_s * 1.20
print("PROMOTE_SCALE_1P00")
'@ | & 'C:\Users\24776\miniforge3\envs\btms\python.exe' -
```

Expected: `PROMOTE_SCALE_1P00`。任一断言失败时保留0.90默认值并报告失败指标，不进入Task 4。

### Task 4: 通过TDD将1.00倍基础工件恢复为默认值

**Files:**
- Modify: `test_p_mpc_operational.py:10-75`
- Modify: `run_p_mpc_operational.py:35-43`

- [ ] **Step 1: 将默认工件回归测试改为1.00倍预期**

Replace the existing default-artifact test with:

```python
def test_default_artifact_is_the_original_displacement_p_model(self):
    from run_p_mpc_operational import displacement_scale_from_artifact

    self.assertEqual(
        DEFAULT_OPERATIONAL_P_ARTIFACT.name,
        "physics_p_heat_generation_corrected.json",
    )
    self.assertIn(
        "thermal_bias_correction_v3",
        DEFAULT_OPERATIONAL_P_ARTIFACT.parts,
    )
    self.assertTrue(DEFAULT_OPERATIONAL_P_ARTIFACT.is_file())
    self.assertEqual(
        displacement_scale_from_artifact(DEFAULT_OPERATIONAL_P_ARTIFACT),
        1.0,
    )
```

- [ ] **Step 2: 运行测试并确认它先失败**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest `
  test_p_mpc_operational.PhysicsPMpcOperationalTest.test_default_artifact_is_the_original_displacement_p_model
```

Expected: FAIL，实际默认文件仍为`physics_p_displacement_scale_0p90.json`。

- [ ] **Step 3: 最小修改默认工件路径**

Change `DEFAULT_OPERATIONAL_P_ARTIFACT` to:

```python
DEFAULT_OPERATIONAL_P_ARTIFACT = (
    PROJECT_ROOT
    / "outputs"
    / "mpc_predictor_low_speed_retrain_v1"
    / "thermal_bias_correction_v3"
    / "physics_p_heat_generation_corrected.json"
)
```

Do not change `displacement_scale_from_artifact()` or `patched_plant_compressor_displacement()`.

- [ ] **Step 4: 运行默认工件测试并确认通过**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest `
  test_p_mpc_operational.PhysicsPMpcOperationalTest.test_default_artifact_is_the_original_displacement_p_model
```

Expected: `Ran 1 test`和`OK`。

- [ ] **Step 5: 运行完整运行入口回归测试**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest `
  test_p_mpc_operational.py `
  test_mpc_physics_p_closed_loop.py `
  test_mpc_params.py `
  test_mpc_comp_dmax_terminal_cost.py
```

Expected: 全部通过；调峰补偿仍为2，调频补偿仍为0，DMAX和权重断言不变。

- [ ] **Step 6: 仅提交默认入口和对应测试**

Run:

```powershell
git add -- run_p_mpc_operational.py test_p_mpc_operational.py
git diff --cached --check
git commit -m "fix: 恢复Physics-P原始压缩机排量"
```

Expected: 提交只包含`run_p_mpc_operational.py`与`test_p_mpc_operational.py`，不包含其他脏工作树文件。

### Task 5: 最终验证、制图与结论冻结

**Files:**
- Create: `outputs/p_mpc_operational_v1/displacement_restore_1p00_v1/figures/gen_fig_displacement_restore_comparison.py`
- Create: `outputs/p_mpc_operational_v1/displacement_restore_1p00_v1/figures/fig_displacement_restore_comparison.png`
- Create: `outputs/p_mpc_operational_v1/displacement_restore_1p00_v1/figures/fig_displacement_restore_comparison.pdf`
- Create: `outputs/p_mpc_operational_v1/displacement_restore_1p00_v1/排量恢复验证结论.md`

- [ ] **Step 1: 使用默认入口进行无仿真的路径核对**

Run:

```powershell
@'
from run_p_mpc_operational import DEFAULT_OPERATIONAL_P_ARTIFACT, displacement_scale_from_artifact

print(DEFAULT_OPERATIONAL_P_ARTIFACT)
print(displacement_scale_from_artifact(DEFAULT_OPERATIONAL_P_ARTIFACT))
assert DEFAULT_OPERATIONAL_P_ARTIFACT.name == "physics_p_heat_generation_corrected.json"
assert displacement_scale_from_artifact(DEFAULT_OPERATIONAL_P_ARTIFACT) == 1.0
'@ | & 'C:\Users\24776\miniforge3\envs\btms\python.exe' -
```

Expected: 输出1.00倍基础工件路径和`1.0`。

- [ ] **Step 2: 绘制0.90与1.00完整工况对照图**

Use the `academic-plotting` skill. The script must read these four CSV files:

```text
outputs/p_mpc_operational_v1/temperature_bias_compensation_v1/peak_h14_bias2_full1280/mpc/peak_physics_p_operational_steps1280_horizon14.csv
outputs/p_mpc_operational_v1/frequency_bias_validation_v1/freq_gain0_full720/mpc/freq_physics_p_operational_steps720_horizon12.csv
outputs/p_mpc_operational_v1/displacement_restore_1p00_v1/peak_full1280/mpc/peak_physics_p_operational_steps1280_horizon14.csv
outputs/p_mpc_operational_v1/displacement_restore_1p00_v1/freq_full720/mpc/freq_physics_p_operational_steps720_horizon12.csv
```

Plot battery average temperature, compressor command, cumulative energy, and solve time. Export both 300-DPI PNG and vector PDF.

- [ ] **Step 3: 写入冻结结论**

The conclusion must state:

```text
Physics-P和详细物理对象已统一恢复原始1.00倍压缩机排量；0.90倍工件仅作为历史实验保留。Candidate B、MPC目标温度、DMAX、权重和P热动态未修改。调峰继续使用在线温度补偿增益2，调频保持补偿关闭。
```

Include the exact peak/frequency MAE, maximum temperature, terminal temperature, energy, solve-time mean/P95, solve success rate, and prediction-domain valid rate from `displacement_restore_summary.csv`.

- [ ] **Step 4: 最终工作树审计**

Run:

```powershell
git status --short --branch
git show --stat --oneline HEAD
```

Expected: 最新代码提交只包含默认运行入口与其回归测试；其他原有未提交文件仍保持原状态。
