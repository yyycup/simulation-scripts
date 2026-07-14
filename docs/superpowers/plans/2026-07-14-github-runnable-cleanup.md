# GitHub可独立运行与本地彻底清理实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补齐GitHub独立运行所需的模型数据、Conda环境和中文说明，在验证并推送后永久删除本地历史归档与临时目录。

**Architecture:** 将运行必需的小型标定数据从被忽略的 `data/` 和 `outputs/` 复制到受Git管理的 `model_data/`，只修改默认路径，不改变模型公式。先提交并验证可运行仓库，再精确停止两个旧测试进程、删除已批准的本地归档/临时目录，最后更新清理记录并再次推送。

**Tech Stack:** Python 3.12、Conda/Miniforge、NumPy、Pandas、SciPy、Matplotlib、PyBaMM、CoolProp、GEKKO、PowerShell、Git。

---

## 文件结构

- Create: `model_data/hppc_params.json` — 电池HPPC参数。
- Create: `model_data/mpc_evaporator_capacity_candidate_b.json` — candidate B蒸发器标定参数。
- Create: `model_data/candidate_b_capacity_limits_grid.csv` — candidate B容量边界表。
- Create: `model_data/mpc_reduced_model_best_theta.json` — MPC降阶模型标定参数。
- Create: `test_model_data_paths.py` — 验证受管模型数据路径和文件存在性。
- Create: `environment.yml` — 可复现的Conda环境。
- Create: `README.md` — 中文项目说明与运行命令。
- Modify: `pack.py` — HPPC参数默认路径。
- Modify: `mpc_evaporator_capacity_model.py` — candidate B参数默认路径。
- Modify: `mpc_flow_direction_strategies.py` — 容量边界和降阶模型默认路径。
- Modify: `.gitignore` — 允许 `model_data/` 中的小型CSV/JSON进入Git。
- Modify after cleanup: `PROJECT_MAP_MIN.md`、`PROJECT_CLEANUP_REPORT.md` — 记录归档永久删除和当前入口。
- Delete after remote verification: `_archive/`、根目录 `tmp*/`、`_gekko_tmp_*/`、`__pycache__/`。

### Task 1: 建立清理前安全基线

**Files:**
- Test: current root Python files and lightweight tests

- [ ] **Step 1: 确认分支、远程和工作区状态**

Run:

```powershell
git status --short --branch
git remote -v
git log -3 --oneline
```

Expected: 当前分支为 `codex/peak-flow-comparison`，跟踪 `origin/main`；除本计划文件外没有未解释改动。

- [ ] **Step 2: 编译现有根目录Python文件**

Run:

```powershell
$files = Get-ChildItem -File -Filter '*.py' | Select-Object -ExpandProperty FullName
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m py_compile @files
```

Expected: exit code `0`。

- [ ] **Step 3: 运行现有核心轻量测试**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q test_btms_runtime test_mpc_evaporator_capacity_model test_final_controller_comparison test_final_controller_configured_pid test_plot_final_mpc_report
```

Expected: 15 tests pass。

### Task 2: 用测试固定新的模型数据路径

**Files:**
- Create: `test_model_data_paths.py`
- Modify: `pack.py`
- Modify: `mpc_evaporator_capacity_model.py`
- Modify: `mpc_flow_direction_strategies.py`

- [ ] **Step 1: 新增失败测试**

Create `test_model_data_paths.py`:

```python
import unittest
from pathlib import Path

import mpc_evaporator_capacity_model as evap_model
import mpc_flow_direction_strategies as flow_mpc
import pack


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DATA_ROOT = PROJECT_ROOT / "model_data"


class ModelDataPathTest(unittest.TestCase):
    def test_default_paths_use_tracked_model_data_directory(self):
        self.assertEqual(pack.HPPC_PARAMS_PATH, MODEL_DATA_ROOT / "hppc_params.json")
        self.assertEqual(
            evap_model.DEFAULT_CALIBRATION_PATH,
            MODEL_DATA_ROOT / "mpc_evaporator_capacity_candidate_b.json",
        )
        self.assertEqual(
            flow_mpc.DEFAULT_EVAPORATOR_CAPACITY_TABLE_PATH,
            MODEL_DATA_ROOT / "candidate_b_capacity_limits_grid.csv",
        )
        self.assertEqual(
            flow_mpc.DEFAULT_REDUCED_MODEL_CALIBRATION_PATH,
            MODEL_DATA_ROOT / "mpc_reduced_model_best_theta.json",
        )

    def test_required_model_data_files_exist(self):
        required = (
            MODEL_DATA_ROOT / "hppc_params.json",
            MODEL_DATA_ROOT / "mpc_evaporator_capacity_candidate_b.json",
            MODEL_DATA_ROOT / "candidate_b_capacity_limits_grid.csv",
            MODEL_DATA_ROOT / "mpc_reduced_model_best_theta.json",
        )
        self.assertEqual([path for path in required if not path.is_file()], [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 验证测试在迁移前失败**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -v test_model_data_paths.ModelDataPathTest.test_default_paths_use_tracked_model_data_directory
```

Expected: FAIL because `pack.HPPC_PARAMS_PATH` is not defined or existing defaults still point to `data/`/`outputs/`。

- [ ] **Step 3: 在 `pack.py` 定义并使用模型数据路径**

Add near the imports:

```python
PROJECT_ROOT = Path(__file__).resolve().parent
HPPC_PARAMS_PATH = PROJECT_ROOT / "model_data" / "hppc_params.json"
```

Replace the inline path in `BatteryPack.load_hppc_params` with:

```python
path = HPPC_PARAMS_PATH
with path.open("r", encoding="utf-8") as handle:
    data = json.load(handle)
```

- [ ] **Step 4: 修改candidate B默认标定路径**

Set in `mpc_evaporator_capacity_model.py`:

```python
DEFAULT_CALIBRATION_PATH = (
    Path(__file__).resolve().parent
    / "model_data"
    / "mpc_evaporator_capacity_candidate_b.json"
)
```

- [ ] **Step 5: 修改MPC容量表和降阶标定路径**

Set in `mpc_flow_direction_strategies.py`:

```python
MODEL_DATA_ROOT = Path(__file__).resolve().parent / "model_data"
DEFAULT_REDUCED_MODEL_CALIBRATION_PATH = (
    MODEL_DATA_ROOT / "mpc_reduced_model_best_theta.json"
)
DEFAULT_EVAPORATOR_CAPACITY_TABLE_PATH = (
    MODEL_DATA_ROOT / "candidate_b_capacity_limits_grid.csv"
)
```

### Task 3: 复制并验证4个受管模型数据文件

**Files:**
- Create: `model_data/hppc_params.json`
- Create: `model_data/mpc_evaporator_capacity_candidate_b.json`
- Create: `model_data/candidate_b_capacity_limits_grid.csv`
- Create: `model_data/mpc_reduced_model_best_theta.json`
- Modify: `.gitignore`

- [ ] **Step 1: 创建目标目录并复制原文件**

Run in one PowerShell process:

```powershell
$root = (Resolve-Path -LiteralPath '.').Path
$target = Join-Path $root 'model_data'
New-Item -ItemType Directory -Path $target -Force | Out-Null
Copy-Item -LiteralPath 'data\hppc_params.json' -Destination (Join-Path $target 'hppc_params.json')
Copy-Item -LiteralPath 'outputs\mpc_evaporator_capacity_candidate_b\mpc_evaporator_capacity_candidate_b.json' -Destination (Join-Path $target 'mpc_evaporator_capacity_candidate_b.json')
Copy-Item -LiteralPath 'outputs\mpc_evaporator_capacity_candidate_b\candidate_b_capacity_limits_grid.csv' -Destination (Join-Path $target 'candidate_b_capacity_limits_grid.csv')
Copy-Item -LiteralPath 'outputs\mpc_reduced_model_calibration\best_theta.json' -Destination (Join-Path $target 'mpc_reduced_model_best_theta.json')
```

Expected: four files exist under `model_data/`；原文件仍存在。

- [ ] **Step 2: 校验源文件与目标文件SHA256**

Expected pairs:

```text
hppc_params.json = B646258BAA3B7CBF0F7142E5E60226E70ED772C689AB655EECF92448888EF186
mpc_evaporator_capacity_candidate_b.json = FDFBA189A9D51F996A4EB78BFFDBC3B7287A3661934DC0E1A9FA5B92C833C9C1
candidate_b_capacity_limits_grid.csv = D20BE679B84573E70A23CF2739C7EEC03628240DFD7A60477D2C52829B85AC59
mpc_reduced_model_best_theta.json = 090232937C21627EA9E50B2464C7817866750867B6962FC96626EB1644A8236F
```

Run:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'model_data\hppc_params.json','model_data\mpc_evaporator_capacity_candidate_b.json','model_data\candidate_b_capacity_limits_grid.csv','model_data\mpc_reduced_model_best_theta.json'
```

- [ ] **Step 3: 允许模型数据CSV/JSON进入Git**

Append after the broad binary/data ignore section in `.gitignore`:

```gitignore
# Small runtime model parameters are intentionally versioned.
!model_data/
!model_data/*.json
!model_data/*.csv
```

- [ ] **Step 4: 运行新路径测试和candidate B测试**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q test_model_data_paths test_mpc_evaporator_capacity_model test_mpc_evaporator_capacity_table test_mpc_reduced_model_calibration
```

Expected: all tests pass。

- [ ] **Step 5: 提交模型数据迁移**

```powershell
git add -- model_data pack.py mpc_evaporator_capacity_model.py mpc_flow_direction_strategies.py test_model_data_paths.py .gitignore
git commit -m "feat: 纳管BTMS运行模型数据"
```

### Task 4: 添加可复现Conda环境

**Files:**
- Create: `environment.yml`

- [ ] **Step 1: 创建环境文件**

Create `environment.yml`:

```yaml
name: btms
channels:
  - conda-forge
dependencies:
  - python=3.12
  - numpy=2.2.3
  - pandas=3.0.3
  - scipy=1.17.1
  - matplotlib=3.10.9
  - pybamm=26.5.0
  - coolprop=7.1.0
  - gekko=1.3.2
  - platformdirs=4.10.0
  - pip=26.1.1
```

- [ ] **Step 2: 验证Conda能够解析环境文件**

Run:

```powershell
& 'C:\Users\24776\miniforge3\Scripts\conda.exe' env create --dry-run -f environment.yml
```

Expected: exit code `0`，solver完成依赖解析但不创建新环境。

- [ ] **Step 3: 提交环境文件**

```powershell
git add -- environment.yml
git commit -m "build: 添加BTMS Conda环境定义"
```

### Task 5: 添加中文README

**Files:**
- Create: `README.md`

- [ ] **Step 1: 创建中文项目说明**

`README.md`必须包含以下可直接执行内容：

````markdown
# 储能电池热管理仿真与控制

本项目用于储能电池热管理系统仿真与控制研究，包含电池热模型、冷板与冷却液动态、制冷循环、On-Off/PID/MPC控制、candidate B蒸发器预测模型、终端代价和预测式流向反转。

## 环境安装

```powershell
conda env create -f environment.yml
conda activate btms
```

## 核心结构

- `thermal_case_simulator.py`：完整工况仿真入口。
- `thermal_batch_config.py`：调峰、调频和控制参数。
- `thermal_control_strategies.py`：控制器创建入口。
- `mpc_flow_direction_strategies.py`：MPC与流向反转主体。
- `thermal_loop.py`：冷却回路、延迟和动态。
- `thermal_system.py`：泵、换热器和制冷循环。
- `model_data/`：运行所需的小型模型标定数据。

## 最终控制器比较

```powershell
python run_final_controller_comparison.py
```

服务器并行入口：

```powershell
python run_final_controller_parallel.py
```

## MPC参数扫描

```powershell
python run_mpc_sensitivity_60.py --help
```

## candidate B诊断

```powershell
python run_evaporator_mpc_capacity_diagnostic.py --help
```

## 最终绘图

```powershell
python plot_final_mpc_report.py
```

## 验证

```powershell
python -m unittest -q test_btms_runtime test_mpc_evaporator_capacity_model test_final_controller_comparison test_final_controller_configured_pid test_plot_final_mpc_report
```

## 注意事项

- Windows入口通过 `btms_runtime.py` 补充Conda环境的 `Library/bin`，以便加载本地DLL。
- `outputs/`、`输出结果/` 和 `data/` 是本地数据/结果目录，不上传Git。
- 长时间GEKKO扫描和完整对比建议在服务器运行。
- 正式论文/PPT结果仍以已确认的原12组完整仿真为准。
````

- [ ] **Step 2: 核对README中的命令对应现有文件**

Run:

```powershell
@('environment.yml','run_final_controller_comparison.py','run_final_controller_parallel.py','run_mpc_sensitivity_60.py','run_evaporator_mpc_capacity_diagnostic.py','plot_final_mpc_report.py') | ForEach-Object { if (-not (Test-Path -LiteralPath $_)) { throw "README引用缺失: $_" } }
```

Expected: no output and exit code `0`。

- [ ] **Step 3: 提交README**

```powershell
git add -- README.md
git commit -m "docs: 添加中文项目说明"
```

### Task 6: 完整验证并推送可运行仓库

**Files:**
- Test: all retained root Python files and selected test modules

- [ ] **Step 1: 编译全部根目录Python文件**

```powershell
$files = Get-ChildItem -File -Filter '*.py' | Select-Object -ExpandProperty FullName
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m py_compile @files
```

Expected: exit code `0`。

- [ ] **Step 2: 运行模型数据与核心回归测试**

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q test_model_data_paths test_btms_runtime test_mpc_evaporator_capacity_model test_mpc_evaporator_capacity_table test_mpc_reduced_model_calibration test_final_comparison_flow_modes test_final_controller_comparison test_final_controller_configured_pid test_final_controller_parallel test_final_controller_parallel_pure test_final_controller_parallel_system_tmp test_plot_final_mpc_report
```

Expected: all collected tests pass，不启动完整仿真或PSO。

- [ ] **Step 3: 检查暂存/提交内容没有禁止目录**

```powershell
$forbidden = git ls-files | Where-Object { $_ -match '^(outputs/|output/|输出结果/|data/|_archive/|_gekko_tmp_|tmp|simulink-agentic-toolkit|\.codex/|\.agents/)' }
if ($forbidden) { $forbidden; throw '发现不应上传的路径' }
```

Expected: no output。

- [ ] **Step 4: 推送当前提交到远程main**

```powershell
git push origin HEAD:main
```

Expected: push succeeds。

- [ ] **Step 5: 验证本地与远程哈希一致**

```powershell
$local = git rev-parse HEAD
$remote = git ls-remote origin refs/heads/main | ForEach-Object { ($_ -split '\s+')[0] }
if ($local -ne $remote) { throw "远程未同步: local=$local remote=$remote" }
```

Expected: no exception。

### Task 7: 精确停止两个旧测试进程

**Files:**
- No repository files modified

- [ ] **Step 1: 重新读取目标进程身份**

```powershell
$expected = @{
  48092 = 'test_pid_local_refinement'
  36696 = 'test_pid_local_refinement'
}
$targets = @()
foreach ($pidValue in $expected.Keys) {
  $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue"
  if ($null -eq $proc) { continue }
  if ($proc.Name -ne 'python.exe' -or $proc.CommandLine -notmatch $expected[$pidValue] -or $proc.CreationDate -lt [datetime]'2026-07-11T20:19:00' -or $proc.CreationDate -gt [datetime]'2026-07-11T20:21:00') {
    throw "进程身份不匹配，拒绝停止 PID $pidValue"
  }
  $targets += $proc
}
$targets | Select-Object ProcessId,CreationDate,Name,CommandLine
```

Expected: zero to two exact matching old test processes；任何身份不匹配都会停止本任务。

- [ ] **Step 2: 只停止精确匹配进程**

```powershell
$targets | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

Expected: target processes no longer exist；other Python processes remain untouched。

### Task 8: 永久删除本地归档和临时目录

**Files:**
- Delete: `_archive/`
- Delete: root `tmp*/`
- Delete: root `_gekko_tmp_*/`
- Delete: root `__pycache__/`

- [ ] **Step 1: 记录删除前统计并验证绝对路径**

```powershell
$root = (Resolve-Path -LiteralPath '.').Path
$candidates = @()
if (Test-Path -LiteralPath '_archive') { $candidates += Get-Item -Force -LiteralPath '_archive' }
$candidates += Get-ChildItem -Directory -Force | Where-Object { $_.Name -eq '__pycache__' -or $_.Name -like 'tmp*' -or $_.Name -like '_gekko_tmp_*' }
foreach ($item in $candidates) {
  if ($item.Parent.FullName -ne $root) { throw "目标不是项目根目录直接子目录: $($item.FullName)" }
  if (-not $item.FullName.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) { throw "目标越出项目目录: $($item.FullName)" }
}
$before = $candidates | ForEach-Object {
  $stats = Get-ChildItem -LiteralPath $_.FullName -File -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum
  [pscustomobject]@{Path=$_.FullName;Files=$stats.Count;Bytes=$stats.Sum}
}
$before
```

Expected: all paths are direct children of the intended workspace。

- [ ] **Step 2: 使用同一PowerShell进程逐个删除已验证目标**

```powershell
foreach ($item in $candidates) {
  Remove-Item -LiteralPath $item.FullName -Recurse -Force
}
```

Expected: all validated targets no longer exist；`outputs/`、`输出结果/`、`data/`、`model_data/` remain。

- [ ] **Step 3: 验证保留目录和模型数据**

```powershell
@('outputs','输出结果','data','model_data') | ForEach-Object { if (-not (Test-Path -LiteralPath $_)) { throw "误删保留目录: $_" } }
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest -q test_model_data_paths test_mpc_evaporator_capacity_model
```

Expected: four directories exist and tests pass。

### Task 9: 更新永久清理记录并完成Git分支整理

**Files:**
- Modify: `PROJECT_MAP_MIN.md`
- Modify: `PROJECT_CLEANUP_REPORT.md`

- [ ] **Step 1: 更新项目地图**

Replace archive-history wording with：

```markdown
历史清理批次已于2026-07-14在完成GitHub可运行基线后永久删除；当前运行代码、模型数据和测试均由Git管理。
```

Add `model_data/` to the default read list and state that `_archive/` no longer exists。

- [ ] **Step 2: 在清理报告追加永久删除结果**

Append the exact `$before` file/byte totals, stopped process IDs, deletion time, post-delete test result, and remote commit hash。明确说明 `outputs/`、`输出结果/`、`data/` 与 `model_data/` 未删除。

- [ ] **Step 3: 提交并推送清理记录**

```powershell
git add -- PROJECT_MAP_MIN.md PROJECT_CLEANUP_REPORT.md
git commit -m "chore: 完成本地历史归档清理"
git push origin HEAD:main
```

- [ ] **Step 4: 将本地分支改名为main**

```powershell
git branch -m main
git branch --set-upstream-to=origin/main main
```

Expected: current local branch is `main` and tracks `origin/main`。

- [ ] **Step 5: 最终状态验证**

```powershell
git status --short --branch
$local = git rev-parse HEAD
$remote = git ls-remote origin refs/heads/main | ForEach-Object { ($_ -split '\s+')[0] }
if ($local -ne $remote) { throw '最终远程哈希不一致' }
```

Expected: clean `main...origin/main` and matching hashes。
