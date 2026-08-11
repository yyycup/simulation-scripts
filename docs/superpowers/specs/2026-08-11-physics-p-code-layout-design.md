# Physics-P 代码布局整合设计

日期：2026-08-11
状态：对话设计已批准，等待书面规格复核
基线：`main` 提交 `1794265622bf73823ef4ba5ef78b6bf247207b3e`

## 1. 背景与决策

Physics-P 已完成闭环验证并合入 `main`。当前功能边界是清楚的：Candidate B 仍是仓库默认预测器，Physics-P 只通过 `run_p_mpc_operational.py` 显式启用；两者的压缩机功率模型和初始温度边界已经隔离。

合并后，Physics-P 的生产代码、辨识脚本、校准脚本、评估脚本、调参脚本和测试仍集中在项目根目录。当前根目录有 113 个 Python 文件，其中 56 个是根目录测试；24 个根目录 Python 文件属于 Physics-P/双预测器工作。正式入口 `run_p_mpc_operational.py` 还从实验入口 `run_p_mpc_short_comparison.py` 导入场景映射、输入定位和汇总函数；正式运行参数文件则位于整体被忽略的 `outputs/` 目录下。

本设计采用已批准的方案 B：物理整理 Physics-P 文件，同时正式退役被移动脚本的旧根目录命令。所有活跃代码、测试和使用文档切换到新的模块命令，不保留根目录兼容包装器。测试只移动，不删除。

## 2. 目标

1. 在根目录只保留 Physics-P 生产运行链和稳定公共模块，使正式入口一眼可识别。
2. 将辨识、校准、评估和调参脚本按职责移动到 `experiments/physics_p/`。
3. 解除正式入口对实验脚本的反向依赖。
4. 将正式 Physics-P 参数资产移动到受版本控制语义明确的 `model_data/`。
5. 将 Physics-P 专属测试集中到 `tests/physics_p/`，保留现有覆盖和测试行为。
6. 统一被移动脚本的调用方式为 `python -m <module>`，修复活跃代码、测试和文档中的旧命令与旧导入。
7. 保持所有控制、热模型、制冷模型、参数、默认选择器和已验证输出语义不变。

## 3. 非目标

本次整合明确不做以下工作：

- 不拆分 3407 行的 `mpc_flow_direction_strategies.py`，不重构 `MPCControllerDual`、`solve_step` 或求解恢复流程。
- 不修改 `thermal_loop.py`、`thermal_system.py`、制冷循环、低速对象边界或风扇逻辑。
- 不改变 Candidate B 默认选择器，也不让 Physics-P 自动替代 Candidate B。
- 不改变 Candidate B 的 legacy 压缩机功率路径、普通仿真的 35 °C 初始状态或 Physics-P 的显式 25 °C 初始状态。
- 不改变 Physics-P 的权重、14/12 步运行 horizon、DMAX、滤波、温度偏差增益、位移缩放契约或求解恢复参数。
- 不删除、合并或弱化测试断言；不因为模型已经调好而移除回归保护。
- 不重跑或覆盖完整调峰/调频结果，不移动现有 worktree 结果目录。
- 不在本次维护提交中同时解决实时 5 秒 deadline、AGC 数据交付或结果发布归档问题。
- 不清理主工作目录中与本任务无关的 TD3 未跟踪文件。

## 4. 必须保持的行为不变量

实施前后必须同时满足：

1. `thermal_case_simulator.py`、`thermal_control_strategies.py` 和 MPC 工厂的默认预测器仍为 `candidate_b`。
2. `run_p_mpc_operational.py` 仍显式传入 `PHYSICS_P`、25 °C 初始状态和正式参数资产。
3. 调峰 Physics-P 的 horizon 保持 14 步，调频保持 12 步；现有场景参数覆盖值逐项相同。
4. Candidate B 与 mixed-integer 路径继续使用 legacy 压缩机功率；只有 Physics-P 使用温度相关功率表达式。
5. 位移缩放仍从参数资产读取，非法或非正有限值仍被拒绝；默认正式资产的缩放值保持 1.0。
6. 所有移动脚本的 CLI 参数名、默认输出目录、默认输入含义和生成文件命名保持不变，唯一有意变化是调用模块路径。
7. 完整结果目录和历史 `outputs/` 数据保持原位，不被移动、删除或重新生成。

## 5. 目标目录

```text
项目根目录/
├─ run_p_mpc_operational.py
├─ p_mpc_run_support.py
├─ mpc_physics_predictor.py
├─ mpc_physics_shadow.py
├─ mpc_predictor_selection.py
├─ model_data/
│  └─ physics_p_operational_v1.json
├─ experiments/
│  ├─ __init__.py
│  └─ physics_p/
│     ├─ __init__.py
│     ├─ README.md
│     ├─ identification/
│     │  ├─ __init__.py
│     │  └─ ...
│     ├─ calibration/
│     │  ├─ __init__.py
│     │  └─ ...
│     ├─ evaluation/
│     │  ├─ __init__.py
│     │  └─ ...
│     └─ tuning/
│        ├─ __init__.py
│        └─ ...
└─ tests/
   ├─ __init__.py
   └─ physics_p/
      ├─ __init__.py
      └─ test_*.py
```

包目录使用显式 `__init__.py`，不依赖隐式 namespace package。这样模块执行、`unittest` discovery、IDE 导航和字符串 patch 目标具有一致的导入名称。

## 6. 根目录生产文件

以下文件保留在根目录：

| 文件 | 角色 | 处理 |
| --- | --- | --- |
| `run_p_mpc_operational.py` | 唯一正式 Physics-P 闭环入口 | 保持 PyCharm/直接脚本可运行；只改公共 helper 和资产路径导入 |
| `p_mpc_run_support.py` | 新增的无实验依赖运行支持模块 | 承载项目根路径、场景映射、输入 CSV 定位和基础运行汇总 |
| `mpc_physics_predictor.py` | 正式 Physics-P 预测器与资产校验 | 保留根目录，不做结构重构 |
| `mpc_physics_shadow.py` | 正式只读 shadow 预测支持 | 保留根目录，不改变算法 |
| `mpc_predictor_selection.py` | Candidate B/Physics-P 选择契约 | 保留根目录，不改变默认值或合法值 |

`p_mpc_run_support.py` 允许依赖稳定根模块和 `thermal_batch_config.py`，但禁止导入 `experiments.*`。它只提取当前 `run_p_mpc_short_comparison.py` 中已被正式入口复用的通用内容：

- `PROJECT_ROOT`；
- `SCENES`；
- `source_csv_for_scene(...)`；
- 列名兼容 helper；
- `summarize_run(...)`。

实验专属的 Candidate B/Physics-P 成对比较、实验默认 artifact、pairwise summary 和短对比执行逻辑继续留在移动后的实验模块中。正式入口不得再导入任何实验模块。

## 7. 实验脚本迁移清单

### 7.1 辨识 `experiments/physics_p/identification/`

| 原根文件 | 新模块 |
| --- | --- |
| `predictor_identification_data.py` | `experiments.physics_p.identification.predictor_identification_data` |
| `generate_predictor_identification_data.py` | `experiments.physics_p.identification.generate_predictor_identification_data` |
| `fit_mpc_physics_predictor.py` | `experiments.physics_p.identification.fit_mpc_physics_predictor` |
| `fit_mpc_lpv_predictor.py` | `experiments.physics_p.identification.fit_mpc_lpv_predictor` |
| `mpc_lpv_predictor.py` | `experiments.physics_p.identification.mpc_lpv_predictor` |
| `evaluate_mpc_predictors.py` | `experiments.physics_p.identification.evaluate_mpc_predictors` |
| `evaluate_p_identification_rollouts.py` | `experiments.physics_p.identification.evaluate_p_identification_rollouts` |

`predictor_identification_data.py`、`mpc_lpv_predictor.py` 和 `evaluate_mpc_predictors.py` 是实验库模块，不宣传为独立正式命令；其余文件保留现有 CLI。

### 7.2 校准 `experiments/physics_p/calibration/`

| 原根文件 | 新模块 |
| --- | --- |
| `build_p_compressor_displacement_variant.py` | `experiments.physics_p.calibration.build_p_compressor_displacement_variant` |
| `build_p_heat_generation_corrected_artifact.py` | `experiments.physics_p.calibration.build_p_heat_generation_corrected_artifact` |
| `calibrate_p_actual_replay.py` | `experiments.physics_p.calibration.calibrate_p_actual_replay` |
| `refine_p_thermal_capacities.py` | `experiments.physics_p.calibration.refine_p_thermal_capacities` |

### 7.3 评估 `experiments/physics_p/evaluation/`

| 原根文件 | 新模块 |
| --- | --- |
| `evaluate_dual_p_shadow.py` | `experiments.physics_p.evaluation.evaluate_dual_p_shadow` |
| `evaluate_p_frozen_mpc_plan.py` | `experiments.physics_p.evaluation.evaluate_p_frozen_mpc_plan` |
| `evaluate_p_shadow_actual_replay.py` | `experiments.physics_p.evaluation.evaluate_p_shadow_actual_replay` |
| `run_p_model_boundary_validation.py` | `experiments.physics_p.evaluation.run_p_model_boundary_validation` |
| `compare_mpc_predictor_formal_results.py` | `experiments.physics_p.evaluation.compare_mpc_predictor_formal_results` |
| `plot_p_peak_displacement_comparison.py` | `experiments.physics_p.evaluation.plot_p_peak_displacement_comparison` |
| `figures/gen_fig_p_mpc_local_formal.py`（当前被忽略、仅存在于已保留的 Physics-P worktree） | `experiments.physics_p.evaluation.plot_p_mpc_local_formal_results` |

`test_plot_p_mpc_local_formal_results.py` 已受 Git 跟踪，但当前 `main` 的 clean checkout 缺少被 `.gitignore` 排除的 `figures/gen_fig_p_mpc_local_formal.py`，因此该测试会在导入阶段失败。实施时必须把已保留 worktree 中的现有 helper 原样纳入新 evaluation 包，再只调整 `PROJECT_ROOT` 来源；不得删除测试、添加 skip 或继续依赖另一个 worktree。

### 7.4 调参与正式实验 `experiments/physics_p/tuning/`

| 原根文件 | 新模块 |
| --- | --- |
| `run_p_mpc_controller_tuning.py` | `experiments.physics_p.tuning.run_p_mpc_controller_tuning` |
| `run_p_mpc_local_formal.py` | `experiments.physics_p.tuning.run_p_mpc_local_formal` |
| `run_p_mpc_short_comparison.py` | `experiments.physics_p.tuning.run_p_mpc_short_comparison` |

这里的“formal”表示历史实验命名，不获得生产入口地位。唯一正式接入入口仍是根目录 `run_p_mpc_operational.py`。

当前 `test_p_mpc_local_formal.py` 有三项 Physics-P 路径测试依赖被
`.gitignore` 排除的旧实验 artifact；clean checkout 在进入 mock 的比较逻辑前就会
因文件不存在而失败。迁移测试时，这三项必须显式传入已跟踪的
`run_p_mpc_operational.DEFAULT_OPERATIONAL_P_ARTIFACT` 作为 hermetic fixture。
这只修复测试环境依赖，不改变 tuning runner 的历史默认 artifact，也不得把旧
`outputs/` artifact 复制或提交进仓库。

同样，`test_p_model_boundary_validation.py` 必须向
`build_boundary_table(...)` 显式传入该已跟踪 operational artifact。boundary runner
继续保留历史 experimental 默认值；测试不得依赖 clean checkout 中不存在的
ignored `outputs/` 文件。

## 8. 导入与路径规则

### 8.1 导入方向

允许的依赖方向为：

```text
run_p_mpc_operational.py
  -> p_mpc_run_support.py
  -> 稳定根模块

experiments.physics_p.*
  -> p_mpc_run_support.py
  -> 稳定根模块
  -> 同一 experiments.physics_p 包内模块

tests.physics_p.*
  -> 稳定根模块或 experiments.physics_p.*
```

禁止以下依赖：

- 生产根模块导入 `experiments.*`；
- 通过 `sys.path` 注入项目根目录；
- 依赖当前工作目录拼接项目根路径；
- 新增循环导入；
- 为维持旧根导入名而创建包装 Python 文件。

### 8.2 具体导入约定

- 实验包内部使用相对导入，例如 `from .predictor_identification_data import ...` 或跨子包的 `from ..evaluation... import ...`。
- 实验模块引用稳定生产模块时使用根模块绝对导入，例如 `from mpc_physics_predictor import ...`。
- 测试引用移动后的实验模块时使用完整绝对包名，例如 `from experiments.physics_p.calibration... import ...`。
- `unittest.mock.patch(...)` 的字符串目标同步改为新模块全名，不能只修改普通 import。
- 所有移动脚本从 `p_mpc_run_support.PROJECT_ROOT` 获得仓库根目录，避免 `Path(__file__).parent` 在移动后改变默认输入/输出位置。

### 8.3 当前耦合的处理

必须至少修复以下现有耦合：

- `run_p_mpc_operational.py` 不再从 `run_p_mpc_short_comparison.py` 导入 `SCENES`、`source_csv_for_scene`、`summarize_run`。
- `build_p_heat_generation_corrected_artifact.py` 和 `calibrate_p_actual_replay.py` 更新为导入移动后的 actual-replay 评估模块。
- `refine_p_thermal_capacities.py` 更新其 actual-replay、fit metric 和实验默认 artifact 导入。
- 辨识生成、拟合和评估模块更新 `predictor_identification_data` 与 LPV 实验模块导入。
- boundary validation、controller tuning 和 local formal 更新 short-comparison helper 的包路径。

实验之间现存的默认 artifact 复用在本次只做包路径更新，不趁目录移动改变实验默认值。消除全部实验内部耦合属于后续独立维护范围。

## 9. 命令迁移与旧命令退役

### 9.1 正式入口保持不变

从项目根目录运行：

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' run_p_mpc_operational.py --help
```

该入口继续支持 PyCharm 直接运行。

### 9.2 移动脚本的新命令

所有移动脚本只支持从项目根目录按模块执行：

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m experiments.physics_p.identification.generate_predictor_identification_data --help
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m experiments.physics_p.identification.fit_mpc_physics_predictor --help
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m experiments.physics_p.calibration.build_p_heat_generation_corrected_artifact --help
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m experiments.physics_p.evaluation.run_p_model_boundary_validation --help
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m experiments.physics_p.evaluation.plot_p_mpc_local_formal_results --help
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m experiments.physics_p.tuning.run_p_mpc_short_comparison --help
```

同一规则适用于第 7 节所有带 CLI 的模块。旧形式 `python <原根文件>.py ...` 正式失效；不创建兼容 wrapper、复制文件或符号链接。

### 9.3 活跃引用修复

实施时更新：

- 所有受 Git 管理的活跃 Python import、patch 字符串和子进程命令；
- `README.md`；
- `PROJECT_MAP_MIN.md`；
- `CODEX_README_MIN.md`；
- 新增 `experiments/physics_p/README.md`，包含完整旧路径到新模块的迁移表。

历史 `docs/superpowers/plans/`、旧设计规格和结果记录保持原文，以保留当时可复现命令和资产路径的事实。新实验 README 和本设计文档明确说明迁移日期与新路径。历史文档中的旧命令不视为活跃悬空引用。

## 10. 正式参数资产迁移

当前正式资产：

```text
outputs/mpc_predictor_low_speed_retrain_v1/thermal_bias_correction_v3/physics_p_heat_generation_corrected.json
```

迁移目标：

```text
model_data/physics_p_operational_v1.json
```

迁移要求：

1. 使用 Git 受控移动，目标文件内容和 SHA-256 必须与迁移前一致；该历史资产含 Python JSON loader 支持的 `NaN` token，不得使用严格 JSON 工具重序列化或规范化。
2. 更新 `run_p_mpc_operational.DEFAULT_OPERATIONAL_P_ARTIFACT` 指向新路径。
3. 更新 `test_p_mpc_operational.py` 的路径契约。
4. 更新 `test_model_data_paths.py`，把新文件列入 clean-checkout 必备模型数据。
5. 旧 `outputs/.../physics_p_heat_generation_corrected.json` 不在当前树保留重复副本；历史可由 Git 追溯。
6. 参数 JSON 内部的来源字段保持可移植，不重新写入本机绝对路径。
7. 其他实验 artifact 仍留在原 `outputs/` 位置，本次不迁移、不重命名。

## 11. 测试迁移

以下 21 个测试文件移动到 `tests/physics_p/`，内容和断言保留：

1. `test_build_p_compressor_displacement_variant.py`
2. `test_build_p_heat_generation_corrected_artifact.py`
3. `test_calibrate_p_actual_replay.py`
4. `test_compare_mpc_predictor_formal_results.py`
5. `test_evaluate_dual_p_shadow.py`
6. `test_evaluate_mpc_predictors.py`
7. `test_evaluate_p_frozen_mpc_plan.py`
8. `test_evaluate_p_shadow_actual_replay.py`
9. `test_generate_predictor_identification_data.py`
10. `test_mpc_lpv_predictor.py`
11. `test_mpc_physics_p_closed_loop.py`
12. `test_mpc_physics_predictor.py`
13. `test_mpc_physics_shadow.py`
14. `test_mpc_predictor_selection.py`
15. `test_p_model_boundary_validation.py`
16. `test_p_mpc_controller_tuning.py`
17. `test_p_mpc_local_formal.py`
18. `test_p_mpc_operational.py`
19. `test_plot_p_mpc_local_formal_results.py`
20. `test_predictor_identification_data.py`
21. `test_refine_p_thermal_capacities.py`

共享边界测试继续留在根目录，包括但不限于：

- `test_mpc_compressor_power_model.py`；
- `test_mpc_solve_recovery.py`；
- `test_thermal_initial_state.py`；
- Candidate B、蒸发器容量和制冷边界相关测试。

标准 Physics-P 测试命令为：

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest discover -s tests/physics_p -t . -p 'test_*.py'
```

从根目录按旧测试模块名运行（例如 `python -m unittest test_p_mpc_operational`）不再是支持接口。活跃文档和 CI/本地验证命令必须更新为 discovery 或 `tests.physics_p.<module>` 全名。

## 12. 文档职责

### `README.md`

- 增加 Physics-P 简介；
- 明确 Candidate B 仍是默认，P 为显式入口；
- 给出正式入口和一个实验模块命令；
- 链接实验目录说明和正式参数资产。

### `PROJECT_MAP_MIN.md`

- 把 `run_p_mpc_operational.py` 列为唯一正式 P 入口；
- 将 `experiments/physics_p/` 定义为辨识/校准/评估/调参代码；
- 将 `tests/physics_p/` 定义为 P 专属回归；
- 继续禁止默认扫描 `outputs/`。

### `CODEX_README_MIN.md`

- 更新 Physics-P 快速路由；
- 移除已不存在 `_archive/` 的陈旧说明；
- 记录移动脚本必须用 `python -m`。

### `experiments/physics_p/README.md`

- 列出四类实验目录；
- 提供第 7 节的完整原文件到新模块映射；
- 说明旧根命令自 2026-08-11 起退役且没有 wrapper；
- 区分正式入口、实验入口和历史结果。

## 13. 实施顺序原则

后续实施计划应把变更拆成可验证的小步，但最终以一个专门维护分支交付：

1. 先建立包目录、`p_mpc_run_support.py` 和支持模块测试，解除正式入口的实验依赖。
2. 移动正式参数资产并验证内容哈希和 clean-checkout 路径。
3. 按 identification、calibration、evaluation、tuning 顺序移动脚本，每一组立即修复 import、patch target、CLI 和编译。
4. 移动 21 个测试并更新统一 discovery 命令。
5. 更新活跃文档和完整命令迁移表。
6. 最后执行静态悬空引用检查、完整定向回归和短闭环 smoke。

任何一步出现行为差异时，先停止并回滚该步；不得通过放宽断言、修改控制参数或删除测试来让整合通过。

## 14. 验证门槛

### 14.1 设计前基线

在提交本设计文档前，基线提交 `1794265` 已运行：

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m unittest `
  test_mpc_predictor_selection `
  test_mpc_compressor_power_model `
  test_mpc_physics_p_closed_loop `
  test_p_mpc_operational `
  test_thermal_initial_state
```

结果：74 项测试通过，耗时 4.946 秒。

### 14.2 实施完成后的必做验证

1. **导入/编译**：编译所有保留生产模块、20 个根目录迁移脚本、1 个补齐的正式绘图 helper 和 21 个移动测试；逐个导入 21 个实验模块。
2. **CLI smoke**：所有带 CLI 的移动模块执行 `python -m ... --help` 成功，不发生相对导入或项目根路径错误。
3. **Physics-P 专属测试**：第 11 节 discovery 命令全部通过，测试数量不得因移动减少。
4. **相邻回归**：重跑 Physics-P 合并时的 277 项定向回归；如果测试路径变化导致命令重写，应记录新旧数量对照。
5. **正式入口 smoke**：调峰与调频各执行至少 6 步，显式确认 predictor 为 `physics_p`、初始状态为 25 °C、horizon 为 14/12，输出写入新的独立临时目录。
6. **行为对比**：在同一提交代码、同一输入和同一参数资产下，对移动前后固定 6 步输出比较关键数值列；允许的差异只有路径/模块元数据，控制指令、温度、功率、能耗、求解状态必须逐值相同或在现有数值序列化精度内相同。
7. **资产验证**：迁移前后 SHA-256 相同，新路径受 Git 跟踪，JSON 可解析且不含 Windows 绝对来源路径。
8. **静态引用检查**：在活跃源码、测试、README 和项目地图中搜索所有旧根文件名；除新迁移表和明确标记的历史文档外不得残留旧命令、旧 import 或旧 patch target。
9. **Git 检查**：`git diff --check` 无新增空白错误；工作树不包含完整 outputs、TD3 文件或无关修改。

完整调峰 1280 步和调频 720 步结果已在先前验证中完成，本次纯目录维护不要求重新跑完整仿真。若 6 步行为对比或 277 项回归出现差异，则必须停止，不能以已有完整结果替代本次回归。

## 15. 回滚

所有移动使用 Git 可追踪 rename；不删除人工代码和测试。若验证失败：

1. 回滚最近一组文件移动和 import 更新；
2. 恢复原资产路径和 `DEFAULT_OPERATIONAL_P_ARTIFACT`；
3. 重跑第 14.1 节基线集合确认恢复；
4. 保留失败证据，另开问题分析，不在本维护变更中调整模型或控制参数。

旧根命令不会作为回滚兼容层保留。真正回滚意味着把文件恢复到原根路径，而不是新增双份入口。

## 16. 验收标准

只有同时满足以下条件，目录整合才算完成：

- 根目录只保留第 6 节列出的 Physics-P 生产文件和唯一正式入口；
- 21 个实验模块全部位于第 7 节目标目录；18 个现有 CLI 按 `python -m` 调用，3 个库模块只用于导入；
- 生产代码不存在对 `experiments.*` 的导入；
- 正式参数资产位于 `model_data/physics_p_operational_v1.json` 且内容未变；
- 21 个 P 专属测试全部保留在 `tests/physics_p/`；
- 活跃代码、测试和文档不存在未迁移的旧根命令；
- 第 14 节全部验证门槛通过；
- Candidate B/Physics-P 隔离、控制参数和短闭环数值输出不变；
- 完整历史结果、无关 TD3 文件和核心热/制冷/MPC 主体没有被触碰。
