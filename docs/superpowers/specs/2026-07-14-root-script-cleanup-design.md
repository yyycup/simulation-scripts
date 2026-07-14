# 根目录脚本与代码增量清理设计

## 目标

在不改变热管理模型、candidate B、MPC/PID 控制逻辑和最终论文数据的前提下，减少项目根目录中的阶段性脚本、重复绘图脚本、配套临时测试和可再生缓存，使当前主流程与最终复现入口更容易识别。

## 范围

本轮只处理项目根目录中的 `run_*.py`、`plot_*.py`、`test_*.py`、兼容入口、探针文件、Python 缓存以及确认无进程占用的 GEKKO/普通临时目录。

以下内容不在本轮范围内：

- `outputs/`、`输出结果/` 和 `data/`；
- 已有 `_archive/` 内容；
- `.agents/`、`.codex/`、`.idea/` 和 `.git_empty_backup/`；
- `simulink-agentic-toolkit/` 与 `simulink-agentic-toolkit-main/`；
- 核心仿真、热模型、制冷模型、控制逻辑和正式参数值。

## 清理策略

采用保守归档方式，不直接删除人工编写的脚本或代码。

1. 盘点根目录候选文件，并检查导入、调用、文档和命令引用。
2. 将候选文件划分为 `KEEP`、`ARCHIVE` 和 `REVIEW`：
   - `KEEP`：核心链路、最终结果复现、当前配置、candidate B、正式绘图及其有效测试；
   - `ARCHIVE`：实验已经结束、功能已被新入口替代、且不存在当前引用的阶段性脚本；
   - `REVIEW`：用途或复现关系不能从代码和文档确定的文件，本轮不移动。
3. 将 `ARCHIVE` 文件移动到 `_archive/cleanup_<时间戳>/root_scripts/`，按 `runs`、`plots`、`tests`、`compat` 分类保存。
4. 仅删除可再生的 `__pycache__`；临时目录必须先检查活动进程并验证解析后的绝对路径位于项目根目录内。
5. 生成或更新清理报告，记录每个文件的原路径、新路径、分类理由以及跳过项。

## 必须保留的核心文件

- `thermal_case_simulator.py`
- `thermal_batch_config.py`
- `thermal_control_strategies.py`
- `mpc_flow_direction_strategies.py`
- `thermal_loop.py`
- `thermal_system.py`
- `pack.py`
- `age_model.py`
- `mpc_evaporator_capacity_model.py`
- `btms_runtime.py`
- candidate B、最终12组控制器比较、终端代价、最终汇报绘图相关入口及其有效测试

## Git 边界

当前目录已经是 Git 仓库，当前分支为 `codex/peak-flow-comparison`，但没有配置远程仓库。大部分工作区文件当前未被 Git 跟踪，因此在首次正式纳管前，不能把 Git 当作这些文件的恢复手段。

本轮清理不自动执行 `git add`、`git commit`、`git remote add` 或 `git push`。远程连接和首次基线提交应作为独立步骤，由用户提供 GitHub/Gitee 仓库地址后执行。

## 验证

清理后执行：

1. 对核心链路及保留入口运行 `py_compile`；
2. 运行与被归档候选相关的轻量测试集合，确认没有引用断裂；
3. 再次搜索被移动文件名，确认正式入口和文档没有悬空引用；
4. 输出清理前后根目录脚本数量和分类清单；
5. 不运行完整仿真、长时间 GEKKO 扫描或 PSO。

## 回滚

人工代码均保留在时间戳归档目录中。若验证发现遗漏，按清理报告中的原路径将对应文件移回；缓存和临时目录不参与回滚。
