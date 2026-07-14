# GitHub可独立运行与本地彻底清理设计

## 目标

把当前GitHub仓库从“已上传主要代码”完善为“克隆后可按说明创建环境并运行核心仿真”，随后删除本地历史归档和可再生临时文件。

本次不改变热模型、制冷循环、candidate B公式、MPC权重、PID参数、终端代价或最终论文结果。

## 中文命名原则

- `README.md`正文、章节标题、项目介绍、运行说明和面向用户的描述使用中文。
- Git提交说明优先使用简洁中文或明确的中英混合表述。
- Python模块名、Conda环境名、命令行参数以及需要跨Windows/Linux服务器使用的内部路径保留ASCII英文，避免编码和命令兼容问题。
- candidate B、MPC、PID、BTMS等已有技术缩写保持不变。

## 运行数据结构

新增受Git管理的 `model_data/`，存放运行必需的小型模型参数：

| 新路径 | 当前来源 | 用途 |
|---|---|---|
| `model_data/hppc_params.json` | `data/hppc_params.json` | 电池HPPC参数，供 `pack.py` 读取 |
| `model_data/mpc_evaporator_capacity_candidate_b.json` | `outputs/mpc_evaporator_capacity_candidate_b/mpc_evaporator_capacity_candidate_b.json` | candidate B蒸发器标定参数 |
| `model_data/candidate_b_capacity_limits_grid.csv` | `outputs/mpc_evaporator_capacity_candidate_b/candidate_b_capacity_limits_grid.csv` | candidate B容量边界诊断和测试数据 |
| `model_data/mpc_reduced_model_best_theta.json` | `outputs/mpc_reduced_model_calibration/best_theta.json` | MPC降阶模型标定参数 |

采用复制后核对哈希的方式迁移，保留原文件直至代码验证和Git推送成功。推送成功后，原文件随本地 `outputs/` 保留，不单独删除；本轮只删除 `_archive/` 和根目录临时目录。

## 代码路径修改

- `pack.py` 改为从 `model_data/hppc_params.json` 读取HPPC参数。
- `mpc_evaporator_capacity_model.py` 改为从 `model_data/mpc_evaporator_capacity_candidate_b.json` 读取candidate B参数。
- `mpc_flow_direction_strategies.py` 改为从 `model_data/` 读取容量边界表和MPC降阶标定参数。
- candidate B计算公式、上下界处理和MPC状态方程保持不变。

## 环境文件

新增 `environment.yml`：

- 环境名为 `btms`；
- 使用 `conda-forge`；
- 固定当前Python主次版本；
- 记录 NumPy、Pandas、SciPy、Matplotlib、PyBaMM、CoolProp、GEKKO、platformdirs 等直接依赖；
- 不记录本机绝对路径、临时目录或用户级配置。

依赖版本从当前可用 `btms` 环境读取。若某个包只能通过pip稳定安装，则放入 `pip:` 子段。

## README

新增中文 `README.md`，至少包含：

1. 项目简介与适用场景；
2. 核心代码结构；
3. Conda环境创建和激活命令；
4. 模型数据说明；
5. 最终12组控制器比较命令；
6. MPC参数扫描、candidate B诊断和最终绘图入口；
7. Windows DLL处理说明；
8. 输出、数据和归档的Git边界；
9. 验证命令和长仿真注意事项。

## 验证

实施后执行：

1. 校验4个新模型数据文件与原文件的SHA256一致；
2. 验证3处默认路径均指向 `model_data/`；
3. 编译全部根目录Python文件；
4. 运行candidate B、降阶标定、最终控制器比较和最终绘图相关轻量测试；
5. 检查Git暂存区不含 `outputs/`、`输出结果/`、`data/`、`_archive/`、临时目录或外部工具；
6. 提交并推送远程 `main`；
7. 比较本地HEAD与远程 `main` 哈希。

## 本地永久清理

只有在代码验证和Git推送成功后才执行：

1. 再次核对PID 48092和36696仍是2026-07-11启动的旧 `test_pid_local_refinement` 进程；
2. 只停止这两个精确匹配的旧进程，不按进程名批量终止Python；
3. 解析并验证 `_archive/`、根目录 `tmp*/`、`_gekko_tmp_*/` 和 `__pycache__/` 的绝对路径均位于项目根目录内；
4. 递归删除上述目录；
5. 不删除 `outputs/`、`输出结果/`、`data/` 或Git管理的 `model_data/`；
6. 记录删除前后的文件数和空间变化；
7. 将本地分支从 `codex/peak-flow-comparison` 改名为 `main`，继续跟踪 `origin/main`。

## 回滚与安全边界

- 模型数据迁移采用复制，不在推送前破坏原文件。
- 代码改动可通过Git提交回滚。
- `_archive/` 和临时目录删除是不可逆操作；用户已明确批准按本设计删除。
- 删除前必须再次确认远程 `main` 已包含新的模型数据、README和环境文件。
- 活跃或身份不匹配的进程不会被终止，相应临时目录也不会强删。
