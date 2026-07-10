# 调峰工况单向流/双向流对照实验设计

## 目标

在相同调峰电流、初始状态、热模型和 MPC 参数下，重新仿真单向冷却液流动与简单阈值双向流动，对比电池包最大温差。

## 实验配置

- 工况：调峰（`scene="peak"`）。
- 控制：MPC，使用当前项目参数。
- 时间步长：使用项目当前的 `SIM_DT=5 s`。
- 单向流：`flow="normal"`，`is_reversed` 始终为 `False`。
- 双向流：`flow="reversed"`，使用 `maybe_reverse_flow()` 的简单实测最大温差阈值逻辑。
- 反转条件：电池包实测最大温差 `> 0.5 ℃`，且距上次反转 `> 200 s`。
- 双向流不使用预测缓冲、综合判据或混合整数流向优化。

## 实现方式

复用 `thermal_case_simulator.simulate_case()` 和已有调峰输入文件，新增一个窄范围对照运行/绘图脚本。脚本顺序执行单向与双向两个完整仿真，然后从主输出 CSV 中读取 `Time (s)`、`Delta_T_cell_C` 和 `Flow Direction d`。

注意：仿真入参必须使用代码实际识别的英文标识 `peak/normal/reversed`，不使用中文显示名称判断是否开启反转。

## 数据与图形输出

在项目内新建独立输出目录 `outputs/peak_single_bidirectional_flow_comparison/`，包含：

- `peak_single_flow.csv`：单向流完整时序数据。
- `peak_bidirectional_flow.csv`：双向流完整时序数据。
- `peak_flow_comparison_summary.csv`：各方案全程最大温差、发生时刻、双向流反转次数。
- `peak_flow_comparison.png`：300 DPI 静态图。
- `peak_flow_comparison.pdf`：矢量图。
- 仿真进度日志。

图形为两行共享时间轴：

1. 上图绘制单向流和双向流的 `Delta_T_cell_C`，并标出 `0.5 ℃` 阈值线和各自峰值。
2. 下图绘制双向流的 `Flow Direction d`（`+1/-1`），在每个反转时刻画竖线。

## 验证

- 确认两份主 CSV 行数、起止时间和输入工况一致。
- 确认单向流的 `Flow Direction d` 全程为 `+1`。
- 从双向流相邻行的流向变化独立计算反转次数。
- 核对每次反转前的温差达到阈值，且相邻反转间隔超过 `200 s`。
- 重新读取导出图片，确认线条、图例、坐标轴和反转标记可见。

## 边界

本任务不修改电池、冷板、制冷循环、MPC 目标函数、参数调度或 PID/PSO 逻辑；仅新增可复现的对照运行与绘图入口。
