# MPC Flow Reversal PPT Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate one compact 2x2 PNG that combines MPC single/double-flow mean temperature, maximum temperature-difference trajectories, maximum temperature-difference bars, and energy bars.

**Architecture:** Extend the existing reproducible matplotlib report script with one focused builder, `plot_mpc_flow_overview_2x2`. It reads the same formal case CSVs and summary table as the existing figures, uses `twinx()` for each upper time-series panel, and is added to the existing PNG-only export pipeline.

**Tech Stack:** Python, pandas, NumPy, matplotlib, unittest

---

### Task 1: Add a failing layout and content test

**Files:**
- Modify: `test_plot_final_mpc_report.py`

- [ ] **Step 1: Import the new builder and write the failing test**

```python
from plot_final_mpc_report import plot_mpc_flow_overview_2x2

def test_flow_overview_is_compact_dual_axis_two_by_two(self):
    figure = plot_mpc_flow_overview_2x2(load_summary())
    self.assertEqual(len(figure.axes), 6)
    self.assertLessEqual(figure.get_size_inches()[1], 6.0)
    self.assertIn("平均温度", figure.axes[0].get_ylabel())
    self.assertIn("最大温差", figure.axes[4].get_ylabel())
    self.assertEqual(len(figure.axes[2].patches), 4)
    self.assertEqual(len(figure.axes[3].patches), 4)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `C:\Users\24776\miniforge3\envs\btms\python.exe -m unittest test_plot_final_mpc_report.py`

Expected: import failure because `plot_mpc_flow_overview_2x2` does not exist.

### Task 2: Implement the compact dual-axis figure

**Files:**
- Modify: `plot_final_mpc_report.py`

- [ ] **Step 1: Add `plot_mpc_flow_overview_2x2(summary=None)`**

```python
def plot_mpc_flow_overview_2x2(summary=None):
    configure_style()
    summary = load_summary() if summary is None else summary
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 5.8))
    twin_axes = []
    for ax, scene in zip(axes[0], SCENES):
        twin = ax.twinx()
        twin_axes.append(twin)
        for flow, color in (("single", "#6BAED6"), ("double", "#08519C")):
            frame = load_case(scene, "mpc", flow)
            linestyle, _ = FLOW_STYLES[flow]
            ax.plot(_minutes(frame), _values(frame, "T_cell_mean_C"), linestyle,
                    color=color, linewidth=1.2, alpha=0.65,
                    label=f"{FLOW_LABELS[flow]}平均温度")
            twin.plot(_minutes(frame), _values(frame, "Delta_T_cell_C"), linestyle,
                      color=color, linewidth=2.0,
                      label=f"{FLOW_LABELS[flow]}最大温差")
        ax.set_xlabel("时间 (min)")
        ax.set_ylabel("电池平均温度 (℃)")
        twin.set_ylabel("最大温差 (℃)")
```

Finish each upper panel and construct the lower panels with the following logic:

```python
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = twin.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2,
                  frameon=False, ncol=2, fontsize=7, loc="best")
        ax.text(0.02, 0.94, SCENE_LABELS[scene], transform=ax.transAxes,
                va="top", fontsize=10)

    labels = [f"{SCENE_LABELS[s]}\n{FLOW_LABELS[f]}" for s, f in CASES]
    colors = ["#9ECAE1", "#3182BD", "#FDD0A2", "#E6550D"]
    for ax, (column, ylabel) in zip(
        axes[1],
        (("max_delta_T_C", "最大温差 (℃)"), ("energy_kWh", "累计能耗 (kWh)")),
    ):
        values = [
            float(summary[(summary.scene_key == scene)
                          & (summary.control == "mpc")
                          & (summary.flow_key == flow)][column].iloc[0])
            for scene, flow in CASES
        ]
        bars = ax.bar(np.arange(len(CASES)), values, color=colors, width=0.68)
        ax.set_xticks(np.arange(len(CASES)), labels)
        ax.set_ylabel(ylabel)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
        ax.margins(y=0.18)
    fig.tight_layout(pad=0.8, h_pad=1.0, w_pad=1.1)
    return fig
```

- [ ] **Step 2: Run the complete plotting test suite**

Run: `C:\Users\24776\miniforge3\envs\btms\python.exe -m unittest test_plot_final_mpc_report.py`

Expected: all tests pass.

### Task 3: Add PNG-only export and verify the artifact

**Files:**
- Modify: `plot_final_mpc_report.py`
- Generate: `输出结果/最终控制器完整对比/服务器_12组_终端代价_candidateB_20260712_134913/MPC内部汇报图/03A_PPT流向反转效果_2x2.png`

- [ ] **Step 1: Add the builder to `generate_all_figures`**

```python
("03A_PPT流向反转效果_2x2", plot_mpc_flow_overview_2x2(summary)),
```

Update the export test expected PNG/path count from five to six.

- [ ] **Step 2: Generate all report PNGs**

Run: `C:\Users\24776\miniforge3\envs\btms\python.exe plot_final_mpc_report.py`

Expected: six PNG paths are printed and no PDF is produced.

- [ ] **Step 3: Inspect the new image**

Check that the upper legends do not obscure data, the right axes clearly say `最大温差 (℃)`, both bottom charts have four labeled bars, and the complete figure remains readable at PPT width.

- [ ] **Step 4: Commit implementation files**

```powershell
git add -- plot_final_mpc_report.py test_plot_final_mpc_report.py docs/superpowers/plans/2026-07-13-mpc-flow-reversal-ppt-figure.md
git commit -m "feat: add compact MPC flow reversal report figure"
```
