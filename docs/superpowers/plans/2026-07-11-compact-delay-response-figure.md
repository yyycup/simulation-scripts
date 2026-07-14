# Compact Delay Response Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create one concise, three-panel PNG from the supplied delayed and delay-bypassed response CSV files.

**Architecture:** Keep the existing CSV files unchanged. Update the dedicated plotting script so it crops data at the compressor step, shifts the time axis to zero, and renders only actual compressor speed, cooling capacity, and cold-plate inlet temperature. The test module will define the exact three plotted fields and their short Chinese labels.

**Tech Stack:** Python 3, pandas, matplotlib, unittest-style direct test functions.

---

### Task 1: Define the compact plotting contract

**Files:**
- Modify: `test_plot_delay_step_response_simple.py:1-13`
- Modify: `plot_delay_step_response_simple.py:12-18`

- [ ] **Step 1: Replace the existing expected row list with a failing three-row test**

```python
from plot_delay_step_response_simple import PLOT_ROWS


def test_compact_plot_has_only_the_three_requested_response_signals():
    assert PLOT_ROWS == [
        ("压缩机实际转速_rpm", "转速 (rpm)"),
        ("制冷量_W", "制冷量 (W)"),
        ("冷板入口冷却液温度_C", "入口温度 (℃)"),
    ]
```

- [ ] **Step 2: Run the test and verify that it fails because the script still contains five rows**

Run:

```powershell
C:\Users\24776\miniforge3\envs\btms\python.exe -c "from test_plot_delay_step_response_simple import test_compact_plot_has_only_the_three_requested_response_signals; test_compact_plot_has_only_the_three_requested_response_signals()"
```

Expected: `AssertionError` because the old list includes cold-plate and battery temperature rows.

- [ ] **Step 3: Replace the plotting row constant with the requested three fields**

```python
PLOT_ROWS = [
    ("压缩机实际转速_rpm", "转速 (rpm)"),
    ("制冷量_W", "制冷量 (W)"),
    ("冷板入口冷却液温度_C", "入口温度 (℃)"),
]
```

- [ ] **Step 4: Re-run the contract test**

Run the command from Step 2.

Expected: no output and process exit code `0`.

### Task 2: Render and verify the compact PNG

**Files:**
- Modify: `plot_delay_step_response_simple.py:21-34`
- Output: `outputs/delay_comparison_smoke/delay_comparison_compact.png`

- [ ] **Step 1: Make the layout compact without changing the two-series comparison**

Replace the figure creation and output name with:

```python
figure, axes = plt.subplots(3, 1, figsize=(7.2, 7.6), sharex=True, constrained_layout=True)
# Keep one solid line per dataset, a subtle dotted vertical line at t=0,
# short Chinese labels, and a two-item legend in every panel.
output_path = args.input_dir / "delay_comparison_compact.png"
```

- [ ] **Step 2: Generate the figure from the two supplied CSV files**

Run:

```powershell
C:\Users\24776\miniforge3\envs\btms\python.exe .\plot_delay_step_response_simple.py --input-dir .\outputs\delay_comparison_smoke --step-time-s 50
```

Expected: a 300 dpi PNG exists at `outputs/delay_comparison_smoke/delay_comparison_compact.png`.

- [ ] **Step 3: Verify the output artifact and CSV preservation**

Run:

```powershell
Get-Item .\outputs\delay_comparison_smoke\delay_comparison_compact.png
Get-Item .\outputs\delay_comparison_smoke\delay_enabled.csv, .\outputs\delay_comparison_smoke\delay_bypassed.csv
```

Expected: all three files exist; only the PNG has a new write time.

- [ ] **Step 4: Commit the focused changes**

```powershell
git add plot_delay_step_response_simple.py test_plot_delay_step_response_simple.py docs/superpowers/specs/2026-07-11-delay-response-figure-design.md docs/superpowers/plans/2026-07-11-compact-delay-response-figure.md
git commit -m "Simplify delay response figure"
```

Expected: one commit containing only the compact-plot implementation, its test, and the approved design records.
