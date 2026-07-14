# Root Script Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely reduce root-level script clutter, preserve reproducibility through timestamped archives, and connect the existing local Git repository to `yyycup/simulation-scripts` without prematurely pushing unreviewed files.

**Architecture:** Treat cleanup as a manifest-driven migration rather than deletion. Read-only dependency checks produce an explicit `KEEP`/`ARCHIVE`/`REVIEW` manifest; only `ARCHIVE` items move into a timestamped archive, generated caches are removed after path and process checks, and lightweight verification checks the surviving main flow. Git remote configuration is independent from staging, committing, and pushing.

**Tech Stack:** PowerShell, Git, Python 3 from the `btms` Conda environment, `rg`, `pytest`, Markdown.

---

## File Structure

- Create: `PROJECT_CLEANUP_MANIFEST_20260714.md` — explicit classification and evidence for every root-level candidate.
- Create: `_archive/cleanup_<timestamp>/root_scripts/{runs,plots,tests,compat}/` — reversible destination for human-authored files.
- Modify: `PROJECT_CLEANUP_REPORT.md` — append moved, preserved, reviewed, deleted-cache, skipped, and verification results.
- Modify: `PROJECT_MAP_MIN.md` — update current primary runners and the new archive location.
- Modify: `.codexignore` — add current root-level GEKKO and ordinary temporary-directory patterns.
- Preserve unchanged: `thermal_case_simulator.py`, `thermal_batch_config.py`, `thermal_control_strategies.py`, `mpc_flow_direction_strategies.py`, `thermal_loop.py`, `thermal_system.py`, `pack.py`, `age_model.py`, `mpc_evaporator_capacity_model.py`, `btms_runtime.py`.
- Preserve unchanged: `outputs/`, `输出结果/`, `data/`, existing `_archive/`, `.agents/`, `.codex/`, `.idea/`, `.git_empty_backup/`, and `simulink-agentic-toolkit*/`.

### Task 1: Connect the existing repository to GitHub without pushing

**Files:**
- Modify: `.git/config` through `git remote add`

- [ ] **Step 1: Confirm the target repository is reachable**

Run:

```powershell
git ls-remote https://github.com/yyycup/simulation-scripts.git
```

Expected: exit code `0`; an empty repository may produce no reference lines.

- [ ] **Step 2: Confirm that `origin` is still absent**

Run:

```powershell
git remote -v
```

Expected: no output.

- [ ] **Step 3: Add the remote**

Run:

```powershell
git remote add origin https://github.com/yyycup/simulation-scripts.git
```

Expected: exit code `0`.

- [ ] **Step 4: Verify fetch and push URLs**

Run:

```powershell
git remote -v
```

Expected:

```text
origin  https://github.com/yyycup/simulation-scripts.git (fetch)
origin  https://github.com/yyycup/simulation-scripts.git (push)
```

- [ ] **Step 5: Do not stage, commit, or push**

Run:

```powershell
git status --branch --short
```

Expected: branch remains `codex/peak-flow-comparison`; untracked files remain visible and no Git index changes are introduced by remote configuration.

### Task 2: Produce an evidence-backed cleanup manifest

**Files:**
- Create: `PROJECT_CLEANUP_MANIFEST_20260714.md`

- [ ] **Step 1: Capture the candidate inventory without scanning excluded result directories**

Run:

```powershell
Get-ChildItem -File |
    Where-Object { $_.Name -like 'run_*.py' -or $_.Name -like 'plot_*.py' -or $_.Name -like 'test_*.py' } |
    Sort-Object Name |
    Select-Object Name,Length,LastWriteTime
```

Expected: a root-only list of runner, plotting, and test scripts.

- [ ] **Step 2: Search source and documentation references for every candidate family**

Run:

```powershell
rg -n --glob '*.py' --glob '*.md' --glob '*.cmd' --glob '!outputs/**' --glob '!输出结果/**' --glob '!data/**' --glob '!_archive/**' --glob '!.git/**' 'run_|plot_|test_|mpc_sensitivity_compat|predictive_delta_t' .
```

Expected: import, command, and documentation evidence without output/archive noise.

- [ ] **Step 3: Write the manifest header and fixed classification rules**

Create `PROJECT_CLEANUP_MANIFEST_20260714.md` with these sections:

```markdown
# 2026-07-14 根目录脚本清理清单

## Classification Rules

- KEEP: core flow, final 12-case reproduction, candidate B, terminal-cost configuration, final report plotting, or a test that protects one of those paths.
- ARCHIVE: completed one-off scan/diagnostic/plot family with no active import or command reference and a newer retained replacement.
- REVIEW: unclear provenance, ambiguous replacement, or any evidence that it may be needed for paper/result reproduction.

## Candidates

| Path | Class | Evidence | Replacement or dependency | Action |
|---|---|---|---|---|

## Generated Directories

| Path | Process check | Resolved-path check | Action |
|---|---|---|---|
```

- [ ] **Step 4: Populate one row for every candidate**

For each file from Step 1, record the exact path, one classification, concrete evidence from Step 2, replacement/dependency if any, and either `preserve`, `archive`, or `review only`.

Expected: no candidate lacks a classification or evidence cell; ambiguous candidates are `REVIEW`, never `ARCHIVE`.

- [ ] **Step 5: Confirm final-result families are preserved**

Run:

```powershell
Select-String -Path PROJECT_CLEANUP_MANIFEST_20260714.md -Pattern 'run_final_controller_comparison|run_final_controller_parallel|plot_final_mpc_report|run_mpc_sensitivity_60|mpc_evaporator_capacity_model'
```

Expected: each family appears as `KEEP` or, if ambiguity remains, `REVIEW`; none is `ARCHIVE`.

### Task 3: Prepare and execute the reversible archive move

**Files:**
- Create: `_archive/cleanup_<timestamp>/root_scripts/runs/`
- Create: `_archive/cleanup_<timestamp>/root_scripts/plots/`
- Create: `_archive/cleanup_<timestamp>/root_scripts/tests/`
- Create: `_archive/cleanup_<timestamp>/root_scripts/compat/`
- Modify: `PROJECT_CLEANUP_REPORT.md`

- [ ] **Step 1: Record counts before moving files**

Run:

```powershell
$before = Get-ChildItem -File | Where-Object { $_.Name -like 'run_*.py' -or $_.Name -like 'plot_*.py' -or $_.Name -like 'test_*.py' }
$before | Group-Object { if ($_.Name -like 'run_*') {'run'} elseif ($_.Name -like 'plot_*') {'plot'} else {'test'} } |
    Select-Object Name,Count
```

Expected: three category counts suitable for the cleanup report.

- [ ] **Step 2: Create a timestamped archive inside the workspace**

Run:

```powershell
$root = (Resolve-Path -LiteralPath '.').Path
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$archive = Join-Path $root "_archive\cleanup_$stamp\root_scripts"
$resolvedArchiveParent = (Resolve-Path -LiteralPath (Join-Path $root '_archive')).Path
if (-not $resolvedArchiveParent.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Archive parent escaped workspace: $resolvedArchiveParent"
}
@('runs','plots','tests','compat') | ForEach-Object {
    New-Item -ItemType Directory -Path (Join-Path $archive $_) -Force | Out-Null
}
```

Expected: four empty category directories under one new cleanup timestamp.

- [ ] **Step 3: Move only manifest rows classified `ARCHIVE`**

For every `ARCHIVE` row, resolve and validate the source path before using `Move-Item -LiteralPath`. Choose the destination using the filename prefix: `run_` to `runs`, `plot_` to `plots`, `test_` to `tests`, and compatibility wrappers to `compat`.

Required safety check for each source:

```powershell
$source = (Resolve-Path -LiteralPath $candidate).Path
if (-not $source.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Source escaped workspace: $source"
}
Move-Item -LiteralPath $source -Destination $destination
```

Expected: only explicit `ARCHIVE` rows move; `KEEP` and `REVIEW` remain at their original paths.

- [ ] **Step 4: Append the move ledger to `PROJECT_CLEANUP_REPORT.md`**

Append a section containing:

```markdown
## 2026-07-14 Incremental Root Script Cleanup

- Scope: root scripts, compatibility wrappers, and generated caches only.
- Excluded: outputs, 输出结果, data, existing archives, and external toolkits.
- Strategy: archive human-authored files; delete only regenerated cache data.

| Original path | Classification | New path or status | Evidence |
|---|---|---|---|
```

Expected: one ledger row per candidate, including preserved and skipped files.

### Task 4: Remove only verified generated cache and temporary directories

**Files:**
- Delete if verified: `__pycache__/`
- Delete if verified and unused: root `_gekko_tmp_*/` and `tmp*/`
- Modify: `.codexignore`
- Modify: `PROJECT_CLEANUP_REPORT.md`

- [ ] **Step 1: Check for active Python, GEKKO, IPOPT, and simulation processes**

Run:

```powershell
Get-CimInstance Win32_Process |
    Where-Object { $_.Name -match 'python|apm|ipopt|gekko' } |
    Select-Object ProcessId,Name,CommandLine
```

Expected: if a process command line references this workspace or a candidate temp directory, record the directory as skipped and do not remove it.

- [ ] **Step 2: Resolve every candidate directory and enforce workspace containment**

Run:

```powershell
$tempCandidates = Get-ChildItem -Directory -Force |
    Where-Object { $_.Name -eq '__pycache__' -or $_.Name -like '_gekko_tmp_*' -or $_.Name -like 'tmp*' }
$tempCandidates | ForEach-Object {
    $resolved = $_.FullName
    if (-not $resolved.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Temporary directory escaped workspace: $resolved"
    }
    [pscustomobject]@{ Name = $_.Name; ResolvedPath = $resolved }
}
```

Expected: every reported path is a direct child of the project root.

- [ ] **Step 3: Remove only unused generated directories**

For each process-free candidate validated in Step 2:

```powershell
Remove-Item -LiteralPath $resolved -Recurse -Force
```

Expected: generated directories disappear; active or ambiguous directories remain and are recorded as skipped.

- [ ] **Step 4: Extend `.codexignore` with current temporary patterns**

Ensure `.codexignore` contains exactly one line for each pattern:

```text
tmp*/
_gekko_tmp_*/
```

Expected: future searches skip both ordinary and explicitly prefixed GEKKO root temp directories.

- [ ] **Step 5: Record removed and skipped directories**

Add each directory to the cleanup report ledger with status `deleted generated cache` or `skipped active/ambiguous`.

### Task 5: Update project navigation after cleanup

**Files:**
- Modify: `PROJECT_MAP_MIN.md`
- Modify: `PROJECT_CLEANUP_REPORT.md`

- [ ] **Step 1: Update the cleanup history in `PROJECT_MAP_MIN.md`**

Add the new timestamped archive path and state that final controller reproduction starts from:

```text
run_final_controller_comparison.py
run_final_controller_parallel.py
run_mpc_sensitivity_60.py
plot_final_mpc_report.py
```

Expected: the project map points to retained root files only.

- [ ] **Step 2: Check project-map paths for existence**

Run:

```powershell
@(
  'run_final_controller_comparison.py',
  'run_final_controller_parallel.py',
  'run_mpc_sensitivity_60.py',
  'plot_final_mpc_report.py'
) | ForEach-Object {
    if (-not (Test-Path -LiteralPath $_)) { throw "Missing retained entry: $_" }
}
```

Expected: exit code `0` with no missing-entry exception.

### Task 6: Verify the cleaned workspace

**Files:**
- Test: retained root Python modules and tests selected by dependency evidence
- Modify: `PROJECT_CLEANUP_REPORT.md`

- [ ] **Step 1: Compile the core main flow**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m py_compile thermal_case_simulator.py thermal_batch_config.py thermal_control_strategies.py mpc_flow_direction_strategies.py thermal_loop.py thermal_system.py pack.py age_model.py mpc_evaporator_capacity_model.py btms_runtime.py
```

Expected: exit code `0` and no output.

- [ ] **Step 2: Run lightweight tests for the retained final path**

Run:

```powershell
& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -m pytest -q test_btms_runtime.py test_mpc_evaporator_capacity_model.py test_final_controller_comparison.py test_final_controller_configured_pid.py test_plot_final_mpc_report.py
```

Expected: all collected tests pass; no full simulation, long GEKKO sweep, or PSO starts.

- [ ] **Step 3: Check for broken root references to archived filenames**

Run one `rg -n` search per moved filename over root `*.py`, `*.md`, and `*.cmd`, excluding `_archive`, results, data, and external toolkits.

Expected: no retained executable or navigation document contains a broken command/import reference. Historical cleanup ledgers may mention archived names and are allowed.

- [ ] **Step 4: Record before/after counts and verification evidence**

Append category counts, compile command result, pytest result, broken-reference result, and the exact archive path to `PROJECT_CLEANUP_REPORT.md`.

- [ ] **Step 5: Review Git changes without staging them**

Run:

```powershell
git status --short
git diff -- .codexignore PROJECT_MAP_MIN.md PROJECT_CLEANUP_REPORT.md
```

Expected: changes match the manifest and report; no files are staged.

### Task 7: Request explicit authorization for the first baseline commit and push

**Files:**
- No files changed in this task.

- [ ] **Step 1: Present the final inclusion and exclusion summary**

Report core files to be tracked, archived files, ignored result/data/temp roots, repository size risks, and test results.

- [ ] **Step 2: Ask for explicit staging/commit/push authorization**

Do not run `git add`, `git commit`, or `git push` until the user explicitly approves the final baseline contents.

- [ ] **Step 3: If approved later, stage only the reviewed baseline**

Use explicit paths or reviewed ignore rules. Do not use an unreviewed blanket `git add .` while outputs, archives, data, or toolkits are visible as untracked.
