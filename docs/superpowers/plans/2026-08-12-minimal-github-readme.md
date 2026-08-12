# Minimal GitHub README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the repository root README with the approved project title and one-paragraph Chinese introduction.

**Architecture:** This is a documentation-only replacement of `README.md`. The implementation must preserve all simulation, controller, model, test, and supporting documentation files unchanged, while enforcing exact content equality with the approved design specification.

**Tech Stack:** GitHub-Flavored Markdown, UTF-8, PowerShell, Git

---

### Task 1: Replace the root README with the approved minimal introduction

**Files:**
- Modify: `README.md`
- Reference: `docs/superpowers/specs/2026-08-12-minimal-github-readme-design.md`

- [ ] **Step 1: Run the exact-content contract and verify the current README fails it**

Run from the repository root:

```powershell
$expected = @'
# 储能电池热管理仿真与控制

本项目面向储能电池热管理系统的建模、仿真与控制研究，集成电池热模型、冷板与冷却液动态、制冷循环，以及 On-Off、PID 和 MPC 控制策略，支持调峰与调频场景分析。
'@

$actual = (Get-Content -LiteralPath README.md -Raw -Encoding UTF8).TrimEnd("`r", "`n")
if ($actual -eq $expected) {
    throw 'Pre-change README unexpectedly already matches the approved minimal content.'
}
Write-Output 'RED: README does not yet match the approved minimal content.'
```

Expected output:

```text
RED: README does not yet match the approved minimal content.
```

- [ ] **Step 2: Replace `README.md` with the exact approved content**

Use `apply_patch` to replace the entire file with:

```markdown
# 储能电池热管理仿真与控制

本项目面向储能电池热管理系统的建模、仿真与控制研究，集成电池热模型、冷板与冷却液动态、制冷循环，以及 On-Off、PID 和 MPC 控制策略，支持调峰与调频场景分析。
```

- [ ] **Step 3: Run the exact-content contract and forbidden-content checks**

Run from the repository root:

```powershell
$expected = @'
# 储能电池热管理仿真与控制

本项目面向储能电池热管理系统的建模、仿真与控制研究，集成电池热模型、冷板与冷却液动态、制冷循环，以及 On-Off、PID 和 MPC 控制策略，支持调峰与调频场景分析。
'@

$actual = (Get-Content -LiteralPath README.md -Raw -Encoding UTF8).TrimEnd("`r", "`n")
if ($actual -ne $expected) {
    throw 'README does not exactly match the approved minimal content.'
}

$forbidden = @(
    'Candidate B',
    'Physics-P',
    '```',
    '| --- |',
    '!['
)
foreach ($token in $forbidden) {
    if ($actual.Contains($token)) {
        throw "README contains forbidden content: $token"
    }
}

Write-Output 'GREEN: README exactly matches the approved minimal content.'
```

Expected output:

```text
GREEN: README exactly matches the approved minimal content.
```

- [ ] **Step 4: Verify the documentation-only diff**

Run:

```powershell
git diff --check -- README.md
git diff --stat
git diff -- README.md
```

Expected:

- `git diff --check` exits with code 0.
- The implementation diff modifies only `README.md`.
- The rendered content consists of one H1 heading and one paragraph.
- No control, model, parameter, test, result, or other documentation file is modified by the implementation.

- [ ] **Step 5: Commit the README replacement**

Run:

```powershell
git add -- README.md
git diff --cached --check
git diff --cached --name-only
git commit -m "docs: simplify repository README"
```

Expected:

- The staged file list contains only `README.md`.
- The commit succeeds with no simulation or test execution required because the change is Markdown-only.
