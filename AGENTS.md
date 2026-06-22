# AGENTS.md

## Project Context

This repository is a battery energy storage system thermal-management simulation and control project. It includes battery thermal modeling, cold-plate and coolant dynamics, refrigeration-cycle modeling, and PID/MPC control strategies for peak-shaving and frequency-regulation operating scenarios.

## Default Navigation

For tasks involving control, MPC, PID, thermal models, or refrigeration models, inspect the main simulation flow, control strategy files, thermal model files, and refrigeration model files first.

Do not default to scanning generated, large, external, or historical directories such as:

- `.codex/`
- `.idea/`
- `data/`
- `outputs/`
- `output/`
- `figures/`
- `analysis_outputs/`
- `node_modules/`
- `slprj/`
- `legacy/`
- `tmp/`
- `simulink-agentic-toolkit/`
- `simulink-agentic-toolkit-main/`

Only inspect those directories when the user explicitly asks for them or when a specific task requires a named file inside them.

## Workflow Rules

Before modifying code, list:

- files planned for inspection
- files that may be modified
- the reason each file is relevant

Wait for user confirmation before broad changes, large refactors, or changes spanning multiple subsystems. Prefer small, verifiable edits that preserve the existing project structure.
