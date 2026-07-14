# Full-Run PID PSO Design

## Goal

Replace the current offline PID random-search workflow with a full-run particle swarm optimization (PSO) workflow that can be launched from PyCharm.

The new workflow must:

- optimize PID directly on full operating profiles, not short representative segments;
- produce exactly two final PID parameter sets:
  - `peak_pid_params`
  - `freq_pid_params`
- evaluate each candidate using both one-way and bidirectional cases for the same scene;
- keep the existing constraint-first, minimum-tracking-error selection rule;
- retire the old random-search scripts as formal tuning entrypoints.

This change does not modify:

- the fixed-rule on-off baseline;
- the outer flow-direction decision logic;
- the plant/simulation model behavior unrelated to PID search;
- online adaptive PID updates during formal simulation.

## Scope

Included:

- add a new PyCharm entry script for full-run PSO PID tuning;
- reuse the existing PID candidate simulation and metric extraction logic where possible;
- reuse the existing feasibility constraints and final metric ranking rule;
- write PSO progress logs and final parameter outputs to `outputs/`;
- remove the old random-search PyCharm scripts from the supported workflow.

Excluded:

- changing on-off thresholds or on-off control structure;
- changing PID control law inside the simulator;
- introducing scene-specific single-direction and bidirectional PID parameter sets;
- adding short-segment screening;
- changing MPC-related code.

## Functional Requirements

### 1. Optimization structure

The optimizer runs two independent searches:

1. `peak` search
2. `freq` search

Each search uses PSO over PID parameter space:

- `kp`
- `ki`
- `kd`

Each particle represents one PID tuple. Each iteration updates particle velocity and position according to standard PSO rules using:

- particle best position;
- swarm global best position;
- configured inertia and cognitive/social gains.

### 2. Candidate evaluation

For one particle evaluation in one scene:

1. collect PID cases for that scene with control mode `pid`;
2. run the full scene for both one-way and bidirectional flow cases;
3. compute merged metrics across both cases:
   - `T_avg_mean_error`
   - `T_avg_min`
   - `T_avg_max`
   - `energy`
   - `compressor_action_count`
4. mark feasibility using the existing constraint policy.

The optimization objective remains:

- first satisfy constraints;
- then minimize `T_avg_mean_error`.

### 3. Constraints

Constraints remain aligned with the current agreed policy:

- reject if `T_avg_max > 28.0 C`;
- for `peak`, reject if `T_avg_min < 24.0 C`;
- for `freq`, reject if `T_avg_min < 24.3 C`;
- reject if `energy` is abnormally high;
- reject if compressor action is too frequent.

To avoid changing the agreed policy surface, the implementation should continue to use the existing helper logic that infers energy/action thresholds from observed candidates unless explicit thresholds are supplied.

### 4. Final outputs

After each scene search completes, write:

- scene summary CSV under `outputs/`;
- scene progress log under `outputs/`;
- best PID tuple.

After both scenes finish, write:

- `outputs/pid_pso_final_pid_params.txt`

with:

- `peak_pid_params = (...)`
- `freq_pid_params = (...)`

## Implementation Approach

### Option chosen

Use a new standalone full-run PSO script rather than mutating the old random-search runner in place.

Why:

- keeps the change isolated;
- reduces risk of breaking the existing evaluation helpers;
- makes deletion of old search entrypoints straightforward;
- gives PyCharm a single new supported launch target.

### Code reuse

The PSO runner should reuse existing logic from:

- `pid_param_search.py`
  - scene normalization
  - case selection
  - PID metric extraction
  - candidate feasibility checks
  - inferred energy/action limits
  - candidate evaluation through `simulate_case`

It may also reuse timeout/progress patterns from:

- `run_pid_two_stage_search_pycharm.py`

but it should not preserve the old two-stage random-search control flow.

### Files to add

- `run_pid_pso_full_search_pycharm.py`

This becomes the new supported PyCharm entrypoint for offline PID tuning.

### Files to retire

The following scripts should be removed from the supported workflow, and if safe, deleted:

- `run_pid_param_search_pycharm.py`
- `run_pid_two_stage_search_pycharm.py`
- `run_pid_two_stage_search_resilient_pycharm.py`
- `run_pid_two_stage_freq_only_pycharm.py`
- `run_pid_two_stage_freq_coarse_batch_pycharm.py`
- `run_pid_two_stage_freq_full_review_pycharm.py`

`pid_param_search.py` should be kept if it still provides reusable shared helpers for PID evaluation and constraints. It should only be deleted if all required shared logic is migrated cleanly into the new PSO module without duplication or ambiguity.

## Logging and Outputs

The new PSO runner should log:

- scene start/end;
- iteration index;
- particle index;
- PID tuple;
- merged metrics;
- feasibility result;
- current scene best PID and objective value.

Outputs should be resumable or at least inspectable from partial logs and CSV summaries so PyCharm termination does not erase all progress visibility.

## Error Handling

The runner should:

- catch per-particle simulation failures;
- record failed particles in the scene summary;
- continue the search unless the scene cannot evaluate any successful particle;
- fail clearly if no feasible particle exists by the end of a scene.

If only one scene succeeds:

- write a partial final file with the completed scene result;
- report the missing scene explicitly in logs.

## Testing

Because full-run thermal simulation is expensive, verification should stay focused:

1. a small smoke configuration for PSO with tiny swarm and iteration counts;
2. checks that:
   - particles are evaluated;
   - summaries are written;
   - final params file is written when both scenes succeed;
   - infeasible scenes fail with a clear message.

If existing test files are unstable or encoding-damaged, add a narrow new test instead of broad repair during this change.

## Risks

### Runtime

Full-run PSO is slower than short-segment screening and slower than small random screening. This is acceptable because the user explicitly chose full-run optimization.

### No feasible frequency solution

The earlier random search showed a real possibility that `freq` may have zero feasible candidates under current constraints. PSO may improve search efficiency, but it does not guarantee feasibility if the feasible region is extremely small or absent.

### Partial legacy removal

Deleting all old search code too aggressively could accidentally remove reusable helper logic. Shared evaluation helpers should be preserved until the new runner is stable and clearly owns the full workflow.

## Acceptance Criteria

The change is complete when:

1. a new PyCharm script can launch full-run PSO PID tuning;
2. the script runs separate `peak` and `freq` PSO searches;
3. each particle is evaluated on full one-way plus bidirectional cases for its scene;
4. the agreed feasibility constraints are enforced;
5. the final result file contains only:
   - `peak_pid_params`
   - `freq_pid_params`
6. the old random-search PyCharm entry scripts are removed from the supported workflow, and deleted where safe.
