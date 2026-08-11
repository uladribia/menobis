---
description: "MENoBiS microcanonical refactor orchestration (HISTORICAL — refactor COMPLETE). Documents how the microcanonical implementation refactor was executed per the practical sparse architecture specification. Retained as a template for future multi-phase refactor orchestration: delegation workflow, gate review, state management. Independent research or small edits should be handled directly, not through this workflow."
argument-hint: "<phase-or-task>"
---

# Microcanonical Refactor — Supervisor Orchestration

> ⚠️ **HISTORICAL — the microcanonical refactor is COMPLETE.** Phases A–H,
> the §35 benchmark matrix, and docs alignment are all merged into
> `microcanonical-refactor` (see `_workspace/refactor-state.md` for the final
> per-phase merge commits). This prompt is retained as a historical record of
> the orchestration workflow used and as a template for future multi-phase
> refactors. If reused for new work, adapt the phase list and state file.

This prompt defined the supervisor workflow for the MENoBiS microcanonical
refactor: delegating to specialist subagents using the `subagent` tool with
`agentScope: "both"`. The workflow structure below is the historical
procedure as executed.

## Phase 0: Context recovery

Before any delegation:

1. If `_workspace/refactor-state.md` exists, read it to recover current state
2. If it does not exist, create it by inspecting the git branch `microcanonical-refactor` and the spec document
3. Load the orchestration skill: read `.pi/skills/refactor-orchestrator/SKILL.md` for detailed phase procedures
4. Read `docs/development/agent-specifications/microcanonical_implementation/11_microcanonical_refactor_practical_final.md` spec (or relevant section)

## Phase 1: Assess

1. Read the current phase state from `_workspace/refactor-state.md`
2. Determine what remains for the current phase
3. Identify the single most actionable bounded task
4. If no phase is in progress, select the first incomplete phase

> Historical note: during the refactor, the incomplete-phase queue was
> B → C → D → E → F → H. All of these are now COMPLETE.

## Phase 2: Deploy

For each bounded task within a phase:

### Step 1: Analysis
Delegate to `architectural-analyst` (single mode, `agentScope: "both"`):
```text
task: "Analyze {specific module/area} against spec §{section}.
       Identify: affected files, current abstractions, obsolete code,
       scalability concerns, migration order, affected tests.
       Write to _workspace/{phase}-{task}-analysis.md"
```

### Step 2: Plan
Read the analyst output. Write a concrete bounded implementation plan.
Record the plan in `_workspace/{phase}-{task}-plan.md`.
Do NOT proceed if the analysis reveals unexpected complexity — report to user.

### Step 3: Implement
Delegate to `implementation-agent` (single mode):
```text
task: "Implement {specific change} on branch refactor/{phase}-{description}.
       Spec reference: §{section}. 
       Files to modify: {paths}.
       Acceptance criteria: {list}.
       Run: cargo fmt --all --check, cargo clippy -p menobis-core --lib --tests,
            cargo test -p menobis-core relevant tests."
```

### Step 4: Test
Delegate to `testing-agent`:
```text
task: "Add tests for {implementation}. 
       Run fast tests and report results.
       Files affected: {paths}."
```

### Step 5: Semantic review
Delegate to `semantic-reviewer`:
```text
task: "Review {implementation} for scientific correctness.
       Spec reference: §{section}.
       Implementation files: {paths}.
       Tests: {test paths}."
```

### Step 6: Integration review
Delegate to `integration-reviewer`:
```text
task: "Review {implementation} for architectural consistency,
       duplication, API regressions, scalability, dead code.
       Files changed: {paths}."
```

### Step 7: Gate decision
Read all HANDOFF sections and review reports.
Decide: PASS or FAIL.

- **PASS**: Merge subbranch into `microcanonical-refactor`. Update `_workspace/refactor-state.md`. Proceed to next task/phase.
- **FAIL**: Write specific issues to `_workspace/{phase}-{task}-repair.md`. Delegate bounded repair iteration. Re-review after fix.

## Phase 3: Report

After each gate decision, write a checkpoint report:
- Phase and task completed
- Subbranch merged
- Tests passing
- Evidence paths
- Next actionable phase/task
- Significant unresolved risks

## Error handling

- Subagent failure (tool error, not review failure): Retry once. If fails again, report to user.
- Critical review finding: Stop the delegating chain. Assess whether to abort phase or repair.
- Chain step failure: Read partial results from `_workspace/`. Decide whether to continue from next step or restart.

## Harness validation test

Use this command to verify project-local agents are discoverable:
```bash
cd {project_root} && pi -e 'use subagent to discover all project agents and list them'
```

## Git branches

- Integration: `microcanonical-refactor` (all phases merged)
- Subbranches: `refactor/<phase>-<description>` (e.g., `refactor/B-occupied-cell-mcmc`)
- Do NOT commit directly to `microcanonical-refactor`
- Use `/skill:commit` for all git commits (Conventional Commits)
- Do NOT merge without gate passing

## Test run commands

```bash
# Fast Rust tests
cargo test -p menobis-core 2>&1 | tail -5

# Clippy
cargo clippy -p menobis-core --lib --tests -- -D warnings 2>&1

# Format check
cargo fmt --all -- --check 2>&1

# Heavy oracle tests
cargo test -p menobis-test-oracles 2>&1 | tail -5

# Python tests (fast suite)
uv run pytest 2>&1 | tail -5
```

## Human-visible checkpoints

Before starting each major phase, report:
- What phase is about to begin
- Current repository/branch state
- Intended changes and files
- Agents that will be delegated
- Acceptance tests/gate criteria
- Significant unresolved risks

Do NOT silently make an architectural decision that contradicts or materially extends the specification. If in doubt, stop and present the choice to the user.
