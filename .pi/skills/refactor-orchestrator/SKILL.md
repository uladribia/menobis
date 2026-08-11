---
name: refactor-orchestrator
description: "Historical orchestration procedure for the MENoBiS microcanonical refactor (COMPLETE). Documents the phase-by-phase delegation, acceptance gates, and supervisor workflow that were used. Retained as reference for future multi-phase refactors."
---

# MENoBiS Microcanonical Refactor — Orchestration Procedure

> ⚠️ **The microcanonical refactor is COMPLETE.** This skill documents the
> orchestration procedure used. It is retained as reference for future
> multi-phase refactors.

This skill defined the procedure for the supervisor (`supervisor` agent) to
orchestrate the microcanonical refactor. It is now a **historical record**:
the delegation patterns (producer-reviewer chain, parallel analysis, state
management) remain useful templates for future multi-phase work, while all
phase-status content below is retrospective.

## References

- **Authoritative spec**: `docs/development/agent-specifications/microcanonical_implementation/11_microcanonical_refactor_practical_final.md`
- **Phase state**: `_workspace/refactor-state.md` (historical; all phases complete)
- **Baseline inventory**: `docs/development/agent-specifications/baseline-sha.txt`
- **Integration branch**: `microcanonical-refactor` (all phases merged)

## Completed phases (from spec §36)

All phases of the refactor were executed and merged into
`microcanonical-refactor`:

| Phase | Status | Merge commit |
|-------|--------|--------------|
| A — Freeze oracle baselines | ✅ DONE | Pre-refactor |
| B — Replace fixed-strength MCMC kernel | ✅ DONE | `8b588c4` |
| C — Add compressed fixed-strength constructor | ✅ DONE | `ec8d60f` |
| D — Add targeted loop repair | ✅ DONE | `8d5961b` |
| E — Add simple B/mask repair | ✅ DONE | `5aae72c` |
| F — Simplify strength-cost fitting | ✅ DONE | `e6dab9a` |
| G — Finish fixed-total Gibbs migration | ✅ DONE | `ae8a2c3` |
| H — Cleanup | ✅ DONE | `a19bce8` |
| §35 Benchmark matrix | ✅ DONE | `5de34ea` |
| Docs alignment | ✅ DONE | `c9aef40` |

The §40 completion gate passed 21/21.

## Phase execution order (historical)

Phases were executed in order B → C → D → E → F → H, with A and G completed
before the orchestration skill was finalized. Each phase gate had to pass
before the next phase started.

## Per-phase record

### Phase A: Freeze oracle baselines

**Spec reference**: §8–10 (oracle crate), baseline inventory

**What was done**: Created `menobis-test-oracles` with exact enumeration and
legacy comparison backends; moved legacy exact fixed-(E,T) rejection/DP code
into the oracle crate; established the baseline inventory
(`baseline-sha.txt`) and the GATE 1–5 audit protocol.

---

### Phase B: Replace fixed-strength MCMC kernel

**Spec reference**: §22–24 (occupied-cell proposal), §23 (Hastings correction), §24 (per-proposal allocation)

**What was done**: Replaced the uniform-coordinate cycle4 kernel in
`occupation_mcmc/move_cycle.rs` with an occupied-cell proposal (selects two
distinct occupied pairs) plus an exact Hastings correction for the
state-dependent selection probability. The legacy uniform-coordinate kernel
was preserved in `menobis-test-oracles` as a reference oracle.

---

### Phase C: Add compressed fixed-strength constructor

**Spec reference**: §11–13 (compressed aggregated matching), §10 (StrengthState), §34 (performance targets)

**What was done**: Added a compressed aggregated fixed-strength constructor
(aggregated residual matching with `remaining_out[i]` / `remaining_in[j]`
active collections, no O(T) stub expansion, no O(N²) pair enumeration) and
moved the legacy `greedy_complete` constructor to the oracle crate.

---

### Phase D: Add targeted loop repair

**Spec reference**: §14–17 (guaranteed ME/W loop repair), §19 (masks), §20 (targeted repair), §21 (failure semantics), §32 (validation)

**What was done**: Implemented guaranteed loop repair for complete loopless
ME/W domains using rectangle transactions and occupied-pair donor selection,
with a loopless feasibility check (`s_i_out + s_i_in <= T`) and bounded
repair configuration (max_steps, max_restarts, failure reporting).
Max-flow routing was removed from the ME/W production path.

---

### Phase E: Add simple B/mask repair

**Spec reference**: §18 (B repair), §19 (masks), §21 (failure semantics), §33 (validation)

**What was done**: Implemented capacity-aware rectangle repair for B
families and forbidden-pair rectangle repair for arbitrary masks, with
bounded randomized restarts. No annealing was added. Failures are reported
via `RepairDidNotConverge` with diagnostics.

---

### Phase F: Simplify strength-cost fitting

**Spec reference**: §25–27 (gamma fitting), §26 (zero-centered bracket expansion)

**What was done**: Removed the fragile variance-based warm start, implemented
zero-centered bracket expansion for the gamma search, and required
sufficient accepted transitions / ESS before accepting a gamma fit.
Nonconverged fits raise `InsufficientMobility`.

---

### Phase G: Finish fixed-total Gibbs migration

**Spec reference**: §14, §28 (pair-Gibbs chain)

**What was done**: Migrated fixed-(E,T) sampling to the shared pair-Gibbs
chain (O(E) memory) and removed the legacy DP/rejection backends from
production (kept in `menobis-test-oracles`).

---

### Phase H: Cleanup

**Spec reference**: §36 Phase H, §40 completion gate

**What was done**: Removed max-flow production routing, the explicit stub
production initializer, the old uniform-coordinate strength kernel, exact
DP/rejection production routing, obsolete gamma warm-start code, migration
flags, duplicated errors, and dead configuration. The §40 completion gate
passed 21/21, including the §35 benchmark matrix (`5de34ea`) and final docs
alignment (`c9aef40`).

## Delegation patterns

### Producer-reviewer chain (single task)

```
subagent(agent: "architectural-analyst", task: "<bounded analysis task>")
    → supervisor reviews output, writes plan
    → subagent(agent: "implementation-agent", task: "<bounded implementation>")
    → subagent(agent: "testing-agent", task: "<test implementation>")
    → subagent(agent: "semantic-reviewer", task: "<review semantic correctness>")
    → subagent(agent: "integration-reviewer", task: "<review architecture>")
    → supervisor: gate decision
```

### Parallel analysis (independent concerns)

```
subagent(parallel: [
    { agent: "architectural-analyst", task: "..." },
    { agent: "semantic-reviewer", task: "review <specific concern>" }
])
    → supervisor reads both
```

### Chain for implement-and-review

```
subagent(chain: [
    { agent: "implementation-agent", task: "..." },
    { agent: "testing-agent", task: "test: {previous}" },
    { agent: "semantic-reviewer", task: "review: {previous}" }
])
```

## State management

- `_workspace/refactor-state.md` was read at every session start and updated
  after each gate passed; it now records ALL phases as COMPLETE.
- Phase evidence was written to `_workspace/<phase>-gate-evidence.md`
  (historical artifacts remain in `_workspace/`).
- If state files are missing, reconstruct from git log (`git log --first-parent`)
  and the spec inventory.
