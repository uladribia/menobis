---
name: refactor-orchestrator
description: "Detailed orchestration procedure for the MENoBiS microcanonical refactor. Defines phase-by-phase delegation, acceptance gates, and the supervisor workflow. Use /skill:refactor-orchestrator to load the full orchestration specification when starting or continuing a refactor session."
---

# MENoBiS Microcanonical Refactor — Orchestration Procedure

This skill defines the detailed procedure for the supervisor (`supervisor` agent) to orchestrate the refactor workflow. Read it at the start of each session and whenever transitioning between phases.

## References

- **Authoritative spec**: `docs/development/agent-specifications/microcanonical_implementation/11_microcanonical_refactor_practical_final.md`
- **Phase state**: `_workspace/refactor-state.md`
- **Baseline inventory**: `docs/development/agent-specifications/baseline-sha.txt`
- **Integration branch**: `microcanonical-refactor` (already exists with Phase 1-5, 8 work)

## Remaining phases (from spec §36)

Current state of the `microcanonical-refactor` branch:

| Phase | Status | Notes |
|-------|--------|-------|
| A — Freeze oracle baselines | ✅ DONE | menobis-test-oracles crate exists |
| B — Replace fixed-strength MCMC kernel | 🔶 PARTIAL | Heap alloc removed; occupied-cell proposal NOT implemented |
| C — Add compressed fixed-strength constructor | 🔶 PARTIAL | Greedy fill exists; may still use stub matching |
| D — Add targeted loop repair | ❌ NOT STARTED | |
| E — Add simple B/mask repair | ❌ NOT STARTED | |
| F — Simplify strength-cost fitting | ❌ NOT STARTED | |
| G — Finish fixed-total Gibbs migration | ✅ DONE | pair-Gibbs chain exists; legacy DP removed |
| H — Cleanup | ❌ NOT STARTED | |

## Phase execution order

Execute phases in order B → C → D → E → F → H.

Do NOT start Phase C before B gates, D before C, etc.

Phase A and G are already complete — skip them.

## Per-phase procedure

### Phase B: Replace fixed-strength MCMC kernel

**Spec reference**: §22–24 (occupied-cell proposal), §23 (Hastings correction), §24 (per-proposal allocation)

**Current state**: The `occupation_mcmc/move_cycle.rs` file exists with the old uniform-coordinate cycle kernel. Per-proposal heap allocation has been removed (commit 84a552b), but the proposal is still uniform-coordinate (selects from full N² domain).

**What needs to happen**:
1. Architectural analysis: inspect `move_cycle.rs`, `chain.rs`, `target.rs`, `state.rs` in `occupation_mcmc/`
2. Implement occupied-cell proposal: select two distinct occupied pairs instead of two random coordinates
3. Implement exact Hastings correction for state-dependent proposal probabilities
4. Keep old kernel available only in heavy oracle tests (temporarily)
5. Validate against exact enumeration and old kernel
6. Remove old kernel from production

**Acceptance gate**:
- [ ] Occupied-cell proposal implemented in `move_cycle.rs`
- [ ] Hastings correction accounts for selection probability
- [ ] Cargo test passes (fast tests)
- [ ] Clippy clean with no warnings
- [ ] Old kernel is oracle-only or removed from production
- [ ] Benchmarks show equal or better proposal acceptance

**Delegation plan**:
1. `architectural-analyst` — analyze current `occupation_mcmc/` vs spec §22-24
2. Supervisor — review analysis, write bounded plan
3. `implementation-agent` — implement occupied-cell proposal + Hastings
4. `testing-agent` — add tests, run fast suite
5. `semantic-reviewer` — verify Hastings ratio and Markov-chain correctness
6. `integration-reviewer` — check architecture, dead code, scalability
7. Supervisor — gate decision

---

### Phase C: Add compressed fixed-strength constructor

**Spec reference**: §11–13 (compressed aggregated matching), §10 (StrengthState), §34 (performance targets)

**Current state**: An initializer exists at `occupation_mcmc/initializer.rs`. Recent Phase 8 work added a "greedy fill" approach and stub-matching router.

**What needs to happen**:
1. Architectural analysis: inspect `initializer.rs`, `route.rs`, `state.rs`
2. Implement aggregated residual matching: maintain `remaining_out[i]` and `remaining_in[j]` with active collections
3. Output must be a `StrengthState` with HashMap-based sparse storage (§10)
4. Must avoid O(T) explicit stub expansion
5. Must avoid O(N²) pair enumeration
6. Validate against old constructor
7. Make old constructor oracle-only

**Acceptance gate**:
- [ ] Constructor uses aggregated residual matching (no stubs, no N²)
- [ ] Output satisfies strengths exactly
- [ ] Storage is O(N + E_occ) not O(N²) or O(T)
- [ ] Cargo test + clippy pass
- [ ] Benchmarks show O(N + E_occ) scaling
- [ ] Old constructor is oracle-only

**Delegation plan**: Same B→C→D→E sequence.

---

### Phase D: Add targeted loop repair

**Spec reference**: §14–17 (guaranteed ME/W loop repair), §19 (masks), §20 (targeted repair), §21 (failure semantics), §32 (validation)

**What needs to happen**:
1. Architectural analysis: inspect current repair/initialization path
2. Implement `repair_self_loops()` using rectangle transaction (§15)
3. Implement efficient donor selection via occupied-pair random attempts (§17)
4. Add loopless feasibility check: `s_i_out + s_i_in <= T` for all i (§14)
5. Add repair config with max_steps, max_restarts, failure reporting (§21)
6. Route complete loopless ME/W through: construction → loop repair → MCMC
7. Validate against heavy max-flow oracle on small systems
8. Remove max-flow routing from ME/W production path

**Acceptance gate**:
- [ ] Loop repair eliminates all self-loops for feasible ME/W
- [ ] Repair terminates deterministically (proven case)
- [ ] Loop mass decreases monotonically
- [ ] No O(N²) domain allocation in repair path
- [ ] Infeasible cases produce clear error, not silent wrong answer
- [ ] Validation against max flow oracle passes
- [ ] Benchmarks show fast repair (<1% of total runtime)

---

### Phase E: Add simple B/mask repair

**Spec reference**: §18 (B repair), §19 (masks), §21 (failure semantics), §33 (validation)

**What needs to happen**:
1. Implement capacity-aware rectangle repair for B
2. Implement forbidden-pair rectangle repair for arbitrary masks
3. Add bounded randomized restarts (§18)
4. Do NOT add annealing initially
5. Validate against heavy feasibility oracle (§33)

**Acceptance gate**:
- [ ] B capacity repair works and detects stalling
- [ ] Mask repair works for simple cases (diagonal, small forbidden sets)
- [ ] Bounded restarts prevent infinite loops
- [ ] Failure reported via `RepairDidNotConverge` with diagnostics
- [ ] No annealing or generic flow solver added to production
- [ ] Validation against heavy oracle shows high success rate for feasible cases

---

### Phase F: Simplify strength-cost fitting

**Spec reference**: §25–27 (gamma fitting), §26 (zero-centered bracket expansion)

**What needs to happen**:
1. Remove fragile variance-based warm start
2. Implement zero-centered bracket expansion (§26)
3. Require sufficient accepted transitions / ESS before accepting gamma fit (§27)
4. Add `InsufficientMobility` error for nonconverged fits

**Acceptance gate**:
- [ ] Variance warm start removed from production
- [ ] Zero-centered bracket expansion implemented
- [ ] Gamma fitting requires real movement evidence
- [ ] Insufficient mobility causes clear error
- [ ] Benchmark shows similar or faster gamma convergence

---

### Phase H: Cleanup

**Spec reference**: §36 Phase H, §40 completion gate

**What needs to happen**:
- Remove max-flow production routing
- Remove complete-domain pair materialization
- Remove explicit stub production initializer
- Remove old uniform-coordinate strength kernel
- Remove exact DP/rejection production routing
- Remove obsolete gamma warm-start code
- Remove migration flags, duplicated errors, dead configuration
- Run full test suite: Rust tests, Python tests, benchmarks, heavy oracle

**Acceptance gate** (§40):
- [ ] No permanent migration flags
- [ ] No duplicate family or cost formulas
- [ ] No unnecessary generic framework
- [ ] Production code is smaller than before the refactor
- [ ] All fast tests pass
- [ ] All Python tests pass (fast suite)
- [ ] Benchmarks pass at N=100, 500, 1000

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

- Always read `_workspace/refactor-state.md` at session start
- Update state after each gate passes
- Write phase evidence to `_workspace/<phase>-gate-evidence.md`
- If phase state file is missing, reconstruct from git log and spec inventory