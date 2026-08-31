# STATUS — Microcanonical Fixed-(s,k): STOPPED at the N=1000 Degree-Repair Gate

> **For the next agent.** This is the review entry point for the
> `feature/microcanonical-fixed-strength-degree` branch. Read this,
> then the spec and the decision record, then the code. Do not assume
> the feature is usable at scale — the initialization repair cannot
> reach the exact degree vector at N ≥ ~30 under the plan's mandated policy.

| | |
|---|---|
| Branch | `feature/microcanonical-fixed-strength-degree` |
| Spec | `agent-specifications/MENoBiS_fixed_sk_implementation_plan_v2.md` |
| Decision record | `docs/decisions/microcanonical-fixed-sk-stop.md` |
| Exact-math oracle | `crates/menobis-test-oracles/tests/fixed_strength_degree_enumeration.rs` |
| N=1000 gate | `crates/menobis-test-oracles/tests/fixed_strength_degree_scalability.rs` |
| Feature status | **STOPPED (algorithmic limitation), never shipped** |

## ⬆ Recovery update (current work — read this first if exploring the repo today)

A recovery effort replaced the initialization approach on a dedicated
branch:

```text
branch: fix/fixed-sk-direct-init-trace-gate   (based on feature/microcanonical-fixed-strength-degree)
plan:   agent-specifications/MENoBiS_fixed_sk_recovery_direct_init_trace_gate.md
```

**Gate A — stationary trace from an exact `(s,k)` state: ✅ VIABLE**
(`docs/decisions/microcanonical-fixed-sk-trace-mobility.md`).  Started on an
already-exact witness, the first-return degree trace is practically mobile at
N=1000 for realistic PA-geographic instances (~3% different/support returns
per trace, ~34 `K_E` per effective return, ~0.2% timeouts).  Uniform-corner
witnesses (zero occupation-1 edges) are immobile — a start-state constraint,
not a kernel defect.

**Gate B — direct exact `(s,k)` constructor: ⛔ blocked at N=1000 for
heterogeneous residuals** (`docs/decisions/microcanonical-fixed-sk-direct-init.md`).
The §13–§22 pipeline (exact-k support → occupation 1 → residual allocation)
works at tiny scale and for uniform-occupation instances (the §28 stress grid
passes), but for realistic heterogeneous residuals **no** generic exact-k
support admits the residual-strengths transport (0/400 k-greedy, 0/64
c-aware, 0/64 proportional-random; verified against the legacy max-flow
oracle).  The extras transport is co-joint: only the witness-like support
carries the required strength-column correlation.  The full-domain extras
transport is feasible and spare (1752 edges, 0 rows over k) — the missing
piece is expressing it on a k-exact support (see the decision record's
§5 options 1–4).

**What works today:** everything up to Gate B's constructor integration — the
trace kernel, the tiny/uniform constructor, fixed pairs, B capacity, and all
prior exact-math gates (see §Regression below).  Phase 8 integration (direct
init into the one-shot sampler) and everything after are intentionally NOT
done: Phase 7 is red.

## TL;DR

Fixed strengths + fixed degrees (`Constraint.STRENGTH_DEGREE`,
microcanonical, ME/B/W) were implemented as a **capped first-return trace
of a degree-distance-biased auxiliary chain whose proposal is the whole
finished fixed-(s,E) kernel `K_E`** — `pi_(s,k) = pi_(s,E)(· | k = k_target)`.
All **mathematical gates pass** (exact `Q`/`R` trace-matrix oracle,
production-vs-exact correspondence, tiny-fiber repair, tiny-N end-to-end).
The **N=1000 gate fails**: the initialization degree repair (the same
degree-biased auxiliary step, `λ = 1.0`, per plan §15/§24) floors at a
strictly positive degree distance that scales ~O(N), so a randomized
exact-E start cannot be brought to the exact degree vector at scale.
Per plan §43.1/§52 and the review decision, the policy is **STOP and
report** — no blind budget raises, no new move family, no second repair
policy, no capability exposure.

## What is on the branch (all green)

| Piece | File (menobis-core `.../occupation_mcmc/`) | Phase |
|---|---|---|
| `ResidualDegreeTarget` + fixed-pair degree subtraction + combined (s,k) validation | `fixed_degrees.rs` | 3 |
| O(1) degree before/after metadata on every 4-cycle proposal | `move_cycle.rs` | 4 |
| Recordable `K_E` (undo-exact flat log, unrecorded path unchanged) | `fixed_edges.rs` | 5 |
| Exact degree oracle: auxiliary matrix `Q`, capped first-return trace matrix `R`, DB/stationarity/connectivity | test-oracles `fixed_strength_degree_enumeration.rs` | 2 |
| `degree_auxiliary_step` (one recorded `K_E` + `min(1, e^{−λΔD})`) | `fixed_degrees.rs` | 6 |
| Degree repair to exact degrees | `fixed_degrees.rs` | 7 |
| Capped first-return trace + sweep (primary kernel) | `fixed_degrees.rs` | 8 |
| One-shot `sample_fixed_strength_degree(_bench)` + O(E) full-invariant validation | `chain.rs` | 9 |
| STOP artifact tests + N-floor characterization | test-oracles `fixed_strength_degree_scalability.rs` | — |

Checks: `cargo test --workspace` **447 passed**; clippy `-D warnings` clean;
`cargo fmt --check` clean; all oracles + heavy fixed-sE N=1000/N=5000 pass.

## The exact math (why the stationary kernel is correct)

- `E_target = Σ k_out`; the degree fiber `A_k ⊂ Ω_E`.
- `pi_(s,k) = pi_(s,E)(· | A_k)`.
- `K_E` reversible for `pi_(s,E)` ⇒ outer degree MH ratio collapses to
  `min(1, exp(−λ·(D(y)−D(x))))` with `D = ½·(Σ|k_out−k*| + Σ|k_in−k*|)` — no
  internal Hastings/bridge recomputation.
- The production kernel is the capped first-return trace of the
  degree-biased auxiliary chain onto `A_k`; timeout restores the origin
  (exact self-loop). Proven numerically: `Q` and `R` satisfy row sums,
  detailed balance, and stationarity on every enumerated tiny fiber; the
  trace connects every underlying-connected fiber at cap 16.
- `K_E` is invoked through an exact recorder so a rejected outer step
  (or a timed-out trace) deterministically undoes the whole excursion
  from a flat `Vec<Cycle4Proposal>` — no state clones.

## The blocker (read `docs/decisions/microcanonical-fixed-sk-stop.md`)

The degree **repair** reuses `degree_auxiliary_step` (plan §15.2, §24).
Probed trajectory (d=8, occ 1..3, λ=1.0, 600k steps):

```text
N=30   initial D=74   floor D=21
N=100  initial D=246  floor D=79
N=200  initial D=514  floor D=154
N=1000 (5M steps × 5 restarts) best D=1791 -> DegreeRepairExhausted
```

λ ∈ {0.5, 1.0, 2.0} all floor; the underlying K_E mobility is healthy
(fixed-sE N=1000 gate passes), so the limitation is the soft
degree-potential descent over the astronomically large exact-E fiber —
a genuine mixing barrier, not a bug. The oracle, correspondence
`P(s1→s2) = 0.475`, tiny-fiber repair, and tiny-N E2E all pass, so the
stationary kernel machinery is sound; **initialization cannot scale**.

## How to verify

```bash
# fast suite (includes the exact degree oracle + repair + trace tests)
cargo test --workspace

# exact (s,k) oracle — the mathematical release gate
cargo test -p menobis-test-oracles --test fixed_strength_degree_enumeration

# pinned STOP artifacts (N=1000 repair exhaustion, N-floor scaling)
cargo test -p menobis-test-oracles --test fixed_strength_degree_scalability -- --ignored --nocapture

# fixed-sE regression (unrecorded K_E path unchanged)
cargo test -p menobis-test-oracles --test fixed_strength_edges_enumeration --test fixed_strength_edges_scalability -- --include-ignored

# static gates
cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all -- --check
```

## Recommended next steps (outside this task's policy)

1. **Constructive degree-obeying initializer** (most promising): build a
   start already on `A_k` (e.g. flow-based or sequential Bernoulli+repair
   honoring `k` before assigning weights), removing the descent entirely.
2. Non-MH greedy/annealed repair (fixed-sE-edge-repair style) — may still
   hit the `D>0` floor; needs explicit approval (plan §52 forbids it in
   the original task).
3. Degree-preserving move-family augmentation — a new proposal law with a
   fresh path-Hastings derivation (explicitly out of the original scope).

Phases 10–13 (sampling-plan priority correction + visible-error guard for
silently-ignored strengths in the factorized router, pyo3 binding, Python
routing, capability exposure) were **not done** — capability must not be
exposed until the scalability gate passes. If the initializer approach
succeeds, pick up at Phase 10 with the router guard landed atomically
with the `SamplingPlan` priority change.