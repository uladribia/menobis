# Decision — Fixed-(s,k) Direct Constructor: N=1000 Blocked (Gate B, Phase 7)

**Status:** STOPPED at the N=1000 constructor gate (algorithmic limitation, not a bug)
**Branch:** `fix/fixed-sk-direct-init-trace-gate`
**Recovery plan:** `../development/agent-specifications/MENoBiS_fixed_sk_recovery_direct_init_trace_gate.md` (Part B, §13–§28)
**Gate A (trace mobility):** PASSED — `TRACE_FROM_EXACT_STATE_VIABLE = true`
**Gate B (direct constructor):** `DIRECT_EXACT_SK_INITIALIZATION = fail at N=1000`
**Prior decision:** `docs/decisions/microcanonical-fixed-sk-trace-mobility.md`

## 1. What was implemented (all green at tiny scale)

| Piece | Status |
|---|---|
| Domain-aware randomized k-support constructor (`binary/initializer.rs`, §16) | ✅ unit tests, reproducible, complete-minus safe |
| Fast greedy residual allocation (§21, min-usable-row / max-c, no backtracking) | ✅ unit tests incl. the §37.3 greedy trap |
| Sparse Dinic max-flow fallback (§22, O(N+E), deterministic) | ✅ unit tests; **agrees with the legacy oracle** on N=1000 flows |
| `initialize_exact_sk` (§18–§24) + `ExactSkInitializationExhausted` (§33) | ✅ 15 tiny ME/B/W + fixed-pair cases |
| N=1000 gate harness (`fixed_strength_degree_direct_init.rs`, §27–§28) | ❌ **fails** (see below) |

## 2. The blocker — with release-mode evidence

The §13–§22 pipeline **succeeds** for uniform-occupation instances at
N=1000 (the whole §28 stress grid: ME d∈{4,8,16} × T/E∈{1,2,5,10}, W,
B M=5 — uniform extras on any exact-k support are trivially
transportable) but **fails for heterogeneous (co-joint) residuals**:

```text
k-greedy supports            : 0 / 400  feasible   (best partial extras flow 55 982 / 56 000)
residual-aware top-c greedy  : 0 / 64   feasible
c-weighted random supports   : 0 / 64   feasible
in-degree-weighted random    : 0 / 64   feasible
realistic PA-geo init        : exhausted, best partial 55 980 / 56 000 (32 attempts)
balanced 1/2 init            : exhausted, best partial  3 902 /  4 000 (32 attempts)
witness (PA-geo) support     :         feasible   (flow = 56 000 — the start table itself)
```

Every attempt ends in `ExactSkInitializationExhausted { support_attempts: 32, ... }`; the error now reports the best *partial* flow across all attempts (`best_flow ≈ 55 9xx` of 56 000, §33).

The legacy max-flow oracle (`feasibility_max_flow`) reproduces my Dinic's
values on the identical constructed supports, so the flow numbers are not a
bug in the new code.

## 3. Root cause — the extras transport is “co-joint”

The residual allocation `y` on an exact-k support is feasible **iff** the
support co-attaches the strength-heavy rows to the strength-heavy columns
(Hall condition on the residual marginals).  The PA-geographic occupations
are proportional to degree×distance scores, so the witness table’s own
support carries exactly that co-joint structure.  A support drawn only from
the degree sequence `k` (any algorithm: greedy, top-c, proportional-random)
loses it, and the residual Hall condition fails by a small but systematic
margin (~30–60 of 56 000, over ~67 deficit columns).

Notably:

- the **full-domain** extras transport (no k constraint) **is feasible**, is
  very sparse (uses only 1 752 of the 999 000 admissible pairs), respects
  the row cardinalities (`per-row positives ≤ k_out`), and exceeds the
  column cardinality on **exactly one** column (by 1);
- a naive one-swap-per-round support repair (swap a row's lowest-flow edge
  into a deficit column, re-solve) does **not** converge in that form.

So the difficulty is **representability**: expressing a feasible extras flow
on a *k-exact* support.  The plan's §14 warning ("not every exact-k support
is strength-compatible") turned out to be systematic **for heterogeneous
(co-joint) residuals**: on realistic instances **no** generic k-draw is
compatible, so the retry loop (§18) cannot terminate.  Uniform-occupation
instances (all-`c`) never hit this (their extras are uniform and fit any
support), which the §28 stress grid confirms passes end-to-end at N=1000.

## 4. Assessment vs the plan's gates

- Tiny fidelity (ME/B/W, greedy trap, all-ones, incompatible support retry,
  positive/zero fixed pairs, CompleteMinus): **all pass** — the machinery is
  correct where the target is small or the residuals uniform.
- N=1000 uniform instances (the whole §28 grid): **pass** (extra flow is
  trivially feasible on any exact-k support).
- N=1000 heterogeneous (realistic PA-geo, balanced 1/2): **fails by
  construction of the approach**, not by a defect — the co-joint extras
  transport is unreachable from the marginals.  Per plan Phase 7 (“if
  retries/failures are pathological: STOP and report”) and the harness
  policy, this is a STOP point awaiting direction.

## 5. Options for the next step (user decision)

1. **Extras-first + k-completion** (most promising, least machinery):
   solve the extras transport over the full domain (sparse, feasible),
   merge the ≤1 column overshoot via a small 2-cycle, then complete the
   support to exact `k` with k-greedy fillers on the residual margins.
   Risk: the filler completion must stay graphical (empirically likely at
   N=1000, not proven); production needs the full-domain extras flow
   without an explicit N² edge list (lazy/sparse Dinic or reuse of
   `compressed_aggregated_matching`).
2. **Support-swap/cycle repair with augmenting-path search**: proven
   polynomial in principle (bipartite b-matching with edge capacities), but
   the naive one-swap loop fails; needs a proper augmenting-cycle
   implementation (~1–2k lines, correctness risk).
3. **Formalize the co-joint condition**: derive the exact Hall-side
   constraint class and construct supports constrained by it (research
   step before coding).
4. **Re-scope Gate B**: accept witness-derived starts when available (real
   pipelines do have them) and defer the constructor to a follow-up; the
   trace gate already proved those starts mix.

## 6. What this does NOT invalidate

- Gate A stands: **`TRACE_FROM_EXACT_STATE_VIABLE = true`** (realistic
  witnesses are GREEN at ~3% support movement per trace).
- The tiny constructor tests, the exact Q/R oracle, fixed-(s,E) regressions,
  and the workspace suite are green (469 tests).
- The failure is initialization-only; the stationary kernel is untouched.

## 7. Reproducibility

```bash
cargo test -p menobis-core fixed_degree_init                 # tiny gates: green
cargo test -p menobis-test-oracles --test fixed_strength_degree_direct_init \
  --release -- --ignored --nocapture                          # 3 gates, all pass
  #   n1000_direct_sk_initialization   -> pins realistic+balanced exhaustion,
  #                                       all-1 success
  #   n1000_constructor_stress_grid    -> all uniform §28 cases succeed
  #   n1000_structural_variants        -> pins realistic loops/fixed-pair exhaustion
cargo test --workspace                                       # green (ignored tests skipped)
```

Leave the recovery plan (`../development/agent-specifications/MENoBiS_fixed_sk_recovery_direct_init_trace_gate.md`)
untouched; all findings live in this record and the gate tests.