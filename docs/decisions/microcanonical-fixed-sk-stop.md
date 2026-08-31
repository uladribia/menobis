# Decision — Fixed-(s,k) Feature Stopped at the N=1000 Degree-Repair Gate

**Status:** stopped (algorithmic limitation, plan §43.1 policy)
**Branch:** `feature/microcanonical-fixed-strength-degree`
**Spec:** `../development/agent-specifications/MENoBiS_fixed_sk_implementation_plan_v2.md`

## 1. Background

The fixed-(s,k) plan implements exact microcanonical strength + degree
sampling by treating the finished fixed-(s,E) kernel `K_E` as the
reversible proposal of a degree-distance-biased auxiliary chain, then
taking a capped first-return trace onto the exact degree fiber
(`pi_(s,k) = pi_(s,E)(· | k = k_target)`).  The degree fiber is reached
at initialization by a **degree repair** that reuses the same auxiliary
step with `λ = 1.0` (§15).

## 2. Evidence gathered

| Gate | Result |
|---|---|
| Exact degree auxiliary matrix `Q` (row sums, DB, stationarity) | ✅ pass |
| Exact capped first-return trace matrix `R` (row sums, DB, stationarity, connectivity on underlying-connected fibers) | ✅ pass |
| Production-vs-exact correspondence (ME N=2 s=[3,3], exact 0.475) | ✅ pass |
| Degree repair on every enumerated tiny fiber (N ≤ 3) | ✅ pass |
| N=1000 degree repair (E≈8000, T>E, sparse d=8) | ❌ **exhausts** |

The N=1000 failure is a `DegreeRepairExhausted` (best degree distance
1791 after 5×10⁶ auxiliary steps × 5 restarts, λ=1.0).

## 3. Inspection per §43.1 (STOP before increasing budgets)

The distance trajectory of the degree-biased walk shows a geometric tail
that **floors at strictly positive D**, with the floor growing roughly
linearly in N:

| N (d=8, occ 1..3) | initial D | floor D (600k steps, λ=1.0) |
|---|---|---|
| 30 | 74 | 21 |
| 100 | 246 | 79 |
| 200 | 514 | 154 |

The λ grid `{0.5, 1.0, 2.0}` all floor (λ=2 reaches 167 at N=200 in
180k steps, then fluctuates around ≈103–125 with 2M steps).  The
underlying `K_E` mobility is healthy (fixed-sE N=1000 gate passes), so
the limitation is the *soft degree-potential walk* itself: from a
randomized exact-E start over an astronomically large fiber, local
support moves cannot descend to the exact target degree vector — every
decreasing direction is blocked by locally-increasing moves, a genuine
mixing barrier.

Per §43.1/§52 the policy is **STOP and report** — not blind budget
increases, not new move families, not a second repair policy (the plan
explicitly forbids those in this task).

## 4. What is and is not delivered

Delivered (all green, on the feature branch):

- `ResidualDegreeTarget` residualization + combined (s,k) validation;
- O(1) per-cycle degree metadata; recordable `K_E` (undo-exact);
- `degree_auxiliary_step` (exact outer MH over `K_E`);
- degree repair + capped first-return trace (exact kernel objects);
- one-shot `sample_fixed_strength_degree` orchestrator with full
  runtime invariant validation;
- the exact `Q`/`R` oracle, tiny-E2E gates, and the pinned STOP
  artifact tests.

NOT delivered: production fixed-(s,k) sampling at N ≥ ~30.  The
stationary trace kernel is exact (oracle-proven), but **initialization
repair cannot scale** under the plan's mandated policy, so the feature
cannot ship.

## 5. Recommended next steps (outside this task's policy)

1. A non-MH initialization-repair policy — e.g. the edge-repair-style
   greedy/annealed descent used by fixed-(s,E) (accept `D_new < D_old`
   always, keep on equality, `exp(−c·ΔD)` otherwise) with restarts —
   noting §52 forbids it *in this task* and it still may not escape the
   `D>0` floors;
2. a constructive degree-obeying initializer (dense/flow-based) that
   produces a start already on the fiber, circumventing repair;
3. a degree-preserving augmentation of the 4-cycle move class — a new
   proposal law requiring a fresh path-Hastings derivation (explicitly
   out of scope per §52).