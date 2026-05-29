---
description: Known solver convergence issues and proposed fixes.
---

# Solver Convergence Issues

## TL;DR

Three convergence problems affect the benchmark under realistic conditions.
All are algorithm-level issues in the Rust solvers, not missing plumbing.

## Issue 1: W strength-edges / strength-degree (all regimes)

**Symptom**: Newton/L-BFGS solver hits max iterations without converging for
heterogeneous PA-geographic inputs at N >= 20.

**Root cause**: The W (geometric/negative-binomial) parametrization requires
`q_ij = x_i * y_j < 1` strictly. With heterogeneous inputs, some pairs want
`q -> 1` (high-weight edges). The Newton step overshoots the boundary, and the
current damping strategy (`damping *= 0.5` on stall) is too conservative to
recover.

**Proposed fix**:
1. **Barrier projection**: Add a log-barrier penalty `- epsilon * sum(log(1 - q_ij))`
   to the objective, keeping all iterates strictly interior.
2. **Adaptive damping with Armijo backtracking**: Replace fixed damping reduction
   with proper line search that respects the `q < 1` constraint.
3. **Warm-start from ME solution**: Initialize W multipliers from the converged
   ME fit (which always converges), transformed via `q_ME = 1 - exp(-r_W)`.

**Impact**: W strength-edges and W strength-degree fits report `converged=False`
but still produce usable (approximate) multipliers. Sampling and filtering work
on the non-converged result with degraded precision.

## Issue 2: ME/B strength-edges (sparse, near-binary regime)

**Symptom**: L-BFGS oscillates without converging when `events_per_edge` is low
(strength/degree ratio near 1.0). Hits 50000 iterations.

**Root cause**: When most edges have weight 1, the zero-inflated model enters a
degenerate regime: `q_ij -> 0` (to produce small expected weight) while
`lambda -> infinity` (to force high occupation). The L-BFGS objective has an
extremely narrow valley in this configuration, causing slow progress with
oscillating step sizes.

**Proposed fix**:
1. **Better initialization**: Start lambda from the analytical ratio
   `E / sum(1 - exp(-q))` which is the maximum-likelihood estimate for the
   homogeneous case.
2. **Scaled variables**: Optimize `log(lambda)` instead of `lambda` to avoid
   the narrow-valley geometry.
3. **Separate the sparse case**: When `mean(s/k) < 2.0`, use a dedicated
   solver that parametrizes via occupation probability directly rather than
   the ZI factorization.

**Impact**: Affects the "sparse" benchmark regime. The "saturated" regime
(higher events_per_edge) converges reliably.

## Issue 3: B strength-edges (sparse, 5% partial)

**Symptom**: Partial B strength-edges with 5% known pairs at N=100 hits 50k
iterations while 0% and 20% converge.

**Root cause**: At 5% known pairs, the excess constraints are almost as hard as
the full problem (only a few pairs removed). The excess computation can produce
slightly imbalanced sequences that make the ZI landscape even flatter. At 20%,
enough pairs are frozen to significantly simplify the free subproblem.

**Proposed fix**: Same as Issue 2 (better initialization for the ZI lambda
search). Additionally, warm-start the partial solver from the full-fit solution
when available.

## Summary table

| Case | Regime | Converges? | Proposed fix |
|---|---|---|---|
| W str-edges | all | rarely | barrier + adaptive damping |
| W str-degree | all | rarely | barrier + adaptive damping |
| ME str-edges | sparse (epe < 3) | sometimes | scaled log-lambda + better init |
| B str-edges | sparse (epe < 3) | sometimes | scaled log-lambda + better init |
| B str-edges 5% partial | sparse | rarely | warm-start from full fit |
| All others | all | reliably | — |
