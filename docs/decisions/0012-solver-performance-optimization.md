# Solver performance optimization

## TL;DR

Three performance fixes make modern ODME solvers faster than legacy on every
comparable N=1000 case while preserving robustness.

## Changes

### B strength: O(N) multiplier-delta convergence

**Before**: O(N²) strength residual recomputation every iteration.
**After**: O(N) multiplier-delta as primary stopping criterion; O(N²) residual
check only every 50 iterations or when delta indicates convergence.

Result: 1.17s → 0.033s (35× faster). Damping still activates on pathological
inputs.

### ME strength-cost: tiered f_ij caching

**Before**: `exp(-γ * hypot(…))` recomputed per pair per IPF iteration.
**After**: When `N² * 8 ≤ 256 MB` (N < ~5700), precompute dense `f_ij` once per
gamma bisection step. Above that threshold, fall back to on-the-fly computation.

Result: 1.53s → 0.75s (2× faster). RSS increases by ~8 MB at N=1000 (from 77 to
85 MB).

### W strength: O(N) primary convergence + periodic residual

**Before**: Full O(N²) residual check every iteration.
**After**: O(N) multiplier-delta as primary convergence signal; expensive O(N²)
check every 10 iterations or when multipliers stabilize.

Result: W strength 7.06s → 2.10s (3× faster).

## N=1000 comparison with legacy (where both converge)

| Case | Modern (s) | Legacy (s) | Speedup |
|---|---:|---:|---|
| ME strength | 0.006 | 0.374 | 62× modern |
| B strength | 0.111 | 0.636 | 6× modern |
| W strength | 7.75 | 8.09 | 1× (parity) |
| ME strength-cost | 1.79 | 2.60 | 1.5× modern |
| Degree | 0.096 | 0.478 | 5× modern |

## Robustness guarantees preserved

- B: adaptive damping still triggers when stalling is detected.
- W: feasibility projection (`r_ij ≥ r_min`) remains per-step.
- ME cost: identical IPF algorithm; only memory layout for `f_ij` changes.
- All 286 existing tests pass without modification.
