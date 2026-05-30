---
description: Comprehensive benchmark results across all families, constraints, and regimes.
---

# Benchmark Results

> TL;DR: **The dense regime (`average_degree = N/5`) is the optimal benchmark
> regime** — it exercises realistic connectivity without triggering pathological
> solver edge cases. ME and B converge at 100% across all constraints at N ≤ 1000.
> W remains broken for zero-inflated constraints (edges, degree).

## Regime analysis

MENoBiS benchmarks span three regimes to probe solver behaviour across the
connectivity spectrum:

| Regime | `average_degree` | `events_per_edge` | Character |
|---|---|---|---|
| Sparse | `3.0` | `3.0` | Low connectivity, `s ≈ k` |
| **Dense** | **`N / 5`** | **`8.0`** | **Moderate connectivity, no node saturates** |
| Saturated | `0.85 × (N - 1)` | `8.0` | Near-complete connectivity, `k ≈ N` |

### Why dense is optimal

- **Sparse** is pathological for degree constraints: when `s ≈ k` (each edge
  has weight ~1), the degree sequence carries no information beyond the
  strength sequence. The degree-constrained solver either converges to the
  same solution as the strength-only solver or struggles with an ill-posed
  problem. This makes sparse an unreliable benchmark for `strength-degree`
  and `strength-edges`.
- **Saturated** is pathological because `k ≈ N - 1` for every node. The degree
  constraint is saturated (each node is connected to nearly every other), and
  the binary support offers no structure for the ZI solvers to optimise
  against. Convergence is fast but the scenario is unrealistic.
- **Dense** (`average_degree = N/5`) gives each node roughly 20% connectivity
  — enough to have meaningful degree structure, but no node reaches N − 1.
  The preferential-attachment generator produces heterogeneous degrees
  (max ~50–60% of N−1 for N ≥ 100), exercising the solvers on a realistic,
  non-pathological input. **ME and B converge at 100%** across all
  constraint types at N ≤ 1000.

### Findings by regime

| Regime | ME convergence | B convergence | W convergence | Notes |
|---|---|---|---|---|
| Sparse | 100% | 94% | 50% | B degree-overflow at N=5000; W ZI fails |
| **Dense** | **100%** | **100%** | **50%** | **Optimal — all ME/B converge** |
| Saturated | 100% | 78% | 50% | B strength-cost IPF plateaus at N≥1000 |

## Convergence by family and constraint

### ME — 100% across all regimes and N ≤ 1000

| Constraint | N=100 | N=1000 |
|---|---|---|
| strength | 0.0002 s, 3 iters | 0.002 s, 3 iters |
| strength-cost | 0.012 s, 29 iters | 1.54 s, 37 iters |
| strength-edges | 0.038 s, 181 iters (11.4×) | 6.74 s, 570 iters (11.8×) |
| strength-degree | 0.40 s, 1754 iters (11.8×) | 61.03 s, 4410 iters (11.6×) |

### B — 100% across all constraints at N ≤ 1000 (dense regime)

| Constraint | N=100 | N=1000 |
|---|---|---|
| strength | 0.001 s, 21 iters | 0.07 s, 22 iters |
| strength-cost | 0.25 s, 31 iters | 30.22 s, 38 iters |
| strength-edges | 0.026 s, 136 iters (10.5×) | 4.06 s, 441 iters (11.4×) |
| strength-degree | 0.29 s, 1369 iters (11.3×) | 40.10 s, 3956 iters (11.6×) |

### W — Only strength and strength-cost converge

W strength-edges and strength-degree **never converge** at any N, regardless
of regime. The W geometric / negative binomial solver enters an unstable
regime for ZI constraints where `q` approaches the upper bound `1.0`.

| Constraint | N=100 | N=1000 |
|---|---|---|
| strength | 0.04 s, 22 iters | — |
| strength-cost | 8.37 s, 11474 iters | — |
| strength-edges | ✗ (max s_err ~ 10⁵) | ✗ |
| strength-degree | ✗ (max s_err ~ 10⁵) | ✗ |

### Memory footprint (dense, N=1000)

| Scenario | RSS (MB) | Notes |
|---|---|---|
| 0% known pairs, no self-loops | ~5–60 | Generation + fit |
| 2% known pairs, no self-loops | ~190 | Partial-fit lookup structures |
| 2% known pairs, self-loops | ~214 | Slightly more pairs to index |
| Sampling | +50 | Temporary edge-table allocation |

Low known-pair fraction has negligible memory impact. The 2% known-pair
spike comes from building the `(source, target) → rate` lookup hashmap for
all free pairs in the partial fit. In the worst case, this can scale as
`O(N²)` for the full-strength partial fit over all node pairs.

## Parallelism

| Solver | Constraints | Parallelism (CPU/wall) |
|---|---|---|
| IPF (sequential) | strength, strength-cost | 1.0× |
| L-BFGS (rayon) | strength-edges, strength-degree | 10–13× |

IPF is inherently sequential. L-BFGS parallelises the `O(N²)` gradient
computation across all cores (14 threads on Intel Ultra 5 125U).

## Partial fitting overhead

| Constraint | 0% kp | 2% kp | Slowdown |
|---|---|---|---|
| strength | 0.002 s | 1.86 s | ~900× (mask overhead) |
| strength-cost | 1.54 s | 11.64 s | ~7.5× |
| strength-edges | 6.74 s | 11.91 s | ~1.8× |
| strength-degree | 61.03 s | 65.18 s | ~1.1× |

Large overhead for non-ZI constraints at 2% kp is due to the IPF solver
iterating over a masked pair list instead of a compact dense array. The
L-BFGS solvers handle masks more efficiently.

## Self-loops comparison (dense, N=1000)

| | No self-loops | Self-loops |
|---|---|---|
| ME strength-cost | 1.54 s | 1.66 s |
| ME strength-edges | 6.74 s (11.8×) | 7.69 s (11.9×) |
| ME strength-degree | 61.03 s (11.6×) | 58.90 s (11.9×) |
| B strength-cost | 30.22 s | 24.24 s |
| B strength-edges | 4.06 s (11.4×) | 2.33 s (12.6×) |
| B strength-degree | 40.10 s (11.6×) | 26.66 s (11.9×) |

Self-loops adds `N` extra node pairs but reduces overall iteration count —
the presence of self-loop edges provides better initial conditions for the
IPF and L-BFGS solvers.

## N=5000 gap analysis

See `PLAN.md` for full gap table. Key gaps remain for B and W at N=5000.

## Known solver limitations

| Case | Status | Root cause |
|---|---|---|
| W strength-edges | Never converges | ZI W formula unstable when `l_ij × G_W(q_ij) >> 1` |
| W strength-degree | Never converges | Per-node multipliers amplify numerical instability |
| W strength-cost partial | Diverges at all N | Conic solver boundary: q → 1.0 for high-strength nodes |
| B strength-cost saturated N ≥ 1000 | Fails within 10k iters | IPF plateau with residual ~10⁻¹⁰ |
| B strength-degree sparse N = 5000 | Overflow | z multipliers ~ 10²⁶² |

## Results files

- `benchmarks/results/all-benchmark.json` — Full merged dataset
- `benchmarks/results/e2e-modern.json` — Latest dense-regime results (ME + B)
