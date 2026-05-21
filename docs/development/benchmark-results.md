# Benchmark Results: E2E Pipeline

> TL;DR: ME models work perfectly at all sizes. B strength works with adequate
> layers. B strength-edges/degree have a critical kernel bug (use ME kernel).
> W/Wnb converge at N≤100 for strength, N≤50 for cost. Degree-events works
> for all families at all sizes.

## Complete convergence matrix

### Fitting convergence (does the solver converge?)

| Constraint | ME | B | W | Wnb | Notes |
|------------|:--:|:-:|:-:|:---:|-------|
| **strength** | ✓ all N | ✓ all N | ✓ N≤100 | ✓ N≤100 | W/Wnb fail at N≥500 |
| **strength-cost** | ✓ all N | ✓ N=25 only | ✓ N≤50 | ✓ N≤25 | Main bottleneck |
| **strength-edges** | ✓ all N | ✓ all N | ✓ all N | ✓ all N | |
| **strength-degree** | ✓ all N | ✓ all N | ✓ all N | ✓ all N | |
| **degree-events** | ✓ all N | ✓ all N | ✓ all N | ✓ all N | Trivial (analytic) |
| **partial strength** | ✓ all N | — | — | — | ME only |
| **partial cost-coord** | ✓ all N | ✓ all N | ✗ | ✗ | W inherits P1 |

### End-to-end correctness (fit → sample → constraint recovery)

| Constraint | ME | B | W | Wnb | Notes |
|------------|:--:|:-:|:-:|:---:|-------|
| **strength** | ✓ | ✓ | △ | △ | W/Wnb: high geometric variance (expected) |
| **strength-cost** | ✓ | ✓ (N=25) | △ | △ | △ = high variance, not bug |
| **strength-edges** | ✓ | **BUG** | △ | △ | B uses ME kernel (P5) |
| **strength-degree** | ✓ | **BUG** | △ | △ | B uses ME kernel (P5) |
| **degree-events** | ✓ | ✓ | ✓ | ✓ | All correct |

Legend:
- ✓ = converges AND samples recover constraints within stochastic tolerance
- △ = converges but single-sample error large due to heavy-tailed distribution
  (NOT a bug; ensemble z-scores confirm correct model)
- **BUG** = silently wrong results (P5: calls ME kernel, applies B sampler)
- ✗ = does not converge

## Strength-only fitting metrics

| N | ME iters | B iters | B layers | W iters | W fit_residual | W tol |
|----:|---------:|--------:|---------:|--------:|---------------:|------:|
| 25 | 4 | 32 | 104 | 10 | 11.5 | 12.2 |
| 50 | 4 | 9 | 76 | 12 | 13.2 | 18.0 |
| 100 | 3 | 6 | 332 | 9 | 73.2 | 163.7 |
| 500 | 3 | 12 | 1092 | ✗ 50k | — | 2720 |
| 1000 | 3 | 10 | 2476 | ✗ | — | — |

## Performance (wall-clock seconds, converged cases only)

| N | ME str | ME cost | ME edges | ME degree | B str | W str | W edges |
|----:|:------:|:-------:|:--------:|:---------:|:-----:|:-----:|:-------:|
| 25 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | 0.01 |
| 50 | <0.01 | 0.02 | 0.01 | <0.01 | <0.01 | <0.01 | 0.02 |
| 100 | <0.01 | 0.06 | 0.02 | 0.01 | <0.01 | 0.01 | 0.08 |
| 500 | 0.02 | 0.96 | 0.31 | 0.29 | 0.02 | ✗ | 5.7 |
| 1000 | 0.07 | 2.6 | 2.2 | 1.1 | 0.04 | ✗ | — |

## Known bugs (critical)

### P5: B strength-edges and B strength-degree use ME kernel

`fit_strength_edges_binomial` calls `_odme.fit_strength_edges_poisson` and
`fit_strength_degree_binomial` calls `_odme.fit_strength_degree_poisson`.
No B-specific Rust kernel exists. The fit "converges" (it's doing ME) but
the B sampler applies binomial formula to ME parameters → wrong results.

**Evidence:** B strength-edges at N=50 produces sample error of 2638 vs
ME error of 56 on the same network. B strength-degree: error 2888 vs ME 190.

## Known limitations (not bugs)

### W/Wnb high single-sample variance

W models use the geometric distribution where Var(t_ij) = q/(1-q)².
For strong nodes (q→1), per-pair variance diverges. Single-sample strength
errors of 100–5000 are expected for correctly fitted models. Ensemble
verification confirms correct mean recovery.

### W/Wnb strength-cost and strength at large N

The Newton coordinate-descent solver stalls at N≥100 (cost) or N≥500
(strength-only). Root cause: adaptive damping hits floor without sufficient
progress. Fix requires Anderson acceleration or L-BFGS.

### B near-saturation IPF

B models require `layers >= 4*ceil(max(s_out)/(N-1))` to avoid multiplier
divergence. With adequate layers, convergence is fast (6–32 iterations).

## Recommendations

| Use case | Recommended family | Constraint type | Max N tested |
|----------|-------------------|-----------------|:------------:|
| Production (any size) | ME | any | 1000+ |
| Bounded weights | B | strength, degree-events | 1000 |
| Bounded weights + cost | B | strength-cost | 25 |
| Heavy-tailed weights | W/Wnb | strength | 100 |
| Heavy-tailed + cost | W/Wnb | strength-cost | 50 |
| Degree structure | any | strength-degree, degree-events | 1000 |
| Partial observations | ME, B | partial cost-coord | 1000 |
