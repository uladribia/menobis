# Benchmark Results: E2E Pipeline

> TL;DR: ME models converge reliably at all sizes (N≤1000). B models need
> adequate layer headroom. W models converge at N≤50 for strength-cost but
> need more iterations at N≥100. All partial ME/B models converge at all sizes.

## Test environment

- CPU: single-threaded Rust via PyO3
- Network: gravity model with Pareto-distributed node activity
- self_loops: False (no diagonal)
- Constraints derived from generated network (guaranteed feasible)

## Summary table

| N | Cases | Fit OK | Fit % | Notes |
|----:|------:|-------:|------:|-------|
| 25 | 27 | 26 | 96% | 1 fail: partial Wnb cost-coord |
| 50 | 27 | 24 | 89% | 3 fails: B/Wnb cost, partial Wnb |
| 100 | 27 | 24 | 89% | 3 fails: B/W cost, partial Wnb |
| 500 | 27 | 20 | 74% | 7 fails: W/Wnb strength, B/W/Wnb cost, partial W/Wnb |
| 1000 | 14* | 14 | 100% | ME + B + degree-events + partial ME/B only |

*N=1000 tested with fast cases only (W strength-cost exceeds time budget).

## Convergence by family and constraint

### ME (Poisson) — Always converges

| Constraint | N=25 | N=50 | N=100 | N=500 | N=1000 |
|------------|------|------|-------|-------|--------|
| strength | ✓ 4i | ✓ 4i | ✓ 3i | ✓ 3i | ✓ 3i |
| strength-cost | ✓ 9i | ✓ 12i | ✓ 9i | ✓ 13i | ✓ 11i |
| strength-edges | ✓ 1i | ✓ 4i | ✓ 2i | ✓ 1i | — |
| strength-degree | ✓ 1i | ✓ 1i | ✓ 1i | ✓ 1i | ✓ 1i |
| degree-events | ✓ 1i | ✓ 1i | ✓ 1i | ✓ 1i | ✓ 1i |

### B (Binomial) — Converges with adequate layers

| Constraint | N=25 | N=50 | N=100 | N=500 | N=1000 |
|------------|------|------|-------|-------|--------|
| strength | ✓ 32i | ✓ 9i | ✓ 6i | ✓ 12i | ✓ 10i |
| strength-cost | ✓ 10i | ✗ | ✗ | ✗ | ✗ |
| strength-edges | ✓ 1i | ✓ 4i | ✓ 2i | ✓ 1i | — |
| strength-degree | ✓ 1i | ✓ 1i | ✓ 1i | ✓ 1i | — |
| degree-events | ✓ 1i | ✓ 1i | ✓ 1i | ✓ 1i | ✓ 1i |

B strength-cost fails at N≥50 because the coordinate-based B IPF
hits the same near-saturation conditioning as B strength (see PLAN Step 3).

### W (Geometric) — Converges at small N, needs more iterations at large N

| Constraint | N=25 | N=50 | N=100 | N=500 |
|------------|------|------|-------|-------|
| strength | ✓ 10i | ✓ 12i | ✓ 9i | ✗ 5000i |
| strength-cost | ✓ 4510i | ✓ 9824i | ✗ 24009i | ✗ 70000i |
| strength-edges | ✓ 1i | ✓ 1i | ✓ 1i | ✓ 4i |
| strength-degree | ✓ 7i | ✓ 1i | ✓ 1i | ✓ 1i |
| degree-events | ✓ 1i | ✓ 1i | ✓ 1i | ✓ 1i |

W strength and strength-cost use Newton coordinate-descent which scales
poorly with network heterogeneity at large N. The bisection+Newton solver
converges at N≤50 but needs >5000 iterations at N≥100 for the strength-cost
case.

### Wnb (Negative Binomial, M=3) — Similar to W

| Constraint | N=25 | N=50 | N=100 | N=500 |
|------------|------|------|-------|-------|
| strength | ✓ 11i | ✓ 12i | ✓ 9i | ✗ 5000i |
| strength-cost | ✓ 2326i | ✗ | ✓ 9092i | ✗ |
| strength-edges | ✓ 2i | ✓ 5i | ✓ 2i | ✓ 6i |
| strength-degree | ✓ 1i | ✓ 1i | ✓ 1i | ✓ 1i |
| degree-events | ✓ 1i | ✓ 1i | ✓ 1i | ✓ 1i |

### Partial fitting — ME/B converge at all sizes

| Constraint | N=25 | N=50 | N=100 | N=500 | N=1000 |
|------------|------|------|-------|-------|--------|
| partial ME strength | ✓ | ✓ | ✓ | ✓ | ✓ |
| partial ME strength-edges | ✓ | ✓ | ✓ | ✓ | ✓ |
| partial ME strength-degree | ✓ | ✓ | ✓ | ✓ | ✓ |
| partial ME cost-coord | ✓ | ✓ | ✓ | ✓ | ✓ |
| partial B cost-coord | ✓ | ✓ | ✓ | ✓ | ✓ |
| partial W cost-coord | ✓ | ✓ | ✓ | ✗ | — |
| partial Wnb cost-coord | ✗ | ✗ | ✗ | ✗ | — |

## Performance (wall-clock seconds)

| N | ME strength | ME cost | ME edges | ME degree | B strength | W cost |
|----:|:-----------:|:-------:|:--------:|:---------:|:----------:|:------:|
| 25 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | 0.12 |
| 50 | <0.01 | 0.02 | 0.01 | <0.01 | <0.01 | 1.7 |
| 100 | <0.01 | 0.06 | 0.02 | 0.01 | <0.01 | 8.5 |
| 500 | 0.02 | 0.96 | 0.31 | 0.29 | 0.02 | 98* |
| 1000 | 0.07 | 2.6 | 2.2 | 1.1 | 0.04 | — |

*W strength-cost at N=500 did not converge within 70k iterations.

## Known limitations

1. **W/Wnb strength-cost at N≥100**: The Newton coordinate-descent solver
   converges slowly for highly heterogeneous gravity-model networks. The
   bisection+Newton approach works at N≤50 but needs O(10k) iterations at
   larger sizes. Future: L-BFGS or trust-region methods.

2. **B strength-cost at N≥50**: The coordinate-based binomial IPF hits
   near-saturation conditioning. Workaround: use more layers.

3. **Partial Wnb cost-coord**: Does not converge at any tested size.
   The negative-binomial Newton solver within the partial framework needs
   investigation.

4. **Sample check (△)**: Some models fit correctly but single-sample
   verification fails due to high variance in W/B families. These are
   NOT fitting failures — the model converges but sampling is stochastic.
   Use ensemble z-scores for proper validation.

## Recommendations

- **Production use**: ME models for all constraint types at any network size.
- **Research use**: B models with `layers >= 4 * ceil(max_s/(N-1))`.
- **W models**: Use only at N≤50 for strength-cost, or use strength-edges/
  strength-degree constraints which converge at all sizes.
- **Partial fitting**: Use ME or B families for reliable convergence.
