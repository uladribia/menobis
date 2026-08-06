# Microcanonical Strength-Cost Benchmarks (Phase 5)

**Date:** August 2026
**Baseline:** `master` after Phase 5 (fixed strengths + expected cost)
**Constraint:** `strength-cost` (exact strengths + expected cost, gamma fitted)

## Command (N=100)

```bash
python -m benchmarks micro \
  --nodes 100 \
  --families me,b,w \
  --regime dense \
  --constraint strength-cost \
  --known-pairs 0.0,0.05 \
  --burn-in-sweeps 30 \
  --sweeps-per-sample 15 \
  --fit-warm-start-sweeps 300 \
  --fit-adaptation-sweeps 300 \
  --fit-estimation-sweeps 150 \
  --fit-samples-per-iteration 50 \
  --fit-max-iterations 40 \
  --fit-cost-tolerance 0.15 \
  --seed 42
```

## Results — N=100, dense regime (T≈16000, observed cost≈5491)

| family | kp% | γ | E[C] | resid | SE | iters | conv | wall(s) | status |
|--------|-----|------|--------|-------|------|-------|-------|---------|--------|
| me | 0 | 2.50 | 6039.3 | 548.3 | 5.14 | 3 | True | 3.0 | ok |
| me | 5 | 50.0 | 4158.1 | 1073.2 | 14.6 | 40 | False | 14.2 | ok |
| b | 0 | 5.00 | 4783.7 | 707.3 | 4.92 | 3 | True | 3.7 | ok |
| b | 5 | 5.00 | 4570.2 | 661.1 | 5.86 | 3 | True | 3.7 | ok |
| w | 0 | 50.0 | 4946.0 | 545.0 | 13.4 | 1 | True | 0.9 | ok |
| w | 5 | 3.13 | 4581.4 | 649.9 | 15.6 | 6 | True | 4.4 | ok |

All six runs pass the benchmark gate: exact out/in-strength recovery, family
occupation bounds, and sampled cost within `5 × fit_cost_tolerance` of the
observed total.  JSON: `benchmarks/results/microcanonical-strength-cost-n100.json`.

## Results — N=500, dense regime (T≈400000, observed cost≈139080)

| family | γ | E[C] | resid | status |
|--------|------|--------|--------|--------|
| me | — | — | — | error: cost not identifiable / bracket not found |
| b | 3.3e4 | 115466 | 23613 | ok (extreme γ, conv=False) |
| w | — | — | — | error: cost not identifiable / bracket not found |

At N=500 the fixed-strength MCMC cost fit is unreliable with the current
sweep budgets: the warm-start cost-variance estimate collapses (the chain
does not mix enough within `warm_start_sweeps` for the total cost to vary),
triggering `CostNotIdentifiable` or a failed bracket for ME and W.  B
completes but with an extreme gamma (3e4) and no convergence.

## Interpretation

- **B is the most robust family**: it converges at γ≈5 for both kp settings
  with residuals ~10% of observed cost and small SEs.  The Binomial
  degeneracy bounds occupations, giving a well-conditioned chain.
- **ME and W are noisier**: at N=100 they land at γ between 2.5 and 50
  depending on the (noisy) warm-start variance estimate.  The results are
  within the loose benchmark tolerance but the fitted gamma is not stable
  run-to-run.
- **Fixed-pair runs (kp=5%)** with ME fail to converge (conv=False) at
  40 iterations: the residualized cost target is hard to match exactly.
  The sampled network still validates (exact strengths, cost within
  tolerance).
- **Scalability**: the cost variance of the fixed-strength chain at γ=0
  shrinks relative to its mean as N grows (central-limit effect), so the
  warm-start gamma `γ₀ = (µ₀ − C_obs) / Var₀` becomes unstable.  Reliable
  fitting needs warm-start sweeps that scale with N.

## Known Limitations

1. The gamma fit is reliable at N ≤ 100 with generous sweeps
   (300/300/150) and tolerance ≥ 0.1; at N = 500 it requires
   per-case tuning that is not yet automated.
2. The `fit_gamma` convergence check uses a Monte-Carlo SE criterion
   (batch-means, `z·SE ≤ tol`); meeting it at tight tolerance needs many
   estimation sweeps.  Non-converged fits now return the best gamma with
   `converged=false` instead of failing the whole run.
3. Fixed-pair residualization can make the residual cost target hard to
   match at tight tolerance (ME kp=5% at N=100 does not converge in 40
   iterations).
4. The ME direct stub-matching backend is disabled when a cost provider is
   present (it cannot sample the cost-tilted distribution), so all
   strength-cost runs use the 4-cycle MCMC chain.
