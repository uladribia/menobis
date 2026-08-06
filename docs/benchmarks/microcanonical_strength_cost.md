# Microcanonical Strength-Cost Benchmarks (Phase 5)

**Date:** August 2026
**Baseline:** `master` after Phase 4 merge `aeb9483bf04e6cf04dbc19b8bbd10a2cfbab23ed`
**Constraint:** `strength-cost` (exact strengths + expected cost, gamma fitted)

## Command

```bash
python -m benchmarks micro \
  --nodes 10,15 \
  --families me,b,w \
  --regime dense \
  --constraint strength-cost \
  --known-pairs 0.0,0.05 \
  --burn-in-sweeps 30 \
  --sweeps-per-sample 15 \
  --seed 42 \
  --no-memory
```

## Results

| N | family | kp% | wall(s) | sampled_edges | status |
|---|--------|-----|---------|---------------|--------|
| 10 | me | 0 | 0.0371 | 48 | ok |
| 10 | me | 5 | 0.0286 | 46 | ok |
| 10 | b | 0 | 0.0356 | 50 | ok |
| 10 | b | 5 | 0.0452 | 54 | ok |
| 10 | w | 0 | 0.0252 | 52 | ok |
| 10 | w | 5 | 0.0232 | 49 | ok |
| 15 | me | 0 | 0.0536 | 194 | ok |
| 15 | me | 5 | 0.0261 | 199 | ok |
| 15 | b | 0 | 0.0330 | 194 | ok |
| 15 | b | 5 | 0.0301 | 199 | ok |
| 15 | w | 0 | 0.0328 | 194 | ok |
| 15 | w | 5 | 0.0339 | 199 | ok |

All 12 runs passed validation: exact out/in-strength recovery and family occupation bounds.

## Gamma-Fit Configuration

The `strength-cost` micro benchmark uses the stochastic-bisection gamma fitter
with these defaults:

| Parameter | Value |
|-----------|-------|
| warm_start_sweeps | 50 |
| adaptation_sweeps | 100 |
| estimation_sweeps | 80 |
| samples_per_iteration | 20 |
| max_iterations | 40 |
| cost tolerance | 0.5 |
| batch_count | 10 |
| confidence_multiplier | 2.09 |

## Notes

- The strength-cost constraint requires larger MCMC effort than plain strength
  because gamma is fitted via stochastic bisection, and each bisection iteration
  runs adaptation + estimation sweeps on a persistent chain.
- Fixed-pair runs (kp=5%) residualize the fixed-pair cost and fit gamma against
  the residual target; exact strengths are still verified on the merged sample.
- Sampling time at N=15 is dominated by the gamma fit; the final sample draw
  itself is fast.

## Known Limitations

- The benchmark does not yet report the fitted gamma value or the expected-cost
  standard error in the row output (the underlying `sample_model` returns only
  the sampled EdgeTable).  These diagnostics are available in the Rust
  `FixedStrengthCostFitResult` and can be surfaced through
  `sample_model_detailed` in a follow-up.
- Larger node counts (N ≥ 50) will need proportionally more fit sweeps; the
  defaults above target N ≤ 20 correctness runs.
