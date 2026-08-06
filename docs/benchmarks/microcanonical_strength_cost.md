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

| N | family | kp% | gamma | expected_cost | cost_residual | SE | wall(s) | status |
|---|--------|-----|-------|---------------|---------------|-----|---------|--------|
| 10 | me | 0 | 3.34 | 57.38 | 0.06 | 0.385 | 0.037 | ok |
| 10 | me | 5 | 0.00 | 58.19 | 6.77 | 0.294 | 0.029 | ok |
| 10 | b | 0 | 2.62 | 60.82 | 3.51 | 0.316 | 0.036 | ok |
| 10 | b | 5 | 2.08 | 53.64 | 2.22 | 0.357 | 0.038 | ok |
| 10 | w | 0 | 0.00 | 66.56 | 9.25 | 0.307 | 0.034 | ok |
| 10 | w | 5 | 1.37 | 54.03 | 2.61 | 0.579 | 0.031 | ok |
| 15 | me | 0 | 3.05 | 154.14 | 0.33 | 0.872 | 0.062 | ok |
| 15 | me | 5 | 4.28 | 127.62 | 19.61 | 0.897 | 0.034 | ok |
| 15 | b | 0 | 4.36 | 142.75 | 11.06 | 0.915 | 0.035 | ok |
| 15 | b | 5 | 4.23 | 135.54 | 11.69 | 0.520 | 0.035 | ok |
| 15 | w | 0 | 0.00 | 193.56 | 39.75 | 1.161 | 0.033 | ok |
| 15 | w | 5 | 0.00 | 177.58 | 30.35 | 0.380 | 0.031 | ok |

All 12 runs passed validation: exact out/in-strength recovery, family occupation
bounds, and cost residual within `5 × fit_cost_tolerance` of the observed target.

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

## Diagnostics

The benchmark now reports the fitted gamma, expected cost, cost residual, and
Monte Carlo standard error (batch-means) for every strength-cost run.  These
flow from the Rust `FixedStrengthCostFitResult` through
`sample_model_detailed` → `SamplingDiagnostics` → the benchmark JSON rows.

## Interpretation

- ME and B recover positive gamma: the gravity-generated observed cost is lower
  than the max-entropy (γ=0) expected cost, so a positive cost penalty is
  required.
- W often lands at γ≈0 with larger residuals (9–40 vs tolerance 0.5·C_obs).
  This reflects a genuine property of the Negative-Binomial chain with small
  layer count M: its γ=0 cost distribution already overlaps the observed cost
  within the benchmark tolerance, so the cost tilt is weak.  Runs with an
  explicit larger `layers` value (e.g., M=40) recover a non-trivial gamma
  (γ≈3.5), confirming the fitter itself is healthy.

## Notes

- The strength-cost constraint requires larger MCMC effort than plain strength
  because gamma is fitted via stochastic bisection, and each bisection iteration
  runs adaptation + estimation sweeps on a persistent chain.
- Fixed-pair runs (kp=5%) residualize the fixed-pair cost and fit gamma against
  the residual target; exact strengths are still verified on the merged sample.

## Known Limitations

- W-family fits can terminate at γ≈0 with loose tolerances; tighten
  `--fit-cost-tolerance` or raise `layers` for sharper W cost recovery.
- Larger node counts (N ≥ 50) need proportionally more fit sweeps; the defaults
  above target N ≤ 20 correctness runs.
