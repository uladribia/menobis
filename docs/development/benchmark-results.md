# Benchmark Results: PA Geographic Pipeline

> TL;DR: canonical PA geographic benchmarks pass generate → fit → sample →
> null-filter for N=100 and N=500. B strength-edges/degree are skipped because
> P5 is still a real kernel bug.

## Command

```bash
uv run python -m benchmarks all --nodes 100,500 \
  --samples 5 --filter-samples 5 \
  --output benchmarks/results/pa-geographic-n100-500
```

## Generated networks

| N | edges | events | max strength |
|---:|---:|---:|---:|
| 100 | 800 | 4,800 | 346 |
| 500 | 4,000 | 24,000 | 551 |

## Fit summary

| N | fit rows | ok | skipped | fit seconds |
|---:|---:|---:|---:|---:|
| 100 | 20 | 18 | 2 | 3.37 |
| 500 | 20 | 18 | 2 | 98.90 |

Skipped cases:

| Case | Reason |
|---|---|
| B strength-edges | P5: wrapper calls ME kernel |
| B strength-degree | P5: wrapper calls ME kernel |

## Slowest fits

| N | case | seconds | iterations |
|---:|---|---:|---:|
| 100 | W strength-cost | 1.63 | 2,845 |
| 100 | Wnb strength-cost | 0.59 | 361 |
| 500 | W strength-cost | 45.79 | 1,135 |
| 500 | W strength-edges | 16.05 | 14 |
| 500 | Wnb strength-edges | 15.80 | 14 |
| 500 | Wnb strength-cost | 13.97 | 342 |

## Null-filter calibration

False-positive rate uses candidate pairs as denominator and filters samples
from the fitted null, not the PA network.

| N | max FPR | mean FPR |
|---:|---:|---:|
| 100 | 0.0379 | 0.0281 |
| 500 | 0.0159 | 0.0140 |

## Interpretation

The PA generator gives less saturated constraints than the previous Pareto-only
benchmarks, so W and B strength-cost converge at N=500 in this run. This does
not close P1/P2: harsher high-strength regimes can still expose those solver
limits.
