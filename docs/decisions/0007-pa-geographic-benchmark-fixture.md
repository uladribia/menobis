---
description: Use PA geographic networks as canonical benchmark fixtures.
---

# 0007 PA geographic benchmark fixture

## Decision

All end-to-end tests and benchmarks use a canonical synthetic fixture:

1. directed binary support from preferential attachment;
2. projected XY coordinates per node;
3. edge scores proportional to origin out-degree and destination in-degree;
4. exponential geographic damping by Euclidean distance;
5. global normalization to an exact positive integer event total.

The fixture is intentionally not sampled from an MENoBiS null model. MENoBiS nulls are
fitted afterward from derived constraints.

## Filter calibration

Filtering is calibrated on samples drawn from the fitted null, not on the PA
fixture itself. The PA fixture supplies constraints; null samples assess false
positive behavior.

## Consequences

| Benefit | Cost |
|---|---|
| realistic heterogeneous constraints | not a closed-form MENoBiS data generator |
| exact density and total events | support generation is benchmark-only |
| spatial costs are always available | computed on the fly to avoid dense matrices |

