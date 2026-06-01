---
description: Fit a null model and filter statistically significant node pairs.
---

# Filter a network

## TL;DR

Fit an independent grand-canonical null model, then classify observed node pairs
as upper-significant, lower-significant, or compatible.

## Choose a starting null

| Data assumption | Recommended first model |
|---|---|
| Distinguishable trips/events, no cost data | ME strength |
| Distinguishable trips/events with projected coordinates | ME strength-cost |
| Binary support is part of the hypothesis | ME strength-degree or strength-edges |
| Bounded layers per pair | B family with `layers=M` |
| Indistinguishable events | W family; check convergence notes first |

!!! tip "Why ME for origin-destination trips?"
    If trips are individual distinguishable events, the ME/Poisson family is the
    natural first null. Switch families only when event nature changes.

## Python example

```python
from menobis.filtering import filter_model
from menobis.models import Constraint, ModelFamily, fit_model
from menobis.utilities.synthetic import (
    derive_synthetic_constraints,
    generate_pa_geographic_network,
)

network = generate_pa_geographic_network(30, average_degree=6.0, seed=11)
constraints = derive_synthetic_constraints(network)

fit = fit_model(
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH_COST,
    strength_out=constraints.strength_out,
    strength_in=constraints.strength_in,
    target_cost=constraints.total_cost,
    coord_x=network.x,
    coord_y=network.y,
    self_loops=False,
)

result = filter_model(
    network.edges,
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH_COST,
    fit=fit,
    coord_x=network.x,
    coord_y=network.y,
    alpha=0.05,
    tail="upper",
)

print(result.upper.edges.num_edges)
```

## Interpret the result

| Field | Meaning |
|---|---|
| `upper` | observed pairs with occupations larger than expected |
| `lower` | positive observed pairs with occupations smaller than expected |
| `compatible` | observed pairs not rejected by the test |
| `absent_lower` | absent pairs that should likely exist; opt-in with `detect_absent=True` |

`FilteredEdges` also stores `upper_pvalue`, `lower_pvalue`, expected occupation,
and occupation probability for each reported pair.

## CLI example

```bash
uv run menobis filter strength-poisson edges.csv \
  --alpha 0.05 --tail upper --output-prefix filtered/
```

For spatial costs, provide projected coordinates:

```bash
uv run menobis filter strength-cost-poisson edges.csv \
  --coordinates xy.csv --target-cost 120.0 --output-prefix filtered/
```

## Next step

Use [Sample ensemble magnitudes](sample-ensemble-magnitudes.md) to test whether a
network-level statistic is special, not just individual pairs.
