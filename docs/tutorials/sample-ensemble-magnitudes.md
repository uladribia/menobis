---
description: Sample null networks and compare observed network magnitudes.
---

# Sample ensemble magnitudes

## TL;DR

Fit a null model once, sample many null networks, compute the same scalar
magnitude on every sample, and compare the observed value to the null ensemble.

!!! note "How many samples?"
    Use `100` samples for a quick check and `1000+` samples for reported
    analyses. Increase the count when tail percentiles matter.

## Python example

```python
import numpy as np

from menobis.analysis import compute_all_stats
from menobis.models import Constraint, ModelFamily, fit_model, sample_model
from menobis.utilities.synthetic import (
    derive_synthetic_constraints,
    generate_pa_geographic_network,
)

network = generate_pa_geographic_network(30, average_degree=6.0, seed=12)
constraints = derive_synthetic_constraints(network)

fit = fit_model(
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    strength_out=constraints.strength_out,
    strength_in=constraints.strength_in,
    self_loops=False,
)


def mean_y2(edges):
    stats = compute_all_stats(edges)
    return float(np.mean(stats.y2_out))

observed = mean_y2(network.edges)
null_values = np.array([
    mean_y2(sample_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        fit=fit,
        seed=seed,
    ))
    for seed in range(100)
])

print(observed, null_values.mean(), np.percentile(null_values, [2.5, 97.5]))
```

## Magnitudes to start with

| Magnitude | MENoBiS helper |
|---|---|
| Y₂ disparity | `compute_all_stats(edges).y2_out` or `.y2_in` |
| Average nearest-neighbour strength | `compute_all_stats(edges).s_nn_out` or `.s_nn_in` |
| Clustering | `clustering_coefficient(edges)` or `weighted_clustering_coefficient(edges)` |
| Occupation histogram | `weight_distribution(edges)` |

## Interpret the comparison

| Observed value | Interpretation |
|---|---|
| inside central interval | magnitude is compatible with the chosen constraints |
| above upper percentile | magnitude is larger than expected under the null |
| below lower percentile | magnitude is smaller than expected under the null |

!!! warning "Null-model dependence"
    A magnitude can look surprising under strength-only constraints and become
    typical after adding cost or degree constraints. Choose constraints before
    interpreting significance.
