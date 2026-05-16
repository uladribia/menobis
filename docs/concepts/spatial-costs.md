---
description: Strength-cost model for spatially constrained multi-edge networks.
---

# Spatial costs

## TL;DR

The strength-cost model (thesis Case 2) constrains both the strength sequence
and the total network cost. It is the doubly-constrained gravity model in
exponential deterrence form.

## Model

Given a cost matrix $d_{ij}$ (e.g., Euclidean distance between nodes),
the expected occupation is:

$$
E[t_{ij}] = x_i \, y_j \, e^{-\gamma \, d_{ij}}
$$

where $x_i$, $y_j$ are node-level Lagrange multipliers fitted to match the
observed strength sequences, and $\gamma \ge 0$ is a scalar multiplier
fitted to match the observed total cost:

$$
C = \sum_{ij} t_{ij} \, d_{ij}.
$$

## Constraints

The model satisfies:

$$
s_i^{out} = \sum_j x_i \, y_j \, e^{-\gamma \, d_{ij}}, \quad
s_j^{in} = \sum_i x_i \, y_j \, e^{-\gamma \, d_{ij}}, \quad
C = \sum_{ij} x_i \, y_j \, d_{ij} \, e^{-\gamma \, d_{ij}}.
$$

## Solver

ODME uses a two-level solver:

1. **Inner loop**: IPF balancing of $x$, $y$ for fixed $\gamma$ to match
   strength constraints.
2. **Outer loop**: adaptive search on $\gamma$ to match the total cost $C$.

## Python API

```python
import numpy as np
from odme.models import fit_strength_cost_me, sample_strength_cost_me

fit = fit_strength_cost_me(
    s_out, s_in,
    cost_sources, cost_targets, cost_values,
    target_cost,
)
sample = sample_strength_cost_me(fit, cost_sources, cost_targets, cost_values, seed=42)
```

## Partial-constraint variant

When some edges have known rates (e.g., from a cutoff), the excess constraints
are computed and the solver fits only the free pairs:

```python
from odme.models.partial import fit_partial_strength_cost_me

result = fit_partial_strength_cost_me(
    s_out, s_in,
    known_source, known_target, known_rate,
    cost_sources, cost_targets, cost_values,
    target_cost,
)
```

See [Partial Constraints](partial-constraints.md) for details.
