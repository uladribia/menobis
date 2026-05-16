---
description: Partial-constraint fitting with known edge probabilities.
---

# Partial constraints

## TL;DR

When some edge rates $t_{ij}$ are known (e.g., heavy edges above a cutoff),
ODME fits the remaining edges using any ME model on the **excess** constraints.

## Problem setup

Partition the set of node pairs into:

- **Known set** $Q$: pairs where $E[t_{ij}] = q_{ij}$ is fixed.
- **Free set** $\bar{Q}$: pairs fitted by the ME model.

The observed constraints apply to the **full** network:

$$
s_i^{out} = \underbrace{\sum_{j \in Q_i} q_{ij}}_{\text{known}} +
\underbrace{\sum_{j \notin Q_i} E[t_{ij}]}_{\text{fitted}}.
$$

The solver works on **excess** constraints:

$$
s_i^{out,\text{free}} = s_i^{out} - \sum_{j \in Q_i} q_{ij}.
$$

Similarly for incoming strengths, degrees, and cost.

## Feasibility

If any excess constraint is negative, the problem is infeasible:

$$
s_i^{out,\text{free}} < 0 \implies \text{reject}.
$$

## Available models

All ME models support partial constraints via masked solvers:

| Model | Function | Excess constraints |
|-------|----------|--------------------|
| Fixed strength | `fit_partial_strength_me` | $s^{free}$ |
| Fixed degree | `fit_partial_degree_me` | $k^{free}$ (known pairs contribute 1 each) |
| Strength + degree | `fit_partial_strength_degree_me` | $s^{free}$, $k^{free}$ |
| Strength + edges | `fit_partial_strength_edges_me` | $s^{free}$, $E^{free} = E - |Q|$ |
| Strength + cost | `fit_partial_strength_cost_me` | $s^{free}$, $C^{free} = C - \sum_Q q_{ij} d_{ij}$ |

## Convenience: cutoff-based splitting

```python
from odme.models.partial import fit_from_network_cutoff

result = fit_from_network_cutoff(
    edges,
    cutoff=10,
    model="strength",     # or "degree", "strength-degree",
                          # "strength-edges", "strength-cost"
    self_loops=False,
)
```

Edges with `weight > cutoff` become known pairs. The rest are fitted.

## Self-loops

When `self_loops=False`, diagonal pairs $(i, i)$ are excluded from both the
known set and the fitted pairs, regardless of the data.

## Normalization

The returned `PartialFitResult.rate` values are **unnormalized expected
counts**. For sampling, convert to a `ProbabilityTable`:

```python
from odme.models import sample_custom_pij_events_poisson

prob = result.as_probability_table()
sample = sample_custom_pij_events_poisson(prob, total_events=T, seed=42)
```

The sampling functions normalize internally by dividing each rate by the total.

## Example: heavy-edge splitting

```python
import numpy as np
from odme.data.frames import EdgeTable
from odme.models.partial import fit_from_network_cutoff
from odme.models import sample_custom_pij_events_poisson

edges = EdgeTable(
    source=np.array([0, 0, 1, 1, 2, 2]),
    target=np.array([1, 2, 0, 2, 0, 1]),
    weight=np.array([50, 3, 40, 4, 30, 5]),
)

result = fit_from_network_cutoff(edges, cutoff=10, model="strength")
sample = sample_custom_pij_events_poisson(
    result.as_probability_table(),
    total_events=edges.total_events,
    seed=42,
)
```
