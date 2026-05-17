---
description: Partial-constraint fitting with known edge rates.
---

# Partial constraints

## TL;DR

Partial fitting fixes some pair rates $q_{ij}$ and fits the remaining pairs on
**free** constraints. ODME names these functions `fit_partial_*` and keeps the
same model names as the full fitters.

## Problem setup

Partition node pairs into:

| Set | Meaning |
|-----|---------|
| $Q$ | known pairs with fixed expectation $\mathbb{E}[t_{ij}] = q_{ij}$ |
| $\bar Q$ | free pairs fitted by a multi-edge model |

Full outgoing strength is split as:

$$
s_i^{out} = \sum_{j:(i,j)\in Q} q_{ij}
+ \sum_{j:(i,j)\in \bar Q} \mathbb{E}[t_{ij}].
$$

The solver uses free, or excess, constraints:

$$
s_i^{out,free} = s_i^{out} - \sum_{j:(i,j)\in Q} q_{ij}.
$$

Incoming strengths, degrees, total binary edges, and total cost are reduced in
the same way.

## Excess constraints

| Constraint | Free value |
|------------|------------|
| Out strength | $s_i^{out} - \sum_{j:(i,j)\in Q} q_{ij}$ |
| In strength | $s_j^{in} - \sum_{i:(i,j)\in Q} q_{ij}$ |
| Out degree | $k_i^{out} - \sum_{j:(i,j)\in Q} \Theta(q_{ij})$ |
| In degree | $k_j^{in} - \sum_{i:(i,j)\in Q} \Theta(q_{ij})$ |
| Binary edges | $E - \sum_{(i,j)\in Q}\Theta(q_{ij})$ |
| Cost | $C - \sum_{(i,j)\in Q} q_{ij} d_{ij}$ |

Any negative free constraint is infeasible and rejected.

## Available models

| ODME model | Function | Free constraints |
|------------|----------|------------------|
| Fixed-strength ME | `fit_partial_strength_poisson` | $s^{free}$ |
| Degree-events ME | `fit_partial_degree_poisson` | $k^{free}$ |
| Strength-degree ME | `fit_partial_strength_degree_poisson` | $s^{free}$, $k^{free}$ |
| Strength-edges ME | `fit_partial_strength_edges_poisson` | $s^{free}$, $E^{free}$ |
| Strength-cost ME | `fit_partial_strength_cost_poisson` | $s^{free}$, $C^{free}$ |

## Self-loops

When `self_loops=False`, diagonal pairs $(i,i)$ are excluded from both known and
free sets.

## Cutoff-based splitting

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

Edges with `weight > cutoff` become known positive pairs. Remaining constraints
are fitted on free pairs.

## Sampling from partial fits

`PartialFitResult.rate` contains unnormalized expected counts, not normalized
probabilities. Convert before custom-probability sampling:

```python
from odme.models import sample_custom_poisson

prob = result.as_probability_table()
sample = sample_custom_poisson(prob, total_events=T, seed=42)
```

Custom-probability samplers normalize their input rates internally.

## Example

```python
import numpy as np
from odme.data.frames import EdgeTable
from odme.models.partial import fit_from_network_cutoff

edges = EdgeTable(
    source=np.array([0, 0, 1, 1, 2, 2]),
    target=np.array([1, 2, 0, 2, 0, 1]),
    weight=np.array([50, 3, 40, 4, 30, 5]),
)

result = fit_from_network_cutoff(edges, cutoff=10, model="strength")
prob = result.as_probability_table()
```
