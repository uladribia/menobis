---
description: Partial-constraint fitting with frozen known node pairs.
---

# Partial constraints

## TL;DR

Partial fitting is not a new model family. It freezes known pair statistics,
computes excess constraints on the free support, and then calls the matching
full solver for the selected family and constraint.

## Problem setup

Partition candidate pairs into known pairs $Q$ and free pairs $\bar Q$.
Known weighted pairs contribute fixed expectations $q_{ij}$.

$$
s_i^{out,free}=s_i^{out}-\sum_{j:(i,j)\in Q} q_{ij}.
$$

Incoming strengths, binary degrees, total binary edges, and total cost are
reduced the same way. Negative excess constraints are infeasible.

## Excess constraints

| Constraint | Free value |
|---|---|
| Out strength | $s_i^{out} - \sum_{j:(i,j)\in Q} q_{ij}$ |
| In strength | $s_j^{in} - \sum_{i:(i,j)\in Q} q_{ij}$ |
| Out degree | $k_i^{out} - \sum_{j:(i,j)\in Q}\Theta(q_{ij})$ |
| In degree | $k_j^{in} - \sum_{i:(i,j)\in Q}\Theta(q_{ij})$ |
| Binary edges | $E - \sum_{(i,j)\in Q}\Theta(q_{ij})$ |
| Cost | $C - \sum_{(i,j)\in Q} q_{ij}d_{ij}$ |

## Required solver pattern

```text
partial fit = compute excess + free support + full family solver
```

The inner solver logic must not be duplicated in a partial path. Masks, excess
computation, and sparse rate-table assembly belong in Rust.

## Current public helpers

| Helper | Family/constraint |
|---|---|
| `fit_partial_strength_poisson` | ME strength |
| `fit_partial_degree_poisson` | ME degree-events occupation |
| `fit_partial_strength_edges_poisson` | ME strength-edges |
| `fit_partial_strength_degree_poisson` | ME strength-degree |
| `fit_partial_strength_cost_poisson` | ME strength-cost |
| `fit_partial_strength_cost_*_coordinates` | ME/B/W coordinate strength-cost |

The Python coordinate helpers for B/W currently do more work in Python than the
AGENTS policy allows. See
[Ontology conformance audit](../development/ontology-conformance-audit.md).

## Cutoff-based splitting

```python
from odme.models.partial import fit_from_network_cutoff

result = fit_from_network_cutoff(
    edges,
    cutoff=10,
    model="strength",  # strength, degree, strength-degree, strength-edges, strength-cost
    self_loops=False,
)
```

Edges with `weight > cutoff` become known positive pairs. Remaining constraints
are fitted on free pairs.

## Sampling from partial fits

`PartialFitResult.rate` stores expected counts for known and free pairs. Use the
custom Poisson sampler for a sparse independent sample:

```python
from odme.models import sample_custom_poisson

rates = result.as_probability_table()
sample = sample_custom_poisson(rates, total_events=T, seed=42)
```
