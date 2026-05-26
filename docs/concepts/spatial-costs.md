---
description: Strength-cost constraints and family-orthogonal cost providers.
---

# Spatial costs

## TL;DR

The strength-cost constraint adds a cost multiplier `gamma` and a pair factor
$f_{ij}=\exp(-\gamma d_{ij})$. Cost providers are orthogonal to ME, B, and W:
the same no-cost, sparse-triple, or coordinate provider should feed every
family kernel.

## Constraint

All strength-cost families match:

$$
s_i^{out}=\sum_j \mathbb{E}[t_{ij}],\qquad
s_j^{in}=\sum_i \mathbb{E}[t_{ij}],\qquad
C=\sum_{ij}\mathbb{E}[t_{ij}]d_{ij}.
$$

When `self_loops=False`, diagonal pairs are omitted before evaluating all sums.
The family-specific expectation is defined in
[Model ontology](model-ontology.md).

## Cost providers

| Provider | Input | Memory target | Use when |
|---|---|---:|---|
| No cost | none | O(1) | strength-only kernels |
| Sparse triples | `cost_sources`, `cost_targets`, `cost_values` | O(K) | arbitrary or incomplete costs |
| Euclidean coordinates | projected `x`, `y` per node | O(N) provider state | complete spatial cost from coordinates |

Coordinates must be projected planar coordinates. MENoBiS does not transform CRS or
compute geodesic distances.

## Required implementation shape

```text
CostProvider + FamilyKernel + ConstraintLayer -> Rust solver/provider
```

The provider computes `f_ij` on demand. The family kernel then maps
`q_ij = x_i y_j f_ij` to an expected weight. This avoids per-family cost wrappers
and dense `N x N` matrices.

## Python examples

```python
from menobis.models import (
    fit_strength_cost_poisson_coordinates,
    fit_strength_cost_binomial_coordinates,
    fit_strength_cost_geometric_coordinates,
)

me = fit_strength_cost_poisson_coordinates(s_out, s_in, x, y, target_cost)
b = fit_strength_cost_binomial_coordinates(s_out, s_in, x, y, target_cost, layers=3)
w = fit_strength_cost_geometric_coordinates(s_out, s_in, x, y, target_cost)
```

## Partial strength-cost

Partial strength-cost fitting must:

1. subtract known weighted-pair contributions from strengths and cost;
2. remove frozen pairs from support;
3. call the corresponding full strength-cost family solver on free pairs;
4. assemble the combined known + free sparse rate table in Rust.

Current Python partial coordinate helpers do not fully satisfy this requirement;
see the [Ontology conformance audit](../development/ontology-conformance-audit.md).
