---
description: Strength-cost model for spatially constrained multi-edge networks.
---

# Spatial costs

## TL;DR

The strength-cost model adds a total-cost Lagrange multiplier $\gamma$ to the
strength constraints. All three families (ME, B, W) share the same cost decay
factor $f_{ij} = e^{-\gamma d_{ij}}$ but differ in expected-weight formula.

Coordinate APIs compute Euclidean distances on the fly in Rust. No dense cost
triples are allocated when using coordinates.

## Family-specific expected weights

| Family | $\mathbb{E}[t_{ij}]$ | Parameters |
|--------|----------------------|------------|
| ME (Poisson) | $x_i y_j f_{ij}$ | $x_i, y_j$ are rate multipliers |
| B (Binomial M) | $\frac{M\, x_i y_j f_{ij}}{1 + x_i y_j f_{ij}}$ | $x_i, y_j$ are rate multipliers |
| W (Geometric/NegBin M) | $\frac{M\, e^{-r_{ij}}}{1 - e^{-r_{ij}}}$ | $r_{ij} = a_i + b_j + \gamma d_{ij}$; returned as $x_i = e^{-a_i}$, $y_j = e^{-b_j}$ |

Here $f_{ij} = e^{-\gamma d_{ij}}$ in all cases.

## Shared constraints

All families satisfy:

$$
s_i^{\text{out}} = \sum_j \mathbb{E}[t_{ij}], \qquad
s_j^{\text{in}} = \sum_i \mathbb{E}[t_{ij}], \qquad
C = \sum_{ij} \mathbb{E}[t_{ij}]\, d_{ij}.
$$

When `self_loops=False`, diagonal pairs $(i,i)$ are omitted from all sums.

## Cost providers

| Mode | Input | Memory | Use when |
|------|-------|--------|----------|
| Sparse triples | `cost_sources, cost_targets, cost_values` | O(K) where K = number of entries | Arbitrary costs, partial support |
| Projected coordinates | `coord_x, coord_y` per node | O(N) | Complete Euclidean spatial costs |

Coordinates must be in a projected CRS (UTM, local metric). ODME does not
transform CRS or compute geodesic distances.

## Solver strategies

| Family | Gamma search | Inner solver | Notes |
|--------|-------------|--------------|-------|
| ME | Bisection | IPF balancing at fixed $\gamma$ | On-the-fly distances |
| B | Bisection | IPF with saturation $z/(1+z)$ | On-the-fly distances |
| W | Exponential-cone (Clarabel) | Conic formulation | Inline distances in CSC assembly |

## Return value semantics

All coordinate functions return `StrengthCostFit(x, y, gamma, family, ...)`.

For ME and B, reconstruct rates as:

```python
f_ij = np.exp(-fit.gamma * distance[i, j])
# ME: rate = fit.x[i] * fit.y[j] * f_ij
# B:  rate = layers * fit.x[i] * fit.y[j] * f_ij / (1 + fit.x[i] * fit.y[j] * f_ij)
```

For W, `x[i] = exp(-a_i)` and `y[j] = exp(-b_j)`:

```python
r_ij = -np.log(fit.x[i]) - np.log(fit.y[j]) + fit.gamma * distance[i, j]
# W:  rate = layers * np.exp(-r_ij) / (1 - np.exp(-r_ij))
```

## Coordinate API

```python
from odme.models import (
    fit_strength_cost_poisson_coordinates,
    fit_strength_cost_binomial_coordinates,
    fit_strength_cost_geometric_coordinates,
    fit_strength_cost_negative_binomial_coordinates,
)

me = fit_strength_cost_poisson_coordinates(s_out, s_in, x, y, target_cost)
b  = fit_strength_cost_binomial_coordinates(s_out, s_in, x, y, target_cost, layers=3)
wg = fit_strength_cost_geometric_coordinates(s_out, s_in, x, y, target_cost)
wnb = fit_strength_cost_negative_binomial_coordinates(s_out, s_in, x, y, target_cost, layers=3)
```

## Partial-constraint variant

```python
from odme.models.partial import (
    fit_partial_strength_cost_poisson_coordinates,
    fit_partial_strength_cost_binomial_coordinates,
    fit_partial_strength_cost_geometric_coordinates,
    fit_partial_strength_cost_negative_binomial_coordinates,
)
```

Partial fitting computes excess strengths and cost, calls the family-specific
full solver on free pairs, and assembles the rate table using the correct
family formula.

## W memory and scalability

The W conic solver requires exponential-cone variables for each pair, making
its memory O(N²) regardless of cost representation. The inline coordinate
version avoids allocating a separate 3×N² cost array by computing distances
during CSC matrix assembly. Runtime is dominated by the conic solver's
interior-point iterations, not distance computation.

At N=500 the W solver takes ~200s (dev build). For larger N, prefer ME/B
coordinate APIs or wait for a non-conic W solver.
