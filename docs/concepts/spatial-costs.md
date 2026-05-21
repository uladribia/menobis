---
description: Strength-cost model for spatially constrained multi-edge networks.
---

# Spatial costs

## TL;DR

The **strength-cost ME** model is thesis case 2. It constrains outgoing and
incoming strengths plus total cost, with expectation
$\mathbb{E}[t_{ij}] = x_i y_j e^{-\gamma d_{ij}}$.

For complete spatial costs, prefer projected XY coordinate APIs that compute
Euclidean distances on the fly. Sparse custom cost triples remain supported, but
dense all-pairs triples can overflow memory at large N.

## Model

Given pair costs $d_{ij}$, usually distances, the expected occupation is:

$$
\mathbb{E}[t_{ij}] = x_i y_j e^{-\gamma d_{ij}}.
$$

Here $x_i$ and $y_j$ are node multipliers and $\gamma \ge 0$ is the scalar cost
multiplier. The observed total cost is:

$$
C = \sum_{ij} t_{ij} d_{ij}.
$$

## Constraints

The fitted expectation satisfies:

$$
s_i^{out} = \sum_j x_i y_j e^{-\gamma d_{ij}},
$$

$$
s_j^{in} = \sum_i x_i y_j e^{-\gamma d_{ij}},
$$

$$
C = \sum_{ij} x_i y_j d_{ij} e^{-\gamma d_{ij}}.
$$

When `self_loops=False`, diagonal pairs $(i,i)$ are omitted from all sums.

## Cost entries

ODME accepts costs as sparse `source,target,cost` entries. For current fitting
and sampling, omitted pairs are treated as cost $d_{ij}=0$. Prefer passing a
complete cost table unless zero-cost missing pairs are intentional.

For large complete spatial supports, pass projected coordinates instead of dense
triples. Coordinates must be in a Euclidean projected CRS such as UTM or a local
metric projection. ODME does not transform CRS and does not compute geodesic
distances.

## Solver

ODME currently uses a two-level structure:

| Level | Method | What it uses |
|-------|--------|--------------|
| Inner | IPF balancing | exact multiplicative structure for $x,y$ at fixed $\gamma$ |
| Outer | adaptive scalar search | warm-started updates of $\gamma$ to match $C$ |

The dual problem is convex under the usual feasible multi-edge setup, but
the current implementation does **not** call a generic convex optimizer. It
exploits separability/IPF and warm starts. Future work can replace the outer
search with bracketed bisection/Brent or a gradient-based convex-dual solve.

## W strength-cost variant

The W ensemble replaces the Poisson mean with geometric or negative-binomial
weights. With `M=1` for geometric and `M>1` for negative binomial,
$q_{ij}=x_i y_j e^{-\gamma d_{ij}}$ and:

$$
\mathbb{E}[t_{ij}] = \frac{M q_{ij}}{1-q_{ij}}, \quad 0 \le q_{ij}<1.
$$

ODME fits this variant with the Clarabel exponential-cone formulation used by
other W strength fits. The result includes solver status, residuals, `max_q`,
and lifted problem-size diagnostics.

## Python API

```python
import numpy as np
from odme.models import (
    fit_strength_cost_geometric,
    fit_strength_cost_poisson,
    sample_strength_cost_geometric,
    sample_strength_cost_poisson,
)

fit = fit_strength_cost_poisson(
    s_out, s_in,
    cost_sources, cost_targets, cost_values,
    target_cost,
)
sample = sample_strength_cost_poisson(fit, cost_sources, cost_targets, cost_values, seed=42)

w_fit = fit_strength_cost_geometric(
    s_out, s_in,
    cost_sources, cost_targets, cost_values,
    target_cost,
)
w_sample = sample_strength_cost_geometric(
    w_fit, cost_sources, cost_targets, cost_values, seed=42,
)
```

## CLI

```bash
odme fit strength-cost-me edges.csv --costs costs.csv --target-cost 120.0
odme generate strength-cost-me edges.csv --costs costs.csv --target-cost 120.0 --seed 42
```

If `--target-cost` is omitted, the CLI computes it from the observed edges and
cost table. Observed edges missing from the cost table contribute zero cost.

## Partial-constraint variant

When some rates are known, ODME subtracts their strength and cost contribution
and fits the free pairs:

```python
from odme.models.partial import fit_partial_strength_cost_poisson

result = fit_partial_strength_cost_poisson(
    s_out, s_in,
    known_source, known_target, known_rate,
    cost_sources, cost_targets, cost_values,
    target_cost,
)
```

See [Partial Constraints](partial-constraints.md).
