---
description: Strength-cost model for spatially constrained multi-edge networks.
---

# Spatial costs

## TL;DR

The **strength-cost ME** model is thesis case 2. It constrains outgoing and
incoming strengths plus total cost, with expectation
$\mathbb{E}[t_{ij}] = x_i y_j e^{-\gamma d_{ij}}$.

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

## Solver

ODME currently uses a two-level structure:

| Level | Method | What it uses |
|-------|--------|--------------|
| Inner | IPF balancing | exact multiplicative structure for $x,y$ at fixed $\gamma$ |
| Outer | adaptive scalar search | warm-started updates of $\gamma$ to match $C$ |

The dual problem is convex under the usual feasible maximum-entropy setup, but
the current implementation does **not** call a generic convex optimizer. It
exploits separability/IPF and warm starts. Future work can replace the outer
search with bracketed bisection/Brent or a gradient-based convex-dual solve.

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
from odme.models.partial import fit_partial_strength_cost_me

result = fit_partial_strength_cost_me(
    s_out, s_in,
    known_source, known_target, known_rate,
    cost_sources, cost_targets, cost_values,
    target_cost,
)
```

See [Partial Constraints](partial-constraints.md).
