---
description: Use projected coordinates for strength-cost MENoBiS null models.
---

# Use spatial costs

## TL;DR

Strength-cost models match strengths and total expected cost. Public MENoBiS APIs
use projected XY coordinates and compute Euclidean pair costs on demand; do not
materialize dense `N x N` cost matrices.

## When to use strength-cost

Use it when distance, travel time proxy, or spatial separation is an explicit
structural constraint. It is common for urban mobility and other
origin-destination networks.

!!! note "Other metrics are possible"
    Euclidean distance is the public built-in provider. Road distance, travel
    time, or other metrics can be added by implementing a Rust cost provider
    that computes one pair cost at a time. See
    [Extending thesis cases](../development/extending-thesis-cases.md#add-a-cost-provider).

Do not use latitude/longitude degrees directly. Project first, then pass planar
`x` and `y`.

## Constraint

For projected coordinates, MENoBiS uses:

$$d_{ij}=\sqrt{(x_i-x_j)^2+(y_i-y_j)^2}$$

and fits:

$$C=\sum_{ij}\mathbb{E}[t_{ij}]d_{ij}.$$

## Python example

```python
from menobis.models import Constraint, ModelFamily, fit_model, sample_model

fit = fit_model(
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH_COST,
    strength_out=strength_out,
    strength_in=strength_in,
    target_cost=total_cost,
    coord_x=x,
    coord_y=y,
    self_loops=False,
)

sample = sample_model(
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH_COST,
    fit=fit,
    coord_x=x,
    coord_y=y,
    seed=42,
)
```

## Scaling warning

| Item | Guidance |
|---|---|
| Coordinate storage | O(N) |
| Fitting sweeps | usually O(N² × iterations) |
| Generation/filtering | streamed over candidate pairs |
| Dense cost matrix | not part of public API |

See [Solvers and scaling](../concepts/solvers-and-scaling.md) before running
large strength-cost fits.
