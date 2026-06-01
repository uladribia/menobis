---
description: Partial-constraint fitting with frozen known node pairs.
---

# Partial constraints

## TL;DR

Partial fitting freezes known node-pair occupations, subtracts their contribution
from the requested constraints, and fits the parent model on the remaining free
support.

!!! note "Not a separate family"
    Partial constraints are a support transformation. The inner solver remains
    the selected ME, B, or W family with the selected constraint layer.

## When to use partial fitting

| Situation | Why partial helps |
|---|---|
| Known structural links | keep trusted node pairs fixed |
| Data integration | combine observed high-confidence pairs with null-model free pairs |
| Filtering with frozen pairs | avoid treating known occupations as random |
| Scenario analysis | remove or keep a selected support while preserving totals |

## Excess constraints

If $Q$ is the set of frozen pairs and $r_{ij}$ is the known occupation, then the
free outgoing strength is:

$$
s_i^{out,free}=s_i^{out}-\sum_{j:(i,j)\in Q} r_{ij}.
$$

Incoming strengths, binary degrees, total binary edges, and total cost are
reduced in the same way. Negative excess means the partial problem is infeasible.

## Public route

Pass `known_source`, `known_target`, and `known_rate` to `fit_model`:

```python
partial_fit = fit_model(
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH_COST,
    strength_out=strength_out,
    strength_in=strength_in,
    target_cost=total_cost,
    coord_x=x,
    coord_y=y,
    known_source=known_source,
    known_target=known_target,
    known_rate=known_rate,
    self_loops=False,
)
```

`partial_fit` is a sparse `PartialFitResult` with `source`, `target`, and `rate`
arrays for frozen and fitted free pairs.

## Implementation rule

```text
partial fit = compute excess constraints + free support + full family solver
```

The partial path should not duplicate solver math. Masks, excess computation,
and sparse rate-table assembly belong in Rust.
