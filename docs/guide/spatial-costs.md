---
description: Using pair costs (spatial distance) in MENoBiS — when to use cost, coordinates, units, and grand-canonical vs microcanonical semantics.
---

# Spatial costs

## TL;DR

A pair cost \(d_{ij}\) modulates how much occupation the model expects on
pair \((i,j)\). With an exponential cost

\[
f_{ij}=e^{-\gamma d_{ij}},
\]

the grand-canonical pair fugacity becomes

\[
q_{ij}=x_i y_j f_{ij}=x_i y_j e^{-\gamma d_{ij}}.
\]

Long pairs receive lower fugacity; how strong that penalty is depends on the
fitted cost multiplier \(\gamma\).

## When to use cost

Use a cost constraint when spatial/pair separation is part of the null: for
example, origin–destination trips decay with distance. Add cost when the
model should reproduce the observed total cost \(C=\sum_{ij}t_{ij}d_{ij}\)
and the decay that produces it — otherwise distance effects may masquerade
as higher-order structure.

## Coordinates and units

The built-in Euclidean provider expects projected coordinates:

- provide `coord_x` and `coord_y`, same length as the node count;
- **project your coordinates before taking Euclidean distances** (use
  projected CRS distances with real length units, not degrees);
- \(d_{ij}\) units determine the units of \(C\);
- \(\gamma\) has inverse-distance units under \(f_{ij}=e^{-\gamma d_{ij}}\).

## Grand-canonical semantics

Fit with `STRENGTH_COST`:

```python
from menobis.models import Constraint, ModelFamily, fit_model

fit = fit_model(
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH_COST,
    strength_out=strength_out,
    strength_in=strength_in,
    coord_x=coord_x,
    coord_y=coord_y,
    target_cost=target_cost,
)
if not fit.converged:
    raise RuntimeError(fit.status)
```

Expected strengths and expected total cost are matched; the fitted `gamma`
minimizes the cost residual. Sampling uses the fitted multipliers:

```python
from menobis.models import Constraint, Ensemble, ModelFamily
from menobis.routing import sample_model

sample = sample_model(
    ensemble=Ensemble.GRAND_CANONICAL,
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH_COST,
    fit=fit,
    coord_x=coord_x,
    coord_y=coord_y,
)
```

The W family keeps every fitted \(q_{ij}\in(0,1)\); the same feasibility
rules as plain strength fitting apply.

## Microcanonical hybrid semantics

The microcanonical `STRENGTH_COST` route is **hybrid**:

- strengths are exact;
- cost is controlled **in expectation** through \(\gamma\),
  \(f_{ij}=e^{-\gamma d_{ij}}\).

```python
sample = sample_model(
    ensemble=Ensemble.MICROCANONICAL,
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH_COST,
    strength_out=strength_out,   # exact
    strength_in=strength_in,     # exact
    coord_x=coord_x,
    coord_y=coord_y,
    target_cost=target_cost,     # matched in expectation
)
```

`sample_model_detailed` reports the gamma fit diagnostics
(`converged`, `gamma`, `expected_cost`, `cost_residual`); inspect them
before trusting the draw.

## Scaling warning

The strength+cost routes are the most expensive: fitting solves for the
node multipliers and \(\gamma\), and the microcanonical variant runs a
gamma-search MCMC loop. Budget your sample counts accordingly
(see [Practical scaling](../performance/scaling.md)).

## Implementing a custom cost provider

Implementing a Rust cost provider is contributor work; see
[Extending MENoBiS](../development/extending-thesis-cases.md).