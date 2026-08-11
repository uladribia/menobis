---
description: Stable public Python API for MENoBiS.
---

# Python API

## TL;DR

Use the unified public entry points first: `fit_model`, `sample_model`, and
`filter_model`. They route by ensemble, family, and constraint.

## Main imports

```python
from menobis.models import Constraint, Ensemble, ModelFamily, fit_model, sample_model
from menobis.filtering import filter_model
from menobis.data.io import read_edges, write_edges
```

## Model selectors

| Selector | Values |
|---|---|
| `ModelFamily` | `ME`, `B`, `W` |
| `Ensemble` | `GRAND_CANONICAL`, `CANONICAL`, `MICROCANONICAL` |
| `Constraint` | `STRENGTH`, `STRENGTH_COST`, `STRENGTH_EDGES`, `STRENGTH_DEGREE`, `DEGREE_EVENTS` |

## Core functions

| Function | Purpose |
|---|---|
| `fit_model(...)` | solve model parameters from constraints |
| `sample_model(...)` | sample a sparse `EdgeTable` from a fit or ME stubs |
| `filter_model(edges, ...)` | classify edges against an independent null |

## Minimal fit/sample/filter

```python
fit = fit_model(
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    strength_out=strength_out,
    strength_in=strength_in,
    self_loops=False,
)

sample = sample_model(
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    fit=fit,
    seed=42,
)

result = filter_model(
    edges,
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    fit=fit,
)
```

## Microcanonical sampling

Microcanonical cases require no fitting step (constraints are hard or matched
in expectation). Validated at **N=1000** across all families and regimes
(see [Microcanonical concept doc](../concepts/microcanonical.md)). ME, B, and W
families are supported where applicable.

### Fixed (E,T) — EDGES_EVENTS

Exact `E` occupied pairs and `T` total events, no fitting step:

```python
net = sample_model(
    ensemble=Ensemble.MICROCANONICAL,
    family=ModelFamily.B,
    constraint=Constraint.EDGES_EVENTS,
    node_count=100,
    target_edges=500,   # exact E
    total_events=1500,  # exact T
    layers=4,           # B/W only
    self_loops=False,
    seed=42,
)
# net has exactly 500 occupied pairs summing to 1500 events
```

Fixed pairs are frozen via `known_source`, `known_target`, `known_occnum`;
their contribution is subtracted from `E`/`T`, the residual is sampled, and
they are merged back. See
[Microcanonical sampling](../concepts/microcanonical.md).

### Fixed (k,T) — DEGREE_EVENTS

Exact out-degree, in-degree, and total events via MCMC:

```python
from menobis.models.spec import Ensemble, Constraint
net = sample_model(
    ensemble=Ensemble.MICROCANONICAL,
    family=ModelFamily.B,
    constraint=Constraint.DEGREE_EVENTS,
    degree_out=degree_out, degree_in=degree_in,
    total_events=5000, layers=4, seed=42,
)
```

### Fixed strengths — STRENGTH

Exact strength sequences via compressed constructor + repair + occupied-cell MCMC:

```python
net = sample_model(
    ensemble=Ensemble.MICROCANONICAL,
    family=ModelFamily.W,
    constraint=Constraint.STRENGTH,
    strength_out=strength_out, strength_in=strength_in,
    layers=2, seed=42,
)
```

### Fixed strengths + expected cost — STRENGTH_COST

Strengths are exact; total cost is matched in expectation by fitting the
cost multiplier `gamma` via stochastic bisection. Returns the sampled
network plus gamma diagnostics:

```python
from menobis.routing import sample_model_detailed
result = sample_model_detailed(
    ensemble=Ensemble.MICROCANONICAL,
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH_COST,
    strength_out=strength_out, strength_in=strength_in,
    coord_x=coord_x, coord_y=coord_y,
    target_cost=observed_cost, seed=42,
)
# result.edges, result.diagnostics.gamma, result.diagnostics.converged
```
they are merged back. See
[Microcanonical fixed-(E,T) sampling](../concepts/microcanonical.md).

## Common data/result types

| Type | Module | Meaning |
|---|---|---|
| `EdgeTable` | `menobis.data.frames` | sparse `source`, `target`, `occ_num` arrays |
| `ProbabilityTable` | `menobis.data.frames` | sparse custom probabilities/rates |
| `FitResult` | `menobis.models` | base fit protocol with diagnostics |
| `StrengthFit` | `menobis.models` | strength multipliers `x`, `y` |
| `StrengthCostFit` | `menobis.models` | `x`, `y`, and `gamma` |
| `StrengthEdgesFit` | `menobis.models` | `x`, `y`, and global edge multiplier |
| `StrengthDegreeFit` | `menobis.models` | strength and degree multipliers |
| `DegreeEventsFit` | `menobis.models` | degree occupation plus positive-occupation intensity |
| `FilterResult` | `menobis.filtering` | upper/lower/compatible/absent classifications |

## Analysis helpers

| Function | Purpose |
|---|---|
| `directed_strengths(edges)` | out/in strengths |
| `directed_degrees(edges)` | out/in binary degrees |
| `compute_all_stats(edges)` | strengths, degrees, Y2, nearest-neighbour stats |
| `occupation_distribution(edges)` | occupation-count histogram |
| `clustering_coefficient(edges)` | binary clustering |
| `occupation_clustering_coefficient(edges)` | occupation-weighted clustering helper |

## Synthetic fixtures

Use these in examples and tests to avoid infeasible arbitrary constraints:

```python
from menobis.utilities.synthetic import (
    derive_synthetic_constraints,
    generate_pa_geographic_network,
)
```
