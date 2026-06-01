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

## Common data/result types

| Type | Module | Meaning |
|---|---|---|
| `EdgeTable` | `menobis.data.frames` | sparse `source`, `target`, `weight` arrays |
| `ProbabilityTable` | `menobis.data.frames` | sparse custom probabilities/rates |
| `FitResult` | `menobis.models` | base fit protocol with diagnostics |
| `StrengthFit` | `menobis.models` | strength multipliers `x`, `y` |
| `StrengthCostFit` | `menobis.models` | `x`, `y`, and `gamma` |
| `StrengthEdgesFit` | `menobis.models` | `x`, `y`, and global edge multiplier |
| `StrengthDegreeFit` | `menobis.models` | strength and degree multipliers |
| `DegreeEventsFit` | `menobis.models` | degree occupation plus positive-weight parameter |
| `FilterResult` | `menobis.filtering` | upper/lower/compatible/absent classifications |

## Analysis helpers

| Function | Purpose |
|---|---|
| `directed_strengths(edges)` | out/in strengths |
| `directed_degrees(edges)` | out/in binary degrees |
| `compute_all_stats(edges)` | strengths, degrees, Y2, nearest-neighbour stats |
| `weight_distribution(edges)` | occupation-count histogram |
| `clustering_coefficient(edges)` | binary clustering |
| `weighted_clustering_coefficient(edges)` | occupation-weighted clustering helper |

## Synthetic fixtures

Use these in examples and tests to avoid infeasible arbitrary constraints:

```python
from menobis.utilities.synthetic import (
    derive_synthetic_constraints,
    generate_pa_geographic_network,
)
```
