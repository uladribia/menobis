---
description: The core MENoBiS workflow — fit a model, sample from it, and inspect fitted and sampled results.
---

# Fit and sample

## TL;DR

```python
from menobis.models import Constraint, Ensemble, ModelFamily, fit_model, sample_model

fit = fit_model(
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    strength_out=strength_out,
    strength_in=strength_in,
)
if not fit.converged:
    raise RuntimeError(fit.status)

sample = sample_model(
    ensemble=Ensemble.GRAND_CANONICAL,
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    fit=fit,
    seed=0,
)
```

## Fit

`fit_model` selects the route by `family`, `constraint`, and optional
`ensemble` (default grand canonical) and returns a typed fit result (e.g.
`StrengthFit`, `StrengthCostFit`, `StrengthEdgesFit`, `StrengthDegreeFit`,
`DegreeEventsFit`, `EdgesEventsFit`). Fit results expose at least:

- `fit.converged` — solver convergence flag;
- `fit.status` — human-readable solver status;
- `fit.self_loops` — the self-loop policy used;
- family-specific Lagrange multipliers (`x`, `y`; `lam`; `z`, `w`; `gamma`;
  `q`; `occupation`) — rarely needed by users.

**Always check `fit.converged` before sampling.** An unconverged fit must
not be silently sampled; the exact failure text is in `fit.status`.

Constraints must be feasible for the chosen model. Derive them from a valid
witness network when possible (see [Getting started](../getting-started.md));
hand-picked sequences are easy to make infeasible.

## Sample

`sample_model` returns an `EdgeTable` — the sparse occupied-pair table with
columns `source target occ_num`.

For metadata about the draw, use `sample_model_detailed`:

```python
from menobis.routing import sample_model_detailed

result = sample_model_detailed(
    ensemble=Ensemble.GRAND_CANONICAL,
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    fit=fit,
    seed=0,
)
edges = result.edges          # EdgeTable
diagnostics = result.diagnostics
print(diagnostics.exactness)  # generation category (see below)
```

`result.diagnostics.exactness` states **how exact** the draw is:

- `exact_independent` — grand-canonical: per-pair draws from the fitted law;
- `exact_direct` — direct sampler from the target distribution (canonical
  and microcanonical `(E,T)` routes);
- `exact_stationary_mcmc` — validated MCMC kernel; finite-run burn-in and
  mixing still matter (see [MCMC diagnostics](../performance/mcmc-diagnostics.md)).

## Seeds and reproducibility

Sampling is deterministic given the `seed` argument: the same call with the
same `seed` and the same fit yields the same `EdgeTable`. Use different
seeds to generate an ensemble of independent draws.

## Canonical

For canonical sampling (ME + STRENGTH), pass the fitted model plus the total
occupation to fix:

```python
sample = sample_model(
    ensemble=Ensemble.CANONICAL,
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    fit=fit,
    total_events=total_events,
)
```

Canonical fixes total occupation \(T\) exactly; strengths stay soft fitted
quantities.

## Microcanonical

Microcanonical routes sample directly from constraints and need **no fit**:

```python
sample = sample_model(
    ensemble=Ensemble.MICROCANONICAL,
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    strength_out=strength_out,
    strength_in=strength_in,
)
```

See [Microcanonical sampling](microcanonical.md) for the per-route
arguments and exactness.

## Shared entry point

All three verbs route through `route_model(verb, ensemble=..., family=...,
constraint=..., **kwargs)`. The dedicated functions `fit_model`,
`sample_model`, `sample_model_detailed`, and `filter_model` are the
documented public endpoints.

## Filtering

Filtering compares observed edges against the fitted null: see
[Filter node pairs](filter-network.md).