---
description: Move model routing above fit, sample, and filter domains.
---

# 0016. Unified verb router owns model workflow dispatch

## TL;DR

MENoBiS routes `fit`, `sample`, and `filter` through `menobis.routing` using
`Verb`, `Ensemble`, `ModelFamily`, and `Constraint`. Model families are ME, B,
and W; `layers` selects the W realization.

## Context

The earlier router lived in `menobis.models.routing`. It mixed fitting,
sampling, and filtering, so the models package imported the filtering package.
That created a circular domain dependency and made the router hard to extend.

## Decision

- Use `menobis.routing.route_model(verb=..., family=..., constraint=...)` as the
  unified orchestration API.
- Keep thin wrappers:
  - `menobis.models.fit_model`
  - `menobis.models.sample_model`
  - `menobis.filtering.filter_model`
- Move shared vocabulary to `menobis.models.spec`.
- Replace the public `Family` enum with `ModelFamily` values `ME`, `B`, and `W`.
- Use `layers=1` for W geometric and `layers>1` for W negative-binomial.
- Do not keep `menobis.models.routing` as a compatibility shim.

## Examples

```python
from menobis.models import Constraint, ModelFamily, fit_model
from menobis.filtering import filter_model

fit = fit_model(
    family=ModelFamily.W,
    constraint=Constraint.STRENGTH,
    strength_out=s_out,
    strength_in=s_in,
    layers=1,
)
result = filter_model(edges, family=ModelFamily.ME, constraint=Constraint.STRENGTH)
```

## Consequences

The model package no longer imports filtering. Adding a route now updates one
orchestration layer while each domain keeps its low-level implementation wrappers.
