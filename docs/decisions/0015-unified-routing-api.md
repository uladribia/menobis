# ADR-0015: Unified Python Routing API

## Status

Superseded by [0016](0016-unified-verb-router.md).

## Context

The Python package exposed 60+ per-family functions (fit, sample, filter) with
inconsistent signatures. Users had to know which specific function to call for
each (family, constraint) combination. Filter functions had inconsistent
signatures — ME auto-fitted internally while B/W/NB required pre-computed
multipliers.

## Decision

Three unified entry points with dispatch tables:

```python
from menobis.models import Constraint, Ensemble, ModelFamily, fit_model, sample_model
from menobis.filtering import filter_model

fit = fit_model(family=ModelFamily.ME, constraint=Constraint.STRENGTH, ...)
edges = sample_model(family=ModelFamily.ME, constraint=Constraint.STRENGTH, fit=fit, ...)
result = filter_model(edges, family=ModelFamily.ME, constraint=Constraint.STRENGTH)
```

Design principles:

1. **Dispatch tables** map `(constraint, family)` to the per-family function.
   Both `fit_model` and `filter_model` use the same pattern.
2. **filter_model always calls fit_model** when no fit is provided. No
   special-casing per family.
3. **All filter functions share uniform signature**: `(edges, fit, *, alpha, ...)`.
   The ME filter no longer auto-fits internally.
4. **FitResult** is the public base class (was `_OptimizationView`) in `types.py`.
5. **Per-family functions remain importable** for advanced use but the router is
   the recommended entry point.
6. **CLI commands use the routers** (`fit_model`, `filter_model`).

## Consequences

- Users only need to know `ModelFamily`, `Constraint`, `Ensemble` enums
- Adding a new family or constraint = one entry in each dispatch table
- Filtering is always: get fit → filter (never auto-fit inside filter)
- ~370 lines of `_fit_*` helpers removed in favor of dispatch tables
- All 310 tests pass with the new unified signatures
