---
description: Router gaps — missing dispatch paths in fit_model/sample_model/filter_model.
---

# Router Gaps

## TL;DR

The unified router (`menobis.routing.{fit_model, sample_model, filter_model}`)
is the intended single dispatch point for all model operations. It currently has
gaps in sampling and filtering for non-strength constraints.

## fit_model — COMPLETE

All 12 family x constraint combinations work:

| Constraint | ME | B | W |
|---|---|---|---|
| strength | ok | ok | ok |
| strength-cost | ok | ok | ok |
| strength-edges | ok | ok | ok |
| strength-degree | ok | ok | ok |

## sample_model — INCOMPLETE

Only `strength` constraint is routed. All other constraints raise
`UnsupportedModelCaseError`:

| Constraint | ME | B | W |
|---|---|---|---|
| strength | ok | ok | ok |
| strength-cost | MISSING | MISSING | MISSING |
| strength-edges | MISSING | MISSING | MISSING |
| strength-degree | MISSING | MISSING | MISSING |

**Root cause**: `sample_model` does not accept `coord_x`/`coord_y` (needed for
cost sampling) and the internal dispatch table (`_sample_model`) only maps
the `STRENGTH` constraint.

**Fix**: Add `coord_x`, `coord_y` parameters to `sample_model`. Extend the
dispatch table in `_sample_model` to call `sample_strength_cost_*`,
`sample_strength_edges_*`, and `sample_strength_degree_*` functions which
already exist as standalone functions.

## filter_model — PARTIAL (kwarg mismatch)

The dispatch table exists for all constraints but passes `self_loops=` to
filter functions that don't accept it. Only `strength` works:

| Constraint | ME | B | W |
|---|---|---|---|
| strength | ok | ok | ok |
| strength-cost | TypeError | TypeError | TypeError |
| strength-edges | TypeError | TypeError | TypeError |
| strength-degree | TypeError | TypeError | TypeError |

**Root cause**: The router passes `self_loops=self_loops` as a keyword argument
to all filter functions, but `filter_strength_cost_*`, `filter_strength_edges_*`,
and `filter_strength_degree_*` don't accept that parameter (they infer it from
the fit result). Also `filter_strength_cost_*` needs `coord_x`/`coord_y`.

**Fix**: Conditionally pass `self_loops` only to filter functions that accept it.
Pass `coord_x`/`coord_y` for cost constraints. Or unify the filter function
signatures to all accept (and ignore) `self_loops`.

## Priority

High — the router must be the only dispatch point per AGENTS.md. The benchmark
currently works around these gaps with manual dispatch for sampling and
filtering. Fixing the router would eliminate ~100 lines of benchmark code and
make the CLI commands (`menobis fit`, `menobis sample`, `menobis filter`)
fully functional for all model cases.

## Estimated effort

- `sample_model`: ~30 lines (add params + extend dispatch table)
- `filter_model`: ~20 lines (fix kwarg passing + add coord params)
- Tests: update `test_menobis_unified_routing.py` with new cases
