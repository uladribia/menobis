# PLAN.md — Outstanding Work

## 1. Router as Single Dispatch Point

The unified router (`menobis.routing.{fit_model, sample_model, filter_model}`)
must become the only entry point for all model operations. Currently it has gaps.

### fit_model — COMPLETE

All 12 family x constraint combinations dispatch correctly.

### sample_model — 9 of 12 MISSING

Only `strength` constraint is routed. Cost, edges, and degree constraints raise
`UnsupportedModelCaseError`.

| Constraint | ME | B | W |
|---|---|---|---|
| strength | ok | ok | ok |
| strength-cost | MISSING | MISSING | MISSING |
| strength-edges | MISSING | MISSING | MISSING |
| strength-degree | MISSING | MISSING | MISSING |

**What to do**:
1. Add `coord_x`, `coord_y` parameters to `sample_model` signature.
2. Extend the dispatch table in `_sample_model` (in `routing.py`) to call the
   existing `sample_strength_cost_*`, `sample_strength_edges_*`, and
   `sample_strength_degree_*` standalone functions.
3. The standalone functions already exist and work — only the routing is missing.

### filter_model — 9 of 12 BROKEN (TypeError)

The dispatch table exists for all constraints but the router passes
`self_loops=` to filter functions that don't accept it. Additionally,
cost-constraint filters need `coord_x`/`coord_y` which the router doesn't pass.

| Constraint | ME | B | W |
|---|---|---|---|
| strength | ok | ok | ok |
| strength-cost | TypeError | TypeError | TypeError |
| strength-edges | TypeError | TypeError | TypeError |
| strength-degree | TypeError | TypeError | TypeError |

**What to do**:
1. Stop passing `self_loops` to filter functions that don't accept it
   (cost/edges/degree infer it from the fit result).
2. Add `coord_x`, `coord_y` to `filter_model` signature and pass them only
   to cost-constraint filter functions.
3. Alternatively, unify all filter function signatures to accept `**kwargs`
   and ignore unknown parameters.

### After fixing the router

Once the router is complete:
- Delete all manual dispatch in `benchmarks/cli.py` (`_sample`, `_filter_one`)
- Replace with `sample_model(...)` and `filter_model(...)` calls
- The benchmark CLI shrinks by ~100 lines
- The CLI commands (`menobis fit`, `menobis sample`, `menobis filter`) become
  fully functional for all 12 model cases

**Effort**: ~50 lines of routing.py changes + test updates.

---


## 3. Solver Convergence Issues

Documented in `docs/decisions/convergence-issues.md`. Summary:

| Case | Status | Proposed fix |
|---|---|---|
| W strength-edges | rarely converges | barrier + adaptive damping |
| W strength-degree | rarely converges | barrier + adaptive damping |
| ME/B strength-edges (very sparse) | fragile | scaled log-lambda + better init |

These are algorithm improvements, not missing plumbing.
