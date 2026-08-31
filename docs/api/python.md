---
description: Concise reference for the public MENoBiS Python API — selectors, entry points, result types, and analysis helpers.
---

# Python API

## TL;DR

The unified public entry points are `fit_model`, `sample_model`,
`sample_model_detailed`, and `filter_model`, all routed by `ensemble`,
`family`, and `constraint`. Supported combinations are the generated
[Supported models](../guide/supported-models.md) matrix.

## Main imports

```python
from menobis.filtering import filter_model
from menobis.models import Constraint, Ensemble, ModelFamily, fit_model, sample_model
from menobis.routing import sample_model_detailed
from menobis.data import read_edges, write_edges
```

## Selectors / enums

| Selector | Values |
|---|---|
| `ModelFamily` | `ME`, `B`, `W` |
| `Ensemble` | `GRAND_CANONICAL`, `CANONICAL`, `MICROCANONICAL` |
| `Constraint` | `STRENGTH`, `STRENGTH_COST`, `STRENGTH_EDGES`, `STRENGTH_DEGREE`, `DEGREE_EVENTS`, `EDGES_EVENTS` |
| `Verb` | `FIT`, `SAMPLE`, `FILTER` |

## Top-level entry points

| Function | Purpose |
|---|---|
| `route_model(verb, ensemble=..., family=..., constraint=..., **kwargs)` | shared router (all verbs) |
| `fit_model(ensemble=..., family=..., constraint=..., ...)` | solve model parameters from constraints |
| `sample_model(ensemble=..., family=..., constraint=..., fit=..., ...)` | sample one `EdgeTable` |
| `sample_model_detailed(ensemble=..., family=..., constraint=..., fit=..., ...)` | sample plus `SamplingResult` metadata |
| `filter_model(edges, family=..., constraint=..., fit=..., ...)` | classify observed edges against the null |

Key keywords shared across verbs:

- selectors: `ensemble`, `family`, `constraint`;
- constraints: `strength_out`, `strength_in`, `degree_out`, `degree_in`,
  `total_events`, `target_edges`, `node_count`, `target_cost` (plus
  `coord_x`, `coord_y` for cost);
- fixed pairs: `known_source`, `known_target`, `known_occnum`
  (see [Fixed / known pairs](../guide/fixed-pairs.md));
- common: `layers` (B/W), `self_loops`, `seed`;
- MCMC: `burn_in_sweeps`, `sweeps_per_sample`;
- filtering: `alpha`, `tail`, `correction`, `detect_absent`.

## Fit

```python
fit = fit_model(
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    strength_out=strength_out,
    strength_in=strength_in,
    self_loops=False,
)
if not fit.converged:
    raise RuntimeError(fit.status)
```

Fit results are typed (`StrengthFit`, `StrengthCostFit`, `StrengthEdgesFit`,
`StrengthDegreeFit`, `DegreeEventsFit`, `EdgesEventsFit`,
`PartialFitResult`); each exposes `converged`, `status`, `self_loops`, and
family-specific multipliers.

## Sample

```python
sample = sample_model(
    ensemble=Ensemble.GRAND_CANONICAL,
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    fit=fit,
    seed=42,
)
```

`sample_model` returns an `EdgeTable`. For the generation method and
exactness, use `sample_model_detailed` — it returns a `SamplingResult`:

```python
result = sample_model_detailed(
    ensemble=Ensemble.GRAND_CANONICAL,
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    fit=fit,
    seed=42,
)
edges = result.edges            # EdgeTable
diagnostics = result.diagnostics
print(diagnostics.exactness)    # generation exactness category
```

`SamplingResult` fields: `edges`, `ensemble`, `family`, `constraint`,
`method`, `exactness`, `seed`, `diagnostics`. For the microcanonical
strength+cost route, `diagnostics` also carries `gamma`, `expected_cost`,
`observed_cost`, `cost_residual`, and `converged`.

## Filter

```python
result = filter_model(
    edges,
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    fit=fit,
    correction="fdr",
)
significant = result.upper.edges
```

See [Filter node pairs](../guide/filter-network.md) for tails, corrections,
and absent-edge options.

## Result types

| Type | Module | Meaning |
|---|---|---|
| `EdgeTable` | `menobis.data` | sparse `source`, `target`, `occ_num` arrays |
| `ProbabilityTable` | `menobis.data` | sparse custom probabilities/rates |
| `FitResult` | `menobis.models` | base fit protocol with diagnostics |
| `StrengthFit` | `menobis.models` | strength multipliers `x`, `y` |
| `StrengthCostFit` | `menobis.models` | `x`, `y`, and `gamma` |
| `StrengthEdgesFit` | `menobis.models` | `x`, `y`, and support multiplier |
| `StrengthDegreeFit` | `menobis.models` | strength/degree multipliers |
| `DegreeEventsFit` | `menobis.models` | degree multipliers, `q`, occupation intensity |
| `EdgesEventsFit` | `menobis.models` | global `q`, `occupation`, `positive_mean` |
| `SamplingResult` | `menobis.models` | sampled edges + metadata |
| `FilterResult` | `menobis.filtering` | upper/lower/compatible/absent classifications |

## Analysis helpers

| Function | Purpose |
|---|---|
| `directed_strengths(edges)` | out/in strengths |
| `directed_degrees(edges)` | out/in binary degrees |
| `compute_all_stats(edges)` | strengths, degrees, Y2, nearest-neighbour stats |
| `occupation_distribution(edges)` | occupation-count histogram |
| `clustering_coefficient(edges)` | binary-support clustering |
| `occupation_clustering_coefficient(edges)` | occupation-based clustering |
| `ensemble_average(...)` / `ensemble_scalar_average(...)` | ensemble aggregation helpers |

Definitions and formulas: [Ensemble statistics](../guide/ensemble-statistics.md).

## Synthetic fixtures

Use these in examples and tests so constraints are feasible by construction:

```python
from menobis.utilities.synthetic import (
    derive_synthetic_constraints,
    generate_pa_geographic_network,
)
```

## Guarantees

The signatures in this page are exercised by `tests/test_docs_examples.py`
and `tests/test_public_docs_contract.py`; if a documentation example
changes, its smoke test changes in the same commit. Supported route
combinations come from the capability registry, never from prose
([Supported models](../guide/supported-models.md)).