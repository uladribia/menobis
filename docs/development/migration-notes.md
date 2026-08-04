# Phase 0 migration notes

**TL;DR** — MENoBiS now uses occupation-number terminology end to end. Public
APIs changed without backward-compatibility shims. This page records the
renames and the supported-route matrix.

## Terminology

| Old | New |
|---|---|
| weighted network | non-binary network |
| weight / weights | occ_num / occ_nums |
| weighted edge | occupied pair |
| `WeightedEdge` | `OccupiedPair` |
| `SampledEdges` | `SampledNetwork` |
| `WeightFamily` | `OccupationFamily` |
| `WeightDistribution` | `OccupationDistribution` |
| `weight_distribution` | `occupation_distribution` |
| `known_rate` | `known_occnum` |
| partial-fit `rate` | `intensity` |
| `weighted_clustering_coefficient` | `occupation_clustering_coefficient` |

Legacy file readers still accept a `weight` column and GraphML `weight`
attribute; writers emit `occ_num`.

## Constraint set

`strength`, `strength_cost`, `strength_edges`, `strength_degree`,
`degree_events`, `edges_events` (new in Phase 0).

## Supported routes

The machine-readable source of truth is `menobis.capabilities.REGISTRY`,
keyed by `(verb, ensemble, family, constraint)`.

| Verb | Ensemble | Families | Constraints |
|---|---|---|---|
| fit | grand-canonical, canonical(ME) | ME, B, W | all six |
| sample | grand-canonical | ME, B, W | all six |
| sample | canonical | ME | strength (exact T) |
| sample | microcanonical | ME | strength (exact s, self-loops only) |
| filter | grand-canonical | ME, B, W | all six |

## New public API

- `fit_model(..., node_count=...)` — required by `edges_events`.
- `sample_model_detailed(...) -> SamplingResult` — adds method/exactness/seed.
- `sample_model(...)` delegates to it.
- `menobis.analysis.analyze(...)` — composable facade.
- `Constraint.EDGES_EVENTS`, `EdgesEventsFit`.

## Generated networks

All generated occupation numbers are positive integers in family support
(B bounded by layers). `EdgeTable` exposes `occ_num`.

## See also

- `docs/development/architecture.md`
- `docs/development/agent-specifications/microcanonical-phase-0.md`
