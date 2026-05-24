---
description: Python API reference for ODME.
---

# Python API

## Data types

| Type | Module | Description |
|------|--------|-------------|
| `EdgeTable` | `odme.data.frames` | Sparse edge table with `source`, `target`, `weight` numpy arrays |
| `ProbabilityTable` | `odme.data.frames` | Sparse table with `source`, `target`, `probability` arrays |
| `FitResult` | `odme.models.fitting` | Fitted strength/degree multipliers `x`, `y`, with `family`, optional `layers`, and diagnostics |
| `StrengthCostFit` | `odme.models.fitting` | Fitted strength-cost model with `x`, `y`, `gamma`, `family`, optional `layers`, and diagnostics |
| `StrengthEdgesFit` | `odme.models.fitting` | Fitted strength-edges model with `x`, `y`, `lam`, `family`, optional `layers`, and diagnostics |
| `StrengthDegreeFit` | `odme.models.fitting` | Fitted strength-degree model with `x`, `y`, `z`, `w` |
| `OptimizationDiagnostics` | `odme.models.fitting` | Shared convergence/status/objective/residual diagnostics |
| `ConicDiagnostics` | `odme.models.fitting` | W solver metrics; historically conic-named and nested under `diagnostics.conic` |
| `FilterResult` | `odme.filtering` | Filtering output with upper, lower, compatible, absent tables |
| `FilteredEdges` | `odme.filtering` | Edge subset with p-values, expected weights, occupation |
| `SyntheticNetwork` | `odme.utilities.synthetic` | Canonical PA geographic network for tests/benchmarks |
| `SyntheticConstraints` | `odme.utilities.synthetic` | Constraints derived from a synthetic network |

## I/O

| Function | Description |
|----------|-------------|
| `read_edges(path)` | Read edges from CSV, TSV, Parquet, Arrow IPC, GraphML, MTX, Pajek |
| `write_edges(edges, path)` | Write edges to CSV, TSV, Parquet, or Arrow IPC |
| `read_probabilities(path)` | Read sparse probability/rate table |

## Analysis

| Function | Description |
|----------|-------------|
| `directed_strengths(edges)` | Out/in strength sequences |
| `directed_degrees(edges)` | Out/in degree sequences |
| `compute_all_stats(edges)` | Full node-level statistics |
| `weight_distribution(edges)` | Edge weight histogram |
| `clustering_coefficient(edges)` | Binary clustering coefficient |
| `weighted_clustering_coefficient(edges)` | Weighted clustering coefficient |
| `ensemble_average(generate=..., analyze=..., repetitions=...)` | Mean/std of per-node statistics across samples |
| `ensemble_scalar_average(generate=..., compute=..., repetitions=...)` | Mean/std of scalar statistics across samples |

See [Network Metrics](network-metrics.md) for optional NetworkX/rustworkx user-side recipes.

## Fitting

| Group | Functions |
|---|---|
| Strength | `fit_strength_poisson`, `fit_strength_binomial`, `fit_strength_geometric`, `fit_strength_negative_binomial` |
| Strength-cost | `fit_strength_cost_poisson`, `fit_strength_cost_binomial`, `fit_strength_cost_geometric`, `fit_strength_cost_negative_binomial` |
| Coordinate strength-cost | `fit_strength_cost_*_coordinates` for ME, B, W, Wnb |
| Strength-edges | `fit_strength_edges_poisson`, `fit_strength_edges_binomial`, `fit_strength_edges_geometric`, `fit_strength_edges_negative_binomial` |
| Strength-degree | `fit_strength_degree_poisson`, `fit_strength_degree_binomial`, `fit_strength_degree_geometric`, `fit_strength_degree_negative_binomial` |
| Degree-events | `fit_degree_events_poisson`, `fit_degree_events_binomial`, `fit_degree_events_geometric`, `fit_degree_events_negative_binomial` |
| Degree-only | `fit_degree_bernoulli` |

Known caveat: B strength-edges and B strength-degree fitting wrappers currently
call ME kernels and must not be used for scientific results until fixed. See the
[ontology audit](../development/ontology-conformance-audit.md).

## Synthetic fixtures

| Function | Description |
|----------|-------------|
| `generate_pa_geographic_network` | Preferential-attachment support with geographic degree-weighted events |
| `derive_synthetic_constraints` | Strengths, degrees, edge count, total events, total cost, B layers |
| `known_pairs_from_network` | Deterministic strongest-edge known pairs for partial tests |

## Generation

| Function | Model | Ensemble |
|----------|-------|----------|
| `sample_strength_poisson` | fixed-strength Poisson | grand-canonical |
| `sample_strength_multinomial` | fixed-strength multinomial | canonical |
| `sample_strength_poisson_multinomial` | Poisson-total multinomial | mixed |
| `sample_strength_stub_matching` | stub-matching | stub_matching |
| `sample_strength_edges_poisson` | strength-edges zero-inflated | grand-canonical |
| `sample_strength_edges_geometric` | W strength-edges zero-inflated | grand-canonical |
| `sample_strength_edges_negative_binomial` | W strength-edges zero-inflated | grand-canonical |
| `sample_strength_degree_poisson` | strength-degree zero-inflated | grand-canonical |
| `sample_strength_cost_poisson` | strength-cost Poisson | grand-canonical |
| `sample_strength_cost_geometric` | W strength-cost geometric | grand-canonical |
| `sample_strength_cost_negative_binomial` | W strength-cost negative binomial | grand-canonical |
| `sample_degree_events_poisson` | degree-events zero-inflated | grand-canonical |
| `sample_custom_poisson` | custom sparse Poisson | grand-canonical |
| `sample_custom_multinomial` | custom sparse multinomial | canonical |

## Filtering

| Function | Null model |
|----------|------------|
| `filter_strength_poisson` | Poisson, auto-fitted |
| `filter_strength_cost_poisson` | Poisson with costs, pre-fitted |
| `filter_strength_edges_poisson` | zero-inflated, pre-fitted |
| `filter_strength_degree_poisson` | zero-inflated, pre-fitted |
| `filter_degree_events_poisson` | zero-inflated; should migrate from manual rate parameters to `DegreeEventsFit` |
| `filter_custom_poisson` | Poisson, user/partial rates |

## Partial fitting

| Function | Model | Constraints |
|----------|-------|-------------|
| `fit_partial_strength_cost_poisson_coordinates` | Poisson partial known weights | strengths + Euclidean cost |
| `fit_partial_strength_cost_binomial_coordinates` | Binomial partial known weights | strengths + Euclidean cost |
| `fit_partial_strength_cost_geometric_coordinates` | Geometric W partial known weights | strengths + Euclidean cost |
| `fit_partial_strength_cost_negative_binomial_coordinates` | Negative-binomial W partial known weights | strengths + Euclidean cost |

See [Filtering API](filtering.md) for detailed parameter tables and examples.
