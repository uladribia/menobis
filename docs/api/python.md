---
description: Python API reference for ODME.
---

# Python API

## Data types

| Type | Module | Description |
|------|--------|-------------|
| `EdgeTable` | `odme.data.frames` | Sparse edge table with `source`, `target`, `weight` numpy arrays |
| `ProbabilityTable` | `odme.data.frames` | Sparse table with `source`, `target`, `probability` arrays |
| `FitResult` | `odme.models.fitting` | Fitted Lagrange multipliers `x`, `y` |
| `StrengthCostFit` | `odme.models.fitting` | Fitted strength-cost model with `x`, `y`, `gamma` |
| `StrengthEdgesFit` | `odme.models.fitting` | Fitted strength-edges model with `x`, `y`, `lam` |
| `StrengthDegreeFit` | `odme.models.fitting` | Fitted strength-degree model with `x`, `y`, `z`, `w` |
| `FilterResult` | `odme.filtering` | Filtering output with upper, lower, compatible, absent tables |
| `FilteredEdges` | `odme.filtering` | Edge subset with p-values, expected weights, occupation |

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

## Fitting

| Function | Model | Constraints |
|----------|-------|-------------|
| `fit_strength_poisson` | Poisson | strengths |
| `fit_degree_bernoulli` | Bernoulli | degrees |
| `fit_strength_edges_poisson` | zero-inflated | strengths + edge count |
| `fit_strength_degree_poisson` | zero-inflated | strengths + degrees |
| `fit_strength_cost_poisson` | Poisson with cost | strengths + spatial cost |

## Generation

| Function | Model | Ensemble |
|----------|-------|----------|
| `sample_strength_poisson` | fixed-strength Poisson | grand-canonical |
| `sample_strength_multinomial` | fixed-strength multinomial | canonical |
| `sample_strength_poisson_multinomial` | Poisson-total multinomial | mixed |
| `sample_strength_stub_matching` | stub-matching | stub_matching |
| `sample_strength_edges_poisson` | strength-edges zero-inflated | grand-canonical |
| `sample_strength_degree_poisson` | strength-degree zero-inflated | grand-canonical |
| `sample_strength_cost_poisson` | strength-cost Poisson | grand-canonical |
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
| `filter_degree_events_poisson` | zero-inflated, manual parameters |
| `filter_custom_poisson` | Poisson, user/partial rates |

See [Filtering API](filtering.md) for detailed parameter tables and examples.
