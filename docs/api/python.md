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
| `StrengthCostMEFit` | `odme.models.fitting` | Fitted strength-cost model with `x`, `y`, `gamma` |
| `StrengthEdgesMEFit` | `odme.models.fitting` | Fitted strength-edges model with `x`, `y`, `lam` |
| `StrengthDegreeMEFit` | `odme.models.fitting` | Fitted strength-degree model with `x`, `y`, `z`, `w` |
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
| `fit_fixed_strength_me` | Poisson | strengths |
| `fit_fixed_degree_binary` | Bernoulli | degrees |
| `fit_strength_edges_me` | ZIP | strengths + edge count |
| `fit_strength_degree_me` | ZIP | strengths + degrees |
| `fit_strength_cost_me` | Poisson with cost | strengths + spatial cost |

## Generation

| Function | Model | Ensemble |
|----------|-------|----------|
| `sample_poisson` | fixed-strength Poisson | grand-canonical |
| `sample_multinomial` | fixed-strength multinomial | canonical |
| `sample_poisson_multinomial` | Poisson-total multinomial | mixed |
| `sample_microcanonical` | stub-matching | microcanonical |
| `sample_strength_edges_me` | strength-edges ZIP | grand-canonical |
| `sample_strength_degree_me` | strength-degree ZIP | grand-canonical |
| `sample_strength_cost_me` | strength-cost Poisson | grand-canonical |
| `sample_fixed_degree_events_me` | degree-events ZIP | grand-canonical |
| `sample_custom_pij_events_poisson` | custom sparse Poisson | grand-canonical |
| `sample_custom_pij_events_multinomial` | custom sparse multinomial | canonical |

## Filtering

| Function | Null model |
|----------|------------|
| `filter_fixed_strength_me` | Poisson, auto-fitted |
| `filter_strength_cost_me` | Poisson with costs, pre-fitted |
| `filter_strength_edges_me` | ZIP, pre-fitted |
| `filter_strength_degree_me` | ZIP, pre-fitted |
| `filter_degree_events_me` | ZIP, manual parameters |
| `filter_custom_rates_poisson` | Poisson, user/partial rates |

See [Filtering API](filtering.md) for detailed parameter tables and examples.
