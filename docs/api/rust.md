---
description: Rust API reference for ODME.
---

# Rust API

The public Rust API lives in the `odme-core` crate under `crates/odme-core/`.

## Modules

| Module | Description |
|--------|-------------|
| `distribution` | `PairDistribution` enum with sampling, expectations, p-values |
| `pairs` | `PairDistributionProvider` trait and all null-model providers |
| `filter` | Observed and absent filtering sinks, Benjamini–Hochberg FDR |
| `generation` | Seeded network samplers for all model cases |
| `fitting` | IPF solvers for strength, degree, and combined constraints |
| `cost` | Strength-cost fitting with spatial distance penalties |
| `graph` | Directed strengths, degrees from sparse edge lists |
| `stats` | Node-level statistics and weight distributions |
| `clustering` | Unweighted and weighted clustering coefficients |

## Key types

| Type | Module | Description |
|------|--------|-------------|
| `PairDistribution` | `distribution` | Poisson or ZeroInflatedPoisson pair distribution |
| `PairDistributionProvider` | `pairs` | Trait: compute distribution for any pair on demand |
| `CandidateSupport` | `pairs` | AllPairs or SparsePairs support declaration |
| `SampledEdges` | `generation` | Sparse sampled edge output |
| `ObservedFilterResult` | `filter` | P-values and expectations for observed edges |
| `AbsentFilterResult` | `filter` | Significant absent pairs |
| `AbsentFilterOptions` | `filter` | Thresholds for absent-edge detection |

## Providers

| Provider | Model | Distribution |
|----------|-------|--------------|
| `FixedStrengthPoissonProvider` | fixed-strength | Poisson |
| `StrengthCostPoissonProvider` | strength-cost | Poisson |
| `StrengthEdgesPoissonProvider` | strength-edges | Poisson (zero-inflated) |
| `StrengthDegreePoissonProvider` | strength-degree | Poisson (zero-inflated) |
| `DegreeEventsPoissonProvider` | degree-events | Poisson (zero-inflated) |
| `SparsePoissonRateProvider` | custom sparse | Poisson |
| `NormalizedSparsePoissonProvider` | custom normalized | Poisson |

## Python bindings

The `odme-python` crate in `crates/odme-python/` exposes `odme-core` functions
to Python via PyO3. Python code should use `odme._odme` only through the typed
wrappers in `src/odme/`.
