---
description: Rust API reference for ODME core abstractions.
---

# Rust API

## TL;DR

`odme-core` owns numerical kernels, fitting loops, pair providers, generation,
and filtering. Python should reach these through PyO3 wrappers only.

## Modules

| Module | Role |
|---|---|
| `distribution` | Pair laws for ME, B, W, and zero-inflated variants |
| `pairs` | `PairDistributionProvider` and candidate support abstractions |
| `fitting` | ME/B/W solvers, support masks, partial excess helpers |
| `generation` | Seeded samplers over provider-backed pair streams |
| `filter` | Observed and absent-edge p-value sinks |
| `graph`, `stats`, `clustering` | Rust-native graph statistics |
| `cost` / support helpers | Cost and distance utilities used by fitting/provider code |

## Key types

| Type | Description |
|---|---|
| `WeightFamily` | ME Poisson, B Binomial(M), W Geometric/NegativeBinomial(M) selector |
| `PairDistribution` | Concrete pair law with sampling, expectation, occupation, p-values |
| `PairDistributionProvider` | Computes a pair distribution on demand |
| `CandidateSupport` | All-pairs or sparse-pairs support declaration |
| `SampledEdges` | Sparse generated edge output |
| `FitResult` and constraint results | Native fit multipliers and convergence flags |

## Provider pipeline

```text
FamilyKernel + ConstraintLayer + CostProvider -> PairDistributionProvider -> sink
```

Generation and filtering should share providers. Fitting should share the same
family kernels and cost providers where possible, instead of duplicating ME/B/W
formula code.

## Current provider coverage

This is generation/filtering provider coverage, not proof that every fitting
solver is conforming.

| Provider concept | Families |
|---|---|
| fixed strength | ME, B, W |
| strength-cost | ME, B, W |
| degree-events | ME, B, W |
| strength-edges | ME, B, W |
| strength-degree | ME, B, W |
| custom sparse rates | ME Poisson |

## Required conformance

- B and W solver implementations must not call ME fitters and relabel results.
- Zero-inflated providers must follow the AGENTS occupation ontology.
- Partial paths must compute excess constraints and call the corresponding full
  solver on the free support.
- Heavy graph/statistical loops stay in Rust.
- External graph libraries remain downstream adapters unless ODME adopts a
  metric as supported core functionality.

See [Network Metrics](network-metrics.md) for optional graph-library extension
recipes and [Ontology conformance audit](../development/ontology-conformance-audit.md)
for current gaps.
