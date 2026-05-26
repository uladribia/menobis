---
description: Mathematical overview of MENoBiS fitting, generation, and filtering.
---

# Maximum entropy models

## TL;DR

MENoBiS fits maximum-entropy null models by matching expected sufficient statistics.
The source ontology is [Model ontology](model-ontology.md): ensemble + family +
constraint layer. This page explains the workflow without repeating all formulas.

## Model choices

| Choice | Meaning |
|---|---|
| Ensemble | Grand-canonical independent pairs; ME-only canonical multinomial; ME-only fixed-strength stub matching. |
| Family | ME/Poisson, B/Binomial(M), W/Negative-Binomial(M). |
| Constraint layer | Strength, strength-cost, strength-edges, strength-degree, degree-events, or partial support. |

Linear constraints depend only on $\mathbb{E}[t_{ij}]$. Constraints involving
$\mathbb{E}[\Theta(t_{ij}>0)]$ are zero-inflated and add occupation
multipliers. See [Model ontology](model-ontology.md) for the canonical formula
tables.

## Fitting

Fitting solves Lagrange multiplier equations. Multipliers usually factor by
origin and destination (`x_i y_j`); cost constraints add
`f_ij = exp(-gamma d_ij)`. Family kernels differ, so each family needs its own
solver even when it reuses the same masking, balancing, and cost-provider
infrastructure.

| Constraint | Shared infrastructure | Family-specific part |
|---|---|---|
| strength | support mask, row/column balancing | ME/B/W mean equation |
| strength-cost | cost provider, gamma search | ME/B/W cost-weighted mean |
| strength-edges | edge-count residual | zero-inflated family kernel |
| strength-degree | degree residuals | zero-inflated family kernel |
| degree-events | Bernoulli occupation balancing | positive-weight family parameter |
| partial | excess constraints and free support | call the matching full solver |

## Generation

Grand-canonical generation streams candidate pairs through a Rust
`PairDistributionProvider`. Canonical and microcanonical ME samplers are separate
because fixed totals or exact strengths couple pairs.

## Filtering

Filtering uses the same provider pipeline as generation to compute expected
weights, occupation probabilities, and p-values. Absent-edge filtering scans
candidate pairs with observed weight zero.

## Numerical policy

- Expose convergence, iterations, status, and residual diagnostics.
- Prefer relative constraint-recovery checks over multiplier-delta checks.
- Keep heavy loops in Rust and return typed Python dataclasses.
- Treat legacy C/Python as reference material, not compatibility behavior.

## Current implementation status

Known mismatches between this ontology and the repository are listed in
[Ontology conformance audit](../development/ontology-conformance-audit.md).
