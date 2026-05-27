# 0003 Provider pipeline for generation and filtering

## TL;DR

Independent MENoBiS models expose pair distributions through providers. Generation
and filtering consume shared distribution and coordinate cost providers without
dense rate or cost matrices.

## Context

MENoBiS null models differ in how they compute pair parameters, but generation and
filtering need the same pair-level information:

- expected weight;
- binary occupation probability;
- random draw;
- lower and upper tail probabilities.

Duplicating this math across samplers and filters makes future models harder to
add and easier to implement inconsistently.

## Decision

Use one Rust-internal pipeline for independent null models:

```text
model parameters -> PairDistributionProvider -> PairDistribution -> sink
```

| Layer | Responsibility |
|-------|----------------|
| model parameters | fitted multipliers, rates, costs, masks |
| cost provider | generate Euclidean pair costs on demand |
| provider | compute one pair distribution on demand |
| distribution | sampling, expectation, occupation, p-values |
| sink | sample edges, filter observed edges, or scan absent edges |

Providers declare their support as either all candidate pairs or sparse pairs.
Sinks choose row chunks for all-pairs support and sparse-entry chunks for sparse
support. Strength-cost public workflows use coordinates and the built-in
Euclidean `PairCostProvider`; MENoBiS intentionally rejects dense cost matrices.

## Consequences

- Generation and filtering share ME/B/W and zero-inflated positive-weight math.
- Cost-constrained public APIs avoid dense matrix allocation by requiring
  projected coordinates.
- Users needing non-Euclidean costs should implement a Rust `PairCostProvider`
  and corresponding model wrapper.
- Filtering reports absent edges by streaming provider support, not by creating
  dense matrices.
- Custom sparse rates preserve input support order before optional truncation.
- Canonical multinomial models remain separate because fixed totals couple
  pairs and violate the independent-pair assumption.

## Validation

Rust and Python regression tests check p-values, absent-edge thresholds, sparse
support order, seeded generation, and public CLI/API behavior. The full check
suite is required after provider changes.
