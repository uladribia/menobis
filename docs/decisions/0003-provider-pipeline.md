# 0003 Provider pipeline for generation and filtering

## TL;DR

Independent ODME models expose pair distributions through providers. Generation
and filtering consume the same provider interface and do not build dense rate
matrices.

## Context

ODME null models differ in how they compute pair parameters, but generation and
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
| provider | compute one pair distribution on demand |
| distribution | sampling, expectation, occupation, p-values |
| sink | sample edges, filter observed edges, or scan absent edges |

Providers declare their support as either all candidate pairs or sparse pairs.
Sinks choose row chunks for all-pairs support and sparse-entry chunks for sparse
support.

## Consequences

- Generation and filtering share Poisson and ZIP/ZTP math.
- Filtering reports absent edges by streaming provider support, not by creating
  dense matrices.
- Custom sparse rates preserve input support order before optional truncation.
- Canonical multinomial models remain separate because fixed totals couple
  pairs and violate the independent-pair assumption.

## Validation

Rust and Python regression tests check p-values, absent-edge thresholds, sparse
support order, seeded generation, and public CLI/API behavior. The full check
suite is required after provider changes.
