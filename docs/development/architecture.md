# MENoBiS architecture (Phase 0)

**TL;DR** — One shared foundation: occupation ontology → shared family/constraint
modules → ensemble backends → shared analysis/filtering. Python is a thin
validating wrapper over Rust kernels.

## Module map (Rust `menobis-core`)

```text
src/
├── distribution.rs        OccupationFamily, PairDistribution (pair laws)
├── constraints/           PairMask, FixedPairs, FixedContributions,
│                          ResidualConstraints, common validation
├── fitting/               Lagrange multiplier solvers (ME/B/W, partial)
├── generation/
│   ├── grandcanonical/    independent-pair sampling
│   ├── canonical/         fixed-total multinomial
│   └── microcanonical/    ME fixed-strength stub matching
├── filter.rs              tail probabilities from the same pair laws
├── graph.rs               OccupiedPair, SparseGraphView
├── stats.rs / clustering.rs  node statistics, clustering (sparse adjacency)
└── pairs.rs               pair-cost and pair-distribution providers
```

## Runtime flow

```text
Python request
    -> capability lookup (menobis/capabilities.py)
    -> input normalization + common validation (constraints/)
    -> fixed-pair contributions + residualization (constraints/)
    -> ensemble backend (grandcanonical | canonical | microcanonical)
    -> SampledNetwork / EdgeTable
    -> shared analysis facade
    -> filtering consumes the same shared pair laws
```

## Dependency direction

```text
shared family / constraint / pair-law modules
        -> fitting
        -> generation
        -> filtering
        -> analysis
```

Filtering never calls generation internally; both consume
`PairDistribution` and the provider abstractions.

## Ensembles

| Ensemble | Constraint treatment | Backends |
|---|---|---|
| grand-canonical | expected (fitted multipliers) | independent pairs, filters |
| canonical | one exact global (T) | multinomial |
| microcanonical | all exact | ME stub matching (self-loops) |

The microcanonical no-self-loop case is intentionally unsupported until the
MCMC backend exists (§11).

## See also

- `docs/development/agent-specifications/microcanonical-phase-0.md`
- `docs/development/migration-notes.md`
