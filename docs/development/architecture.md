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
│   ├── grandcanonical/    independent-pair sampling (fitted multipliers)
│   ├── canonical/         fixed-total multinomial
│   └── microcanonical/    exact-constraint generation
│       ├── occupation_mcmc/     fixed-strength MCMC + compressed constructor + repair
│       ├── conditional/         fixed-total pair-Gibbs chain (shared by E,T and k,T)
│       ├── binary/              binary support sampling (degree-events)
│       ├── support/             uniform support sampling (edges-events)
│       ├── mcmc/                shared MCMC config, counters, outcome types
│       └── route.rs             constraint dispatcher
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
| microcanonical | exact (strengths, E, T, k, T); expected cost (STRENGTH_COST) | compressed constructor + repair + occupied-cell MCMC (strength); pair-Gibbs chain (E,T/k,T); stub matching, DP, max flow (oracle only) |

Self-loops are supported across microcanonical backends (guaranteed loop repair).

## See also

- Historical phase specifications are archived in git history
