# MENoBiS architecture

**TL;DR** — One shared foundation: occupation ontology → shared
family/constraint modules → ensemble backends → shared analysis/filtering.
Python is a thin validating wrapper over Rust kernels. The authoritative
list of supported routes is the
[generated capability matrix](../guide/supported-models.md), not any prose
table in this file.

## Module map (Rust `menobis-core`)

```text
src/
├── model/                 family/problem/sampling-plan ontology
├── distribution.rs        OccupationFamily, PairDistribution (pair laws)
├── constraints/           PairMask, FixedPairs, FixedContributions,
│                          ResidualConstraints, common validation
├── fitting/               Lagrange multiplier solvers (ME/B/W, partial, conic W)
├── generation/
│   ├── grandcanonical/    independent-pair sampling (fitted multipliers)
│   ├── canonical/         fixed-total multinomial
│   └── microcanonical/
│       ├── route.rs             constraint dispatcher (shared pipeline)
│       ├── occupation_mcmc/     fixed-strength routes
│       │   ├── compressed.rs    compressed fixed-strength state
│       │   ├── chain.rs         occupied-cell (4-cycle) chain
│       │   ├── fixed_edges.rs   fixed-(s,E): local exact-E kernel + censored bridge
│       │   ├── fixed_degrees.rs fixed-(s,k): capped first-return degree trace (K_E)
│       │   ├── fixed_degree_init.rs  extras-first combinatorial constructor
│       │   ├── cost.rs / cost_fit.rs  strength+cost chain and gamma fitting
│       │   └── repair.rs        initialization-only repair
│       ├── conditional/         fixed-total pair-Gibbs chain (E,T; occupations for k,T)
│       ├── binary/              binary support sampling (degree sequences)
│       ├── support/             uniform support sampling (edges-events)
│       └── mcmc/                shared MCMC config, counters, outcome types
├── filter.rs              tail probabilities from the same pair laws
├── graph.rs               OccupiedPair, SparseGraphView
├── stats.rs / clustering.rs  node statistics, clustering (sparse adjacency)
└── pairs.rs               pair-cost and pair-distribution providers
```

## Runtime flow

```text
Python request
    -> capability lookup (menobis/capabilities.py)
    -> route_model verbs (fit | sample | filter)
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
| canonical | one exact global (T) | multinomial (ME strength) |
| microcanonical | exact (strengths, E, T, k, T); expected cost (STRENGTH_COST hybrid) | occupation MCMC (s), fixed edges + bridge (s,E), extras-first + degree trace (s,k), fixed-total Gibbs (E,T / k,T), uniform support (E,T), cost-biased gamma (s+cost) |

Self-loop policies are supported across microcanonical backends; the
admissible-pair domain follows the policy.

## Exactness

Exactness categories per route are assigned in `menobis/routing.py`
(`sample_model_detailed`) and surfaced on `result.diagnostics.exactness`;
the [generated capability matrix](../guide/supported-models.md) lists them
per route. Theoretical claims are always "exact stationary MCMC" (kernel
law), never "exact samples" without qualification
([Validation](../performance/validation.md)).

## Contributor documentation

- [Extending MENoBiS](extending-thesis-cases.md) — how to add families,
  constraints, and samplers;
- [Microcanonical algorithms](microcanonical-algorithms.md) — per-route
  constructors and kernels;
- [Testing](testing.md) — the test/validation workflow;
- [Release process](release-process.md) — CI and publishing workflow.