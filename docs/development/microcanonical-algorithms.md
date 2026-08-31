---
description: Contributor index of the microcanonical sampling algorithms and their constructors, kernels, and exactness proofs.
---

# Microcanonical algorithms

## TL;DR

All microcanonical routes share the two-stage philosophy documented in
[Microcanonical sampling](../guide/microcanonical.md) — **construct one
feasible state, then sample the target measure on the constraint fiber**.
The concrete constructors and kernels differ per constraint; this page maps
each route to its algorithm and proof.

| Constraint | Constructor | Sampling kernel | Exactness | Detailed page |
|---|---|---|---|---|
| `(E,T)` | feasible support/occupations | direct conditional sampler (uniform support + fixed-total Gibbs for ME) | exact direct | — |
| `(k,T)` | exact-degree support + occupations | degree-support MCMC + occupation allocator, fixed total | exact stationary MCMC | — |
| `s` | compressed feasible table | 4-cycle occupation MCMC on the residual state | exact stationary MCMC | — |
| `(s,E)` | exact feasible state (biased initialization-only repair for residual E) | local exact-\(E\) 4-cycle kernel + censored bridge mixture | exact stationary MCMC | [fixed (s,E)](microcanonical-fixed-strength-edges.md) |
| `(s,k)` | extras-first combinatorial exact constructor | capped first-return degree trace of the fixed-\((s,E)\) kernel \(K_E\) | exact stationary MCMC | [fixed (s,k)](microcanonical-fixed-strength-degree.md) |
| `s + cost` | feasible strength state | cost-biased chain with fitted gamma; strengths exact, cost expected | hybrid | [fixed strength+cost] — see [Spatial costs](../guide/spatial-costs.md) |

## Shared philosophy → concrete algorithms

The common view is:

```text
constraints
   |
   v
construct one feasible state
   |
   v
sample the target ensemble around/on the corresponding constraint fiber
   |
   v
diagnostics / burn-in / thinning / independent draws as appropriate
```

Every route instantiates it differently:

- `(E,T)` constructs a uniform support/occupation allocation and draws
  directly from the conditional law;
- `(k,T)` constructs a support with exact degrees and allocates occupations
  under a fixed total;
- `s` compresses the fixed-strength state and runs the occupied-cell
  (4-cycle) chain;
- `(s,E)` builds an exact-\(E\) state and mixes the local kernel with the
  censored bridge;
- `(s,k)` constructs extras-first (combinatorially, no MCMC) and traces the
  degree-distance auxiliary chain onto the degree fiber;
- `s + cost` fits \(\gamma\) so expected cost matches, then runs the
  cost-biased chain with strengths exact.

Construction is **not required to be reversible**; sampling exactness is a
property of the sampling law/kernel, not of how the initial feasible state
was found ([Validation](../performance/validation.md)).

## Where the code lives

- shared capability registry: `menobis.capabilities` (`backend` column);
- routing: `menobis/routing.py` (`sample_model_detailed` exactness
  assignment);
- Rust kernels: `crates/menobis-core` (constructors, chains, trace kernels,
  gamma fitting);
- pyo3 bindings: `crates/menobis-python`.

## Extending the routes

When adding or changing a microcanonical route, follow the contributor
policy: document exactness classification, feasibility, diagnostics, and
tested scales ([Contributor documentation policy](extending-thesis-cases.md)),
update the capability registry, and regenerate the support tables
([Supported models](../guide/supported-models.md)).