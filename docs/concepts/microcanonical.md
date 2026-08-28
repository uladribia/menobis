---
description: Microcanonical sampling cases, backends, pipeline, and scaling.
---

# Microcanonical sampling

**TL;DR** — Microcanonical sampling for four constraint families (ME/B/W),
validated at **N=1000** in the benchmark matrix: hard constraints exact,
expected-cost constraints in expectation.

When soft (grand-canonical) and hard (microcanonical) formulations of the same
null hypothesis are interchangeable for theoretical calculations depends on
occupation scale and constraint type; see
[Ensemble equivalence at large occupation](ensemble-equivalence.md).

!!! warning "Validation status"
    Validated at **N=1000** for fixed strengths and strengths+cost (all
    families, sparse/dense); fixed-(E,T) and fixed-(k,T) run on the shared
    pair-Gibbs chain at N≥1000. Gamma fitting at dense N=1000 is slow
    (≈20 min per cell); MCMC cases need sweep-budget tuning.

## Implemented cases

| Case | Constraint | Backend | Exactness |
|---|---|---|---|
| fixed (E,T) | `EDGES_EVENTS` | pair-Gibbs chain (production); DP/rejection (oracle) | exact |
| fixed (k,T) | `DEGREE_EVENTS` | MCMC binary support + shared pair-Gibbs chain | exact at stationarity |
| fixed strengths | `STRENGTH` | compressed aggregated constructor → loop/capacity repair → occupied-cell MCMC | exact at stationarity |
| fixed strengths + exact edge count | `STRENGTH_EDGES` | same constructor/repair → edge-count repair → local 4-cycle + censored-bridge MCMC | exact at stationarity |
| fixed strengths + expected cost | `STRENGTH_COST` | same + gamma fitting (zero-centered bracket expansion + stochastic bisection) | exact strengths; cost in expectation |

Families: `ME` (Poisson), `B` (Binomial on M), `W` (NegBin on M; M=1 geometric); B/W take a `layers` argument.

## Pipeline

```text
generator (PA-geographic) -> derive constraints -> construct -> repair -> MCMC -> sample
```

A PA-geographic generator supplies a feasible weighted network; constraints
(strengths, degrees, edge counts, total cost) are derived from it; the
compressed aggregated constructor builds an exact-strength sparse occupation
table (no `O(T)` stub expansion, `O(N²)` enumeration, or max flow); loop,
capacity, and admissibility repairs make it feasible; the occupied-cell
Metropolis chain samples the family degeneracy, thinned after burn-in.

## How it works

The target measure is the family degeneracy conditioned on the constraints:
ME pairs weigh `∝ 1/tᵢ!` on `tᵢ ≥ 0`; B (M layers) `∝ C(M, tᵢ)` on
`0 ≤ tᵢ ≤ M`; W (M layers) `∝ C(M+tᵢ−1, tᵢ)` on `tᵢ ≥ 0`.

The fixed-(E,T) and fixed-(k,T) cases share the **pair-Gibbs chain**: each
step redraws the split of two uniformly chosen cells from the exact family
conditional — always accepted, preserving total `T`, positivity, and B
capacity. Fixed-strength and strength-cost use the occupied-cell Metropolis
chain over the repaired sparse occupation table.

## Usage: fixed (E,T)

Exact `E` occupied pairs and `T` total events, no fitting step:

```python
from menobis.models.spec import Constraint, Ensemble, ModelFamily
from menobis.routing import sample_model

net = sample_model(
    ensemble=Ensemble.MICROCANONICAL,
    family=ModelFamily.ME,
    constraint=Constraint.EDGES_EVENTS,
    node_count=100, target_edges=500, total_events=3000, seed=42,
)
```

Pass `layers` for B or W. Freeze pairs with `known_source`/`known_target`/
`known_occnum`; their contributions are subtracted and merged back.

## Usage: fixed (k,T)

Exact out-degree, in-degree, and total events:

```python
net = sample_model(
    ensemble=Ensemble.MICROCANONICAL,
    family=ModelFamily.ME,
    constraint=Constraint.DEGREE_EVENTS,
    degree_out=degree_out, degree_in=degree_in, total_events=3000,
    self_loops=False, seed=42,
)
```

## Usage: fixed strengths

Exact strength sequences (no fitting step):

```python
net = sample_model(
    ensemble=Ensemble.MICROCANONICAL,
    family=ModelFamily.B,
    constraint=Constraint.STRENGTH,
    strength_out=strength_out, strength_in=strength_in,
    layers=4, seed=42,
)
```

All families use the compressed constructor → repair → occupied-cell MCMC
path; `burn_in_sweeps` / `sweeps_per_sample` tune the chain.

## Usage: fixed strengths + exact edge count (s, E)

Exact strength sequences **and** an exact number of occupied ordered pairs
`target_edges` (no fitting step).  The returned network has exactly
`target_edges` positive pairs and exactly the requested strengths; `E` is
smaller than the total occupation `T`, so `target_edges` is the number of
pairs with `occ_num > 0`, not the number of events.

```python
net = sample_model(
    ensemble=Ensemble.MICROCANONICAL,
    family=ModelFamily.W,
    constraint=Constraint.STRENGTH_EDGES,
    strength_out=strength_out, strength_in=strength_in,
    target_edges=1200, layers=3, self_loops=False, seed=42,
)
```

- Families: `ME`, `B`, `W` (pass `layers` for B/W).
- `self_loops=True` admits diagonal pairs; `False` forbids them.
- Freeze pairs with `known_source`/`known_target`/`known_occnum`;
  positive fixed pairs count toward `target_edges` and their coordinates
  are excluded from the residual domain (zero-occupation fixed pairs stay
  frozen at zero).  Rust performs the residualization exactly once.

The sampler is an **exact stationary MCMC**: the transition kernel has the
exact family-degeneracy-on-(s,E) distribution as its stationary law; finite
burn-in remains an ordinary MCMC concern.  See
[Fixed strengths + exact edge count](fixed-strength-edges.md) for the
stationary-target argument.

## Usage: fixed strengths + expected cost

Strengths are exact; total cost is matched in expectation by fitting the
cost multiplier `gamma` via zero-centered bracket expansion and stochastic
bisection, then sampling:

```python
edges, diagnostics = sample_model_detailed(
    ensemble=Ensemble.MICROCANONICAL,
    family=ModelFamily.W,
    constraint=Constraint.STRENGTH_COST,
    strength_out=strength_out, strength_in=strength_in,
    coord_x=coord_x, coord_y=coord_y,
    target_cost=observed_cost, layers=2, seed=42,
)
```

## Feasibility

- fixed (E,T): `0 ≤ E ≤ L`, `E = 0 ⟺ T = 0`, `T ≥ E`, B: `T ≤ M·E`.
- fixed strengths: balanced totals; B occupations ≤ `M`; the constructor
  plus targeted repairs (loop, capacity, admissibility) generate a feasible
  state for supported configurations; infeasible inputs are rejected at the
  constraint validation layer.
- fixed (s,E): necessary bounds are validated up front (`E ≤ T`, `E ≤`
  admissible pairs, B capacity bounds, positive node lower bounds); a
  target passing them may still be infeasible for a particular sparse
  domain, in which case repair exhausts with a structured error (never an
  approximate edge count).  Loopless perfect-matchings are an inherent
  4-cycle-kernel limitation.
- strength-cost: cost must be identifiable; extreme targets can fail the gamma bracket.

## Scaling

See [Scaling (GC + MC)](../development/scalability.md) for the authoritative
operating range table, complexity analysis, and benchmark matrix results.
Memory stays `O(E_occ)` — see the Memory section below.

## Validation

Exact enumeration on tiny systems; conditioned grand-canonical identity
`P_GC(t | E,T) = P_MC(t | E,T)`; E2E constraint recovery on synthetic
networks; benchmark matrix at N=100/500/1000 across ME/B/W × sparse/dense.

## Memory

No `O(N²)` pair lists are materialised; production state is `O(E_occ)`;
fixed-total DP tables and rejection backends live only in the oracle crate
(`menobis-test-oracles`).

!!! note "Deferred microcanonical cases"
    Fixed `(s,E)` (strength + binary edges) and fixed `(s,k)` (strength +
    degree sequence) are not currently implemented. See [`development/todos.md`](../development/todos.md).
