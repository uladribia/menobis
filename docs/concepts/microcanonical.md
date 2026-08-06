---
description: Microcanonical sampling cases, backends, and experimental status.
---

# Microcanonical sampling

**TL;DR** — MENoBiS implements microcanonical sampling for four constraint
families across ME, B, and W. **The whole ensemble is experimental and
validated only for small N (≈10–100 nodes).** Hard constraints are satisfied
exactly; expected-cost constraints are satisfied in expectation.

!!! warning "Experimental status"
    All microcanonical samplers are **experimental and intended for small
    networks (N ≲ 100)**. They have not been validated at production scale.
    Use the grand-canonical ensemble for large-N scientific work. MCMC-based
    cases require sweep-budget tuning and may fail to converge for larger or
    tighter problems.

## Implemented cases

| Case | Constraint | Backend | Exactness |
|---|---|---|---|
| fixed (E,T) | `EDGES_EVENTS` | direct: support selection + occupation allocation (rejection + DP) | exact |
| fixed (k,T) | `DEGREE_EVENTS` | MCMC support (double-edge switch) + occupation allocation | exact at stationarity |
| fixed strengths | `STRENGTH` | ME: stub matching (direct) or MCMC; B/W: MCMC | exact at stationarity |
| fixed strengths + expected cost | `STRENGTH_COST` | MCMC + gamma stochastic bisection | exact strengths; cost in expectation |

Families: `ME` (Poisson), `B` (Binomial on M layers), `W` (NegBin on M
layers; M=1 is geometric). B/W take a `layers` argument.

## How it works

The microcanonical ensemble fixes observables *exactly* rather than in
expectation. The target measure is the family degeneracy conditioned on the
constraints:

| Family | Weight per pair | Support |
|---|---|---|
| ME | `∝ 1/tᵢ!` | `tᵢ ≥ 0` |
| B (M layers) | `∝ C(M, tᵢ)` | `0 ≤ tᵢ ≤ M` |
| W (M layers) | `∝ C(M+tᵢ−1, tᵢ)` | `tᵢ ≥ 0` |

The fixed-(E,T) sampler factorises exactly into uniform support selection
plus positive-occupation allocation. The fixed-(k,T), fixed-strength, and
strength-cost cases use a 4-cycle/double-edge-switch Metropolis chain whose
stationary distribution is the target ensemble.

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
# net has exactly 500 occupied pairs and total occupation 3000
```

Pass `layers` for B or W. Freeze pairs with `known_source`, `known_target`,
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

ME with self-loops routes to direct stub matching; ME without self-loops and
B/W use the MCMC backend. `burn_in_sweeps` / `sweeps_per_sample` tune the
chain.

## Usage: fixed strengths + expected cost

Strengths are exact; total cost is matched in expectation by fitting the
cost multiplier `gamma` via stochastic bisection, then sampling:

```python
edges, diagnostics = sample_model_detailed(
    ensemble=Ensemble.MICROCANONICAL,
    family=ModelFamily.W,
    constraint=Constraint.STRENGTH_COST,
    strength_out=strength_out, strength_in=strength_in,
    coord_x=coord_x, coord_y=coord_y,
    target_cost=observed_cost, layers=2, seed=42,
)
# diagnostics carries gamma, expected cost, SE, converged
```

## Feasibility

- fixed (E,T): `0 ≤ E ≤ L`, `E = 0 ⟺ T = 0`, `T ≥ E`, B additionally
  `T ≤ M·E`.
- fixed strengths: total out-strength must equal total in-strength; B
  occupations cannot exceed `M` layers.
- strength-cost: the cost must be identifiable; extreme targets can make
  the gamma bracket fail.

## Experimental limitations

| Limitation | Detail |
|---|---|
| Small N only | validated on N ≈ 10–100; not production-ready at larger N |
| MCMC mixing | fixed-strength and strength-cost chains need generous sweep budgets and may mix poorly on tight or large problems |
| Strength-cost fit | gamma fit is reliable at N ≤ 100 with tuned sweeps; at N = 500 it requires per-case tuning (see [strength-cost benchmark](../benchmarks/microcanonical_strength_cost.md)) |
| ME/W cost variance | the warm-start cost-variance estimate can collapse, triggering `CostNotIdentifiable` |
| W hard regimes | fixed-(E,T) with large E and high T may exceed the rejection work budget and error out |

## Validation

- exact enumeration on tiny systems;
- conditioned grand-canonical identity `P_GC(t | E,T) = P_MC(t | E,T)`;
- E2E constraint recovery on synthetic networks (dense/sparse).

## Memory

No O(N²) pair lists are materialised. Fixed-(E,T) DP tables are capped
(~16 MB); larger problems fall back to bounded rejection.
