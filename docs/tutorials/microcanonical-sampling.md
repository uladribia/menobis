---
description: Step-by-step microcanonical sampling examples with code.
---

# Microcanonical sampling tutorial

## TL;DR

Generate networks with exact constraints using the microcanonical ensemble:
fixed edges+events, fixed degree+events, fixed strengths, or fixed strengths
+ expected cost. No fitting step required (except cost, which uses a separate
gamma fit).

## Pipeline

See [Microcanonical sampling concepts](../concepts/microcanonical.md#pipeline) for the shared pipeline description (derive → construct → repair → sample → thin). The examples below follow the same sequence.

## When to use which constraint

| Constraint | Fixed quantities | When to use |
|---|---|---|
| `EDGES_EVENTS` | Exact edge count + total events | Baseline null model with fixed density |
| `DEGREE_EVENTS` | Exact out/in degree + total events | Preserve node activity heterogeneity |
| `STRENGTH` | Exact out/in strength sequences | Weighted‑network analogue of Configuration Model |
| `STRENGTH_COST` | Exact strengths + target cost ≈ observed | Spatial or distance‑constrained mobility |

!!! note "Deferred cases"
    Fixed `(s,E)` and fixed `(s,k)` microcanonical constraints are not
    currently implemented. See [`development/todos.md`](../development/todos.md).

## 0. Setup

```python
import numpy as np
from menobis.models.spec import Constraint, Ensemble, ModelFamily
from menobis.routing import sample_model, sample_model_detailed
from menobis.utilities.synthetic import generate_pa_geographic_network, derive_synthetic_constraints
```

## 1. Fixed (E,T)

```python
net = sample_model(
    ensemble=Ensemble.MICROCANONICAL,
    family=ModelFamily.ME,
    constraint=Constraint.EDGES_EVENTS,
    node_count=100,
    target_edges=500,
    total_events=3000,
    seed=42,
)
print(f"Edges: {net.num_edges}, Total events: {net.total_events}")
```

## 2. Fixed (k,T)

```python
net = sample_model(
    ensemble=Ensemble.MICROCANONICAL,
    family=ModelFamily.ME,
    constraint=Constraint.DEGREE_EVENTS,
    degree_out=np.array([5, 3, 2]),
    degree_in=np.array([3, 4, 3]),
    total_events=20,
    self_loops=False,
    seed=42,
)
```

## 3. Fixed strengths

```python
net = sample_model(
    ensemble=Ensemble.MICROCANONICAL,
    family=ModelFamily.W,
    constraint=Constraint.STRENGTH,
    strength_out=np.array([10, 5, 5], dtype=np.uint64),
    strength_in=np.array([5, 8, 7], dtype=np.uint64),
    layers=3,
    seed=42,
)
# out/in strengths are exactly preserved
```

## 4. Fixed strengths + expected cost

```python
edges, diagnostics = sample_model_detailed(
    ensemble=Ensemble.MICROCANONICAL,
    family=ModelFamily.B,
    constraint=Constraint.STRENGTH_COST,
    strength_out=np.array([3, 4, 2], dtype=np.uint64),
    strength_in=np.array([4, 3, 2], dtype=np.uint64),
    coord_x=np.array([0.0, 0.5, 1.0]),
    coord_y=np.array([0.0, 0.5, 0.0]),
    target_cost=5.0,
    layers=4,
    seed=42,
)
print(f"gamma = {diagnostics.gamma:.3f}, converged = {diagnostics.converged}")
```

## 5. From a synthetic network (production workflow)

```python
net = generate_pa_geographic_network(50, average_degree=5, seed=42)
c = derive_synthetic_constraints(net)

sample = sample_model(
    ensemble=Ensemble.MICROCANONICAL,
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    strength_out=c.strength_out,
    strength_in=c.strength_in,
    seed=7,
)
```

## Feasibility

| Constraint | Condition |
|---|---|
| `EDGES_EVENTS` | 0 ≤ E ≤ N(N−1), T ≥ E, B: T ≤ M·E |
| `STRENGTH` | Σ s_out = Σ s_in; B: occupancy ≤ M layers |
| `STRENGTH_COST` | Must be cost‑identifiable (extreme targets may fail) |

Chains have tunable `burn_in_sweeps` and `sweeps_per_sample`. Strength‑cost
uses `adaptation_sweeps`, `estimation_sweeps`, `max_iterations` for the
gamma bisection. Defaults cover most cases; increase for tighter problems.

## Scaling notes

| Stage | Sparse N=1000 | Dense N=1000 |
|---|---|---|
| Construction | < 5 ms | < 5 ms |
| Repairs | < 1 ms | < 75 ms (B dense) |
| MCMC burn‑in + thin | 3–5 s | 33–260 s |
| Gamma fit (cost only) | 11–30 s | 200–1200 s |

All state is O(E_occ) — no N² structures. See the [scaling page](../development/scalability.md) for the full operating range across N, families, and regimes.