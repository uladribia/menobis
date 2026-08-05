# Microcanonical fixed-(E,T) sampling

**TL;DR** — MENoBiS can sample exact microcanonical networks with a fixed
number of occupied pairs `E` and fixed total occupation `T`, for the ME, B,
and W families, with no fitting step. The sampled network satisfies the hard
constraints exactly.

## What it does

The microcanonical ensemble fixes one or more observables *exactly* rather
than in expectation. The `EDGES_EVENTS` microcanonical case fixes:

- `E` — the number of occupied pairs (binary edges),
- `T` — the total occupation (sum of all `t_ij`).

The target measure is the family degeneracy, conditioned on `(E, T)`:

| Family | Target | Support |
|---|---|---|
| ME | `P(t) ∝ ∏ 1/tᵢ!` | `tᵢ ≥ 0` |
| B (M layers) | `P(t) ∝ ∏ C(M, tᵢ)` | `0 ≤ tᵢ ≤ M` |
| W (M layers) | `P(t) ∝ ∏ C(M+tᵢ−1, tᵢ)` | `tᵢ ≥ 0` |

## How sampling works

The sampler factorises exactly into two independent steps:

1. **Uniform support selection** — pick `E` of the `L` admissible pairs
   uniformly (Floyd's algorithm, no N² pair list materialised).
2. **Positive occupation allocation** — draw positive integers
   `(t_1, …, t_E)` summing to `T` with weight proportional to the family
   degeneracy.

Step 2 uses a hybrid of two exact backends:

- a **fast rejection proposal** (≤ 20 attempts) built on a microscopic
  combinatorial object:
  - ME: multinomial labels;
  - B: uniform binary-cell subset;
  - W: uniform weak composition (stars-and-bars);
- an **exact DP fallback** (bounded composition table) used when rejection
  is unlikely or retries are exhausted.

Both backends target the same exact distribution, so backend selection only
affects performance, never correctness. If the DP table would exceed its
memory budget, the sampler retries rejection with a work-bounded attempt
budget, and returns a clear error for genuinely hard parameter points.

## Usage

```python
from menobis.models.spec import Constraint, Ensemble, ModelFamily
from menobis.routing import sample_model

net = sample_model(
    ensemble=Ensemble.MICROCANONICAL,
    family=ModelFamily.ME,
    constraint=Constraint.EDGES_EVENTS,
    node_count=100,
    target_edges=500,
    total_events=3000,
    seed=42,
)
# net has exactly 500 occupied pairs and total occupation 3000
```

For B or W, pass `layers`:

```python
net_b = sample_model(
    ensemble=Ensemble.MICROCANONICAL, family=ModelFamily.B,
    constraint=Constraint.EDGES_EVENTS,
    node_count=100, target_edges=500, total_events=1500, layers=4, seed=42,
)
```

### Fixed pairs

Pass `known_source`, `known_target`, and `known_occnum` to freeze specific
pairs. Their contribution is subtracted from `E` and `T`, the residual is
sampled, and the fixed pairs are merged back:

```python
net = sample_model(
    ensemble=Ensemble.MICROCANONICAL, family=ModelFamily.ME,
    constraint=Constraint.EDGES_EVENTS,
    node_count=100, target_edges=500, total_events=3000,
    known_source=[0, 5], known_target=[1, 6], known_occnum=[3, 7],
    seed=42,
)
```

## Feasibility

The residual problem `(E, T)` must satisfy:

- `0 ≤ E ≤ L` (at most one occupation per admissible pair);
- `E = 0 ⟺ T = 0`;
- `T ≥ E` (every occupied pair has at least one event);
- B additionally requires `T ≤ M·E` (no pair can exceed its layers).

## Validation

The samplers are validated by:

- exact enumeration on tiny systems;
- the conditioned grand-canonical identity
  `P_GC(t | E, T) = P_MC(t | E, T)`;
- E2E constraint recovery on synthetic networks (dense and sparse regimes).

## Memory

No O(N²) memory is used: admissible pairs are mapped from a linear index on
the fly. The exact DP tables are capped (~16 MB); larger problems fall back
to bounded rejection.
