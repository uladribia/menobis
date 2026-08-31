---
description: Get started with MENoBiS — install from source, generate an observed network, fit, sample, and inspect one statistic.
---

# Getting started

## TL;DR

MENoBiS fits maximum-entropy null models for directed non-binary networks:
you derive structural constraints from an observed network, fit a model,
sample null networks, and compare.

## Install from source

MENoBiS is not distributed on PyPI yet; install from source (a Rust
toolchain is required):

```bash
git clone https://github.com/uladribia/menobis.git
cd menobis
uv sync
uv run maturin develop --release -m crates/menobis-python/Cargo.toml
```

Then check the CLI version:

```bash
uv run menobis --version
```

## 1. Generate a small observed network

Use the built-in synthetic generator (preferential-attachment geometry with
positive integer occupations):

```python
from menobis.utilities.synthetic import generate_pa_geographic_network

network = generate_pa_geographic_network(30, average_degree=6.0, seed=7)
```

## 2. Inspect the EdgeTable

```python
edges = network.edges
print(edges.num_edges)      # occupied pairs E
print(edges.total_events)   # total occupation T
print(edges.source[:5])     # source column
print(edges.target[:5])     # target column
print(edges.occ_num[:5])    # occupation column
```

This confirms the canonical schema `source target occ_num`.

## 3. Derive constraints from the observed network

```python
from menobis.utilities.synthetic import derive_synthetic_constraints

c = derive_synthetic_constraints(network)
strength_out = c.strength_out
strength_in = c.strength_in
```

Constraints derived from a real network are feasible by construction — the
way to build honest examples (see [Constraints](science/constraints.md)).

## 4. Fit a grand-canonical ME strength model

```python
from menobis.models import Constraint, ModelFamily, fit_model

fit = fit_model(
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    strength_out=strength_out,
    strength_in=strength_in,
)
```

## 5. Check convergence

```python
if not fit.converged:
    raise RuntimeError(fit.status)
```

Never sample or filter from an unconverged fit.

## 6. Sample 10 null networks

```python
from menobis.models import Ensemble
from menobis.routing import sample_model

samples = [
    sample_model(
        ensemble=Ensemble.GRAND_CANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        fit=fit,
        seed=r,
    )
    for r in range(10)
]
```

Each sample is a sparse `EdgeTable` with the same schema as the observed
network; different seeds give different draws.

## 7. Compute one high-level statistic

```python
from menobis.analysis import compute_all_stats

observed_stat = compute_all_stats(edges).y2_out.mean()
ensemble_stat = [compute_all_stats(s).y2_out.mean() for s in samples]
print("observed mean Y2:", observed_stat)
print("ensemble mean Y2:", sum(ensemble_stat) / len(ensemble_stat))
```

See [Ensemble statistics](guide/ensemble-statistics.md) for the metric
definitions.

## 8. One microcanonical sample

Microcanonical sampling fixes the requested constraints exactly and needs
no fit. Here, fixed strengths from the same observed network:

```python
single = sample_model(
    ensemble=Ensemble.MICROCANONICAL,
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    strength_out=c.strength_out.astype("uint64"),
    strength_in=c.strength_in.astype("uint64"),
    seed=1,
)
```

Every microcanonical draw reproduces the strengths exactly. See
[Microcanonical sampling](guide/microcanonical.md) for all routes.

## Where next?

- [Choose a model](guide/choose-model.md) — the decision order;
- [Supported models](guide/supported-models.md) — the generated capability
  matrix;
- [Fit and sample](guide/fit-and-sample.md) — the API workflow;
- [Filter node pairs](guide/filter-network.md) — significance filtering.