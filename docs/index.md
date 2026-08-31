---
description: MENoBiS — Max Entropy NOn Binary Suite for null modeling of directed non-binary networks.
---

# MENoBiS

**MENoBiS fits maximum-entropy null models for directed non-binary
networks**: it models integer node-pair occupations \(t_{ij}\), fits
structural constraints in expectation or exactly, samples null ensembles,
and flags statistically surprising node pairs.

## What MENoBiS does

1. **Fit** null models — ME, B, and W occupation families under
   grand-canonical, canonical, and microcanonical ensembles, with six
   structural constraint types (strengths, degrees, occupied-pair counts,
   total events, pair costs, and their combinations).
2. **Sample** null ensembles — exact direct samplers and validated
   stationary-MCMC kernels; use the draws to compare observed network
   statistics with the null.
3. **Filter** observed node pairs — per-pair p-values against the fitted
   null, with multiple-testing corrections and absent-edge detection.

## Minimal vocabulary

- **non-binary network** — a directed network with integer pair
  occupations \(t_{ij}\);
- **occupation number** \(t_{ij}\) — integer event count on pair \((i,j)\);
- **occupied pair** — a pair with \(t_{ij}>0\);
- **binary support** — the indicator \(a_{ij}=\mathbf 1[t_{ij}>0]\);
- **strength / degree** — occupation / support sums per node.

See [Notation](science/notation.md) for the full symbol table.

## Start here by task

| Goal | Start here |
|---|---|
| Install and run the first fit/sample pipeline | [Getting started](getting-started.md) |
| Decide which null model to use | [Choose a model](guide/choose-model.md) |
| What is actually supported today | [Supported models](guide/supported-models.md) |
| Fit and sample a model | [Fit and sample](guide/fit-and-sample.md) |
| Filter significant node pairs | [Filter node pairs](guide/filter-network.md) |
| Understand the mathematics | [Scientific foundations](science/notation.md) |
| Runtime and memory expectations | [Practical scaling](performance/scaling.md) |
| Work on the code | [Development](development/architecture.md) |

## Installation status

Source/development installation (Rust toolchain required; not yet on PyPI):

```bash
git clone https://github.com/uladribia/menobis.git
cd menobis
uv sync
uv run maturin develop --release -m crates/menobis-python/Cargo.toml
uv run menobis --version
```

This is a **source/development installation**, not a generic package
install.

## One tiny example

```python
from menobis.analysis import compute_all_stats
from menobis.models import Constraint, Ensemble, ModelFamily, fit_model
from menobis.routing import sample_model
from menobis.utilities.synthetic import (
    derive_synthetic_constraints,
    generate_pa_geographic_network,
)

network = generate_pa_geographic_network(30, average_degree=6.0, seed=7)
c = derive_synthetic_constraints(network)

fit = fit_model(
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    strength_out=c.strength_out,
    strength_in=c.strength_in,
)
assert fit.converged

sample = sample_model(
    ensemble=Ensemble.GRAND_CANONICAL,
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    fit=fit,
    seed=0,
)
print(compute_all_stats(sample).y2_out.mean())
```

This example is executable — the full pipeline is in
[Getting started](getting-started.md).

## About

MENoBiS implements the non-binary maximum-entropy framework of Oleguer
Sagarra's doctoral thesis (see [References and thesis](science/references.md)).

The codebase is Rust for computation with thin typed Python wrappers;
see [Architecture](development/architecture.md) for contributors.

!!! note "Agentic coding disclosure"
    MENoBiS was coded and documented with help from agentic coding workflows
    using the Pi coding agent and several LLM providers. Human maintainers
    directed, reviewed, tested, and accepted the changes.