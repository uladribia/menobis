---
description: MENoBiS — Max Entropy NOn Binary Suite for non-binary network null models.
---

# MENoBiS

## TL;DR

MENoBiS fits maximum-entropy null models for directed non-binary networks,
filters surprising node-pair occupations, and samples null ensembles for
network-magnitude checks.

!!! note "Terminology"
    In MENoBiS, **non-binary** means integer-valued node-pair occupations
    $t_{ij}\in\{0,1,2,\ldots\}$. Weighted, multi-edge, and aggregated binary
    interpretations are different event families, not synonyms.

!!! info "Agentic coding disclosure"
    MENoBiS was coded and documented with help from agentic coding workflows
    using the Pi coding agent and several LLM providers. Human maintainers
    directed, reviewed, tested, and accepted the changes.

## Why MENoBiS?

Origin-destination matrices and other non-binary networks often have strong
low-order structure: large origins, large destinations, spatial costs, and
binary support constraints. MENoBiS helps separate those expected effects from
statistically significant higher-order structure.

## Start by goal

| Goal | Start here |
|---|---|
| Filter observed node pairs | [Filter a network](tutorials/filter-network.md) |
| Sample null networks and compare magnitudes | [Sample ensemble magnitudes](tutorials/sample-ensemble-magnitudes.md) |
| Pick a null model | [Choose a null model](concepts/choose-null-model.md) |
| Understand the equations | [Equations](concepts/equations.md) |
| Use frozen known pairs | [Partial constraints](concepts/partial-constraints.md) |
| Estimate runtime and memory | [Solvers and scaling](concepts/solvers-and-scaling.md) |
| Extend MENoBiS | [Extending thesis cases](development/extending-thesis-cases.md) |

## Minimal install

```bash
git clone https://github.com/uladribia/menobis.git
cd menobis
uv sync
uv run maturin develop --release -m crates/menobis-python/Cargo.toml
uv run menobis --version
```

## Main public Python workflow

```python
from menobis.models import Constraint, ModelFamily, fit_model, sample_model
from menobis.filtering import filter_model

fit = fit_model(
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    strength_out=strength_out,
    strength_in=strength_in,
    self_loops=False,
)

sample = sample_model(
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    fit=fit,
    seed=42,
)

filtered = filter_model(
    edges,
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    fit=fit,
)
```

Use [Getting started](getting-started.md) for a full feasible example.

## Citation

MENoBiS is based on the thesis and papers listed in
[Thesis and citations](thesis-context.md). The repository includes
`CITATION.cff` for GitHub citation metadata.
