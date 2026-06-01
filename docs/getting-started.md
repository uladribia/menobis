---
description: First MENoBiS workflow with feasible non-binary constraints.
---

# Getting started

## TL;DR

Use sparse edge tables with non-negative integer occupations. Derive constraints
from an observed network, fit a null model, then filter or sample from it.

## Install for development

```bash
git clone https://github.com/uladribia/menobis.git
cd menobis
uv sync
uv run maturin develop --release -m crates/menobis-python/Cargo.toml
```

## Input edge table

MENoBiS reads directed edge lists with columns:

| Column | Meaning |
|---|---|
| `source` | non-negative integer origin node id |
| `target` | non-negative integer destination node id |
| `weight` | non-negative integer occupation; zero rows are ignored |

Supported formats: CSV, TSV, Parquet, Arrow IPC, GraphML, Matrix Market, Pajek.

!!! warning "Use feasible constraints"
    User-facing examples should derive constraints from a real or synthetic
    network. Avoid arbitrary strength vectors unless feasibility is proven.

## First Python workflow

```python
from menobis.analysis import directed_strengths
from menobis.filtering import filter_model
from menobis.models import Constraint, ModelFamily, fit_model, sample_model
from menobis.utilities.synthetic import generate_pa_geographic_network

network = generate_pa_geographic_network(
    node_count=30,
    average_degree=6.0,
    events_per_edge=8.0,
    seed=7,
    self_loops=False,
)
edges = network.edges
strengths = directed_strengths(edges)

fit = fit_model(
    family=ModelFamily.ME,
    constraint=Constraint.STRENGTH,
    strength_out=strengths.out,
    strength_in=strengths.incoming,
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
    alpha=0.05,
)
```

## First CLI workflow

The installed `menobis` CLI currently exposes `fit`, `generate`, and `filter`.
For real-data smoke testing, use the repository script below: it downloads a
prepared dataset, fits selected nulls, optionally samples, and estimates filter
false-positive rates.

```bash
uv run python scripts/fetch_data.py download openflights
uv run python scripts/evaluate_real_data.py openflights \
  --families me,b --constraints strength --sample --filter-samples 3
```

You can also run individual commands:

```bash
uv run menobis fit strength-poisson data/openflights.csv --json
uv run menobis generate strength-poisson data/openflights.csv \
  --seed 42 --output sample.csv
uv run menobis filter strength-poisson data/openflights.csv \
  --output-prefix filtered/
```

## Next steps

- Use [Filter a network](tutorials/filter-network.md) for edge significance.
- Use [Sample ensemble magnitudes](tutorials/sample-ensemble-magnitudes.md) for
  null distributions of network-level statistics.
- Use [Choose a null model](concepts/choose-null-model.md) before changing
  family, ensemble, or constraints.
