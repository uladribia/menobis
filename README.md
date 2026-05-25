# ODME

## TL;DR

ODME is a Rust + Python toolkit for maximum-entropy null models of directed,
weighted origin-destination networks. It fits model parameters, samples null
networks, and filters statistically significant edges.

## Install for development

```bash
git clone <repo-url> ODME_lite
cd ODME_lite
uv sync
uv run maturin develop --release -m crates/odme-python/Cargo.toml
```

## Quickstart: Python

```python
import numpy as np
from odme.models import fit_strength_poisson, sample_strength_poisson

strength_out = np.array([10.0, 20.0, 30.0])
strength_in = np.array([15.0, 25.0, 20.0])

fit = fit_strength_poisson(strength_out, strength_in)
sample = sample_strength_poisson(fit.x, fit.y, seed=42)

print(sample.source, sample.target, sample.weight)
```

## Quickstart: CLI

```bash
uv run odme --help
uv run odme analyze strengths tests/sample.tr --json
uv run odme fit strength-poisson tests/sample.tr --json
uv run odme generate strength-poisson tests/sample.tr --output sample.csv
uv run odme filter strength-poisson tests/sample.tr --alpha 0.05
```

## Model families

| Family | Weight law | Typical use |
|---|---|---|
| ME | Poisson / multinomial | Multi-edge distinguishable events |
| B | Binomial(M) | Bounded layer/event count |
| W | Geometric / negative-binomial | Weighted non-binary null models |

Supported constraints include strengths, degree-events, strengths plus total
edges, strengths plus degrees, and strengths plus cost.

## Benchmarks

Run repository benchmarks without installing an ODME benchmark entry point:

```bash
uv run python -m benchmarks fit --nodes 100 --families me,b,w,wnb \
  --constraints strength --output benchmarks/results/example
```

Large all-case runs are expensive. Run them in chunks and keep benchmark code in
`benchmarks/`, not under `src/odme`.

## Documentation

```bash
uv run mkdocs serve
uv run mkdocs build --strict
```

Start with:

- `docs/getting-started.md`
- `docs/concepts/maximum-entropy-models.md`
- `docs/development/benchmarking.md`
- `docs/development/contributing-and-extending.md`

## Repository layout

The modern ODME implementation lives under `src/`, `crates/`, `tests/`, and
`docs/`. Thesis-era C/Python folders were removed from the active tree; git
history remains the archive.
