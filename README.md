# MENoBiS

## TL;DR

MENoBiS (**Max Entropy NOn Binary Suite**) is a Rust + Python library for
maximum-entropy null models of directed non-binary networks. It fits constraints,
samples null networks, filters statistically significant node pairs, and computes
network magnitudes.

## Agentic coding disclosure

MENoBiS was coded and documented with help from agentic coding workflows using
the Pi coding agent and several LLM providers. Human maintainers directed,
reviewed, tested, and accepted the changes.

## Install for development

```bash
git clone https://github.com/uladribia/menobis.git
cd menobis
uv sync
uv run maturin develop --release -m crates/menobis-python/Cargo.toml
```

## Python quickstart

Use feasible constraints derived from a network:

```python
from menobis.analysis import directed_strengths
from menobis.models import Constraint, ModelFamily, fit_model, sample_model
from menobis.utilities.synthetic import generate_pa_geographic_network

network = generate_pa_geographic_network(30, average_degree=6.0, seed=7)
strengths = directed_strengths(network.edges)

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
```

## CLI quickstart

```bash
uv run menobis --help
uv run python scripts/fetch_data.py download openflights
uv run python scripts/evaluate_real_data.py openflights \
  --families me,b --constraints strength --sample --filter-samples 3
uv run menobis fit strength-poisson data/openflights.csv --json
uv run menobis generate strength-poisson data/openflights.csv --output sample.csv
uv run menobis filter strength-poisson data/openflights.csv --output-prefix filtered/
```

## Documentation

```bash
uv run mkdocs serve
uv run mkdocs build --strict
```

Public documentation starts at `docs/index.md` and is published to:

<https://uladribia.github.io/menobis/>

## Model families

| Family | Meaning | Pair law |
|---|---|---|
| ME | distinguishable events | Poisson |
| B | bounded aggregated binary layers | Binomial(M) |
| W | indistinguishable events | geometric / negative-binomial |

## Repository layout

| Path | Purpose |
|---|---|
| `crates/menobis-core` | Rust kernels and solvers |
| `crates/menobis-python` | PyO3 extension |
| `src/menobis` | Python API and CLI |
| `tests` | Python test suite |
| `docs` | MkDocs site and notebooks |
| `benchmarks` | repository benchmark harness |

## Citation

Use `CITATION.cff` or see `docs/thesis-context.md`.
