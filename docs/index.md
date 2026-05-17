---
description: ODME — Origin-Destination Multi-Edge network models.
---

# ODME

ODME is a modern Rust + Python library for maximum-entropy multi-edge network
models. It implements the statistical mechanics framework from the
[thesis](https://hdl.handle.net/10803/400560) with fast Rust kernels and
ergonomic Python workflows.

## What ODME does

| Capability | Description |
|------------|-------------|
| **Fit** | Solve Lagrange multipliers for strength, degree, cost, and combined constraints |
| **Generate** | Sample networks from fitted null models (Poisson, ZIP, multinomial, microcanonical) |
| **Filter** | Identify statistically significant edges against independent null models |
| **Analyze** | Compute strengths, degrees, weight distributions, and node-level statistics |
| **Convert** | Read and write CSV, TSV, Parquet, and Arrow IPC edge tables |

## Quick start

```bash
uv pip install -e .
uv run maturin develop
uv run odme --version
```

```python
from odme.data.io import read_edges
from odme.analysis import directed_strengths
from odme.models import fit_strength_poisson, sample_strength_poisson

edges = read_edges("network.csv")
s = directed_strengths(edges)
fit = fit_strength_poisson(s.out, s.incoming)
sample = sample_strength_poisson(fit.x, fit.y, seed=42)
```

See [Getting Started](getting-started.md) for installation and workflow details.

## Documentation map

| Section | Contents |
|---------|----------|
| [Concepts](concepts/multi-edge-networks.md) | Scientific background and model equations |
| [CLI](cli/analyze.md) | Command-line interface reference |
| [API](api/python.md) | Python and Rust API reference |
| [Decisions](decisions/0001-rust-python-architecture.md) | Architectural decision records |
| [Development](development/testing.md) | Testing, benchmarking, and release workflows |
