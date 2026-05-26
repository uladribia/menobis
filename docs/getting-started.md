---
description: Quick start guide for MENoBiS.
---

# Getting started

## Installation

```bash
uv pip install -e .
```

Or with maturin for development:

```bash
uv run maturin develop
```

## Verify

```bash
uv run menobis --version
uv run pytest
cargo test --workspace
```

## Supported input formats

MENoBiS accepts networks with non-negative integer weights. Zero-weight edges are
ignored. The in-memory representation is an `EdgeTable` with numpy arrays.

| Format | Extensions |
|--------|-----------|
| CSV | `.csv` |
| TSV | `.tsv`, `.tab` |
| Parquet | `.parquet`, `.pq` |
| Arrow IPC | `.arrow`, `.ipc`, `.feather` |
| GraphML | `.graphml` |
| Matrix Market | `.mtx`, `.mm` |
| Pajek | `.net`, `.paj` |

## Quick workflow

```python
import numpy as np
from menobis.data.io import read_edges
from menobis.analysis import directed_strengths
from menobis.models import fit_strength_poisson, sample_strength_poisson

edges = read_edges("network.csv")
s = directed_strengths(edges)
fit = fit_strength_poisson(s.out, s.incoming)
sample = sample_strength_poisson(fit.x, fit.y, seed=42)
```

## CLI

```bash
menobis analyze strengths network.csv --json
menobis fit strength-poisson network.csv --output fit.csv
menobis generate strength-poisson network.csv --seed 42 --output sample.csv
```
