---
description: Fetch real-world datasets for MENoBiS examples and evaluation.
---

# Fetching real-world data

## TL;DR

Use `scripts/fetch_data.py` from the repository root to download small
MENoBiS-compatible edge tables into `data/`.

## Commands

```bash
uv run python scripts/fetch_data.py list
uv run python scripts/fetch_data.py download openflights
uv run python scripts/fetch_data.py download email-eu
```

## Built-in datasets

| Name | Use | Notes |
|---|---|---|
| `openflights` | primary OD example | directed airport route network with integer weights |
| `email-eu` | quick smoke test | directed communication support, weights default to 1 |

OpenFlights also writes projected coordinate arrays for strength-cost examples.

## Output files

| File | Meaning |
|---|---|
| `data/{name}.csv` | MENoBiS edge table with `source,target,weight` |
| `data/{name}.yml` | dataset summary |
| `data/openflights_coords.npz` | projected `x`, `y` coordinates when available |

## Evaluate a dataset

```bash
uv run python scripts/evaluate_real_data.py openflights \
  --families me,b --constraints strength --sample
```

Use W models cautiously on large real datasets; inspect convergence diagnostics
before using the result scientifically.
