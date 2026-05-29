---
description: Fetching real-world OD datasets for use with MENoBiS.
---

# Fetching real-world data

## TL;DR

Use `scripts/fetch_data.py` to download and prepare real-world directed
weighted networks for MENoBiS. Run it from the project root:

```bash
uv run python scripts/fetch_data.py list
uv run python scripts/fetch_data.py openflights
uv run python scripts/fetch_data.py email-eu
```

Datasets are stored in `data/` (gitignored) as MENoBiS-compatible CSV edge
tables plus YAML summaries.

## Built-in datasets

| Name | Nodes | Edges | Weighted | Description |
|---|---|---|---|---|
| `openflights` | ~6 000 | ~37 000 | Yes | Global airport OD network, weighted by airlines per route |
| `email-eu` | ~1 000 | ~26 000 | No (all w=1) | Email-Eu-core communication network from SNAP |

### OpenFlights (primary OD dataset)

The **openflights** dataset is the recommended real-data test case. It
represents a global Origin-Destination flow network:

- **Nodes**: Airports with IATA/ICAO codes (mapped to integer IDs)
- **Edges**: Directed routes between airports
- **Weight**: Number of airlines serving each directed route (integer ≥ 1)
- **Size**: 6 063 nodes, 36 768 edges, total weight 66 607

This dataset is suitable for testing MENoBiS strength-based fitting on
real-world OD flow data:

```bash
uv run python scripts/fetch_data.py download openflights

# Analyze
uv run python -m menobis analyze strengths data/openflights.csv --json

# Fit
uv run python -m menobis fit strength-poisson data/openflights.csv --json

# Generate a sample
uv run python -m menobis generate strength-poisson data/openflights.csv \
    --seed 42 --output data/openflights-sample.csv
```

### Email-Eu-core (simple test case)

A small directed communication network from SNAP. All edges have weight 1.
Good for quick CI or smoke tests:

```bash
uv run python scripts/fetch_data.py download email-eu
```

## CLI reference

```
Usage: fetch_data.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  list      List available built-in datasets.
  download  Download and prepare a dataset for MENoBiS.
```

### download

```
Usage: fetch_data.py download [OPTIONS] DATASET [URL]

Arguments:
  DATASET  Dataset name (see `list`) or 'url' for custom URL.
  [URL]    Custom URL (required when dataset='url').

Options:
  -o, --output-dir PATH  Output directory [default: data/]
  -c, --cache-dir PATH   Cache for raw downloads [default: ~/.cache/menobis-datasets/]
  -f, --format TEXT      Force format (mtx, csv, tsv) for custom URLs.
  --force                Re-download even if cached.
  --help                 Show help.
```

### Custom URLs

Download any directed weighted network from a public URL. Supported formats
are Matrix Market (`.mtx`), CSV (`.csv`), and TSV (`.tsv`):

```bash
uv run python scripts/fetch_data.py url https://example.com/graph.mtx \
    --format mtx
```

CSV/TSV files must have at least two columns (source, target). A third
column is used as weight; otherwise weight defaults to 1.

## Output format

Each dataset produces two files in the output directory:

| File | Format | Description |
|---|---|---|
| `{name}.csv` | CSV with header `source,target,weight` | MENoBiS edge table |
| `{name}.yml` | YAML | Summary statistics |

The CSV files are compatible with all MENoBiS commands (`analyze`, `fit`,
`generate`, `filter`, `convert`).

### Mercator coordinates (OpenFlights only)

The OpenFlights download also produces a compressed coordinate file
`openflights_coords.npz` containing Web Mercator projected airport
coordinates (``x``, ``y`` arrays). These are needed for the
``strength-cost`` constraint family.

## Evaluating models on real data

Use `scripts/evaluate_real_data.py` to run the MENoBiS pipeline
(fit, sample, filter) on real datasets and see convergence behavior
and precision.

```bash
# Quick test: email-eu, ME and B, strength-only
uv run python scripts/evaluate_real_data.py email-eu \
    --families me,b --constraints strength

# OpenFlights strength fit across families
uv run python scripts/evaluate_real_data.py openflights \
    --families me,b --constraints strength

# With sampling and FPR estimation
uv run python scripts/evaluate_real_data.py email-eu \
    --families me --constraints strength \
    --sample --filter-samples 3

# Cost constraint (requires coordinates)
uv run python scripts/evaluate_real_data.py openflights \
    --families me --constraints strength-cost

# Save results as JSON
uv run python scripts/evaluate_real_data.py openflights \
    --families me,b --constraints strength \
    --output results/openflights-eval.json
```

### evaluate_real_data.py CLI

```
Usage: evaluate_real_data.py [OPTIONS] [DATASET]

Arguments:
  [DATASET]  Dataset name or path [default: openflights]

Options:
  -f, --families TEXT      Model families: me,b,w [default: me,b,w]
  -c, --constraints TEXT   Constraint types [default: strength,strength-edges,
                                strength-degree]
  --sample/--no-sample     Generate a sample from the fit
  --filter-samples INT     Null samples for FPR [default: 0]
  --alpha FLOAT            Upper-tail alpha for filtering [default: 0.05]
  -t, --tolerance FLOAT    Solver tolerance [default: 1e-6]
  --max-iterations INT     Max solver iterations [default: 10000]
  -o, --output PATH        Output JSON path
  --json                   Print JSON to stdout
  --no-header              Suppress column header
```

### Known convergence notes

- **W (geometric)**: For large N (N > 1000), the W solver may converge
  slowly or not at all. This is a known issue documented in the AGENTS.md
  and `docs/decisions/convergence-issues.md`. Stick to N ≤ 1000 for W.
- **ME (Poisson)**: Fast and reliable for all constraint types.
- **B (binomial)**: Generally fast but strength-edges may require more
  iterations for large networks.
- **strength-degree**: Can be slow for N > 1000; start with the tolerance
  at 1e-4 for exploratory runs.

## Dependencies

The script requires Python 3.11+ with `numpy` and `typer`. Both are
dependencies of MENoBiS and available when running via `uv run`.

To run standalone (outside the MENoBiS project):

```bash
pip install numpy typer
python scripts/fetch_data.py list
```
