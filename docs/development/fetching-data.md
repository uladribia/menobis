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

## Dependencies

The script requires Python 3.11+ with `numpy` and `typer`. Both are
dependencies of MENoBiS and available when running via `uv run`.

To run standalone (outside the MENoBiS project):

```bash
pip install numpy typer
python scripts/fetch_data.py list
```
