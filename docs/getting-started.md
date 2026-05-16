# Getting started

The first implementation milestone exposes a minimal Python package, Typer CLI, and Rust workspace.

```bash
uv run odme --version
cargo test --workspace
```

## Supported weighted network inputs

ODME accepts networks with non-negative integer weights. Zero-weight edges are ignored during normalization because they do not represent existing weighted edges. The canonical in-memory representation is a Polars dataframe with columns:

- `source`
- `target`
- `weight`

Current readers support:

- CSV (`.csv`)
- TSV (`.tsv`, `.tab`)
- Parquet (`.parquet`, `.pq`)
- Arrow IPC / Feather (`.arrow`, `.ipc`, `.feather`)
- GraphML (`.graphml`)
- Matrix Market coordinate matrices (`.mtx`, `.mm`)
- Pajek (`.net`, `.paj`)

Formats with one-based node identifiers, such as Matrix Market and Pajek, are converted to zero-based ODME node identifiers.
