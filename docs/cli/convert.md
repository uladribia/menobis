---
description: Convert between supported ODME edge-list formats.
---

# `odme convert`

## TL;DR

Convert edge tables between CSV, TSV, Parquet, and Arrow IPC formats.

## Usage

```bash
odme convert edges.csv -o edges.parquet
odme convert edges.parquet -o edges.tsv
odme convert edges.arrow -o edges.csv
```

## Options

| Option | Meaning |
|--------|---------|
| `-o`, `--output` | Output file path (format inferred from extension) |
| `--quiet` | Suppress progress messages |

## Supported formats

| Extension | Format |
|-----------|--------|
| `.csv` | Comma-separated values |
| `.tsv`, `.tab` | Tab-separated values |
| `.parquet`, `.pq` | Apache Parquet |
| `.arrow`, `.ipc`, `.feather` | Arrow IPC |
