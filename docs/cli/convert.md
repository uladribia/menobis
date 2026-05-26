---
description: Convert between supported MENoBiS edge-list formats.
---

# `menobis convert`

## TL;DR

Convert edge tables between CSV, TSV, Parquet, and Arrow IPC formats.

## Usage

```bash
menobis convert edges.csv -o edges.parquet
menobis convert edges.parquet -o edges.tsv
menobis convert edges.arrow -o edges.csv
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
