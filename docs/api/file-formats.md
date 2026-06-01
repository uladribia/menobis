---
description: MENoBiS file schemas and supported input/output formats.
---

# File formats

## TL;DR

The canonical network file is a sparse directed edge table with integer
occupations. Zero occupations are dropped and negative or fractional values are
rejected.

## Edge table schema

| Column | Type | Rule |
|---|---|---|
| `source` | integer | non-negative node id |
| `target` | integer | non-negative node id |
| `weight` | integer | non-negative occupation; zero rows are ignored |

Example CSV:

```csv
source,target,weight
0,1,12
0,2,3
2,1,7
```

## Supported edge formats

| Extension | Format | Read | Write |
|---|---|:---:|:---:|
| `.csv` | CSV | yes | yes |
| `.tsv`, `.tab` | TSV | yes | yes |
| `.parquet`, `.pq` | Apache Parquet | yes | yes |
| `.arrow`, `.ipc`, `.feather` | Arrow IPC | yes | yes |
| `.graphml` | GraphML | yes | no |
| `.mtx`, `.mm` | Matrix Market | yes | no |
| `.net`, `.paj` | Pajek | yes | no |

## Probability/rate table schema

Custom sparse probabilities use:

| Column | Meaning |
|---|---|
| `source` | source node id |
| `target` | target node id |
| `probability` | value in `[0, 1]` |

Some custom Poisson filter paths use the same sparse pair idea with a `rate`
column at the CLI boundary.

## Python I/O

```python
from menobis.data.io import read_edges, write_edges

edges = read_edges("network.csv")
write_edges(edges, "network.parquet")
```

!!! note "CLI scope"
    The installed CLI currently exposes `fit`, `generate`, and `filter`.
    Use the Python I/O functions for format conversion.
