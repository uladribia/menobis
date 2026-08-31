---
description: MENoBiS file schemas and supported input/output formats.
---

# File formats

## TL;DR

The canonical network file is a sparse directed edge table with integer
occupations under the column schema `source target occ_num`. Zero
occupations are dropped; negative or fractional values are rejected at the
boundary.

## Canonical edge-table schema

| Column | Type | Rule |
|---|---|---|
| `source` | non-negative integer | source node id |
| `target` | non-negative integer | target node id |
| `occ_num` | non-negative integer | occupation number; zero rows are ignored |

Canonical CSV:

```csv
source,target,occ_num
0,1,12
0,2,3
2,1,7
```

Readers may accept `weight` as an **input alias** where the current I/O code
explicitly supports it; **writers always emit `occ_num`**. `weight` is never
presented as the canonical writer schema.

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
from menobis.data import read_edges, write_edges

edges = read_edges("network.csv")
write_edges(edges, "network.parquet")
```

Round-trip guarantee: writing then reading back preserves `source`,
`target`, and positive `occ_num` entries.

!!! note "CLI scope"
    The Python API is the authoritative full model interface; CLI command
    strings expose a convenience subset and may retain command names that
    do not mirror the model ontology exactly (see
    [CLI overview](../cli/fit.md)). Use the Python I/O functions for format
    conversion.