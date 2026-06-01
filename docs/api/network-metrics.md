---
description: Computing network metrics without coupling MENoBiS to graph libraries.
---

# Network metrics

## TL;DR

MENoBiS computes core directed non-binary statistics in Rust. NetworkX and
rustworkx are optional user choices: install them in your own environment and
convert an `EdgeTable` at the boundary of your application.

## Built-in MENoBiS metrics

| API | Runtime | Purpose |
|---|---|---|
| `directed_strengths(edges)` | Rust via Python | out/in event totals |
| `directed_degrees(edges)` | Rust via Python | out/in binary degrees |
| `compute_all_stats(edges)` | Rust via Python | strengths, degrees, Y2, nearest-neighbour stats |
| `weight_distribution(edges)` | Rust via Python | occupation-count histogram |
| `clustering_coefficient(edges)` | Rust via Python | binary clustering |
| `weighted_clustering_coefficient(edges)` | Rust via Python | occupation-weighted clustering helper |

```python
from menobis.analysis import compute_all_stats
from menobis.data.io import read_edges

edges = read_edges("edges.csv")
stats = compute_all_stats(edges)
print(stats.strength_out)
```

## Optional NetworkX recipe

NetworkX is useful for exploratory Python-only metrics. Keep this adapter in
your project so MENoBiS stays independent of NetworkX release cycles.

```bash
python -m pip install networkx
```

```python
import networkx as nx

from menobis.data.io import read_edges

edges = read_edges("edges.csv")
graph = nx.DiGraph()
graph.add_weighted_edges_from(
    zip(edges.source.tolist(), edges.target.tolist(), edges.weight.tolist())
)
centrality = nx.pagerank(graph, weight="weight")
```

## Optional rustworkx recipe

```python
import rustworkx as rx

from menobis.data.io import read_edges

edges = read_edges("edges.csv")
graph = rx.PyDiGraph(multigraph=True)
if len(edges) > 0:
    graph.add_nodes_from(range(int(max(edges.source.max(), edges.target.max())) + 1))
for source, target, weight in zip(edges.source, edges.target, edges.weight, strict=True):
    graph.add_edge(int(source), int(target), int(weight))
```

!!! note "Dependency policy"
    External graph libraries remain downstream adapters unless MENoBiS adopts a
    metric as supported core functionality with Rust tests and docs.
