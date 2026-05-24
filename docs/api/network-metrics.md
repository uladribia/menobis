---
description: Computing network metrics without coupling ODME to graph libraries.
---

# Network metrics

## TL;DR

ODME computes core weighted directed statistics in Rust. NetworkX and rustworkx
are optional user choices: install them in your own environment and convert an
`EdgeTable` at the boundary of your application.

## Built-in ODME metrics

| API | Runtime | Purpose |
|---|---|---|
| `directed_strengths(edges)` | Rust via Python | out/in weighted strengths |
| `directed_degrees(edges)` | Rust via Python | out/in binary degrees |
| `compute_all_stats(edges)` | Rust via Python | strengths, degrees, Y2, nearest-neighbour stats |
| `weight_distribution(edges)` | Rust via Python | edge-weight histogram |
| `clustering_coefficient(edges)` | Rust via Python | binary clustering |
| `weighted_clustering_coefficient(edges)` | Rust via Python | weighted clustering |

```python
from odme.analysis import compute_all_stats
from odme.data.io import read_edges

edges = read_edges("edges.csv")
stats = compute_all_stats(edges)
print(stats.strength_out)
```

Rust users can call `odme-core` graph/stat modules directly:

```rust
use odme_core::graph::{directed_strengths, WeightedEdge};

let edges = vec![WeightedEdge::new(0, 1, 3), WeightedEdge::new(1, 2, 4)];
let strengths = directed_strengths(3, &edges);
assert_eq!(strengths.out, vec![3, 4, 0]);
```

## Optional NetworkX recipe

NetworkX is useful for exploratory Python-only metrics. Keep this adapter in
your project so ODME stays independent of NetworkX release cycles.

```bash
python -m pip install networkx
```

```python
import networkx as nx

from odme.data.io import read_edges

edges = read_edges("edges.csv")
graph = nx.DiGraph()
graph.add_weighted_edges_from(
    zip(edges.source.tolist(), edges.target.tolist(), edges.weight.tolist())
)
centrality = nx.pagerank(graph, weight="weight")
```

## Optional rustworkx recipe

rustworkx can be useful when you want Python access to Rust graph algorithms.
Keep conversion code outside ODME and treat node identifiers as your own API.

```bash
python -m pip install rustworkx
```

```python
import rustworkx as rx

from odme.data.io import read_edges

edges = read_edges("edges.csv")
graph = rx.PyDiGraph(multigraph=True)
if len(edges) > 0:
    graph.add_nodes_from(range(int(max(edges.source.max(), edges.target.max())) + 1))
for source, target, weight in zip(edges.source, edges.target, edges.weight, strict=True):
    graph.add_edge(int(source), int(target), int(weight))
```

## Rust-side extension guidance

If ODME's Rust metrics are insufficient, create a downstream crate or binary
that depends on `odme-core` plus your chosen graph crate. Convert
`WeightedEdge` streams at that boundary. Do not add those graph crates to ODME
unless a metric becomes a supported ODME feature with Rust tests and docs.
