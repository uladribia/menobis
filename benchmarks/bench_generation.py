"""Compatibility generation benchmark using PA geographic networks."""

from __future__ import annotations

import time

from benchmarks.bench_fitting import _benchmark_sizes
from odme.synthetic import generate_pa_geographic_network


def bench_generation(max_n: int = 5000, repeats: int = 3, nodes: list[int] | None = None) -> list[dict[str, object]]:
    """Benchmark canonical synthetic network generation."""
    results: list[dict[str, object]] = []
    for node_count in _benchmark_sizes(max_n, nodes):
        times = []
        edge_count = 0
        for repeat in range(repeats):
            started = time.perf_counter()
            network = generate_pa_geographic_network(
                node_count,
                average_degree=8.0,
                events_per_edge=6.0,
                seed=10_000 + node_count + repeat,
                self_loops=False,
            )
            times.append(time.perf_counter() - started)
            edge_count = network.edges.num_edges
        results.append(
            {
                "name": "pa-geographic",
                "n": node_count,
                "edges": edge_count,
                "time": sum(times) / len(times),
            }
        )
    return results


def plot_generation(_results: list[dict[str, object]]) -> None:
    """Compatibility no-op; canonical results are JSON-first."""
