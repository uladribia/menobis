"""Compatibility filter benchmark using fitted PA-constraint null samples."""

from __future__ import annotations

from pathlib import Path

from benchmarks.cli import run_pipeline
from benchmarks.fit import CONSTRAINTS, FAMILIES
from benchmarks.types import BenchmarkOptions


def bench_filter(max_n: int = 500, ensemble: int = 5, nodes: list[int] | None = None) -> list[dict[str, object]]:
    """Benchmark filter calibration on samples from fitted null models."""
    selected = tuple(n for n in (nodes or [100, 500]) if n <= max_n)
    rows = run_pipeline(
        BenchmarkOptions(
            nodes=selected,
            families=FAMILIES,
            constraints=CONSTRAINTS,
            seed=10_000,
            average_degree=8.0,
            density=None,
            events_per_edge=6.0,
            tolerance_factor=0.02,
            max_iterations=5_000,
            sample_count=0,
            filter_sample_count=ensemble,
            alpha=0.05,
            self_loops=False,
        ),
        stage="filter",
        output=Path("benchmarks/results/canonical-filter"),
    )
    return [row.to_json() for row in rows]


def plot_filter(_results: list[dict[str, object]]) -> None:
    """Compatibility no-op; canonical results are JSON-first."""
