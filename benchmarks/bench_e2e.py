"""Compatibility wrapper for the canonical PA geographic E2E benchmark."""

from __future__ import annotations

from pathlib import Path

from benchmarks.cli import run_pipeline
from benchmarks.fit import CONSTRAINTS, FAMILIES
from benchmarks.types import BenchmarkOptions


def bench_e2e(
    max_n: int = 500,
    nodes: list[int] | None = None,
    ensemble: int = 5,
    tolerance: float = 0.02,
) -> list[dict[str, object]]:
    """Run generate → fit → sample → null-filter using canonical PA networks."""
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
            tolerance_factor=tolerance,
            max_iterations=5_000,
            sample_count=ensemble,
            filter_sample_count=ensemble,
            alpha=0.05,
            self_loops=False,
        ),
        stage="all",
        output=Path("benchmarks/results/canonical-e2e"),
    )
    return [row.to_json() for row in rows]


def plot_e2e(_results: list[dict[str, object]]) -> None:
    """Compatibility no-op; canonical results are JSON-first."""


if __name__ == "__main__":
    bench_e2e()
