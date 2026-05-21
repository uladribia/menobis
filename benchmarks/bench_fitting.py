"""Compatibility fitting benchmark API backed by the canonical pipeline."""

from __future__ import annotations

from pathlib import Path

from benchmarks.cli import run_pipeline
from benchmarks.fit import CONSTRAINTS, FAMILIES
from benchmarks.types import BenchmarkOptions


def fitting_case_registry(include_partial: bool = True) -> list[dict[str, object]]:
    """Return fitting benchmark case metadata for coverage tests."""
    cases: list[dict[str, object]] = []
    specs = (
        ("ME", "poisson", None),
        ("B", "binomial", 3),
        ("W", "geometric", None),
        ("W", "negative-binomial", 3),
    )
    for ensemble, family, layers in specs:
        for constraint in CONSTRAINTS:
            cases.append(
                {
                    "ensemble": ensemble,
                    "family": family,
                    "constraint": constraint,
                    "layers": layers,
                    "partial": False,
                }
            )
    if include_partial:
        for ensemble, family, layers in specs:
            for constraint in CONSTRAINTS:
                cases.append(
                    {
                        "ensemble": ensemble,
                        "family": family,
                        "constraint": constraint,
                        "layers": layers,
                        "partial": True,
                    }
                )
    return cases


def parse_nodes(nodes: str | None) -> list[int] | None:
    """Parse comma-separated benchmark sizes."""
    return None if nodes is None else [int(part) for part in nodes.split(",") if part]


def parse_floats(values: str | None) -> list[float] | None:
    """Parse comma-separated floats."""
    return None if values is None else [float(part) for part in values.split(",") if part]


def _benchmark_sizes(max_n: int, nodes: list[int] | None = None) -> list[int]:
    """Return requested benchmark sizes."""
    return sorted({n for n in (nodes or [100, 500]) if n <= max_n})


def bench_all(
    max_n: int = 500,
    tolerance: float = 0.02,
    verbose: int = 0,
    nodes: list[int] | None = None,
) -> list[dict[str, object]]:
    """Run canonical fitting benchmarks on PA geographic constraints."""
    del verbose
    rows = run_pipeline(
        _options(max_n=max_n, tolerance=tolerance, nodes=nodes, samples=0),
        stage="fit",
        output=Path("benchmarks/results/canonical-fit"),
    )
    return [row.to_json() for row in rows]


def bench_partial(
    max_n: int = 500,
    tolerance: float = 0.02,
    verbose: int = 0,
    nodes: list[int] | None = None,
    known_fractions: list[float] | None = None,
) -> list[dict[str, object]]:
    """Partial fitting benchmarks are pending canonical PA integration."""
    del max_n, tolerance, verbose, nodes, known_fractions
    return []


def save_results(results: list[dict[str, object]], output_dir: str = "benchmarks/results") -> None:
    """Save compatibility benchmark results."""
    import json

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "fitting_benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def plot_results(_results: list[dict[str, object]], _output_dir: str | Path) -> None:
    """Compatibility no-op; canonical results are JSON-first."""


def _options(max_n: int, tolerance: float, nodes: list[int] | None, samples: int) -> BenchmarkOptions:
    return BenchmarkOptions(
        nodes=tuple(_benchmark_sizes(max_n, nodes)),
        families=FAMILIES,
        constraints=CONSTRAINTS,
        seed=10_000,
        average_degree=8.0,
        density=None,
        events_per_edge=6.0,
        tolerance_factor=tolerance,
        max_iterations=5_000,
        sample_count=samples,
        filter_sample_count=samples,
        alpha=0.05,
        self_loops=False,
    )
