"""Shared helpers for ODME benchmark scripts."""

from __future__ import annotations

import gc
import resource
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np

from odme.analysis import directed_strengths
from odme.data.frames import EdgeTable, ProbabilityTable
from odme.models import fit_strength_poisson, sample_strength_poisson

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

T = TypeVar("T")


@dataclass(frozen=True)
class SparseCosts:
    """Sparse pair costs plus observed target cost for a network."""

    source: np.ndarray
    target: np.ndarray
    value: np.ndarray
    target_cost: float


def max_rss_mb() -> float:
    """Return maximum resident set size in MiB for the current process."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def time_call(function: Callable[..., T], *args: object, **kwargs: object) -> float:
    """Run a callable after GC and return elapsed seconds."""
    gc.collect()
    start = time.perf_counter()
    function(*args, **kwargs)
    return time.perf_counter() - start


def balanced_pareto_strengths(
    node_count: int,
    average_strength: int,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create balanced heterogeneous out/in strength sequences."""
    rng = np.random.default_rng(seed)
    total = node_count * average_strength
    raw_out = rng.pareto(1.5, size=node_count) + 1.0
    raw_in = rng.pareto(1.5, size=node_count) + 1.0
    strength_out = np.round(raw_out / raw_out.sum() * total).astype(np.uint64)
    strength_in = np.round(raw_in / raw_in.sum() * total).astype(np.uint64)
    _balance_integer_sequences(strength_out, strength_in)
    return strength_out, strength_in


def pareto_strength_network(
    node_count: int,
    average_strength: int,
    *,
    seed: int = 42,
) -> EdgeTable:
    """Sample a fixed-strength Poisson network from Pareto-like strengths."""
    strength_out, strength_in = balanced_pareto_strengths(
        node_count, average_strength, seed=seed
    )
    fit = fit_strength_poisson(strength_out, strength_in)
    return sample_strength_poisson(fit.x, fit.y, seed=seed)


def edge_strengths(edges: EdgeTable, node_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return balanced integer strengths for a known node count."""
    strengths = directed_strengths(edges)
    strength_out = np.zeros(node_count, dtype=np.uint64)
    strength_in = np.zeros(node_count, dtype=np.uint64)
    strength_out[: len(strengths.out)] = strengths.out
    strength_in[: len(strengths.incoming)] = strengths.incoming
    _balance_integer_sequences(strength_out, strength_in)
    return strength_out, strength_in


def sparse_probability_table(node_count: int, degree: int) -> ProbabilityTable:
    """Build deterministic sparse custom p_ij support with O(N * degree) entries."""
    degree = max(1, min(degree, node_count))
    sources = np.repeat(np.arange(node_count, dtype=np.uint64), degree)
    offsets = np.arange(1, degree + 1, dtype=np.uint64)
    targets = (sources.reshape(node_count, degree) + offsets.reshape(1, degree))
    return ProbabilityTable(
        source=sources,
        target=(targets % node_count).reshape(-1).astype(np.uint64),
        probability=np.ones(node_count * degree, dtype=np.float64),
    )


def sparse_exponential_costs(
    edges: EdgeTable,
    node_count: int,
    degree: int,
    *,
    seed: int = 42,
    target_fraction: float = 0.9,
) -> SparseCosts:
    """Build deterministic sparse costs and a feasible observed-cost target."""
    rng = np.random.default_rng(seed)
    degree = max(1, min(degree, node_count))
    sources = np.repeat(np.arange(node_count, dtype=np.uint64), degree)
    offsets = np.arange(1, degree + 1, dtype=np.uint64)
    targets = (sources.reshape(node_count, degree) + offsets.reshape(1, degree))
    targets = (targets % node_count).reshape(-1).astype(np.uint64)
    values = rng.exponential(1.0, size=len(sources)).astype(np.float64)
    target_cost = target_fraction * observed_sparse_cost(edges, sources, targets, values)
    if target_cost <= 0.0:
        target_cost = target_fraction * float(np.mean(values) * edges.total_events)
    return SparseCosts(sources, targets, values, target_cost)


def complete_euclidean_costs(
    edges: EdgeTable,
    node_count: int,
    *,
    seed: int = 99,
) -> SparseCosts:
    """Build complete directed non-self-loop Euclidean costs."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, 10.0, size=(node_count, 2))
    pairs = [(i, j) for i in range(node_count) for j in range(node_count) if i != j]
    sources = np.fromiter((i for i, _ in pairs), dtype=np.uint64, count=len(pairs))
    targets = np.fromiter((j for _, j in pairs), dtype=np.uint64, count=len(pairs))
    values = np.fromiter(
        (float(np.linalg.norm(positions[i] - positions[j])) for i, j in pairs),
        dtype=np.float64,
        count=len(pairs),
    )
    return SparseCosts(sources, targets, values, observed_sparse_cost(edges, sources, targets, values))


def observed_sparse_cost(
    edges: EdgeTable,
    cost_sources: np.ndarray,
    cost_targets: np.ndarray,
    cost_values: np.ndarray,
) -> float:
    """Compute observed cost using zero for pairs absent from sparse costs."""
    cost_by_pair = {
        (int(src), int(tgt)): float(cost)
        for src, tgt, cost in zip(cost_sources, cost_targets, cost_values, strict=True)
    }
    return sum(
        float(weight) * cost_by_pair.get((int(src), int(tgt)), 0.0)
        for src, tgt, weight in zip(edges.source, edges.target, edges.weight, strict=True)
    )


def _balance_integer_sequences(out: np.ndarray, incoming: np.ndarray) -> None:
    diff = int(out.sum()) - int(incoming.sum())
    if diff > 0:
        incoming[np.argmax(incoming)] += abs(diff)
    elif diff < 0:
        out[np.argmax(out)] += abs(diff)
