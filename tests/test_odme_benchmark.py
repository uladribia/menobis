"""Pytest benchmark: regression tests against baselines.

Reads thresholds from ``benchmarks/regression_baselines.json`` so CI can use
machine-specific baselines without hardcoded constants in test code.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from odme.analysis import directed_strengths
from odme.analysis.stats import compute_all_stats
from odme.data.frames import EdgeTable
from odme.filtering import filter_fixed_strength_me
from odme.models import fit_fixed_strength_me, sample_microcanonical, sample_poisson

BASELINES_PATH = (
    Path(__file__).resolve().parent.parent / "benchmarks" / "regression_baselines.json"
)


def _load_threshold(name: str) -> float:
    """Load max_seconds for a named benchmark from the baselines file."""
    data = json.loads(BASELINES_PATH.read_text(encoding="utf-8"))
    return float(data["thresholds"][name]["max_seconds"])


def _large_network(n: int = 10000, avg_s: int = 100) -> EdgeTable:
    """Build a synthetic Pareto-strength network at given scale."""
    rng = np.random.default_rng(42)
    raw = rng.pareto(1.5, size=n) + 1.0
    p_out = raw / raw.sum()
    raw = rng.pareto(1.5, size=n) + 1.0
    p_in = raw / raw.sum()
    total = n * avg_s
    s_out = np.round(p_out * total).astype(np.uint64)
    s_in = np.round(p_in * total).astype(np.uint64)
    diff = int(s_out.sum()) - int(s_in.sum())
    if diff > 0:
        s_in[np.argmax(s_in)] += abs(diff)
    elif diff < 0:
        s_out[np.argmax(s_out)] += abs(diff)
    fit = fit_fixed_strength_me(s_out, s_in)
    return sample_poisson(fit.x, fit.y, seed=42)


@pytest.fixture(scope="module")
def large_edges() -> EdgeTable:
    """Module-scoped fixture for a large synthetic network."""
    return _large_network()


def test_analysis_10k(large_edges: EdgeTable) -> None:
    """Analysis at N=10000 completes within baseline threshold."""
    threshold = _load_threshold("analysis_10k")
    start = time.perf_counter()
    directed_strengths(large_edges)
    compute_all_stats(large_edges)
    elapsed = time.perf_counter() - start
    assert elapsed < threshold, f"analysis took {elapsed:.2f}s (limit {threshold}s)"


def test_fitting_strength_10k(large_edges: EdgeTable) -> None:
    """Fixed-strength fitting at N=10000 completes within baseline threshold."""
    threshold = _load_threshold("fitting_strength_10k")
    s = directed_strengths(large_edges)
    start = time.perf_counter()
    fit_fixed_strength_me(s.out, s.incoming)
    elapsed = time.perf_counter() - start
    assert elapsed < threshold, f"fitting took {elapsed:.2f}s (limit {threshold}s)"


def test_poisson_generation_10k(large_edges: EdgeTable) -> None:
    """Poisson generation at N=10000 completes within baseline threshold."""
    threshold = _load_threshold("poisson_generation_10k")
    s = directed_strengths(large_edges)
    fit = fit_fixed_strength_me(s.out, s.incoming)
    start = time.perf_counter()
    sample_poisson(fit.x, fit.y, seed=0)
    elapsed = time.perf_counter() - start
    msg = f"Poisson generation took {elapsed:.2f}s (limit {threshold}s)"
    assert elapsed < threshold, msg


def test_microcanonical_generation_10k(large_edges: EdgeTable) -> None:
    """Microcanonical generation at N=10000 completes within baseline threshold."""
    threshold = _load_threshold("microcanonical_generation_10k")
    s = directed_strengths(large_edges)
    start = time.perf_counter()
    sample_microcanonical(s.out, s.incoming, seed=0)
    elapsed = time.perf_counter() - start
    msg = f"microcanonical took {elapsed:.2f}s (limit {threshold}s)"
    assert elapsed < threshold, msg


def test_filtering_10k(large_edges: EdgeTable) -> None:
    """Filtering at N=10000 completes within baseline threshold."""
    threshold = _load_threshold("filtering_10k")
    start = time.perf_counter()
    filter_fixed_strength_me(large_edges, alpha=0.05)
    elapsed = time.perf_counter() - start
    assert elapsed < threshold, f"filtering took {elapsed:.2f}s (limit {threshold}s)"
