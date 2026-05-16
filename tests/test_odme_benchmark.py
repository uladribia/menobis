"""Pytest benchmark: ensures N=10000 operations complete in reasonable time."""

import numpy as np
import pytest

from odme.analysis import directed_strengths
from odme.analysis.stats import compute_all_stats
from odme.data.frames import EdgeTable
from odme.models import (
    fit_fixed_strength_me,
    sample_microcanonical,
    sample_poisson,
)


def _large_network(n: int = 10000, avg_s: int = 100) -> EdgeTable:
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
    return _large_network()


def test_analysis_10k(large_edges: EdgeTable) -> None:
    """Analysis at N=10000 completes under 1 second."""
    import time

    start = time.perf_counter()
    directed_strengths(large_edges)
    compute_all_stats(large_edges)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"analysis took {elapsed:.2f}s"


def test_fitting_strength_10k(large_edges: EdgeTable) -> None:
    """Fixed-strength fitting at N=10000 is instant."""
    import time

    s = directed_strengths(large_edges)
    start = time.perf_counter()
    fit_fixed_strength_me(s.out, s.incoming)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.1, f"fitting took {elapsed:.2f}s"


def test_poisson_generation_10k(large_edges: EdgeTable) -> None:
    """Poisson generation at N=10000 completes under 5 seconds."""
    import time

    s = directed_strengths(large_edges)
    fit = fit_fixed_strength_me(s.out, s.incoming)
    start = time.perf_counter()
    sample_poisson(fit.x, fit.y, seed=0)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"Poisson generation took {elapsed:.2f}s"


def test_microcanonical_generation_10k(large_edges: EdgeTable) -> None:
    """Microcanonical generation at N=10000 completes under 5 seconds."""
    import time

    s = directed_strengths(large_edges)
    start = time.perf_counter()
    sample_microcanonical(s.out, s.incoming, seed=0)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"microcanonical generation took {elapsed:.2f}s"
