"""Tests for partial-constraint fitting with known p_ij pairs."""

import numpy as np
import pytest

from odme.analysis import directed_strengths
from odme.data.frames import EdgeTable
from odme.models.partial import fit_from_network_cutoff, fit_partial_strength_me


def _small_network() -> EdgeTable:
    """Network with heterogeneous weights: some large, some small."""
    return EdgeTable(
        source=np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.uint64),
        target=np.array([1, 2, 3, 0, 2, 3, 0, 1, 3], dtype=np.uint64),
        weight=np.array([50, 3, 2, 40, 4, 1, 30, 5, 3], dtype=np.uint64),
    )


def test_partial_strength_recovers_constraints() -> None:
    """Fitted partial model recovers full strength sequence."""
    edges = _small_network()
    s = directed_strengths(edges)
    # Fix the 3 heaviest edges as known.
    known_source = np.array([0, 1, 2], dtype=np.uint64)
    known_target = np.array([1, 0, 0], dtype=np.uint64)
    known_weight = np.array([50, 40, 30], dtype=np.uint64)

    result = fit_partial_strength_me(
        s.out.astype(float),
        s.incoming.astype(float),
        known_source,
        known_target,
        known_weight.astype(float),
    )

    # The combined rate table should recover strengths.
    n = 4
    expected_matrix = np.zeros((n, n))
    for src, tgt, rate in zip(result.source, result.target, result.rate, strict=True):
        expected_matrix[int(src), int(tgt)] = rate
    np.testing.assert_allclose(
        expected_matrix.sum(axis=1), s.out.astype(float), atol=1.0, rtol=0.05
    )
    np.testing.assert_allclose(
        expected_matrix.sum(axis=0), s.incoming.astype(float), atol=1.0, rtol=0.05
    )


def test_partial_strength_known_rates_preserved() -> None:
    """Known pair rates appear unchanged in result."""
    edges = _small_network()
    s = directed_strengths(edges)
    known_source = np.array([0], dtype=np.uint64)
    known_target = np.array([1], dtype=np.uint64)
    known_weight = np.array([50.0])

    result = fit_partial_strength_me(
        s.out.astype(float),
        s.incoming.astype(float),
        known_source,
        known_target,
        known_weight,
    )
    # Find rate for (0,1).
    for src, tgt, rate in zip(result.source, result.target, result.rate, strict=True):
        if int(src) == 0 and int(tgt) == 1:
            np.testing.assert_allclose(rate, 50.0, rtol=1e-10)
            break


def test_partial_rejects_infeasible_excess() -> None:
    """If known pairs exceed observed strength, reject."""
    with pytest.raises(ValueError, match="exceed"):
        fit_partial_strength_me(
            np.array([10.0, 10.0]),
            np.array([10.0, 10.0]),
            np.array([0], dtype=np.uint64),
            np.array([1], dtype=np.uint64),
            np.array([20.0]),  # exceeds s_out[0]=10
        )


def test_fit_from_network_cutoff_strength() -> None:
    """Convenience method splits by cutoff and fits."""
    edges = _small_network()
    result = fit_from_network_cutoff(edges, cutoff=10, model="strength")
    assert result.source.shape[0] > 0
    assert result.rate.shape[0] > 0

    # Ensemble check: sample and verify strengths.
    from odme.models import sample_custom_pij_events_poisson

    s = directed_strengths(edges)
    total = edges.total_events
    repetitions = 200
    sampled_out: list[np.ndarray] = []
    for seed in range(repetitions):
        sample = sample_custom_pij_events_poisson(
            result.as_probability_table(), total_events=total, seed=seed
        )
        sampled_out.append(directed_strengths(sample).out.astype(float))
    mean = np.vstack(sampled_out).mean(axis=0)
    np.testing.assert_allclose(mean, s.out.astype(float), atol=3.0, rtol=0.1)
