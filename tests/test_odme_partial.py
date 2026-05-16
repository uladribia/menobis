"""Tests for partial-constraint fitting with known p_ij pairs."""

import numpy as np
import pytest

from odme.analysis import directed_degrees, directed_strengths
from odme.data.frames import EdgeTable
from odme.models.partial import (
    fit_from_network_cutoff,
    fit_partial_degree_me,
    fit_partial_strength_cost_me,
    fit_partial_strength_degree_me,
    fit_partial_strength_edges_me,
    fit_partial_strength_me,
)


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
            np.array([20.0]),
        )


def test_partial_no_self_loops() -> None:
    """With self_loops=False, no diagonal pairs in fitted output."""
    edges = _small_network()
    s = directed_strengths(edges)
    result = fit_partial_strength_me(
        s.out.astype(float),
        s.incoming.astype(float),
        np.array([0], dtype=np.uint64),
        np.array([1], dtype=np.uint64),
        np.array([50.0]),
        self_loops=False,
    )
    for src, tgt in zip(result.source, result.target, strict=True):
        assert int(src) != int(tgt), f"self-loop found: ({src}, {tgt})"


def test_fit_from_network_cutoff_strength() -> None:
    """Convenience method splits by cutoff and fits."""
    edges = _small_network()
    result = fit_from_network_cutoff(edges, cutoff=10, model="strength")
    assert result.source.shape[0] > 0
    assert result.rate.shape[0] > 0

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


def test_fit_from_network_cutoff_no_self_loops() -> None:
    """Cutoff method respects self_loops=False."""
    edges = _small_network()
    result = fit_from_network_cutoff(
        edges, cutoff=10, model="strength", self_loops=False
    )
    for src, tgt in zip(result.source, result.target, strict=True):
        assert int(src) != int(tgt)


def test_partial_degree_recovers_constraints() -> None:
    """Partial fixed-degree model recovers excess degrees."""
    edges = _small_network()
    k = directed_degrees(edges)
    known_source = np.array([0, 1], dtype=np.uint64)
    known_target = np.array([1, 0], dtype=np.uint64)
    result = fit_partial_degree_me(
        k.out.astype(float),
        k.incoming.astype(float),
        known_source,
        known_target,
        self_loops=False,
    )
    assert result.source.shape[0] > 0
    for src, tgt in zip(result.source, result.target, strict=True):
        assert int(src) != int(tgt)


def test_partial_strength_degree_recovers_constraints() -> None:
    """Partial strength-degree model produces valid output."""
    edges = _small_network()
    s = directed_strengths(edges)
    k = directed_degrees(edges)
    known_source = np.array([0], dtype=np.uint64)
    known_target = np.array([1], dtype=np.uint64)
    known_rate = np.array([50.0])
    result = fit_partial_strength_degree_me(
        s.out.astype(float),
        s.incoming.astype(float),
        k.out.astype(float),
        k.incoming.astype(float),
        known_source,
        known_target,
        known_rate,
        self_loops=False,
    )
    assert result.source.shape[0] > 0
    for src, tgt, rate in zip(result.source, result.target, result.rate, strict=True):
        if int(src) == 0 and int(tgt) == 1:
            np.testing.assert_allclose(rate, 50.0, rtol=1e-10)
            break


def test_partial_strength_edges_recovers_constraints() -> None:
    """Partial strength-edges model produces valid output."""
    edges = _small_network()
    s = directed_strengths(edges)
    known_source = np.array([0], dtype=np.uint64)
    known_target = np.array([1], dtype=np.uint64)
    known_rate = np.array([50.0])
    result = fit_partial_strength_edges_me(
        s.out.astype(float),
        s.incoming.astype(float),
        known_source,
        known_target,
        known_rate,
        float(edges.num_edges),
        self_loops=False,
    )
    assert result.source.shape[0] > 0
    for src, tgt in zip(result.source, result.target, strict=True):
        assert int(src) != int(tgt)


def test_fit_from_network_cutoff_all_models() -> None:
    """Cutoff method works for all model variants."""
    edges = _small_network()
    for model in ["strength", "degree", "strength-degree", "strength-edges"]:
        result = fit_from_network_cutoff(
            edges, cutoff=10, model=model, self_loops=False
        )
        assert result.source.shape[0] > 0
        for src, tgt in zip(result.source, result.target, strict=True):
            assert int(src) != int(tgt), f"{model}: self-loop found"


def test_partial_strength_cost_recovers_constraints() -> None:
    """Partial strength-cost model recovers strengths."""
    n = 4
    edges = _small_network()
    s = directed_strengths(edges)
    # Build cost matrix.
    rng = np.random.default_rng(99)
    positions = rng.uniform(0.0, 10.0, size=(n, 2))
    c_src: list[int] = []
    c_tgt: list[int] = []
    c_val: list[float] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c_src.append(i)
            c_tgt.append(j)
            c_val.append(float(np.linalg.norm(positions[i] - positions[j])))
    cost_src = np.array(c_src)
    cost_tgt = np.array(c_tgt)
    cost_val = np.array(c_val)
    cost_map = {
        (int(s_val), int(t_val)): float(cv)
        for s_val, t_val, cv in zip(cost_src, cost_tgt, cost_val, strict=True)
    }
    target_cost = sum(
        float(w) * cost_map.get((int(s_val), int(t_val)), 0.0)
        for s_val, t_val, w in zip(
            edges.source, edges.target, edges.weight, strict=True
        )
    )
    known_source = np.array([0], dtype=np.uint64)
    known_target = np.array([1], dtype=np.uint64)
    known_rate = np.array([50.0])

    result = fit_partial_strength_cost_me(
        s.out.astype(float),
        s.incoming.astype(float),
        known_source,
        known_target,
        known_rate,
        cost_src,
        cost_tgt,
        cost_val,
        target_cost,
        self_loops=False,
    )
    assert result.source.shape[0] > 0
    # Known pair preserved.
    for src, tgt, rate in zip(result.source, result.target, result.rate, strict=True):
        if int(src) == 0 and int(tgt) == 1:
            np.testing.assert_allclose(rate, 50.0, rtol=1e-10)
            break
