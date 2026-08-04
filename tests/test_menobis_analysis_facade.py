"""P0.8 tests: consolidated analysis facade and sparse graph view."""

import numpy as np

from menobis.analysis import analyze
from menobis.analysis.common import node_count
from menobis.analysis.facade import AnalysisResult
from menobis.data.frames import EdgeTable


def _triangle() -> EdgeTable:
    """3-node directed triangle with occupations."""
    return EdgeTable(
        source=np.array([0, 1, 2], dtype=np.uint64),
        target=np.array([1, 2, 0], dtype=np.uint64),
        occ_num=np.array([3, 4, 5], dtype=np.uint64),
    )


def test_node_count_is_shared() -> None:
    edges = _triangle()
    assert node_count(edges) == 3
    assert (
        node_count(
            EdgeTable(
                np.array([], dtype=np.uint64),
                np.array([], dtype=np.uint64),
                np.array([], dtype=np.uint64),
            )
        )
        == 0
    )


def test_analyze_returns_only_requested_metrics() -> None:
    edges = _triangle()
    r = analyze(edges, strengths=True, degrees=True)
    assert isinstance(r, AnalysisResult)
    assert r.requested == frozenset({"strengths", "degrees"})
    assert r.node_count == 3
    assert r.strengths is not None
    assert r.degrees is not None
    assert r.stats is None
    assert r.clustering is None


def test_analyze_full_pass() -> None:
    edges = _triangle()
    r = analyze(
        edges,
        strengths=True,
        degrees=True,
        stats=True,
        distribution=True,
        clustering=True,
        occupation_clustering=True,
    )
    assert r.stats is not None
    assert r.stats.strength_out.tolist() == [3, 4, 5]
    assert r.distribution is not None
    assert r.distribution.occ_num.tolist() == [3, 4, 5]
    assert r.clustering is not None
    assert r.clustering.values is not None
    assert r.occupation_clustering is not None
    assert r.occupation_clustering.values is not None


def test_analyze_matches_direct_calls() -> None:
    from menobis.analysis import (
        clustering_coefficient,
        directed_strengths,
        occupation_distribution,
    )

    edges = _triangle()
    r = analyze(edges, strengths=True, distribution=True, clustering=True)
    direct_s = directed_strengths(edges)
    direct_d = occupation_distribution(edges)
    direct_c = clustering_coefficient(edges)
    assert r.strengths is not None
    assert r.distribution is not None
    assert r.clustering is not None
    np.testing.assert_array_equal(r.strengths.out, direct_s.out)
    np.testing.assert_array_equal(r.distribution.occ_num, direct_d.occ_num)
    np.testing.assert_array_equal(r.clustering.values, direct_c.values)


def test_analyze_does_not_run_clustering_by_default() -> None:
    """Clustering is algorithmically distinct and opt-in."""
    edges = _triangle()
    r = analyze(edges, strengths=True)
    assert r.clustering is None
