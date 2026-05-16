"""Tests for analysis summary functions."""

import numpy as np

from odme.analysis import directed_degrees, directed_strengths
from odme.data.frames import normalize_edges


def test_directed_strengths() -> None:
    edges = normalize_edges(np.array([0, 1]), np.array([1, 2]), np.array([3, 4]))
    s = directed_strengths(edges)
    np.testing.assert_array_equal(s.out, [3, 4, 0])
    np.testing.assert_array_equal(s.incoming, [0, 3, 4])


def test_directed_degrees() -> None:
    edges = normalize_edges(
        np.array([0, 0, 1]), np.array([1, 2, 2]), np.array([3, 4, 5])
    )
    d = directed_degrees(edges)
    np.testing.assert_array_equal(d.out, [2, 1, 0])
    np.testing.assert_array_equal(d.incoming, [0, 1, 2])
