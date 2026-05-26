"""Tests for strength-degree invariants."""

import numpy as np
import pytest

from menobis.analysis import directed_degrees, directed_strengths
from menobis.data.frames import normalize_edges


def test_positive_integer_weights_force_strength_at_least_degree() -> None:
    edges = normalize_edges(
        np.array([0, 0, 1, 2]),
        np.array([1, 2, 2, 0]),
        np.array([1, 3, 1, 4]),
    )

    strengths = directed_strengths(edges)
    degrees = directed_degrees(edges)

    assert np.all(strengths.out >= degrees.out)
    assert np.all(strengths.incoming >= degrees.incoming)


def test_normalize_edges_rejects_fractional_weights() -> None:
    with pytest.raises(ValueError, match="integer"):
        normalize_edges(
            np.array([0]),
            np.array([1]),
            np.array([0.5]),
        )
