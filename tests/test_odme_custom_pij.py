"""Tests for original custom p_ij generation case."""

import numpy as np
import pytest

from odme.data.frames import normalize_probabilities
from odme.models import (
    sample_custom_pij_events_multinomial,
    sample_custom_pij_events_poisson,
)


def test_probability_table_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        normalize_probabilities(np.array([0]), np.array([1]), np.array([1.5]))


def test_custom_pij_events_multinomial_preserves_total_and_support() -> None:
    probabilities = normalize_probabilities(
        np.array([0, 1]), np.array([1, 2]), np.array([0.25, 0.75])
    )
    sample = sample_custom_pij_events_multinomial(
        probabilities, total_events=100, seed=42
    )

    assert sample.total_events == 100
    assert set(zip(sample.source.tolist(), sample.target.tolist(), strict=True)) <= {
        (0, 1),
        (1, 2),
    }


def test_custom_pij_events_poisson_is_reproducible() -> None:
    probabilities = normalize_probabilities(
        np.array([0, 1]), np.array([1, 2]), np.array([0.25, 0.75])
    )
    first = sample_custom_pij_events_poisson(probabilities, total_events=100, seed=42)
    second = sample_custom_pij_events_poisson(probabilities, total_events=100, seed=42)

    np.testing.assert_array_equal(first.source, second.source)
    np.testing.assert_array_equal(first.target, second.target)
    np.testing.assert_array_equal(first.weight, second.weight)
