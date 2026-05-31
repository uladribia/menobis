"""Sampling API refactor characterization tests."""

import numpy as np

from menobis.models.generation import (
    _sample_strength_negative_binomial as sample_strength_negative_binomial,
)
from menobis.models.generation import (
    _sample_strength_stub_matching as sample_strength_stub_matching,
)


def test_stub_matching_replaces_stub_matching_name() -> None:
    """The exact-strength stub-matching sampler is exposed under its precise name."""
    sample = sample_strength_stub_matching(
        np.array([2, 1], dtype=np.uint64),
        np.array([1, 2], dtype=np.uint64),
        seed=7,
    )

    assert sample.weight.sum() == 3
    out_strength = np.bincount(sample.source, weights=sample.weight, minlength=2)
    in_strength = np.bincount(sample.target, weights=sample.weight, minlength=2)
    assert out_strength.tolist() == [2, 1]
    assert in_strength.tolist() == [1, 2]


def test_negative_binomial_public_name_is_spelled_out() -> None:
    """Public sampling names use negative_binomial rather than negative_binomial."""
    sample = sample_strength_negative_binomial(
        np.array([0.2, 0.3], dtype=np.float64),
        np.array([0.4, 0.5], dtype=np.float64),
        layers=2,
        seed=11,
    )

    assert np.all(sample.weight >= 0)
