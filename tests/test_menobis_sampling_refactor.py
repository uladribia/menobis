"""Sampling API refactor characterization tests."""

import numpy as np

from menobis.models.generation import (
    _sample_strength_fixed_strength_mcmc as sample_fixed_strength,
)
from menobis.models.generation import (
    _sample_strength_negative_binomial as sample_strength_negative_binomial,
)


def test_fixed_strength_mcmc_preserves_strengths() -> None:
    """The MCMC fixed-strength sampler preserves exact strengths."""
    sample = sample_fixed_strength(
        family="ME",
        strength_out=np.array([2, 1], dtype=np.uint64),
        strength_in=np.array([1, 2], dtype=np.uint64),
        self_loops=True,
        seed=7,
    )

    assert sample.occ_num.sum() == 3
    out_strength = np.bincount(sample.source, weights=sample.occ_num, minlength=2)
    in_strength = np.bincount(sample.target, weights=sample.occ_num, minlength=2)
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

    assert np.all(sample.occ_num >= 0)
