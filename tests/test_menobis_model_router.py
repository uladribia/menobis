"""Unified model routing tests."""

import numpy as np
import pytest

from menobis.data.frames import EdgeTable
from menobis.models import (
    Constraint,
    Ensemble,
    ModelFamily,
    StrengthFit,
    UnsupportedModelCaseError,
    fit_model,
    sample_model,
)

S_OUT = np.array([3.0, 2.0], dtype=np.float64)
S_IN = np.array([2.0, 3.0], dtype=np.float64)


def test_fit_model_grandcanonical_strength_me() -> None:
    """Grand-canonical ME fixed strengths routes to Poisson fitter."""
    fit = fit_model(
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=S_OUT,
        strength_in=S_IN,
    )
    assert isinstance(fit, StrengthFit)
    assert fit.family == "poisson"


def test_sample_model_canonical_me_strength_uses_multinomial() -> None:
    """Canonical ME fixed-strength sampling has fixed total events."""
    fit = fit_model(
        ensemble=Ensemble.CANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=S_OUT,
        strength_in=S_IN,
    )
    sample = sample_model(
        ensemble=Ensemble.CANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        fit=fit,
        total_events=50,
        seed=7,
    )
    assert isinstance(sample, EdgeTable)
    assert sample.total_events == 50


def test_sample_model_microcanonical_me_strength_uses_stub_matching() -> None:
    """Microcanonical ME fixed-strength sampling preserves strengths exactly."""
    sample = sample_model(
        ensemble=Ensemble.MICROCANONICAL,
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH,
        strength_out=np.array([2, 1], dtype=np.uint64),
        strength_in=np.array([1, 2], dtype=np.uint64),
        seed=9,
    )
    assert sample.total_events == 3
    s_out = np.bincount(sample.source, weights=sample.weight, minlength=2)
    s_in = np.bincount(sample.target, weights=sample.weight, minlength=2)
    assert s_out.tolist() == [2, 1]
    assert s_in.tolist() == [1, 2]


def test_canonical_rejects_non_me_family() -> None:
    """Canonical ensemble is ME-only in MENoBiS."""
    with pytest.raises(UnsupportedModelCaseError, match=r"canonical.*ME"):
        fit_model(
            ensemble=Ensemble.CANONICAL,
            family=ModelFamily.B,
            constraint=Constraint.STRENGTH,
            strength_out=S_OUT,
            strength_in=S_IN,
            layers=3,
        )


def test_microcanonical_rejects_non_strength_constraint() -> None:
    """Microcanonical support is only ME fixed strengths."""
    with pytest.raises(UnsupportedModelCaseError, match=r"microcanonical.*ME"):
        sample_model(
            ensemble=Ensemble.MICROCANONICAL,
            family=ModelFamily.ME,
            constraint=Constraint.DEGREE_EVENTS,
            strength_out=np.array([1], dtype=np.uint64),
            strength_in=np.array([1], dtype=np.uint64),
        )


def test_fit_model_rejects_invalid_enum_value() -> None:
    """Constructing a ModelFamily from an invalid string raises ValueError."""
    with pytest.raises(ValueError):
        ModelFamily("invalid_family")
