"""Tests verifying full API coherence across ME, W, and B families."""

import numpy as np
import pytest

from odme.models import (
    DegreeEventsFit,
    FitResult,
    StrengthCostFit,
    StrengthDegreeFit,
    StrengthEdgesFit,
    fit_degree_bernoulli,
    fit_degree_events_binomial,
    fit_degree_events_geometric,
    fit_degree_events_negative_binomial,
    fit_degree_events_poisson,
    fit_strength_binomial,
    fit_strength_cost_binomial,
    fit_strength_cost_geometric,
    fit_strength_cost_negative_binomial,
    fit_strength_cost_poisson,
    fit_strength_degree_binomial,
    fit_strength_degree_geometric,
    fit_strength_degree_negative_binomial,
    fit_strength_degree_poisson,
    fit_strength_edges_binomial,
    fit_strength_edges_geometric,
    fit_strength_edges_negative_binomial,
    fit_strength_edges_poisson,
    fit_strength_geometric,
    fit_strength_negative_binomial,
    fit_strength_poisson,
)
from odme.models.types import OptimizationDiagnostics

S_OUT = np.array([2.0, 2.0])
S_IN = np.array([2.0, 2.0])
K_OUT = np.array([1.0, 1.0])
K_IN = np.array([1.0, 1.0])
C_SRC = np.array([0, 1], dtype=np.uint64)
C_TGT = np.array([1, 0], dtype=np.uint64)
C_VAL = np.array([1.0, 1.0])


@pytest.mark.parametrize(
    "func,expected_family",
    [
        (lambda: fit_strength_poisson(S_OUT, S_IN), "poisson"),
        (lambda: fit_strength_geometric(S_OUT, S_IN), "geometric"),
        (
            lambda: fit_strength_negative_binomial(S_OUT, S_IN, layers=3),
            "negative_binomial",
        ),
        (lambda: fit_strength_binomial(S_OUT, S_IN, layers=3), "binomial"),
        (lambda: fit_degree_bernoulli(K_OUT, K_IN), "bernoulli"),
    ],
)
def test_strength_fits_return_fit_result_with_all_fields(func, expected_family) -> None:
    result = func()
    assert isinstance(result, FitResult)
    assert result.family == expected_family
    assert isinstance(result.self_loops, bool)
    assert result.diagnostics is not None
    assert isinstance(result.diagnostics, OptimizationDiagnostics)
    assert result.diagnostics.converged == result.converged
    assert result.diagnostics.iterations == result.iterations


@pytest.mark.parametrize(
    "func,expected_family",
    [
        (
            lambda: fit_strength_cost_poisson(S_OUT, S_IN, C_SRC, C_TGT, C_VAL, 1.0),
            "poisson",
        ),
        (
            lambda: fit_strength_cost_geometric(S_OUT, S_IN, C_SRC, C_TGT, C_VAL, 1.0),
            "geometric",
        ),
        (
            lambda: fit_strength_cost_negative_binomial(
                S_OUT, S_IN, C_SRC, C_TGT, C_VAL, 1.0, layers=3
            ),
            "negative_binomial",
        ),
        (
            lambda: fit_strength_cost_binomial(
                S_OUT, S_IN, C_SRC, C_TGT, C_VAL, 1.0, layers=3
            ),
            "binomial",
        ),
    ],
)
def test_strength_cost_fits_return_consistent_shape(func, expected_family) -> None:
    result = func()
    assert isinstance(result, StrengthCostFit)
    assert result.family == expected_family
    assert isinstance(result.self_loops, bool)
    assert isinstance(result.gamma, float)
    assert result.diagnostics is not None


@pytest.mark.parametrize(
    "func,expected_family",
    [
        (lambda: fit_strength_edges_poisson(S_OUT, S_IN, 1.5), "poisson"),
        (lambda: fit_strength_edges_geometric(S_OUT, S_IN, 1.5), "geometric"),
        (
            lambda: fit_strength_edges_negative_binomial(S_OUT, S_IN, 1.5, layers=3),
            "negative_binomial",
        ),
        (lambda: fit_strength_edges_binomial(S_OUT, S_IN, 1.5, layers=3), "binomial"),
    ],
)
def test_strength_edges_fits_return_consistent_shape(func, expected_family) -> None:
    result = func()
    assert isinstance(result, StrengthEdgesFit)
    assert result.family == expected_family
    assert isinstance(result.self_loops, bool)
    assert isinstance(result.lam, float)
    assert result.diagnostics is not None


@pytest.mark.parametrize(
    "func,expected_family",
    [
        (lambda: fit_strength_degree_poisson(S_OUT, S_IN, K_OUT, K_IN), "poisson"),
        (lambda: fit_strength_degree_geometric(S_OUT, S_IN, K_OUT, K_IN), "geometric"),
        (
            lambda: fit_strength_degree_negative_binomial(
                S_OUT, S_IN, K_OUT, K_IN, layers=3
            ),
            "negative_binomial",
        ),
        (
            lambda: fit_strength_degree_binomial(S_OUT, S_IN, K_OUT, K_IN, layers=3),
            "binomial",
        ),
    ],
)
def test_strength_degree_fits_return_consistent_shape(func, expected_family) -> None:
    result = func()
    assert isinstance(result, StrengthDegreeFit)
    assert result.family == expected_family
    assert isinstance(result.self_loops, bool)
    assert result.z is not None
    assert result.w is not None
    assert result.diagnostics is not None


@pytest.mark.parametrize(
    "func,expected_family",
    [
        (lambda: fit_degree_events_poisson(K_OUT, K_IN, 4), "poisson"),
        (lambda: fit_degree_events_geometric(K_OUT, K_IN, 4), "geometric"),
        (
            lambda: fit_degree_events_negative_binomial(K_OUT, K_IN, 4, layers=3),
            "negative_binomial",
        ),
        (lambda: fit_degree_events_binomial(K_OUT, K_IN, 4, layers=3), "binomial"),
    ],
)
def test_degree_events_fits_return_consistent_shape(func, expected_family) -> None:
    result = func()
    assert isinstance(result, DegreeEventsFit)
    assert result.family == expected_family
    assert isinstance(result.self_loops, bool)
    assert isinstance(result.q, float)
    assert isinstance(result.positive_mean, float)
    assert result.diagnostics is not None
