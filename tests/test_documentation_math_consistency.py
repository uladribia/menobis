"""Mathematical consistency tests for documented distribution equations (§52).

These tests verify that the equations published in the scientific
documentation (``science/event-families.md``, ``science/constraints.md``)
match the behaviour of the implemented code:

- ME pair mean:      E[t] = q
- B pair mean:       E[t] = M q / (1 + q)
- W pair mean:       E[t] = M q / (1 - q)
- B(M=1) invariant:  t in {0,1}, hence s = k
- Zero-inflated G_F layer: P(t>0) and E[t] expressed through the support
  fugacity, cross-checked against the per-edge null expectations returned
  by the filter kernel.

Method: symmetric strength sequences make every fitted pair parameter
collapse to a single q, so a grand-canonical sample's empirical mean over
all candidate pairs estimates the documented pair mean directly.

Tolerances: Monte Carlo standard error of a pair mean over L = N(N-1)
pairs is below 0.01 for the chosen sizes; the single-draw sampling checks
use 3% relative tolerance (ME, B) and 5% (W, which has larger variance).
The zero-inflated cross-checks are deterministic and use machine tolerance.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from menobis.data.frames import EdgeTable
from menobis.filtering import filter_model
from menobis.models import Constraint, Ensemble, ModelFamily, fit_model, sample_model
from menobis.models.types import StrengthEdgesFit, StrengthFit

N = 200
L = N * (N - 1)


def _constant_strength(value: float) -> np.ndarray:
    """Symmetric constant strength sequence on N nodes."""
    return np.full(N, value)


def _empirical_pair_mean(fit: object, family: ModelFamily, *, seed: int = 0) -> float:
    """Sample once from the fitted model and average over all pairs."""
    sample = sample_model(
        ensemble=Ensemble.GRAND_CANONICAL,
        family=family,
        constraint=Constraint.STRENGTH,
        fit=fit,
        seed=seed,
    )
    return float(sample.occ_num.sum()) / L


def _pair_q(fit: StrengthFit) -> float:
    """Per-pair fugacity q = x_i y_j (constant across pairs by symmetry)."""
    x0 = float(fit.x[0])
    y0 = float(fit.y[0])
    return x0 * y0


def test_me_pair_mean_matches_q() -> None:
    """ME: documented E[t] = q matches the implemented Poisson mean."""
    strengths = _constant_strength(398.0)  # q = 398/(N-1) = 2
    fit = cast(
        "StrengthFit",
        fit_model(
            family=ModelFamily.ME,
            constraint=Constraint.STRENGTH,
            strength_out=strengths,
            strength_in=strengths,
            self_loops=False,
        ),
    )
    assert fit.converged, fit.status
    q = _pair_q(fit)
    assert math.isclose(q, 2.0, rel_tol=1e-9)
    empirical = _empirical_pair_mean(fit, ModelFamily.ME)
    assert math.isclose(empirical, q, rel_tol=0.03), (
        f"empirical ME pair mean {empirical} vs documented q {q}"
    )


def test_b_pair_mean_matches_m_q_over_1_plus_q() -> None:
    """B: documented E[t] = M q / (1 + q) matches the binomial mean."""
    layers = 10
    strengths = _constant_strength(398.0)  # q = 0.25 with M=10
    fit = cast(
        "StrengthFit",
        fit_model(
            family=ModelFamily.B,
            constraint=Constraint.STRENGTH,
            strength_out=strengths,
            strength_in=strengths,
            self_loops=False,
            layers=layers,
        ),
    )
    assert fit.converged, fit.status
    q = _pair_q(fit)
    documented = layers * q / (1.0 + q)
    empirical = _empirical_pair_mean(fit, ModelFamily.B)
    assert math.isclose(empirical, documented, rel_tol=0.03), (
        f"empirical B pair mean {empirical} vs documented {documented} (q={q})"
    )


def test_w_pair_mean_matches_m_q_over_1_minus_q() -> None:
    """W: documented E[t] = M q / (1 - q) matches the negative-binomial mean."""
    layers = 1
    strengths = _constant_strength(398.0)  # q = 398/(398+N-1) = 2/3
    fit = cast(
        "StrengthFit",
        fit_model(
            family=ModelFamily.W,
            constraint=Constraint.STRENGTH,
            strength_out=strengths,
            strength_in=strengths,
            self_loops=False,
            layers=layers,
        ),
    )
    assert fit.converged, fit.status
    q = _pair_q(fit)
    documented = layers * q / (1.0 - q)
    empirical = _empirical_pair_mean(fit, ModelFamily.W)
    assert math.isclose(empirical, documented, rel_tol=0.05), (
        f"empirical W pair mean {empirical} vs documented {documented} (q={q})"
    )


def test_b_m1_invariant_strength_equals_degree() -> None:
    """B(M=1) is Bernoulli: t in {0,1}, so sampled s equals k per node."""
    strengths = _constant_strength(50.0)
    fit = cast(
        "StrengthFit",
        fit_model(
            family=ModelFamily.B,
            constraint=Constraint.STRENGTH,
            strength_out=strengths,
            strength_in=strengths,
            self_loops=False,
            layers=1,
        ),
    )
    assert fit.converged, fit.status
    sample = sample_model(
        ensemble=Ensemble.GRAND_CANONICAL,
        family=ModelFamily.B,
        constraint=Constraint.STRENGTH,
        fit=fit,
        seed=1,
    )
    assert int(np.max(sample.occ_num)) == 1, "B(M=1) occupations must be 0/1"
    strengths_out = np.bincount(sample.source, weights=sample.occ_num)
    strengths_in = np.bincount(sample.target, weights=sample.occ_num)
    degrees_out = np.bincount(sample.source, minlength=N)
    degrees_in = np.bincount(sample.target, minlength=N)
    np.testing.assert_allclose(strengths_out, degrees_out, atol=1e-9)
    np.testing.assert_allclose(strengths_in, degrees_in, atol=1e-9)


def _gf_and_deriv(q: float) -> tuple[float, float]:
    """ME zero-inflated partition factors G(q)=e^q-1 and G'(q)=e^q."""
    return math.expm1(q), math.exp(q)


def test_zero_inflated_strength_edges_formulas() -> None:
    """Zero-inflated ME: documented G_F formulas match filter expectations."""
    rng = np.random.default_rng(5)
    n = 30
    strengths = rng.integers(1, 8, size=n).astype(np.float64)
    target_edges = int(strengths.sum()) // 2  # feasible: 0 < E <= total strength
    fit = cast(
        "StrengthEdgesFit",
        fit_model(
            family=ModelFamily.ME,
            constraint=Constraint.STRENGTH_EDGES,
            strength_out=strengths,
            strength_in=strengths,
            target_edges=target_edges,
            self_loops=False,
        ),
    )
    assert fit.converged, fit.status
    lam = float(fit.lam)

    # Filter the observed table to obtain per-edge null expectations.
    observed = filter_model(
        _dense_witness_edges(n, strengths),
        family=ModelFamily.ME,
        constraint=Constraint.STRENGTH_EDGES,
        fit=fit,
        self_loops=False,
    )
    per_pair: dict[tuple[int, int], tuple[float, float]] = {}
    for bucket in (observed.upper, observed.lower, observed.compatible):
        for k in range(bucket.edges.num_edges):
            key = (int(bucket.edges.source[k]), int(bucket.edges.target[k]))
            per_pair[key] = (float(bucket.expected[k]), float(bucket.occupation[k]))

    checked = 0
    max_expected_err = 0.0
    max_occupation_err = 0.0
    for (i, j), (expected, occupation) in per_pair.items():
        q = float(fit.x[i]) * float(fit.y[j])
        g, gprime = _gf_and_deriv(q)
        documented_p_positive = (lam * g) / (1.0 + lam * g)
        documented_mean = (lam * q * gprime) / (1.0 + lam * g)
        max_occupation_err = max(
            max_occupation_err, abs(documented_p_positive - occupation)
        )
        max_expected_err = max(max_expected_err, abs(documented_mean - expected))
        checked += 1
    assert checked > 0
    assert max_occupation_err < 1e-9, (
        f"documented P(t>0) deviates from filter occupation by {max_occupation_err}"
    )
    assert max_expected_err < 1e-9, (
        f"documented E[t] deviates from filter expectation by {max_expected_err}"
    )

    # Conditional mean E[t | t>0] = q G'/G is independent of the support fugacity.
    (i0, j0), (expected0, occupation0) = next(iter(per_pair.items()))
    q0 = float(fit.x[i0]) * float(fit.y[j0])
    conditional_documented = q0 * math.exp(q0) / math.expm1(q0)
    conditional_filter = expected0 / occupation0
    assert math.isclose(conditional_documented, conditional_filter, rel_tol=1e-9)


def _dense_witness_edges(n: int, strengths: np.ndarray) -> EdgeTable:
    """Small occupied-pair table for the zero-inflated cross-check."""
    rng = np.random.default_rng(11)
    sources: list[int] = []
    targets: list[int] = []
    occs: list[int] = []
    for i in range(n):
        out_alloc = strengths[i]
        while out_alloc > 0:
            j = int(rng.integers(0, n))
            if i == j:
                continue
            take = min(out_alloc, int(rng.integers(1, out_alloc + 1)))
            sources.append(i)
            targets.append(j)
            occs.append(take)
            out_alloc -= take
    return EdgeTable(
        source=np.asarray(sources, dtype=np.uint64),
        target=np.asarray(targets, dtype=np.uint64),
        occ_num=np.asarray(occs, dtype=np.uint64),
    )


@given(st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=20, deadline=None)
def test_me_mean_formula_holds_for_varied_q(q: float) -> None:
    """ME pair mean equals q for arbitrary fitted fugacities."""
    strengths = _constant_strength(q * (N - 1))
    fit = cast(
        "StrengthFit",
        fit_model(
            family=ModelFamily.ME,
            constraint=Constraint.STRENGTH,
            strength_out=strengths,
            strength_in=strengths,
            self_loops=False,
        ),
    )
    assert fit.converged, fit.status
    fitted_q = _pair_q(fit)
    assert math.isclose(fitted_q, q, rel_tol=1e-6)
