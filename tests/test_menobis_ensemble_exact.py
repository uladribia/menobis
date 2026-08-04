"""P0.9 exact ensemble-equivalence tests.

Separate suites for:
1. exact conditioning identities (canonical total, stub strength);
2. direct-sampler correctness: ME stub matching matches the theoretical
   degeneracy-weighted measure on small state spaces;
3. asymptotic observable convergence is covered by the smoke test in
   test_menobis_ensemble_equivalence.py (not duplicated here).
"""

import math

import numpy as np
import pytest

from menobis.analysis import directed_strengths
from menobis.models.fitting import _fit_strength_poisson as fit_strength_poisson
from menobis.models.generation import (
    _sample_strength_multinomial as sample_strength_multinomial,
)
from menobis.models.generation import (
    _sample_strength_stub_matching as sample_strength_stub_matching,
)

# ---------------------------------------------------------------------------
# Exact small-state enumeration: P(T | s) proportional to T! / prod t_ij!
# ---------------------------------------------------------------------------


def _enumerate_marginal_matrices(
    n: int, s_out: np.ndarray, s_in: np.ndarray
) -> list[np.ndarray]:
    """Enumerate all n×n non-negative integer matrices with the given margins."""
    total = int(s_out.sum())
    assert int(s_in.sum()) == total

    matrices: list[np.ndarray] = []

    def cells(
        i: int, j: int, mat: np.ndarray, rem_out: np.ndarray, rem_in: np.ndarray
    ) -> None:
        if i == n:
            if np.all(rem_out == 0) and np.all(rem_in == 0):
                matrices.append(mat.copy())
            return
        if j == n:
            if rem_out[i] == 0:
                cells(i + 1, 0, mat, rem_out, rem_in)
            return
        max_val = min(rem_out[i], rem_in[j])
        for v in range(max_val + 1):
            mat[i, j] = v
            ro = rem_out.copy()
            ri = rem_in.copy()
            ro[i] -= v
            ri[j] -= v
            cells(i, j + 1, mat, ro, ri)
        mat[i, j] = 0

    cells(0, 0, np.zeros((n, n), dtype=np.uint64), s_out.copy(), s_in.copy())
    return matrices


def _me_state_weight(matrix: np.ndarray) -> float:
    """ME degeneracy weight d(T) = T! / prod t_ij! (T! constant for fixed T)."""
    total = int(matrix.sum())
    return math.factorial(total) / math.prod(
        math.factorial(int(v)) for v in matrix.flat
    )


def _stub_matching_frequencies(
    s_out: np.ndarray, s_in: np.ndarray, samples: int = 4000
) -> dict[tuple[tuple[int, ...], ...], int]:
    """Empirical stub-matching frequency per occupation matrix."""
    counts: dict[tuple[tuple[int, ...], ...], int] = {}
    for seed in range(samples):
        sample = sample_strength_stub_matching(s_out, s_in, seed=seed)
        mat = np.zeros((len(s_out), len(s_out)), dtype=np.uint64)
        for s, t, o in zip(sample.source, sample.target, sample.occ_num, strict=True):
            mat[int(s), int(t)] += int(o)
        counts[tuple(tuple(row) for row in mat)] = (
            counts.get(tuple(tuple(row) for row in mat), 0) + 1
        )
    return counts


@pytest.mark.parametrize(
    "s_out,s_in",
    [
        (np.array([1, 1]), np.array([1, 1])),  # T=2, two matrices
        (np.array([2, 1]), np.array([1, 2])),  # T=3
        (np.array([1, 2, 1]), np.array([2, 1, 1])),  # T=4
    ],
)
def test_stub_matching_matches_me_degeneracy_measure(
    s_out: np.ndarray, s_in: np.ndarray
) -> None:
    """Stub matching frequencies must match P(T|s) ∝ T!/∏t_ij! on small states.

    Uses self-loops (the regime where stub matching is uniform over labelled
    matchings, thesis §2.1).
    """
    matrices = _enumerate_marginal_matrices(len(s_out), s_out, s_in)
    assert len(matrices) >= 2, "state space too small for a meaningful test"

    weights = np.array([_me_state_weight(m) for m in matrices], dtype=float)
    p_theory = weights / weights.sum()

    counts = _stub_matching_frequencies(s_out, s_in, samples=4000)
    n_samples = sum(counts.values())
    frequencies = np.array(
        [counts.get(tuple(tuple(row) for row in m), 0) / n_samples for m in matrices],
        dtype=float,
    )

    # Proportion test: with 4000 samples, expected counts are >= 500 for the
    # uniform case; tolerance reflects binomial sampling noise (3 sigma).
    expected_counts = p_theory * n_samples
    observed_counts = frequencies * n_samples
    tol = 3.0 * np.sqrt(np.maximum(expected_counts, 1.0))
    for i, m in enumerate(matrices):
        assert abs(observed_counts[i] - expected_counts[i]) <= tol[i], (
            f"matrix {m.tolist()}: observed {observed_counts[i]:.0f} vs "
            f"expected {expected_counts[i]:.0f} (p={p_theory[i]:.3f})"
        )


def test_small_state_space_has_expected_size() -> None:
    """Sanity: T=2 N=2 self-loops has exactly two marginal matrices."""
    matrices = _enumerate_marginal_matrices(2, np.array([1, 1]), np.array([1, 1]))
    assert len(matrices) == 2
    # Both have the same degeneracy: T!/(1!1!) = 2.
    w = [_me_state_weight(m) for m in matrices]
    np.testing.assert_allclose(w, [2.0, 2.0])


# ---------------------------------------------------------------------------
# Exact conditioning identities
# ---------------------------------------------------------------------------


def test_canonical_preserves_exact_total_events() -> None:
    """Canonical multinomial conditioning on T is exact."""
    s_out = np.array([3, 4])
    s_in = np.array([4, 3])
    fit = fit_strength_poisson(s_out, s_in)
    for total in (20, 50):
        for seed in range(10):
            sample = sample_strength_multinomial(
                fit.x, fit.y, total_events=total, seed=seed
            )
            assert sample.total_events == total


def test_stub_matching_preserves_exact_strengths_exact() -> None:
    """Stub matching conditioning on the strength sequence is exact."""
    for s_out, s_in in [
        (np.array([1, 1]), np.array([1, 1])),
        (np.array([2, 1, 1]), np.array([1, 2, 1])),
        (np.array([3, 2]), np.array([2, 3])),
    ]:
        for seed in range(20):
            sample = sample_strength_stub_matching(s_out, s_in, seed=seed)
            actual = directed_strengths(sample)
            np.testing.assert_array_equal(actual.out, s_out)
            np.testing.assert_array_equal(actual.incoming, s_in)
            assert sample.total_events == int(s_out.sum())


# ---------------------------------------------------------------------------
# Direct sampler correctness
# ---------------------------------------------------------------------------


def test_stub_matching_occupations_are_positive_and_bounded() -> None:
    """Stub-matching outputs are valid occupation matrices (positive, finite)."""
    s_out = np.array([2, 3, 4])
    s_in = np.array([4, 3, 2])
    for seed in range(50):
        sample = sample_strength_stub_matching(s_out, s_in, seed=seed)
        assert len(sample) <= len(s_out) * len(s_out)
        assert np.all(sample.occ_num >= 1)
        assert sample.occ_num.dtype == np.uint64


def test_canonical_occupations_are_non_negative_integers() -> None:
    s_out = np.array([3, 4])
    s_in = np.array([4, 3])
    fit = fit_strength_poisson(s_out, s_in)
    sample = sample_strength_multinomial(fit.x, fit.y, total_events=50, seed=0)
    assert np.all(sample.occ_num >= 1)
    assert sample.occ_num.dtype == np.uint64
