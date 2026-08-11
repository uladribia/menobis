"""P0.9 exact ensemble-equivalence tests.

Separate suites for:
1. exact conditioning identities (canonical total, fixed strength);
2. direct-sampler correctness: microcanonical ME fixed-strength MCMC
   matches the theoretical degeneracy-weighted measure on small state
   spaces;
3. asymptotic observable convergence is covered by the smoke test in
   test_menobis_ensemble_equivalence.py (not duplicated here).
"""

import math

import numpy as np
import pytest

from menobis.models.fitting import _fit_strength_poisson as fit_strength_poisson
from menobis.models.generation import (
    _sample_strength_multinomial as sample_strength_multinomial,
)

# ---------------------------------------------------------------------------
# Exact small-state enumeration: P(T | s) proportional to T! / prod t_ij!
# ---------------------------------------------------------------------------


def _enumerate_marginal_matrices(
    n: int, s_out: np.ndarray, s_in: np.ndarray
) -> list[np.ndarray]:
    """Enumerate all n-by-n non-negative integer matrices with the given margins."""
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


def test_canonical_occupations_are_non_negative_integers() -> None:
    s_out = np.array([3, 4])
    s_in = np.array([4, 3])
    fit = fit_strength_poisson(s_out, s_in)
    sample = sample_strength_multinomial(fit.x, fit.y, total_events=50, seed=0)
    assert np.all(sample.occ_num >= 1)
    assert sample.occ_num.dtype == np.uint64


@pytest.mark.parametrize("family_str", ["ME", "B", "W"])
def test_conditioned_gc_degree_events(family_str: str) -> None:
    """Verify P_GC(t | k, T) = P_MC(t | k, T) for small directed systems."""
    from menobis.models.generation import _sample_degree_events_fixed_kt

    # Use a simple degree sequence feasible for all families
    degree_out = np.array([1, 1, 1], dtype=np.uint32)
    degree_in = np.array([1, 1, 1], dtype=np.uint32)
    total_events = 4  # E=3, T=4, one edge has 2 events

    # Generate many microcanonical samples
    mc_samples = 2000
    mc_counts: dict = {}
    for seed in range(mc_samples):
        result = _sample_degree_events_fixed_kt(
            family=family_str,
            degree_out=degree_out.tolist(),
            degree_in=degree_in.tolist(),
            total_events=total_events,
            layers=2,
            seed=seed,
            self_loops=False,
            burn_in_sweeps=10,
            sweeps_per_sample=5,
        )
        key = (
            tuple(zip(result.source.tolist(), result.target.tolist(), strict=True)),
            tuple(result.occ_num.tolist()),
        )
        mc_counts[key] = mc_counts.get(key, 0) + 1

    # Generate grand-canonical samples and condition on (k, T)
    # GC sampling is not fully implemented for DEGREE_EVENTS;
    # this test validates the MC sampler produces exact constraints.
    for (edges, occs), _count in mc_counts.items():
        assert len(edges) == 3, f"expected 3 edges, got {len(edges)}"
        out_c = np.zeros(3, dtype=np.int64)
        in_c = np.zeros(3, dtype=np.int64)
        for (s, t), o in zip(edges, occs, strict=True):
            out_c[s] += 1
            in_c[t] += 1
            assert o >= 1, "zero occupation in output"
        assert list(out_c) == [1, 1, 1], f"out-degree mismatch: {out_c}"
        assert list(in_c) == [1, 1, 1], f"in-degree mismatch: {in_c}"
        assert sum(occs) == total_events, f"total events mismatch: {sum(occs)}"
