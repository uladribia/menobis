"""Property tests for strength-cost ME model: fixed strength + fixed total cost."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from odme.models import fit_strength_cost_poisson, sample_strength_cost_poisson


def _build_cost_entries(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sources: list[int] = []
    targets: list[int] = []
    costs: list[float] = []
    for i in range(n):
        for j in range(n):
            sources.append(i)
            targets.append(j)
            costs.append(float(abs(i - j) + 1))
    return np.array(sources), np.array(targets), np.array(costs)


def _expectations(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float,
    cost_src: np.ndarray,
    cost_tgt: np.ndarray,
    cost_val: np.ndarray,
    n: int,
) -> tuple[np.ndarray, float]:
    expected = np.zeros((n, n))
    total_cost = 0.0
    for idx in range(len(cost_src)):
        i, j = int(cost_src[idx]), int(cost_tgt[idx])
        d = cost_val[idx]
        e = x[i] * y[j] * np.exp(-gamma * d)
        expected[i, j] = e
        total_cost += e * d
    return expected, total_cost


@given(
    gamma_val=st.floats(
        min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False
    ),
    scale=st.floats(
        min_value=2.0, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(deadline=None, max_examples=10)
def test_strength_cost_fit_recovers_strengths_and_cost(
    gamma_val: float, scale: float
) -> None:
    n = 4
    true_x = np.array([scale, scale * 0.6, scale * 0.3, scale * 0.1])
    true_y = np.array([scale * 0.8, scale * 0.5, scale * 0.4, scale * 0.3])
    cost_src, cost_tgt, cost_val = _build_cost_entries(n)

    expected, target_cost = _expectations(
        true_x, true_y, gamma_val, cost_src, cost_tgt, cost_val, n
    )
    s_out = expected.sum(axis=1)
    s_in = expected.sum(axis=0)

    fit = fit_strength_cost_poisson(
        s_out, s_in, cost_src, cost_tgt, cost_val, target_cost
    )
    fitted_expected, fitted_cost = _expectations(
        fit.x, fit.y, fit.gamma, cost_src, cost_tgt, cost_val, n
    )

    np.testing.assert_allclose(fitted_expected.sum(axis=1), s_out, atol=0.5, rtol=0.05)
    np.testing.assert_allclose(fitted_expected.sum(axis=0), s_in, atol=0.5, rtol=0.05)
    np.testing.assert_allclose(fitted_cost, target_cost, atol=0.5, rtol=0.05)

    # Verify per-pair equation: E[t_ij] = x_i y_j exp(-gamma d_ij)
    for idx in range(len(cost_src)):
        i, j = int(cost_src[idx]), int(cost_tgt[idx])
        d = cost_val[idx]
        analytical = fit.x[i] * fit.y[j] * np.exp(-fit.gamma * d)
        np.testing.assert_allclose(
            fitted_expected[i, j],
            analytical,
            rtol=1e-10,
            err_msg=f"equation mismatch at ({i},{j})",
        )


def test_strength_cost_sample_is_reproducible() -> None:
    n = 4
    s_out = np.array([20.0, 15.0, 10.0, 5.0])
    s_in = np.array([12.0, 13.0, 12.0, 13.0])
    cost_src, cost_tgt, cost_val = _build_cost_entries(n)
    target_cost = 80.0

    fit = fit_strength_cost_poisson(
        s_out, s_in, cost_src, cost_tgt, cost_val, target_cost
    )
    first = sample_strength_cost_poisson(fit, cost_src, cost_tgt, cost_val, seed=42)
    second = sample_strength_cost_poisson(fit, cost_src, cost_tgt, cost_val, seed=42)

    np.testing.assert_array_equal(first.source, second.source)
    np.testing.assert_array_equal(first.target, second.target)
    np.testing.assert_array_equal(first.weight, second.weight)
    assert np.all(first.weight >= 1)


def test_strength_cost_ensemble_recovers_strengths() -> None:
    n = 4
    s_out = np.array([30.0, 20.0, 15.0, 10.0])
    s_in = np.array([18.0, 20.0, 18.0, 19.0])
    cost_src, cost_tgt, cost_val = _build_cost_entries(n)
    target_cost = 120.0

    fit = fit_strength_cost_poisson(
        s_out, s_in, cost_src, cost_tgt, cost_val, target_cost
    )

    repetitions = 300
    sampled_out: list[np.ndarray] = []
    for seed in range(repetitions):
        sample = sample_strength_cost_poisson(
            fit, cost_src, cost_tgt, cost_val, seed=seed
        )
        from odme.analysis import directed_strengths

        sampled_out.append(directed_strengths(sample).out.astype(float))

    stacked = np.vstack(sampled_out)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0, ddof=1)
    se = std / np.sqrt(repetitions)
    tol = 6.0 * se + 1.0
    assert np.all(np.abs(mean - s_out) <= tol + 0.08 * np.abs(s_out))
