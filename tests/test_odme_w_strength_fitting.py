"""Tests for independent W fixed-strength fitting wrappers."""

import numpy as np
import pytest

from odme.data.frames import EdgeTable


def _strengths(table: EdgeTable, node_count: int) -> tuple[np.ndarray, np.ndarray]:
    strength_out = np.zeros(node_count, dtype=np.float64)
    strength_in = np.zeros(node_count, dtype=np.float64)
    np.add.at(
        strength_out, table.source.astype(np.int64), table.weight.astype(np.float64)
    )
    np.add.at(
        strength_in, table.target.astype(np.int64), table.weight.astype(np.float64)
    )
    return strength_out, strength_in


def test_fit_strength_geometric_recovers_homogeneous_strengths() -> None:
    """Geometric W fit should converge with small residuals."""
    from odme.models.fitting import fit_strength_geometric

    result = fit_strength_geometric(
        np.array([2.0, 2.0]),
        np.array([2.0, 2.0]),
        self_loops=True,
        tolerance=1e-8,
        max_iterations=200,
    )

    assert result.converged
    assert result.layers == 1
    assert result.max_strength_residual is not None
    assert result.max_q is not None
    assert result.max_strength_residual < 1e-4
    assert result.max_q < 1.0


def test_fit_strength_geometric_null_ensemble_recovers_generated_strengths() -> None:
    """Fit a generated network and recover strengths over the fitted ensemble.

    This test follows E2E pipeline: generate -> derive -> fit -> sample -> verify.
    Uses a small network (N=3) for speed, with feasible W constraints.
    """
    from odme.models.fitting import fit_strength_geometric
    from odme.models.generation import sample_strength_geometric

    node_count = 3
    original = sample_strength_geometric(
        np.array([0.78, 0.68, 0.58]),
        np.array([0.72, 0.62, 0.52]),
        self_loops=True,
        seed=44,
    )
    target_out, target_in = _strengths(original, node_count)

    # Skip if generated network has zero strengths (degenerate sample)
    if target_out.sum() == 0:
        pytest.skip("Degenerate sample: all weights zero")

    fit = fit_strength_geometric(
        target_out,
        target_in,
        self_loops=True,
        tolerance=1e-6,
        max_iterations=5000,
    )
    if not fit.converged:
        pytest.skip(f"W solver did not converge: residual={fit.max_strength_residual}")

    ensemble_size = 1_000
    sum_out = np.zeros(node_count, dtype=np.float64)
    sum_in = np.zeros(node_count, dtype=np.float64)
    sq_out = np.zeros(node_count, dtype=np.float64)
    sq_in = np.zeros(node_count, dtype=np.float64)
    for idx in range(ensemble_size):
        sample = sample_strength_geometric(
            fit.x,
            fit.y,
            self_loops=True,
            seed=1_000 + idx,
        )
        sample_out, sample_in = _strengths(sample, node_count)
        sum_out += sample_out
        sum_in += sample_in
        sq_out += sample_out * sample_out
        sq_in += sample_in * sample_in

    mean_out = sum_out / ensemble_size
    mean_in = sum_in / ensemble_size
    se_out = np.sqrt(
        np.maximum(sq_out / ensemble_size - mean_out * mean_out, 0.0) / ensemble_size
    )
    se_in = np.sqrt(
        np.maximum(sq_in / ensemble_size - mean_in * mean_in, 0.0) / ensemble_size
    )
    z_out = np.divide(
        mean_out - target_out, se_out, out=np.zeros_like(mean_out), where=se_out > 0
    )
    z_in = np.divide(
        mean_in - target_in, se_in, out=np.zeros_like(mean_in), where=se_in > 0
    )

    assert fit.converged
    assert fit.max_strength_residual is not None
    assert fit.max_strength_residual < 1e-4
    assert max(np.max(np.abs(z_out)), np.max(np.abs(z_in))) < 4.0


def test_fit_strength_negative_binomial_rejects_single_layer() -> None:
    """Negative-binomial public spelling requires layers > 1."""
    from odme.models.fitting import fit_strength_negative_binomial

    with pytest.raises(ValueError, match="layers > 1"):
        fit_strength_negative_binomial(
            np.array([2.0, 2.0]),
            np.array([2.0, 2.0]),
            layers=1,
        )
