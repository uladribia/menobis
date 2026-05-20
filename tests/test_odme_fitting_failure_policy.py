"""General fitting failure-policy tests."""

from collections.abc import Callable

import numpy as np
import pytest

from odme.models import (
    fit_degree_events_geometric,
    fit_degree_events_negative_binomial,
    fit_strength_cost_geometric,
    fit_strength_cost_poisson,
    fit_strength_edges_geometric,
    fit_strength_edges_poisson,
    fit_strength_geometric,
)


def test_fits_reject_nonfinite_sequences_at_boundary() -> None:
    """All fitting wrappers should reject non-finite constraints early."""
    with pytest.raises(ValueError, match="finite sequences"):
        fit_strength_geometric(np.array([1.0, np.nan]), np.array([1.0, np.nan]))


def test_strength_edges_capacity_failure_is_shared_across_families() -> None:
    """Strength-edges fits share explicit capacity validation."""
    s_out = np.array([10.0, 10.0])
    s_in = np.array([10.0, 10.0])

    with pytest.raises(ValueError, match="candidate-pair capacity"):
        fit_strength_edges_poisson(s_out, s_in, target_edges=4.0, self_loops=True)

    with pytest.raises(ValueError, match="candidate-pair capacity"):
        fit_strength_edges_geometric(s_out, s_in, target_edges=4.0, self_loops=True)


@pytest.mark.parametrize(
    "fit_func",
    [fit_strength_cost_poisson, fit_strength_cost_geometric],
)
def test_strength_cost_rejects_bad_cost_entries_across_families(
    fit_func: Callable[..., object],
) -> None:
    """Strength-cost fits share sparse-cost validation."""
    s_out = np.array([1.0, 1.0])
    s_in = np.array([1.0, 1.0])

    with pytest.raises(ValueError, match="same length"):
        fit_func(
            s_out,
            s_in,
            np.array([0, 1], dtype=np.uint64),
            np.array([0], dtype=np.uint64),
            np.array([1.0, 1.0]),
            target_cost=1.0,
        )

    with pytest.raises(ValueError, match="finite and non-negative"):
        fit_func(
            s_out,
            s_in,
            np.array([0], dtype=np.uint64),
            np.array([0], dtype=np.uint64),
            np.array([np.nan]),
            target_cost=1.0,
        )


def test_degree_events_rejects_capacity_and_layers_consistently() -> None:
    """Degree-events fits reject explicit infeasibility before native code."""
    with pytest.raises(ValueError, match="boundary or infeasible"):
        fit_degree_events_geometric(
            np.array([2.0, 1.0]),
            np.array([1.5, 1.5]),
            total_events=3,
            self_loops=False,
        )

    with pytest.raises(ValueError, match="layers > 1"):
        fit_degree_events_negative_binomial(
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            total_events=2,
            layers=1,
        )
