"""Harder Pareto-sequence workflow tests for W fitting and sampling."""

from collections.abc import Callable

import numpy as np
import pytest

from menobis.data.frames import EdgeTable
from menobis.models import (
    StrengthCostFit,
    StrengthEdgesFit,
    StrengthFit,
    fit_degree_events_geometric,
    fit_degree_events_negative_binomial,
    fit_strength_cost_geometric,
    fit_strength_cost_negative_binomial,
    fit_strength_edges_geometric,
    fit_strength_edges_negative_binomial,
    fit_strength_geometric,
    fit_strength_negative_binomial,
    sample_degree_events_geometric,
    sample_degree_events_negative_binomial,
    sample_strength_cost_geometric,
    sample_strength_cost_negative_binomial,
    sample_strength_edges_geometric,
    sample_strength_edges_negative_binomial,
    sample_strength_geometric,
    sample_strength_negative_binomial,
)

NODE_COUNT = 10
TOTAL_STRENGTH = 30.0
ENSEMBLE_SIZE = 300
SELF_LOOPS = True
LAYERS = 3


def _pareto_strengths() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20240520)
    raw = rng.pareto(2.3, NODE_COUNT) + 1.0
    raw = np.clip(raw, 0.0, np.quantile(raw, 0.9))
    out = raw / raw.sum() * TOTAL_STRENGTH
    inc = np.roll(raw[::-1], 3)
    inc = inc / inc.sum() * TOTAL_STRENGTH
    return out.astype(np.float64), inc.astype(np.float64)


def _complete_costs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12345)
    sources, targets = np.meshgrid(
        np.arange(NODE_COUNT), np.arange(NODE_COUNT), indexing="ij"
    )
    values = rng.lognormal(mean=0.0, sigma=0.35, size=(NODE_COUNT, NODE_COUNT))
    values = values / values.mean()
    return (
        sources.ravel().astype(np.uint64),
        targets.ravel().astype(np.uint64),
        values.ravel().astype(np.float64),
    )


def _strengths(table: EdgeTable) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros(NODE_COUNT, dtype=np.float64)
    inc = np.zeros(NODE_COUNT, dtype=np.float64)
    np.add.at(out, table.source.astype(np.int64), table.weight.astype(np.float64))
    np.add.at(inc, table.target.astype(np.int64), table.weight.astype(np.float64))
    return out, inc


def _degrees(table: EdgeTable) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros(NODE_COUNT, dtype=np.float64)
    inc = np.zeros(NODE_COUNT, dtype=np.float64)
    np.add.at(out, table.source.astype(np.int64), 1.0)
    np.add.at(inc, table.target.astype(np.int64), 1.0)
    return out, inc


def _total_cost(table: EdgeTable, cost_values: np.ndarray) -> float:
    costs = cost_values.reshape((NODE_COUNT, NODE_COUNT))
    return float(
        sum(
            costs[int(source), int(target)] * float(weight)
            for source, target, weight in zip(
                table.source, table.target, table.weight, strict=True
            )
        )
    )


def _edge_count(table: EdgeTable) -> float:
    return float(table.num_edges)


def _assert_ensemble_node_stat(
    sampler: Callable[[int], EdgeTable],
    expected_out: np.ndarray,
    expected_in: np.ndarray,
    statistic: Callable[[EdgeTable], tuple[np.ndarray, np.ndarray]],
    *,
    z_limit: float = 4.5,
) -> None:
    sum_out = np.zeros(NODE_COUNT, dtype=np.float64)
    sum_in = np.zeros(NODE_COUNT, dtype=np.float64)
    sq_out = np.zeros(NODE_COUNT, dtype=np.float64)
    sq_in = np.zeros(NODE_COUNT, dtype=np.float64)
    for index in range(ENSEMBLE_SIZE):
        out, inc = statistic(sampler(10_000 + index))
        sum_out += out
        sum_in += inc
        sq_out += out * out
        sq_in += inc * inc
    mean_out = sum_out / ENSEMBLE_SIZE
    mean_in = sum_in / ENSEMBLE_SIZE
    se_out = np.sqrt(
        np.maximum(sq_out / ENSEMBLE_SIZE - mean_out * mean_out, 0.0) / ENSEMBLE_SIZE
    )
    se_in = np.sqrt(
        np.maximum(sq_in / ENSEMBLE_SIZE - mean_in * mean_in, 0.0) / ENSEMBLE_SIZE
    )
    z_out = np.divide(
        mean_out - expected_out, se_out, out=np.zeros_like(mean_out), where=se_out > 0
    )
    z_in = np.divide(
        mean_in - expected_in, se_in, out=np.zeros_like(mean_in), where=se_in > 0
    )
    assert np.max(np.abs(z_out)) < z_limit
    assert np.max(np.abs(z_in)) < z_limit


def _assert_ensemble_strengths(
    sampler: Callable[[int], EdgeTable],
    expected_out: np.ndarray,
    expected_in: np.ndarray,
    *,
    z_limit: float = 4.5,
) -> None:
    _assert_ensemble_node_stat(
        sampler, expected_out, expected_in, _strengths, z_limit=z_limit
    )


def _expected_cost_from_strength_fit(
    fit: StrengthFit,
    cost_values: np.ndarray,
    layers: int,
) -> float:
    costs = cost_values.reshape((NODE_COUNT, NODE_COUNT))
    total = 0.0
    for i in range(NODE_COUNT):
        for j in range(NODE_COUNT):
            q = fit.x[i] * fit.y[j]
            total += costs[i, j] * layers * q / (1.0 - q)
    return float(total)


@pytest.mark.parametrize("family", ["geometric", "negative_binomial"])
def test_pareto_strength_fit_then_simulate_recovers_strengths(family: str) -> None:
    """Independent W fits handle a Pareto strength sequence at N=10."""
    s_out, s_in = _pareto_strengths()
    if family == "geometric":
        fit = fit_strength_geometric(s_out, s_in, self_loops=SELF_LOOPS)
    else:
        fit = fit_strength_negative_binomial(
            s_out, s_in, layers=LAYERS, self_loops=SELF_LOOPS
        )

    def sampler(seed: int) -> EdgeTable:
        if family == "geometric":
            return sample_strength_geometric(fit.x, fit.y, seed=seed)
        return sample_strength_negative_binomial(fit.x, fit.y, layers=LAYERS, seed=seed)

    assert fit.converged
    assert fit.max_q is not None and fit.max_q < 1.0
    _assert_ensemble_strengths(sampler, s_out, s_in)


@pytest.mark.parametrize("family", ["geometric", "negative_binomial"])
def test_pareto_strength_cost_fit_then_simulate_recovers_constraints(
    family: str,
) -> None:
    """W strength-cost fits align Pareto strengths, costs, and simulations."""
    s_out, s_in = _pareto_strengths()
    c_src, c_tgt, c_val = _complete_costs()
    base = fit_strength_geometric(s_out, s_in, self_loops=SELF_LOOPS)
    target_cost = _expected_cost_from_strength_fit(base, c_val, 1)

    if family == "geometric":
        fit = fit_strength_cost_geometric(
            s_out, s_in, c_src, c_tgt, c_val, target_cost, self_loops=SELF_LOOPS
        )
    else:
        nb_base = fit_strength_negative_binomial(
            s_out, s_in, layers=LAYERS, self_loops=SELF_LOOPS
        )
        target_cost = _expected_cost_from_strength_fit(nb_base, c_val, LAYERS)
        fit = fit_strength_cost_negative_binomial(
            s_out,
            s_in,
            c_src,
            c_tgt,
            c_val,
            target_cost,
            layers=LAYERS,
            self_loops=SELF_LOOPS,
        )

    def sampler(seed: int) -> EdgeTable:
        if family == "geometric":
            return sample_strength_cost_geometric(fit, c_src, c_tgt, c_val, seed=seed)
        return sample_strength_cost_negative_binomial(
            fit, c_src, c_tgt, c_val, layers=LAYERS, seed=seed
        )

    assert isinstance(fit, StrengthCostFit)
    assert fit.converged
    _assert_ensemble_strengths(sampler, s_out, s_in, z_limit=5.0)
    costs = np.array(
        [_total_cost(sampler(30_000 + index), c_val) for index in range(ENSEMBLE_SIZE)]
    )
    assert (
        abs(costs.mean() - target_cost) / costs.std(ddof=1) * np.sqrt(ENSEMBLE_SIZE)
        < 5.0
    )


@pytest.mark.parametrize("family", ["geometric", "negative_binomial"])
def test_pareto_degree_events_fit_then_simulate_recovers_constraints(
    family: str,
) -> None:
    """W degree-events fits align non-saturating Pareto-like degrees and events."""
    s_out, s_in = _pareto_strengths()
    degree_total = 14.0
    k_out = s_out / s_out.sum() * degree_total
    k_in = s_in / s_in.sum() * degree_total
    assert np.max(k_out) < NODE_COUNT / 2
    assert np.max(k_in) < NODE_COUNT / 2

    if family == "geometric":
        fit = fit_degree_events_geometric(
            k_out, k_in, int(TOTAL_STRENGTH), self_loops=SELF_LOOPS
        )
    else:
        fit = fit_degree_events_negative_binomial(
            k_out,
            k_in,
            int(TOTAL_STRENGTH),
            layers=LAYERS,
            self_loops=SELF_LOOPS,
        )

    def sampler(seed: int) -> EdgeTable:
        if family == "geometric":
            return sample_degree_events_geometric(fit, seed=seed)
        return sample_degree_events_negative_binomial(fit, seed=seed)

    assert fit.converged
    _assert_ensemble_node_stat(sampler, k_out, k_in, _degrees, z_limit=5.0)
    totals = np.array(
        [sampler(70_000 + index).weight.sum() for index in range(ENSEMBLE_SIZE)]
    )
    assert (
        abs(totals.mean() - TOTAL_STRENGTH)
        / totals.std(ddof=1)
        * np.sqrt(ENSEMBLE_SIZE)
        < 5.0
    )


@pytest.mark.parametrize("family", ["geometric", "negative_binomial"])
def test_pareto_strength_degree_fit_then_simulate_recovers_constraints(
    family: str,
) -> None:
    """W strength-degree fits recover strengths and degrees on Pareto sequences."""
    s_out, s_in = _pareto_strengths()
    k_total = 14.0
    k_out = s_out / s_out.sum() * k_total
    k_in = s_in / s_in.sum() * k_total
    assert np.max(k_out) < NODE_COUNT / 2
    assert np.max(k_in) < NODE_COUNT / 2
    # Ensure s >= k
    scale = min((s_out / k_out).min(), (s_in / k_in).min())
    if scale < 1.0:
        k_out = k_out * scale * 0.9
        k_in = k_in * (k_out.sum() / k_in.sum())

    from menobis.models import (
        fit_strength_degree_geometric,
        fit_strength_degree_negative_binomial,
        sample_strength_degree_geometric,
        sample_strength_degree_negative_binomial,
    )

    if family == "geometric":
        fit = fit_strength_degree_geometric(
            s_out, s_in, k_out, k_in, self_loops=SELF_LOOPS
        )
    else:
        fit = fit_strength_degree_negative_binomial(
            s_out, s_in, k_out, k_in, layers=LAYERS, self_loops=SELF_LOOPS
        )

    def sampler(seed: int) -> EdgeTable:
        if family == "geometric":
            return sample_strength_degree_geometric(fit, seed=seed)
        return sample_strength_degree_negative_binomial(fit, layers=LAYERS, seed=seed)

    assert fit.converged
    _assert_ensemble_strengths(sampler, s_out, s_in, z_limit=5.0)
    _assert_ensemble_node_stat(sampler, k_out, k_in, _degrees, z_limit=5.0)


@pytest.mark.parametrize("family", ["geometric", "negative_binomial"])
def test_pareto_strength_edges_fit_then_simulate_recovers_constraints(
    family: str,
) -> None:
    """W strength-edges fits avoid binary-degree saturation on Pareto strengths."""
    s_out, s_in = _pareto_strengths()
    target_edges = 14.0
    assert target_edges / (NODE_COUNT * NODE_COUNT) < 0.2

    if family == "geometric":
        fit = fit_strength_edges_geometric(
            s_out, s_in, target_edges, self_loops=SELF_LOOPS
        )
    else:
        fit = fit_strength_edges_negative_binomial(
            s_out, s_in, target_edges, layers=LAYERS, self_loops=SELF_LOOPS
        )

    def sampler(seed: int) -> EdgeTable:
        if family == "geometric":
            return sample_strength_edges_geometric(fit, seed=seed)
        return sample_strength_edges_negative_binomial(fit, layers=LAYERS, seed=seed)

    assert isinstance(fit, StrengthEdgesFit)
    assert fit.converged
    assert fit.max_q is not None and fit.max_q < 1.0
    _assert_ensemble_strengths(sampler, s_out, s_in, z_limit=5.0)
    edges = np.array(
        [_edge_count(sampler(50_000 + index)) for index in range(ENSEMBLE_SIZE)]
    )
    assert (
        abs(edges.mean() - target_edges) / edges.std(ddof=1) * np.sqrt(ENSEMBLE_SIZE)
        < 5.0
    )
