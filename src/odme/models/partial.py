"""Partial-constraint fitting: some p_ij known, rest fitted from ME models.

This implements a generalization of the thesis ``fitter_pij`` case. Given an
observed network, certain heavy edges (with ``t_ij > cutoff``) are treated as
having known expected rates. The remaining pairs are fitted using a
maximum-entropy model subject to the observed constraints minus the known-pair
contributions.

The procedure is:

1. Identify the set Q of known pairs (e.g., by weight cutoff).
2. Compute **excess** constraints: observed strengths (and optionally cost)
   minus the contribution from Q pairs.
3. Build a mask that excludes Q pairs (and optionally the diagonal for
   no-self-loops) from the IPF summations.
4. Fit Lagrange multipliers ``x, y`` (and optionally ``gamma``) on the excess
   constraints over the free pairs only.
5. Combine the known rates with the fitted rates ``x_i y_j`` (or
   ``x_i y_j exp(-gamma  d_ij)``) into a single rate table.
6. The result can be normalized to probabilities ``p_ij = rate_ij / sum(rate)``
   and sampled using ``sample_custom_poisson`` or ``_multinomial``.

**Self-loops**: when ``self_loops=False``, diagonal pairs ``(i, i)`` are
excluded from both the mask and the fitted pairs, regardless of whether they
appear as known pairs.

**Normalization**: the returned ``PartialFitResult.rate`` values are
**unnormalized expected counts** (not probabilities). The sampling functions
``sample_custom_pij_events_*`` normalize internally. To obtain the
probability matrix, divide each rate by the sum of all rates.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import odme._odme as _odme
from odme.analysis import directed_degrees, directed_strengths
from odme.data.frames import EdgeTable, ProbabilityTable


@dataclass(frozen=True)
class PartialFitResult:
    """Combined rate table from partial-constraint fitting."""

    source: NDArray[np.uint64]
    target: NDArray[np.uint64]
    rate: NDArray[np.float64]

    def as_probability_table(self) -> ProbabilityTable:
        """Convert to ProbabilityTable for sampling (rates as weights)."""
        return ProbabilityTable(
            source=self.source,
            target=self.target,
            probability=self.rate,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_mask(
    n: int,
    known_source: NDArray[np.uint64],
    known_target: NDArray[np.uint64],
    self_loops: bool,
) -> list[bool]:
    """Build NxN boolean mask: True = skip this pair in IPF."""
    mask = [False] * (n * n)
    for s, t in zip(known_source, known_target, strict=True):
        mask[int(s) * n + int(t)] = True
    if not self_loops:
        for i in range(n):
            mask[i * n + i] = True
    return mask


def _infer_n(
    sequence: NDArray[np.float64],
    known_source: NDArray[np.uint64],
    known_target: NDArray[np.uint64],
) -> int:
    n = len(sequence)
    if known_source.size > 0:
        n = max(n, int(known_source.max()) + 1, int(known_target.max()) + 1)
    return n


def _compute_excess(
    out_seq: NDArray[np.float64],
    in_seq: NDArray[np.float64],
    known_source: NDArray[np.uint64],
    known_target: NDArray[np.uint64],
    known_contrib: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    excess_out = out_seq.copy()
    excess_in = in_seq.copy()
    for s, t, c in zip(known_source, known_target, known_contrib, strict=True):
        excess_out[int(s)] -= c
        excess_in[int(t)] -= c
    if np.any(excess_out < -1e-6) or np.any(excess_in < -1e-6):
        msg = "known pair contributions exceed observed constraints; infeasible"
        raise ValueError(msg)
    return np.maximum(excess_out, 0.0), np.maximum(excess_in, 0.0)


def _balance_excess(
    excess_out: NDArray[np.float64],
    excess_in: NDArray[np.float64],
) -> None:
    """Nudge excess sequences to balance (in-place)."""
    diff = excess_out.sum() - excess_in.sum()
    if abs(diff) > 1e-6:
        if diff > 0:
            excess_in[np.argmax(excess_in)] += diff
        else:
            excess_out[np.argmax(excess_out)] -= diff


def _pad_to_n(arr: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    if len(arr) < n:
        return np.pad(arr, (0, n - len(arr)))
    return arr


def _build_result(
    n: int,
    known_source: NDArray[np.uint64],
    known_target: NDArray[np.uint64],
    known_rate: NDArray[np.float64],
    mask: list[bool],
    free_rate_fn: Callable[[int, int], float],
) -> PartialFitResult:
    """Combine known rates and fitted free-pair rates into one table."""
    sources: list[int] = []
    targets: list[int] = []
    rates: list[float] = []
    for s, t, r in zip(known_source, known_target, known_rate, strict=True):
        if r > 0:
            sources.append(int(s))
            targets.append(int(t))
            rates.append(float(r))
    for i in range(n):
        for j in range(n):
            if mask[i * n + j]:
                continue
            rate = free_rate_fn(i, j)
            if rate > 0:
                sources.append(i)
                targets.append(j)
                rates.append(rate)
    return PartialFitResult(
        source=np.array(sources, dtype=np.uint64),
        target=np.array(targets, dtype=np.uint64),
        rate=np.array(rates, dtype=np.float64),
    )


def _warn_if_not_converged(name: str, converged: bool, iters: int) -> None:
    if not converged:
        warnings.warn(
            f"{name} did not converge after {iters} iterations",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Public partial fitters
# ---------------------------------------------------------------------------


def fit_partial_strength_poisson(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    known_source: NDArray[np.integer],
    known_target: NDArray[np.integer],
    known_rate: NDArray[np.floating],
    *,
    self_loops: bool = True,
    tolerance: float = 1e-8,
    max_iterations: int = 10000,
) -> PartialFitResult:
    """Fit fixed-strength ME with some p_ij pairs known."""
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_src = np.asarray(known_source, dtype=np.uint64)
    k_tgt = np.asarray(known_target, dtype=np.uint64)
    k_rate = np.asarray(known_rate, dtype=np.float64)
    n = _infer_n(s_out, k_src, k_tgt)
    s_out = _pad_to_n(s_out, n)
    s_in = _pad_to_n(s_in, n)
    excess_out, excess_in = _compute_excess(s_out, s_in, k_src, k_tgt, k_rate)
    mask = _build_mask(n, k_src, k_tgt, self_loops)
    if excess_out.sum() <= 0:
        return _build_result(n, k_src, k_tgt, k_rate, mask, lambda i, j: 0.0)
    _balance_excess(excess_out, excess_in)
    x_l, y_l, converged, iters = _odme.fit_masked_strength_poisson(
        excess_out.tolist(), excess_in.tolist(), mask, tolerance, max_iterations
    )
    _warn_if_not_converged("fit_partial_strength_poisson", converged, iters)
    x = np.asarray(x_l, dtype=np.float64)
    y = np.asarray(y_l, dtype=np.float64)
    return _build_result(n, k_src, k_tgt, k_rate, mask, lambda i, j: x[i] * y[j])


def fit_partial_degree_poisson(
    degree_out: NDArray[np.floating],
    degree_in: NDArray[np.floating],
    known_source: NDArray[np.integer],
    known_target: NDArray[np.integer],
    *,
    self_loops: bool = True,
    tolerance: float = 1e-10,
    max_iterations: int = 50000,
) -> PartialFitResult:
    """Fit fixed-degree binary ME with some pairs known."""
    k_out = np.asarray(degree_out, dtype=np.float64)
    k_in = np.asarray(degree_in, dtype=np.float64)
    k_src = np.asarray(known_source, dtype=np.uint64)
    k_tgt = np.asarray(known_target, dtype=np.uint64)
    n = _infer_n(k_out, k_src, k_tgt)
    k_out = _pad_to_n(k_out, n)
    k_in = _pad_to_n(k_in, n)
    known_binary = np.ones(len(k_src))
    excess_out, excess_in = _compute_excess(k_out, k_in, k_src, k_tgt, known_binary)
    mask = _build_mask(n, k_src, k_tgt, self_loops)
    _balance_excess(excess_out, excess_in)
    x_l, y_l, converged, iters = _odme.fit_masked_degree_bernoulli(
        excess_out.tolist(), excess_in.tolist(), mask, tolerance, max_iterations
    )
    _warn_if_not_converged("fit_partial_degree_poisson", converged, iters)
    x = np.asarray(x_l, dtype=np.float64)
    y = np.asarray(y_l, dtype=np.float64)
    known_rate = np.ones(len(k_src))

    def _degree_rate(i: int, j: int) -> float:
        z = x[i] * y[j]
        return z / (1.0 + z)

    return _build_result(n, k_src, k_tgt, known_rate, mask, _degree_rate)


def fit_partial_strength_degree_poisson(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    degree_out: NDArray[np.floating],
    degree_in: NDArray[np.floating],
    known_source: NDArray[np.integer],
    known_target: NDArray[np.integer],
    known_rate: NDArray[np.floating],
    *,
    self_loops: bool = True,
    tolerance: float = 1e-10,
    max_iterations: int = 50000,
) -> PartialFitResult:
    """Fit exact ME fixed-strength-degree with some pairs known."""
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_out = np.asarray(degree_out, dtype=np.float64)
    k_in = np.asarray(degree_in, dtype=np.float64)
    k_src = np.asarray(known_source, dtype=np.uint64)
    k_tgt = np.asarray(known_target, dtype=np.uint64)
    k_rate_arr = np.asarray(known_rate, dtype=np.float64)
    n = _infer_n(s_out, k_src, k_tgt)
    s_out = _pad_to_n(s_out, n)
    s_in = _pad_to_n(s_in, n)
    k_out = _pad_to_n(k_out, n)
    k_in = _pad_to_n(k_in, n)
    excess_s_out, excess_s_in = _compute_excess(s_out, s_in, k_src, k_tgt, k_rate_arr)
    known_binary = np.ones(len(k_src))
    excess_k_out, excess_k_in = _compute_excess(k_out, k_in, k_src, k_tgt, known_binary)
    mask = _build_mask(n, k_src, k_tgt, self_loops)
    if excess_s_out.sum() <= 0:
        return _build_result(n, k_src, k_tgt, k_rate_arr, mask, lambda i, j: 0.0)
    _balance_excess(excess_s_out, excess_s_in)
    _balance_excess(excess_k_out, excess_k_in)
    x_l, y_l, z_l, w_l, converged, iters = _odme.fit_masked_strength_degree_poisson(
        excess_s_out.tolist(),
        excess_s_in.tolist(),
        excess_k_out.tolist(),
        excess_k_in.tolist(),
        mask,
        tolerance,
        max_iterations,
    )
    _warn_if_not_converged("fit_partial_strength_degree_poisson", converged, iters)
    x = np.asarray(x_l, dtype=np.float64)
    y = np.asarray(y_l, dtype=np.float64)
    z = np.asarray(z_l, dtype=np.float64)
    w = np.asarray(w_l, dtype=np.float64)

    def _case4_rate(i: int, j: int) -> float:
        u = x[i] * y[j]
        v = z[i] * w[j]
        exp_u = np.exp(u)
        den = 1.0 + v * (exp_u - 1.0)
        return v * u * exp_u / den if den > 0 else 0.0

    return _build_result(n, k_src, k_tgt, k_rate_arr, mask, _case4_rate)


def fit_partial_strength_edges_poisson(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    known_source: NDArray[np.integer],
    known_target: NDArray[np.integer],
    known_rate: NDArray[np.floating],
    target_edges: float,
    *,
    self_loops: bool = True,
    tolerance: float = 1e-10,
    max_iterations: int = 50000,
) -> PartialFitResult:
    """Fit fixed-strength + total-edge-count ME with some pairs known."""
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_src = np.asarray(known_source, dtype=np.uint64)
    k_tgt = np.asarray(known_target, dtype=np.uint64)
    k_rate_arr = np.asarray(known_rate, dtype=np.float64)
    n = _infer_n(s_out, k_src, k_tgt)
    s_out = _pad_to_n(s_out, n)
    s_in = _pad_to_n(s_in, n)
    excess_out, excess_in = _compute_excess(s_out, s_in, k_src, k_tgt, k_rate_arr)
    excess_edges = target_edges - float(len(k_src))
    if excess_edges < -1e-6:
        msg = "known pairs exceed target edge count; infeasible"
        raise ValueError(msg)
    excess_edges = max(excess_edges, 0.0)
    mask = _build_mask(n, k_src, k_tgt, self_loops)
    if excess_out.sum() <= 0 or excess_edges <= 0:
        return _build_result(n, k_src, k_tgt, k_rate_arr, mask, lambda i, j: 0.0)
    _balance_excess(excess_out, excess_in)

    from odme.models.fitting import fit_strength_edges_poisson as _fit_se

    fit = _fit_se(
        excess_out,
        excess_in,
        excess_edges,
        self_loops=self_loops,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    lam = fit.lam
    fx = fit.x
    fy = fit.y

    def _case3_rate(i: int, j: int) -> float:
        u = fx[i] * fy[j]
        exp_u = np.exp(u)
        den = 1.0 + lam * (exp_u - 1.0)
        return lam * u * exp_u / den if den > 0 else 0.0

    return _build_result(n, k_src, k_tgt, k_rate_arr, mask, _case3_rate)


def fit_partial_strength_cost_poisson(
    strength_out: NDArray[np.floating],
    strength_in: NDArray[np.floating],
    known_source: NDArray[np.integer],
    known_target: NDArray[np.integer],
    known_rate: NDArray[np.floating],
    cost_sources: NDArray[np.integer],
    cost_targets: NDArray[np.integer],
    cost_values: NDArray[np.floating],
    target_cost: float,
    *,
    self_loops: bool = True,
    tolerance: float = 1e-6,
    max_iterations: int = 5000,
) -> PartialFitResult:
    """Fit fixed-strength + fixed-cost ME with some p_ij pairs known."""
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_src = np.asarray(known_source, dtype=np.uint64)
    k_tgt = np.asarray(known_target, dtype=np.uint64)
    k_rate_arr = np.asarray(known_rate, dtype=np.float64)
    c_src = np.asarray(cost_sources, dtype=np.int64)
    c_tgt = np.asarray(cost_targets, dtype=np.int64)
    c_val = np.asarray(cost_values, dtype=np.float64)
    n = _infer_n(s_out, k_src, k_tgt)
    s_out = _pad_to_n(s_out, n)
    s_in = _pad_to_n(s_in, n)
    cost_map: dict[tuple[int, int], float] = {}
    for cs, ct, cv in zip(c_src, c_tgt, c_val, strict=True):
        cost_map[(int(cs), int(ct))] = float(cv)
    known_cost = sum(
        float(r) * cost_map.get((int(s), int(t)), 0.0)
        for s, t, r in zip(k_src, k_tgt, k_rate_arr, strict=True)
    )
    excess_cost = max(target_cost - known_cost, 0.0)
    if target_cost - known_cost < -1e-6:
        msg = "known pair costs exceed target cost; infeasible"
        raise ValueError(msg)
    excess_out, excess_in = _compute_excess(s_out, s_in, k_src, k_tgt, k_rate_arr)
    mask = _build_mask(n, k_src, k_tgt, self_loops)
    if excess_out.sum() <= 0:
        return _build_result(n, k_src, k_tgt, k_rate_arr, mask, lambda i, j: 0.0)
    _balance_excess(excess_out, excess_in)
    free_src = [
        int(cs)
        for cs, ct in zip(c_src, c_tgt, strict=True)
        if int(cs) < n and int(ct) < n and not mask[int(cs) * n + int(ct)]
    ]
    free_tgt = [
        int(ct)
        for cs, ct in zip(c_src, c_tgt, strict=True)
        if int(cs) < n and int(ct) < n and not mask[int(cs) * n + int(ct)]
    ]
    free_cost = [
        float(cv)
        for cs, ct, cv in zip(c_src, c_tgt, c_val, strict=True)
        if int(cs) < n and int(ct) < n and not mask[int(cs) * n + int(ct)]
    ]

    from odme.models.fitting import fit_strength_cost_poisson

    fit = fit_strength_cost_poisson(
        excess_out,
        excess_in,
        np.array(free_src, dtype=np.int64),
        np.array(free_tgt, dtype=np.int64),
        np.array(free_cost, dtype=np.float64),
        excess_cost,
        self_loops=self_loops,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    gamma = fit.gamma
    fx = fit.x
    fy = fit.y

    def _cost_rate(i: int, j: int) -> float:
        d = cost_map.get((i, j), 0.0)
        return fx[i] * fy[j] * np.exp(-gamma * d)

    return _build_result(n, k_src, k_tgt, k_rate_arr, mask, _cost_rate)


# ---------------------------------------------------------------------------
# Convenience: split by cutoff
# ---------------------------------------------------------------------------


def fit_from_network_cutoff(
    edges: EdgeTable,
    cutoff: float,
    model: str = "strength",
    *,
    self_loops: bool = True,
    cost_sources: NDArray[np.integer] | None = None,
    cost_targets: NDArray[np.integer] | None = None,
    cost_values: NDArray[np.floating] | None = None,
    target_cost: float | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 10000,
) -> PartialFitResult:
    """Split an observed network by weight cutoff and fit partial constraints."""
    s = directed_strengths(edges)
    heavy = edges.weight > cutoff
    known_source = edges.source[heavy]
    known_target = edges.target[heavy]
    known_rate = edges.weight[heavy].astype(np.float64)

    if model == "strength":
        return fit_partial_strength_poisson(
            s.out.astype(np.float64),
            s.incoming.astype(np.float64),
            known_source,
            known_target,
            known_rate,
            self_loops=self_loops,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    if model == "degree":
        k = directed_degrees(edges)
        return fit_partial_degree_poisson(
            k.out.astype(np.float64),
            k.incoming.astype(np.float64),
            known_source,
            known_target,
            self_loops=self_loops,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    if model == "strength-degree":
        k = directed_degrees(edges)
        return fit_partial_strength_degree_poisson(
            s.out.astype(np.float64),
            s.incoming.astype(np.float64),
            k.out.astype(np.float64),
            k.incoming.astype(np.float64),
            known_source,
            known_target,
            known_rate,
            self_loops=self_loops,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    if model == "strength-edges":
        return fit_partial_strength_edges_poisson(
            s.out.astype(np.float64),
            s.incoming.astype(np.float64),
            known_source,
            known_target,
            known_rate,
            float(edges.num_edges),
            self_loops=self_loops,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    if model == "strength-cost":
        if cost_sources is None or cost_targets is None or cost_values is None:
            msg = "strength-cost requires cost_sources, cost_targets, cost_values"
            raise ValueError(msg)
        if target_cost is None:
            cost_map: dict[tuple[int, int], float] = {}
            for cs, ct, cv in zip(
                np.asarray(cost_sources),
                np.asarray(cost_targets),
                np.asarray(cost_values),
                strict=True,
            ):
                cost_map[(int(cs), int(ct))] = float(cv)
            target_cost = sum(
                float(w) * cost_map.get((int(sv), int(tv)), 0.0)
                for sv, tv, w in zip(
                    edges.source, edges.target, edges.weight, strict=True
                )
            )
        return fit_partial_strength_cost_poisson(
            s.out.astype(np.float64),
            s.incoming.astype(np.float64),
            known_source,
            known_target,
            known_rate,
            cost_sources,
            cost_targets,
            cost_values,
            target_cost,
            self_loops=self_loops,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    msg = (
        f"unsupported model: {model!r}. "
        "Use 'strength', 'degree', 'strength-degree', "
        "'strength-edges', or 'strength-cost'."
    )
    raise ValueError(msg)


__all__ = [
    "PartialFitResult",
    "fit_from_network_cutoff",
    "fit_partial_degree_poisson",
    "fit_partial_strength_cost_poisson",
    "fit_partial_strength_degree_poisson",
    "fit_partial_strength_edges_poisson",
    "fit_partial_strength_poisson",
]
