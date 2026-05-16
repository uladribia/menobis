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
   and sampled using ``sample_custom_pij_events_poisson`` or ``_multinomial``.

**Self-loops**: when ``self_loops=False``, diagonal pairs ``(i, i)`` are
excluded from both the mask and the fitted pairs, regardless of whether they
appear as known pairs.

**Normalization**: the returned ``PartialFitResult.rate`` values are
**unnormalized expected counts** (not probabilities). The sampling functions
``sample_custom_pij_events_*`` normalize internally. To obtain the
probability matrix, divide each rate by the sum of all rates.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import odme._odme as _odme
from odme.analysis import directed_degrees, directed_strengths
from odme.data.frames import EdgeTable, ProbabilityTable


@dataclass(frozen=True)
class PartialFitResult:
    """Combined rate table from partial-constraint fitting.

    Contains rates for all pairs: known pairs at their fixed values,
    fitted pairs at their model-predicted values. Rates are unnormalized
    expected counts. Use ``.as_probability_table()`` to convert to a
    ``ProbabilityTable`` for sampling.
    """

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


def _build_mask(
    n: int,
    known_source: NDArray[np.uint64],
    known_target: NDArray[np.uint64],
    self_loops: bool,
) -> list[bool]:
    """Build NxN boolean mask: True = skip this pair in IPF.

    Masks known pairs and, if ``self_loops=False``, the diagonal.
    """
    mask = [False] * (n * n)
    for s, t in zip(known_source, known_target, strict=True):
        mask[int(s) * n + int(t)] = True
    if not self_loops:
        for i in range(n):
            mask[i * n + i] = True
    return mask


def _infer_n(
    strength_out: NDArray[np.float64],
    known_source: NDArray[np.uint64],
    known_target: NDArray[np.uint64],
) -> int:
    n = len(strength_out)
    if known_source.size > 0:
        n = max(n, int(known_source.max()) + 1, int(known_target.max()) + 1)
    return n


def _compute_excess(
    strength_out: NDArray[np.float64],
    strength_in: NDArray[np.float64],
    known_source: NDArray[np.uint64],
    known_target: NDArray[np.uint64],
    known_rate: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    excess_out = strength_out.copy()
    excess_in = strength_in.copy()
    for s, t, r in zip(known_source, known_target, known_rate, strict=True):
        excess_out[int(s)] -= r
        excess_in[int(t)] -= r
    if np.any(excess_out < -1e-6) or np.any(excess_in < -1e-6):
        msg = "known pair rates exceed observed strengths; problem is infeasible"
        raise ValueError(msg)
    excess_out = np.maximum(excess_out, 0.0)
    excess_in = np.maximum(excess_in, 0.0)
    return excess_out, excess_in


def _combine_rates(
    n: int,
    known_source: NDArray[np.uint64],
    known_target: NDArray[np.uint64],
    known_rate: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    mask: list[bool],
    *,
    gamma: float = 0.0,
    cost_map: dict[tuple[int, int], float] | None = None,
) -> PartialFitResult:
    sources: list[int] = []
    targets: list[int] = []
    rates: list[float] = []
    # Known pairs.
    for s, t, r in zip(known_source, known_target, known_rate, strict=True):
        if r > 0:
            sources.append(int(s))
            targets.append(int(t))
            rates.append(float(r))
    # Fitted pairs (skip masked).
    for i in range(n):
        for j in range(n):
            if mask[i * n + j]:
                continue
            if gamma != 0.0 and cost_map is not None:
                d = cost_map.get((i, j), 0.0)
                rate = x[i] * y[j] * np.exp(-gamma * d)
            else:
                rate = x[i] * y[j]
            if rate > 0:
                sources.append(i)
                targets.append(j)
                rates.append(rate)
    return PartialFitResult(
        source=np.array(sources, dtype=np.uint64),
        target=np.array(targets, dtype=np.uint64),
        rate=np.array(rates, dtype=np.float64),
    )


def fit_partial_degree_me(
    degree_out: NDArray[np.floating],
    degree_in: NDArray[np.floating],
    known_source: NDArray[np.integer],
    known_target: NDArray[np.integer],
    *,
    self_loops: bool = True,
    tolerance: float = 1e-10,
    max_iterations: int = 50000,
) -> PartialFitResult:
    """Fit fixed-degree binary ME with some pairs known.

    Known pairs contribute fixed binary edges to the degree constraints.
    Free pairs are fitted as ``p_ij = x_i y_j / (1 + x_i y_j)``.

    Args:
        degree_out: Observed outgoing degree per node.
        degree_in: Observed incoming degree per node.
        known_source: Source node ids of known pairs.
        known_target: Target node ids of known pairs.
        self_loops: Whether to allow self-loop pairs in the fitted part.
        tolerance: Solver tolerance.
        max_iterations: Maximum solver iterations.

    Returns:
        PartialFitResult with binary probabilities as rates.
    """
    k_out = np.asarray(degree_out, dtype=np.float64)
    k_in = np.asarray(degree_in, dtype=np.float64)
    k_src = np.asarray(known_source, dtype=np.uint64)
    k_tgt = np.asarray(known_target, dtype=np.uint64)
    n = _infer_n(k_out, k_src, k_tgt)
    if len(k_out) < n:
        k_out = np.pad(k_out, (0, n - len(k_out)))
        k_in = np.pad(k_in, (0, n - len(k_in)))
    # Known pairs each contribute 1 to degree.
    known_rate = np.ones(len(k_src))
    excess_out, excess_in = _compute_excess(k_out, k_in, k_src, k_tgt, known_rate)
    mask = _build_mask(n, k_src, k_tgt, self_loops)
    # Balance excess.
    diff = excess_out.sum() - excess_in.sum()
    if abs(diff) > 1e-6:
        if diff > 0:
            excess_in[np.argmax(excess_in)] += diff
        else:
            excess_out[np.argmax(excess_out)] -= diff
    x_list, y_list, converged, iters = _odme.fit_masked_binary_degrees(
        excess_out.tolist(), excess_in.tolist(), mask, tolerance, max_iterations
    )
    if not converged:
        warnings.warn(
            f"fit_partial_degree_me did not converge after {iters} iterations",
            stacklevel=2,
        )
    x = np.asarray(x_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.float64)
    # Combine: known pairs get rate=1 (binary), free pairs get p_ij.
    sources: list[int] = []
    targets: list[int] = []
    rates: list[float] = []
    for s, t in zip(k_src, k_tgt, strict=True):
        sources.append(int(s))
        targets.append(int(t))
        rates.append(1.0)
    for i in range(n):
        for j in range(n):
            if mask[i * n + j]:
                continue
            z = x[i] * y[j]
            p = z / (1.0 + z)
            if p > 0:
                sources.append(i)
                targets.append(j)
                rates.append(p)
    return PartialFitResult(
        source=np.array(sources, dtype=np.uint64),
        target=np.array(targets, dtype=np.uint64),
        rate=np.array(rates, dtype=np.float64),
    )


def fit_partial_strength_degree_me(
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
    """Fit exact ME fixed-strength-degree with some pairs known.

    Known pairs contribute their rates to strength and binary edge to
    degree constraints. Free pairs are fitted using the thesis Case 4 equation.

    Args:
        strength_out: Observed outgoing strength per node.
        strength_in: Observed incoming strength per node.
        degree_out: Observed outgoing degree per node.
        degree_in: Observed incoming degree per node.
        known_source: Source node ids of known pairs.
        known_target: Target node ids of known pairs.
        known_rate: Expected event counts for known pairs.
        self_loops: Whether to allow self-loop pairs in the fitted part.
        tolerance: Solver tolerance.
        max_iterations: Maximum solver iterations.

    Returns:
        PartialFitResult with combined rate table.
    """
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_out = np.asarray(degree_out, dtype=np.float64)
    k_in = np.asarray(degree_in, dtype=np.float64)
    k_src = np.asarray(known_source, dtype=np.uint64)
    k_tgt = np.asarray(known_target, dtype=np.uint64)
    k_rate = np.asarray(known_rate, dtype=np.float64)
    n = _infer_n(s_out, k_src, k_tgt)
    if len(s_out) < n:
        s_out = np.pad(s_out, (0, n - len(s_out)))
        s_in = np.pad(s_in, (0, n - len(s_in)))
        k_out = np.pad(k_out, (0, n - len(k_out)))
        k_in = np.pad(k_in, (0, n - len(k_in)))
    excess_s_out, excess_s_in = _compute_excess(s_out, s_in, k_src, k_tgt, k_rate)
    known_binary = np.ones(len(k_src))
    excess_k_out, excess_k_in = _compute_excess(k_out, k_in, k_src, k_tgt, known_binary)
    mask = _build_mask(n, k_src, k_tgt, self_loops)
    if excess_s_out.sum() <= 0:
        return _combine_rates(n, k_src, k_tgt, k_rate, np.zeros(n), np.zeros(n), mask)
    # Balance.
    for seq_out, seq_in in [(excess_s_out, excess_s_in), (excess_k_out, excess_k_in)]:
        d = seq_out.sum() - seq_in.sum()
        if abs(d) > 1e-6:
            if d > 0:
                seq_in[np.argmax(seq_in)] += d
            else:
                seq_out[np.argmax(seq_out)] -= d
    x_l, y_l, z_l, w_l, converged, iters = _odme.fit_masked_strength_degree_me(
        excess_s_out.tolist(),
        excess_s_in.tolist(),
        excess_k_out.tolist(),
        excess_k_in.tolist(),
        mask,
        tolerance,
        max_iterations,
    )
    if not converged:
        warnings.warn(
            f"fit_partial_strength_degree_me did not converge after {iters} iterations",
            stacklevel=2,
        )
    x = np.asarray(x_l, dtype=np.float64)
    y = np.asarray(y_l, dtype=np.float64)
    z = np.asarray(z_l, dtype=np.float64)
    w = np.asarray(w_l, dtype=np.float64)
    # Combine with Case 4 rates.
    sources: list[int] = []
    targets: list[int] = []
    rates: list[float] = []
    for s, t, r in zip(k_src, k_tgt, k_rate, strict=True):
        if r > 0:
            sources.append(int(s))
            targets.append(int(t))
            rates.append(float(r))
    for i in range(n):
        for j in range(n):
            if mask[i * n + j]:
                continue
            u = x[i] * y[j]
            v = z[i] * w[j]
            exp_u = np.exp(u)
            den = 1.0 + v * (exp_u - 1.0)
            rate = v * u * exp_u / den if den > 0 else 0.0
            if rate > 0:
                sources.append(i)
                targets.append(j)
                rates.append(rate)
    return PartialFitResult(
        source=np.array(sources, dtype=np.uint64),
        target=np.array(targets, dtype=np.uint64),
        rate=np.array(rates, dtype=np.float64),
    )


def fit_partial_strength_edges_me(
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
    """Fit fixed-strength + total-edge-count ME with some pairs known.

    Known pairs contribute their rates to strengths and 1 binary edge each
    to the edge count. Free pairs are fitted using thesis Case 3.

    Args:
        strength_out: Observed outgoing strength per node.
        strength_in: Observed incoming strength per node.
        known_source: Source node ids of known pairs.
        known_target: Target node ids of known pairs.
        known_rate: Expected event counts for known pairs.
        target_edges: Observed total binary edge count.
        self_loops: Whether to allow self-loop pairs in the fitted part.
        tolerance: Solver tolerance.
        max_iterations: Maximum solver iterations.

    Returns:
        PartialFitResult with combined rate table.
    """
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_src = np.asarray(known_source, dtype=np.uint64)
    k_tgt = np.asarray(known_target, dtype=np.uint64)
    k_rate = np.asarray(known_rate, dtype=np.float64)
    n = _infer_n(s_out, k_src, k_tgt)
    if len(s_out) < n:
        s_out = np.pad(s_out, (0, n - len(s_out)))
        s_in = np.pad(s_in, (0, n - len(s_in)))
    excess_out, excess_in = _compute_excess(s_out, s_in, k_src, k_tgt, k_rate)
    excess_edges = target_edges - float(len(k_src))
    if excess_edges < -1e-6:
        msg = "known pairs exceed target edge count; problem is infeasible"
        raise ValueError(msg)
    excess_edges = max(excess_edges, 0.0)
    mask = _build_mask(n, k_src, k_tgt, self_loops)
    if excess_out.sum() <= 0 or excess_edges <= 0:
        return _combine_rates(n, k_src, k_tgt, k_rate, np.zeros(n), np.zeros(n), mask)
    diff = excess_out.sum() - excess_in.sum()
    if abs(diff) > 1e-6:
        if diff > 0:
            excess_in[np.argmax(excess_in)] += diff
        else:
            excess_out[np.argmax(excess_out)] -= diff

    from odme.models.fitting import fit_strength_edges_me as _fit_se

    fit = _fit_se(
        excess_out,
        excess_in,
        excess_edges,
        self_loops=self_loops,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    # Reconstruct rates using Case 3 equation.
    sources: list[int] = []
    targets: list[int] = []
    rates: list[float] = []
    for s, t, r in zip(k_src, k_tgt, k_rate, strict=True):
        if r > 0:
            sources.append(int(s))
            targets.append(int(t))
            rates.append(float(r))
    for i in range(n):
        for j in range(n):
            if mask[i * n + j]:
                continue
            u = fit.x[i] * fit.y[j]
            exp_u = np.exp(u)
            den = 1.0 + fit.lam * (exp_u - 1.0)
            rate = fit.lam * u * exp_u / den if den > 0 else 0.0
            if rate > 0:
                sources.append(i)
                targets.append(j)
                rates.append(rate)
    return PartialFitResult(
        source=np.array(sources, dtype=np.uint64),
        target=np.array(targets, dtype=np.uint64),
        rate=np.array(rates, dtype=np.float64),
    )


def fit_partial_strength_me(
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
    """Fit fixed-strength ME with some p_ij pairs known.

    Known pairs contribute their fixed rates to the strength constraints.
    The remaining pairs are fitted using masked IPF so that the full
    strength sequence is recovered.

    Args:
        strength_out: Observed outgoing strength per node.
        strength_in: Observed incoming strength per node.
        known_source: Source node ids of known pairs.
        known_target: Target node ids of known pairs.
        known_rate: Expected event counts for known pairs.
        self_loops: Whether to allow self-loop pairs in the fitted part.
        tolerance: Solver tolerance.
        max_iterations: Maximum solver iterations.

    Returns:
        PartialFitResult with combined rate table.
    """
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_src = np.asarray(known_source, dtype=np.uint64)
    k_tgt = np.asarray(known_target, dtype=np.uint64)
    k_rate = np.asarray(known_rate, dtype=np.float64)

    n = _infer_n(s_out, k_src, k_tgt)
    if len(s_out) < n:
        s_out = np.pad(s_out, (0, n - len(s_out)))
        s_in = np.pad(s_in, (0, n - len(s_in)))

    excess_out, excess_in = _compute_excess(s_out, s_in, k_src, k_tgt, k_rate)
    mask = _build_mask(n, k_src, k_tgt, self_loops)

    if excess_out.sum() <= 0:
        return _combine_rates(n, k_src, k_tgt, k_rate, np.zeros(n), np.zeros(n), mask)

    # Balance excess so solver doesn't reject.
    diff = excess_out.sum() - excess_in.sum()
    if abs(diff) > 1e-6:
        if diff > 0:
            excess_in[np.argmax(excess_in)] += diff
        else:
            excess_out[np.argmax(excess_out)] -= diff

    x_list, y_list, converged, iters = _odme.fit_masked_strength(
        excess_out.tolist(), excess_in.tolist(), mask, tolerance, max_iterations
    )
    if not converged:
        warnings.warn(
            f"fit_partial_strength_me did not converge after {iters} iterations",
            stacklevel=2,
        )
    x = np.asarray(x_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.float64)
    return _combine_rates(n, k_src, k_tgt, k_rate, x, y, mask)


def fit_partial_strength_cost_me(
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
    """Fit fixed-strength + fixed-cost ME with some p_ij pairs known.

    Like ``fit_partial_strength_me`` but also fits ``gamma`` to match total cost.
    The fitted expectation for free pairs is ``E[t_ij] = x_i y_j exp(-gamma  d_ij)``.

    Args:
        strength_out: Observed outgoing strength per node.
        strength_in: Observed incoming strength per node.
        known_source: Source node ids of known pairs.
        known_target: Target node ids of known pairs.
        known_rate: Expected event counts for known pairs.
        cost_sources: Source node ids for cost entries.
        cost_targets: Target node ids for cost entries.
        cost_values: Cost/distance values.
        target_cost: Observed total cost ``C = Σ t_ij d_ij``.
        self_loops: Whether to allow self-loop pairs in the fitted part.
        tolerance: Solver tolerance.
        max_iterations: Maximum solver iterations.

    Returns:
        PartialFitResult with combined rate table including cost deterrence.
    """
    s_out = np.asarray(strength_out, dtype=np.float64)
    s_in = np.asarray(strength_in, dtype=np.float64)
    k_src = np.asarray(known_source, dtype=np.uint64)
    k_tgt = np.asarray(known_target, dtype=np.uint64)
    k_rate = np.asarray(known_rate, dtype=np.float64)
    c_src = np.asarray(cost_sources, dtype=np.int64)
    c_tgt = np.asarray(cost_targets, dtype=np.int64)
    c_val = np.asarray(cost_values, dtype=np.float64)

    n = _infer_n(s_out, k_src, k_tgt)
    if len(s_out) < n:
        s_out = np.pad(s_out, (0, n - len(s_out)))
        s_in = np.pad(s_in, (0, n - len(s_in)))

    # Build cost lookup.
    cost_map: dict[tuple[int, int], float] = {}
    for cs, ct, cv in zip(c_src, c_tgt, c_val, strict=True):
        cost_map[(int(cs), int(ct))] = float(cv)

    # Excess cost: total cost minus contribution from known pairs.
    known_cost = sum(
        float(r) * cost_map.get((int(s), int(t)), 0.0)
        for s, t, r in zip(k_src, k_tgt, k_rate, strict=True)
    )
    excess_cost = target_cost - known_cost
    if excess_cost < -1e-6:
        msg = "known pair costs exceed target cost; problem is infeasible"
        raise ValueError(msg)
    excess_cost = max(excess_cost, 0.0)

    excess_out, excess_in = _compute_excess(s_out, s_in, k_src, k_tgt, k_rate)
    mask = _build_mask(n, k_src, k_tgt, self_loops)

    if excess_out.sum() <= 0:
        return _combine_rates(n, k_src, k_tgt, k_rate, np.zeros(n), np.zeros(n), mask)

    # Balance excess.
    diff = excess_out.sum() - excess_in.sum()
    if abs(diff) > 1e-6:
        if diff > 0:
            excess_in[np.argmax(excess_in)] += diff
        else:
            excess_out[np.argmax(excess_out)] -= diff

    # Filter cost entries to only free pairs.
    free_src: list[int] = []
    free_tgt: list[int] = []
    free_cost: list[float] = []
    for cs, ct, cv in zip(c_src, c_tgt, c_val, strict=True):
        i, j = int(cs), int(ct)
        if i < n and j < n and not mask[i * n + j]:
            free_src.append(i)
            free_tgt.append(j)
            free_cost.append(float(cv))

    from odme.models.fitting import fit_strength_cost_me

    fit = fit_strength_cost_me(
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

    return _combine_rates(
        n,
        k_src,
        k_tgt,
        k_rate,
        fit.x,
        fit.y,
        mask,
        gamma=fit.gamma,
        cost_map=cost_map,
    )


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
    """Split an observed network by weight cutoff and fit partial constraints.

    Edges with ``weight > cutoff`` are treated as known pairs with rates
    equal to their observed weight. The remaining structure is fitted using
    the specified ME model on the excess constraints.

    For the ``"strength-cost"`` model, a cost matrix and target cost must
    be provided. The target cost defaults to the observed total cost
    ``C = Σ t_ij d_ij`` if not given.

    Args:
        edges: Observed weighted edge table.
        cutoff: Weight threshold; edges above this are fixed.
        model: Model variant: ``"strength"`` or ``"strength-cost"``.
        self_loops: Whether self-loop pairs are allowed in the fitted part.
        cost_sources: Source node ids for cost entries (strength-cost only).
        cost_targets: Target node ids for cost entries (strength-cost only).
        cost_values: Cost/distance values (strength-cost only).
        target_cost: Total cost constraint (strength-cost only; defaults
            to observed).
        tolerance: Solver tolerance.
        max_iterations: Maximum solver iterations.

    Returns:
        PartialFitResult with combined rate table.
    """
    s = directed_strengths(edges)

    heavy = edges.weight > cutoff
    known_source = edges.source[heavy]
    known_target = edges.target[heavy]
    known_rate = edges.weight[heavy].astype(np.float64)

    if model == "strength":
        return fit_partial_strength_me(
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
        return fit_partial_degree_me(
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
        return fit_partial_strength_degree_me(
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
        return fit_partial_strength_edges_me(
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
            msg = "strength-cost model requires cost_sources, cost_targets, cost_values"
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
                float(w) * cost_map.get((int(s_val), int(t_val)), 0.0)
                for s_val, t_val, w in zip(
                    edges.source, edges.target, edges.weight, strict=True
                )
            )
        return fit_partial_strength_cost_me(
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
    "fit_partial_degree_me",
    "fit_partial_strength_cost_me",
    "fit_partial_strength_degree_me",
    "fit_partial_strength_edges_me",
    "fit_partial_strength_me",
]
