"""Fitting benchmark logic for all ME/W/B families."""

import json
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.common import (
    complete_costs,
    compute_degree_residual,
    compute_strength_residual,
    degrees_from_strengths,
    pareto_strengths,
)

CONSTRAINTS = (
    "strength",
    "degree-events",
    "strength-edges",
    "strength-degree",
    "strength-cost",
)


def fitting_case_registry(include_partial=True):
    """Return expected fitting benchmark case metadata."""
    cases = []
    for ensemble, family, layers in (
        ("ME", "poisson", None),
        ("B", "binomial", 3),
        ("W", "geometric", None),
        ("W", "negative-binomial", 3),
    ):
        for constraint in CONSTRAINTS:
            cases.append(
                {
                    "ensemble": ensemble,
                    "family": family,
                    "constraint": constraint,
                    "layers": layers,
                    "partial": False,
                }
            )
    if include_partial:
        for ensemble, family, layers in (
            ("ME", "poisson", None),
            ("B", "binomial", 3),
            ("W", "geometric", None),
            ("W", "negative-binomial", 3),
        ):
            for constraint in CONSTRAINTS:
                cases.append(
                    {
                        "ensemble": ensemble,
                        "family": family,
                        "constraint": constraint,
                        "layers": layers,
                        "partial": True,
                    }
                )
    return cases


def _case_name(ensemble, family, constraint, layers=None, partial=False, solver=None):
    prefix = "partial " if partial else ""
    name = f"{prefix}{ensemble} {family} {constraint}"
    if layers is not None:
        name = f"{name} M={layers}"
    if solver is not None:
        name = f"{name} [{solver}]"
    return name


def _independent_expected_weights(fit):
    """Return dense expected weights for an independent strength fit."""
    q = np.outer(np.asarray(fit.x), np.asarray(fit.y))
    family = getattr(fit, "family", "poisson")
    layers = getattr(fit, "layers", None) or 1
    if family in {"geometric", "negative_binomial"}:
        return layers * q / np.maximum(1.0 - q, 1e-15)
    if family == "binomial":
        return layers * q / (1.0 + q)
    return q


def _target_cost_from_independent_fit(fit, cost_values, n):
    """Compute a feasible target cost from a same-family independent fit."""
    return float(np.sum(cost_values.reshape(n, n) * _independent_expected_weights(fit)))


def _known_pairs_from_top_rates(base, n, fraction):
    """Select known weighted pairs from the largest independent rates."""
    capacity = n * n
    n_known = int(capacity * fraction)
    rates_all = np.outer(base.x, base.y).ravel()
    pair_order = np.argsort(rates_all)[::-1][:n_known]
    return (
        (pair_order // n).astype(np.uint64),
        (pair_order % n).astype(np.uint64),
        rates_all[pair_order],
    )


def _known_pairs_from_random_rates(base, n, fraction, seed=12345):
    """Select deterministic random known weighted pairs from independent rates."""
    capacity = n * n
    n_known = int(capacity * fraction)
    rates_all = np.outer(base.x, base.y).ravel()
    rng = np.random.default_rng(seed)
    pair_order = rng.choice(capacity, n_known, replace=False)
    return (
        (pair_order // n).astype(np.uint64),
        (pair_order % n).astype(np.uint64),
        rates_all[pair_order],
    )


def _feasible_binary_known_pairs(
    strength_out, strength_in, degree_out, degree_in, fraction
):
    """Select known occupied pairs that leave feasible strength-degree excess."""
    n = len(strength_out)
    row_capacity = np.floor(np.minimum(strength_out, degree_out)).astype(int)
    col_capacity = np.floor(np.minimum(strength_in, degree_in)).astype(int)
    target = max(1, int(min(row_capacity.sum(), col_capacity.sum()) * fraction))
    sources = []
    targets = []
    row_remaining = row_capacity.copy()
    col_remaining = col_capacity.copy()
    pairs = [
        (i, j, min(strength_out[i], strength_in[j])) for i in range(n) for j in range(n)
    ]
    pairs.sort(key=lambda item: item[2], reverse=True)
    for source, target_node, _score in pairs:
        if len(sources) >= target:
            break
        if row_remaining[source] > 0 and col_remaining[target_node] > 0:
            sources.append(source)
            targets.append(target_node)
            row_remaining[source] -= 1
            col_remaining[target_node] -= 1
    return (
        np.array(sources, dtype=np.uint64),
        np.array(targets, dtype=np.uint64),
        np.ones(len(sources), dtype=np.float64),
    )


def _infer_metadata(name):
    """Infer structured metadata for legacy benchmark case names."""
    if name.startswith("ME "):
        ensemble, family = "ME", "poisson"
    elif name.startswith("B "):
        ensemble, family = "B", "binomial"
    elif name.startswith("W "):
        ensemble = "W"
        family = "negative-binomial" if "negative-binomial" in name else "geometric"
    else:
        return {}

    constraint = None
    for candidate in (
        "degree-events",
        "strength-edges",
        "strength-degree",
        "strength-cost",
        "strength",
    ):
        if candidate in name:
            constraint = candidate
            break
    if constraint is None:
        return {}

    solver = None
    lowered = name.lower()
    if "clarabel" in lowered:
        solver = "clarabel"
    elif "coord" in lowered or "coordinate" in lowered:
        solver = "coordinate"
    elif "degree-events" in lowered:
        solver = "scalar-ipf"
    elif ensemble in {"ME", "B"}:
        solver = "ipf"

    return {
        "ensemble": ensemble,
        "family": family,
        "constraint": constraint,
        "layers": 3 if "M=3" in name or "(M=3)" in name else None,
        "partial": False,
        "solver": solver,
    }


def run_fit(
    name,
    func,
    kwargs,
    n,
    s_out=None,
    s_in=None,
    k_out=None,
    k_in=None,
    verbose=0,
    metadata=None,
):
    if verbose >= 2:
        kwargs = {**kwargs, "verbose": 2}
    t0 = time.perf_counter()
    fit = func(**kwargs)
    elapsed = time.perf_counter() - t0
    diag = getattr(fit, "diagnostics", None)
    max_s = diag.max_strength_residual if diag and diag.max_strength_residual else None
    max_q = diag.conic.max_q if diag and diag.conic else None

    # Compute residuals post-hoc
    if max_s is None and s_out is not None:
        max_s = compute_strength_residual(fit, s_out, s_in)
    max_k = None
    if k_out is not None:
        max_k = compute_degree_residual(fit, k_out, k_in)

    result = {
        "name": name,
        "n": n,
        "time": elapsed,
        "converged": fit.converged,
        "iterations": fit.iterations,
        "max_strength_residual": max_s,
        "max_degree_residual": max_k,
        "max_q": max_q,
        "status": "converged" if fit.converged else "not_converged",
        "failure_reason": None
        if fit.converged
        else "solver did not report convergence",
        "expected_failure": False,
    }
    result.update(_infer_metadata(name))
    if metadata:
        result.update(metadata)
    status = "✓" if fit.converged else "✗"
    resid_s = f"{max_s:.2e}" if max_s is not None else "—"
    resid_k = f"{max_k:.2e}" if max_k is not None else "—"
    q_str = f"{max_q:.4f}" if max_q is not None else "—"
    print(
        f"  {status} {name:<38} N={n:<4} {elapsed:>7.2f}s "
        f"iter={fit.iterations:<5} s_res={resid_s:<10} k_res={resid_k:<10} q={q_str}"
    )
    sys.stdout.flush()
    return result


def parse_nodes(nodes):
    """Parse comma-separated benchmark sizes."""
    return [int(part) for part in nodes.split(",")] if nodes else None


def parse_floats(values):
    """Parse comma-separated floating-point values."""
    return [float(part) for part in values.split(",")] if values else None


def _benchmark_sizes(max_n, nodes=None):
    """Return requested benchmark sizes."""
    if nodes is not None:
        return sorted({int(n) for n in nodes if int(n) <= max_n})
    return sorted({n for n in [10, 25, 50, 100, 200, 500, 1000] if n <= max_n})


def bench_all(max_n=1000, tolerance=1e-4, verbose=0, nodes=None):
    from odme.models import (
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

    conic_max = min(max_n, 150)
    coord_max = max_n
    all_sizes = _benchmark_sizes(max_n, nodes)
    results = []

    for n in all_sizes:
        total = n * 6.0
        s_out, s_in = pareto_strengths(n, total)
        k_out, k_in = degrees_from_strengths(s_out, s_in)
        target_edges = n * 1.8
        total_events = int(total)

        print(
            f"\n{'=' * 80}\nN = {n}  (T = {total:.0f}, E_target = {target_edges:.0f})\n{'=' * 80}"
        )

        # --- ME (Poisson) ---
        results.append(
            run_fit(
                "ME strength",
                fit_strength_poisson,
                {"strength_out": s_out, "strength_in": s_in, "tolerance": tolerance},
                n,
                s_out,
                s_in,
                verbose=verbose,
            )
        )

        results.append(
            run_fit(
                "ME strength-edges",
                fit_strength_edges_poisson,
                {
                    "strength_out": s_out,
                    "strength_in": s_in,
                    "target_edges": target_edges,
                    "tolerance": tolerance,
                },
                n,
                s_out,
                s_in,
                verbose=verbose,
            )
        )

        results.append(
            run_fit(
                "ME strength-degree",
                fit_strength_degree_poisson,
                {
                    "strength_out": s_out,
                    "strength_in": s_in,
                    "degree_out": k_out,
                    "degree_in": k_in,
                    "tolerance": tolerance,
                },
                n,
                s_out,
                s_in,
                k_out,
                k_in,
                verbose=verbose,
            )
        )

        results.append(
            run_fit(
                "ME degree-events",
                fit_degree_events_poisson,
                {
                    "degree_out": k_out,
                    "degree_in": k_in,
                    "total_events": total_events,
                    "tolerance": tolerance,
                },
                n,
                k_out=k_out,
                k_in=k_in,
                verbose=verbose,
            )
        )

        if n <= 200:
            c_src, c_tgt, c_val = complete_costs(n)
            base = fit_strength_poisson(s_out, s_in)
            target_cost = float(np.sum(c_val.reshape(n, n) * np.outer(base.x, base.y)))
            results.append(
                run_fit(
                    "ME strength-cost",
                    fit_strength_cost_poisson,
                    {
                        "strength_out": s_out,
                        "strength_in": s_in,
                        "cost_sources": c_src,
                        "cost_targets": c_tgt,
                        "cost_values": c_val,
                        "target_cost": target_cost,
                        "tolerance": tolerance,
                    },
                    n,
                    s_out,
                    s_in,
                    verbose=verbose,
                )
            )

        # --- B (Binomial) ---
        results.append(
            run_fit(
                "B strength (M=3)",
                fit_strength_binomial,
                {
                    "strength_out": s_out,
                    "strength_in": s_in,
                    "layers": 3,
                    "tolerance": tolerance,
                },
                n,
                s_out,
                s_in,
                verbose=verbose,
            )
        )

        results.append(
            run_fit(
                "B strength-edges (M=3)",
                fit_strength_edges_binomial,
                {
                    "strength_out": s_out,
                    "strength_in": s_in,
                    "target_edges": target_edges,
                    "layers": 3,
                    "tolerance": tolerance,
                },
                n,
                s_out,
                s_in,
                verbose=verbose,
            )
        )

        results.append(
            run_fit(
                "B degree-events (M=3)",
                fit_degree_events_binomial,
                {
                    "degree_out": k_out,
                    "degree_in": k_in,
                    "total_events": total_events,
                    "layers": 3,
                    "tolerance": tolerance,
                },
                n,
                k_out=k_out,
                k_in=k_in,
                verbose=verbose,
            )
        )

        results.append(
            run_fit(
                "B strength-degree (M=3)",
                fit_strength_degree_binomial,
                {
                    "strength_out": s_out,
                    "strength_in": s_in,
                    "degree_out": k_out,
                    "degree_in": k_in,
                    "layers": 3,
                    "tolerance": tolerance,
                },
                n,
                s_out,
                s_in,
                k_out,
                k_in,
                verbose=verbose,
            )
        )

        if n <= 200:
            c_src, c_tgt, c_val = complete_costs(n)
            base_me = fit_strength_poisson(s_out, s_in)
            target_cost_me = float(
                np.sum(c_val.reshape(n, n) * np.outer(base_me.x, base_me.y))
            )
            results.append(
                run_fit(
                    "B strength-cost (M=3)",
                    fit_strength_cost_binomial,
                    {
                        "strength_out": s_out,
                        "strength_in": s_in,
                        "cost_sources": c_src,
                        "cost_targets": c_tgt,
                        "cost_values": c_val,
                        "target_cost": target_cost_me,
                        "layers": 3,
                        "tolerance": tolerance,
                    },
                    n,
                    s_out,
                    s_in,
                    verbose=verbose,
                )
            )

        # --- W (Geometric + negative binomial) ---
        for (
            family,
            layers,
            fit_strength,
            fit_cost,
            fit_edges,
            fit_degree,
            fit_events,
        ) in (
            (
                "geometric",
                None,
                fit_strength_geometric,
                fit_strength_cost_geometric,
                fit_strength_edges_geometric,
                fit_strength_degree_geometric,
                fit_degree_events_geometric,
            ),
            (
                "negative-binomial",
                3,
                fit_strength_negative_binomial,
                fit_strength_cost_negative_binomial,
                fit_strength_edges_negative_binomial,
                fit_strength_degree_negative_binomial,
                fit_degree_events_negative_binomial,
            ),
        ):
            layer_kwargs = {"layers": layers} if layers is not None else {}
            label_layers = f" M={layers}" if layers is not None else ""

            if n <= conic_max:
                results.append(
                    run_fit(
                        f"W {family} strength{label_layers} [clarabel]",
                        fit_strength,
                        {
                            "strength_out": s_out,
                            "strength_in": s_in,
                            "tolerance": tolerance,
                            **layer_kwargs,
                        },
                        n,
                        s_out,
                        s_in,
                        verbose=verbose,
                        metadata={
                            "ensemble": "W",
                            "family": family,
                            "constraint": "strength",
                            "layers": layers,
                            "partial": False,
                            "solver": "clarabel",
                        },
                    )
                )

                if n <= 100:
                    c_src, c_tgt, c_val = complete_costs(n)
                    base = fit_strength(
                        s_out, s_in, tolerance=tolerance, **layer_kwargs
                    )
                    target_cost = _target_cost_from_independent_fit(base, c_val, n)
                    results.append(
                        run_fit(
                            f"W {family} strength-cost{label_layers} [clarabel]",
                            fit_cost,
                            {
                                "strength_out": s_out,
                                "strength_in": s_in,
                                "cost_sources": c_src,
                                "cost_targets": c_tgt,
                                "cost_values": c_val,
                                "target_cost": target_cost,
                                "tolerance": tolerance,
                                **layer_kwargs,
                            },
                            n,
                            s_out,
                            s_in,
                            verbose=verbose,
                            metadata={
                                "ensemble": "W",
                                "family": family,
                                "constraint": "strength-cost",
                                "layers": layers,
                                "partial": False,
                                "solver": "clarabel",
                            },
                        )
                    )

            if n <= coord_max:
                results.append(
                    run_fit(
                        f"W {family} strength-edges{label_layers} [coordinate]",
                        fit_edges,
                        {
                            "strength_out": s_out,
                            "strength_in": s_in,
                            "target_edges": target_edges,
                            "tolerance": tolerance,
                            "max_iterations": 500,
                            **layer_kwargs,
                        },
                        n,
                        s_out,
                        s_in,
                        verbose=verbose,
                        metadata={
                            "ensemble": "W",
                            "family": family,
                            "constraint": "strength-edges",
                            "layers": layers,
                            "partial": False,
                            "solver": "coordinate",
                        },
                    )
                )

                results.append(
                    run_fit(
                        f"W {family} strength-degree{label_layers} [coordinate]",
                        fit_degree,
                        {
                            "strength_out": s_out,
                            "strength_in": s_in,
                            "degree_out": k_out,
                            "degree_in": k_in,
                            "tolerance": tolerance,
                            "max_iterations": 500,
                            **layer_kwargs,
                        },
                        n,
                        s_out,
                        s_in,
                        k_out,
                        k_in,
                        verbose=verbose,
                        metadata={
                            "ensemble": "W",
                            "family": family,
                            "constraint": "strength-degree",
                            "layers": layers,
                            "partial": False,
                            "solver": "coordinate",
                        },
                    )
                )

            results.append(
                run_fit(
                    f"W {family} degree-events{label_layers} [scalar-ipf]",
                    fit_events,
                    {
                        "degree_out": k_out,
                        "degree_in": k_in,
                        "total_events": total_events,
                        "tolerance": tolerance,
                        **layer_kwargs,
                    },
                    n,
                    k_out=k_out,
                    k_in=k_in,
                    verbose=verbose,
                    metadata={
                        "ensemble": "W",
                        "family": family,
                        "constraint": "degree-events",
                        "layers": layers,
                        "partial": False,
                        "solver": "scalar-ipf",
                    },
                )
            )

    return results


def _subtract_known_strengths(
    strength_out, strength_in, known_source, known_target, known_weight
):
    """Return residual strengths after fixed known weighted-pair contributions."""
    residual_out = np.asarray(strength_out, dtype=np.float64).copy()
    residual_in = np.asarray(strength_in, dtype=np.float64).copy()
    np.add.at(residual_out, known_source.astype(int), -known_weight)
    np.add.at(residual_in, known_target.astype(int), -known_weight)
    residual_out = np.maximum(residual_out, 0.0)
    residual_in = np.maximum(residual_in, 0.0)
    diff = residual_out.sum() - residual_in.sum()
    if abs(diff) > 1e-9:
        if diff > 0:
            residual_in[np.argmax(residual_in)] += diff
        else:
            residual_out[np.argmax(residual_out)] -= diff
    return residual_out, residual_in


def _partial_case_metadata(ensemble, family, constraint, layers, fraction, solver):
    """Structured metadata shared by all partial benchmark cases."""
    return {
        "ensemble": ensemble,
        "family": family,
        "constraint": constraint,
        "layers": layers,
        "partial": True,
        "known_weight_fraction": fraction,
        "solver": solver,
    }


def bench_partial(
    max_n=1000, tolerance=1e-4, verbose=0, nodes=None, known_fractions=None
):
    """Benchmark partial known-weight fitting across ME/B/W families."""
    from odme.models import (
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

    fractions = known_fractions or [0.05, 0.40]
    sizes = _benchmark_sizes(min(max_n, 200), nodes)
    family_specs = (
        (
            "ME",
            "poisson",
            None,
            {},
            fit_strength_poisson,
            fit_strength_edges_poisson,
            fit_strength_degree_poisson,
            fit_strength_cost_poisson,
            fit_degree_events_poisson,
        ),
        (
            "B",
            "binomial",
            3,
            {"layers": 3},
            fit_strength_binomial,
            fit_strength_edges_binomial,
            fit_strength_degree_binomial,
            fit_strength_cost_binomial,
            fit_degree_events_binomial,
        ),
        (
            "W",
            "geometric",
            None,
            {},
            fit_strength_geometric,
            fit_strength_edges_geometric,
            fit_strength_degree_geometric,
            fit_strength_cost_geometric,
            fit_degree_events_geometric,
        ),
        (
            "W",
            "negative-binomial",
            3,
            {"layers": 3},
            fit_strength_negative_binomial,
            fit_strength_edges_negative_binomial,
            fit_strength_degree_negative_binomial,
            fit_strength_cost_negative_binomial,
            fit_degree_events_negative_binomial,
        ),
    )
    results = []

    for n in sizes:
        total = n * 6.0
        strength_out, strength_in = pareto_strengths(n, total)
        base = fit_strength_poisson(strength_out, strength_in)
        capacity = n * n
        for fraction in fractions:
            known_source, known_target, known_weight = _known_pairs_from_random_rates(
                base, n, fraction, seed=10_000 + n + int(fraction * 1000)
            )
            residual_out, residual_in = _subtract_known_strengths(
                strength_out, strength_in, known_source, known_target, known_weight
            )
            residual_total = float(residual_out.sum())
            known_edges = float(len(known_source))
            residual_edges = max(1.0, min(n * 1.8, residual_total * 0.10))
            residual_degree_out, residual_degree_in = degrees_from_strengths(
                residual_out, residual_in, frac=2.0
            )
            residual_events = int(
                max(np.ceil(residual_total), np.ceil(residual_degree_out.sum() + 1.0))
            )
            cost_sources, cost_targets, cost_values = complete_costs(n)

            print(
                f"\n--- Partial known-weight fitting N={n} "
                f"known={fraction:.0%} ({len(known_source)}/{capacity}) ---"
            )

            for (
                ensemble,
                family,
                layers,
                layer_kwargs,
                fit_strength,
                fit_edges,
                fit_degree,
                fit_cost,
                fit_events,
            ) in family_specs:
                label_layers = f" M={layers}" if layers is not None else ""
                name_prefix = f"partial {ensemble} {family}"

                results.append(
                    run_fit(
                        f"{name_prefix} strength{label_layers}",
                        fit_strength,
                        {
                            "strength_out": residual_out,
                            "strength_in": residual_in,
                            "tolerance": tolerance,
                            **layer_kwargs,
                        },
                        n,
                        residual_out,
                        residual_in,
                        verbose=verbose,
                        metadata=_partial_case_metadata(
                            ensemble,
                            family,
                            "strength",
                            layers,
                            fraction,
                            "known-weight",
                        ),
                    )
                )

                results.append(
                    run_fit(
                        f"{name_prefix} strength-edges{label_layers}",
                        fit_edges,
                        {
                            "strength_out": residual_out,
                            "strength_in": residual_in,
                            "target_edges": residual_edges,
                            "tolerance": tolerance,
                            **layer_kwargs,
                        },
                        n,
                        residual_out,
                        residual_in,
                        verbose=verbose,
                        metadata={
                            **_partial_case_metadata(
                                ensemble,
                                family,
                                "strength-edges",
                                layers,
                                fraction,
                                "known-weight",
                            ),
                            "known_edges": known_edges,
                            "full_target_edges": known_edges + residual_edges,
                        },
                    )
                )

                results.append(
                    run_fit(
                        f"{name_prefix} strength-degree{label_layers}",
                        fit_degree,
                        {
                            "strength_out": residual_out,
                            "strength_in": residual_in,
                            "degree_out": residual_degree_out,
                            "degree_in": residual_degree_in,
                            "tolerance": tolerance,
                            "max_iterations": 500,
                            **layer_kwargs,
                        },
                        n,
                        residual_out,
                        residual_in,
                        residual_degree_out,
                        residual_degree_in,
                        verbose=verbose,
                        metadata=_partial_case_metadata(
                            ensemble,
                            family,
                            "strength-degree",
                            layers,
                            fraction,
                            "known-weight",
                        ),
                    )
                )

                cost_strength_fit = (
                    fit_strength_poisson if family == "binomial" else fit_strength
                )
                cost_strength_kwargs = {} if family == "binomial" else layer_kwargs
                independent = cost_strength_fit(
                    residual_out,
                    residual_in,
                    tolerance=tolerance,
                    **cost_strength_kwargs,
                )
                target_cost = _target_cost_from_independent_fit(
                    independent, cost_values, n
                )
                results.append(
                    run_fit(
                        f"{name_prefix} strength-cost{label_layers}",
                        fit_cost,
                        {
                            "strength_out": residual_out,
                            "strength_in": residual_in,
                            "cost_sources": cost_sources,
                            "cost_targets": cost_targets,
                            "cost_values": cost_values,
                            "target_cost": target_cost,
                            "tolerance": tolerance,
                            **layer_kwargs,
                        },
                        n,
                        residual_out,
                        residual_in,
                        verbose=verbose,
                        metadata=_partial_case_metadata(
                            ensemble,
                            family,
                            "strength-cost",
                            layers,
                            fraction,
                            "known-weight",
                        ),
                    )
                )

                results.append(
                    run_fit(
                        f"{name_prefix} degree-events{label_layers}",
                        fit_events,
                        {
                            "degree_out": residual_degree_out,
                            "degree_in": residual_degree_in,
                            "total_events": residual_events,
                            "tolerance": tolerance,
                            **layer_kwargs,
                        },
                        n,
                        k_out=residual_degree_out,
                        k_in=residual_degree_in,
                        verbose=verbose,
                        metadata=_partial_case_metadata(
                            ensemble,
                            family,
                            "degree-events",
                            layers,
                            fraction,
                            "known-weight",
                        ),
                    )
                )

    return results


def plot_results(results, output_dir: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    solvers = {}
    for r in results:
        name = r["name"]
        if name not in solvers:
            solvers[name] = {"n": [], "time": [], "resid": []}
        solvers[name]["n"].append(r["n"])
        solvers[name]["time"].append(r["time"])
        solvers[name]["resid"].append(r["max_strength_residual"])

    fig, ax = plt.subplots(figsize=(12, 7))
    for name, data in sorted(solvers.items()):
        if max(data["time"]) > 0.001:
            ax.plot(data["n"], data["time"], "o-", label=name, markersize=4)
    ax.set_xlabel("N (nodes)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Fitting time vs network size (release mode)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fitting_scaling.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot: {output_dir / 'fitting_scaling.png'}")

    solvers_r = {
        k: v for k, v in solvers.items() if any(r is not None for r in v["resid"])
    }
    if solvers_r:
        fig, ax = plt.subplots(figsize=(12, 7))
        for name, data in sorted(solvers_r.items()):
            rs = [r for r in data["resid"] if r is not None]
            ns = [data["n"][i] for i, r in enumerate(data["resid"]) if r is not None]
            if rs:
                ax.plot(ns, rs, "o-", label=name, markersize=4)
        ax.set_xlabel("N (nodes)")
        ax.set_ylabel("Max strength residual")
        ax.set_title("Constraint recovery vs network size")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "fitting_residuals.png", dpi=150)
        plt.close(fig)
        print(f"Plot: {output_dir / 'fitting_residuals.png'}")


def save_results(results, output_dir="benchmarks/results"):
    """Save benchmark results to JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "fitting_benchmark.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults: {out / 'fitting_benchmark.json'}")


def print_summary(results):
    """Print results summary table."""
    print(f"\n{'=' * 80}\nSUMMARY\n{'=' * 80}")
    print(
        f"{'Solver':<40} {'N':>4} {'Time':>7} {'Conv':>4} {'s_resid':>10} {'k_resid':>10}"
    )
    print("-" * 80)
    for r in results:
        sr = (
            f"{r['max_strength_residual']:.1e}"
            if r["max_strength_residual"]
            else "\u2014"
        )
        kr = (
            f"{r['max_degree_residual']:.1e}"
            if r.get("max_degree_residual")
            else "\u2014"
        )
        conv = "✓" if r["converged"] else "✗"
        print(
            f"{r['name']:<40} {r['n']:>4} {r['time']:>6.2f}s "
            f"{conv:>4} {sr:>10} {kr:>10}"
        )
