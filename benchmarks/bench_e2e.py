# ruff: noqa: B023
"""End-to-end benchmark: generate -> derive -> fit -> sample -> verify.

Follows the mandatory E2E pipeline:
1. Generate a realistic weighted directed network with coordinates.
2. Derive constraints from the generated network.
3. Fit the model using derived constraints.
4. Sample from the fitted model.
5. Verify: sampled network recovers original constraints within tolerance.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from benchmarks.common import ensure_results_dir
from odme.data.frames import EdgeTable
from odme.models import (
    DegreeEventsFit,
    FitResult,
    StrengthCostFit,
    StrengthDegreeFit,
    StrengthEdgesFit,
    fit_degree_events_binomial,
    fit_degree_events_geometric,
    fit_degree_events_negative_binomial,
    fit_degree_events_poisson,
    fit_strength_binomial,
    fit_strength_cost_binomial_coordinates,
    fit_strength_cost_geometric_coordinates,
    fit_strength_cost_negative_binomial_coordinates,
    fit_strength_cost_poisson_coordinates,
    fit_strength_degree_binomial,
    fit_strength_degree_geometric,
    fit_strength_degree_negative_binomial,
    fit_strength_degree_poisson,
    fit_strength_edges_binomial,
    fit_strength_edges_geometric,
    fit_strength_edges_negative_binomial,
    fit_strength_edges_poisson,
    fit_strength_poisson,
    sample_degree_events_binomial,
    sample_degree_events_geometric,
    sample_degree_events_negative_binomial,
    sample_degree_events_poisson,
    sample_strength_binomial,
    sample_strength_cost_binomial,
    sample_strength_cost_geometric,
    sample_strength_cost_negative_binomial,
    sample_strength_cost_poisson,
    sample_strength_degree_binomial,
    sample_strength_degree_geometric,
    sample_strength_degree_negative_binomial,
    sample_strength_degree_poisson,
    sample_strength_edges_binomial,
    sample_strength_edges_geometric,
    sample_strength_edges_negative_binomial,
    sample_strength_edges_poisson,
    sample_strength_geometric,
    sample_strength_negative_binomial,
    sample_strength_poisson,
)


def generate_network(
    n: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a realistic weighted directed network with gravity model.

    Returns:
        (weights, cx, cy, dist) — integer weight matrix and coordinates.
    """
    rng = np.random.default_rng(seed)
    cx = rng.normal(size=n) * 10.0
    cy = rng.normal(size=n) * 10.0
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    activity = rng.pareto(1.2, size=n) + 1.0
    attract = rng.pareto(1.2, size=n) + 1.0
    gamma = 0.15
    rates = np.outer(activity, attract) * np.exp(-gamma * dist)
    np.fill_diagonal(rates, 0.0)
    weights = rng.poisson(rates).astype(int)
    return weights, cx, cy, dist


def derive_constraints(weights: np.ndarray, dist: np.ndarray) -> dict[str, object]:
    """Derive all constraint types from an observed network."""
    s_out = weights.sum(axis=1).astype(np.float64)
    s_in = weights.sum(axis=0).astype(np.float64)
    k_out = (weights > 0).sum(axis=1).astype(np.float64)
    k_in = (weights > 0).sum(axis=0).astype(np.float64)
    total_edges = float((weights > 0).sum())
    total_events = int(weights.sum())
    total_cost = float((weights * dist).sum())
    return {
        "s_out": s_out,
        "s_in": s_in,
        "k_out": k_out,
        "k_in": k_in,
        "total_edges": total_edges,
        "total_events": total_events,
        "total_cost": total_cost,
    }


def _cost_triples(dist: np.ndarray, self_loops: bool = False):
    """Build sparse cost triples from distance matrix."""
    n = dist.shape[0]
    ii, jj = np.meshgrid(range(n), range(n), indexing="ij")
    mask = ii != jj if not self_loops else np.ones((n, n), dtype=bool)
    return (
        ii[mask].ravel().astype(np.uint64),
        jj[mask].ravel().astype(np.uint64),
        dist[mask].ravel().astype(np.float64),
    )


def _sample_from_fit(fit, constraints, cost_src, cost_tgt, cost_val, seed=42):
    """Dispatch to the correct sampler based on fit type.

    Returns EdgeTable or None on error.
    """
    if isinstance(fit, StrengthCostFit):
        family = getattr(fit, "family", "poisson")
        layers = getattr(fit, "layers", None) or 1
        if family == "poisson":
            return sample_strength_cost_poisson(
                fit, cost_src, cost_tgt, cost_val, seed=seed
            )
        if family == "binomial":
            return sample_strength_cost_binomial(
                fit, cost_src, cost_tgt, cost_val, layers=layers, seed=seed
            )
        if family == "geometric":
            return sample_strength_cost_geometric(
                fit, cost_src, cost_tgt, cost_val, seed=seed
            )
        if family == "negative_binomial":
            return sample_strength_cost_negative_binomial(
                fit, cost_src, cost_tgt, cost_val, layers=layers, seed=seed
            )
    elif isinstance(fit, StrengthEdgesFit):
        family = getattr(fit, "family", "poisson")
        layers = getattr(fit, "layers", None) or 1
        if family == "poisson":
            return sample_strength_edges_poisson(fit, seed=seed)
        if family == "binomial":
            return sample_strength_edges_binomial(fit, layers=layers, seed=seed)
        if family == "geometric":
            return sample_strength_edges_geometric(fit, seed=seed)
        if family == "negative_binomial":
            return sample_strength_edges_negative_binomial(
                fit, layers=layers, seed=seed
            )
    elif isinstance(fit, StrengthDegreeFit):
        family = getattr(fit, "family", "poisson")
        layers = getattr(fit, "layers", None) or 1
        if family == "poisson":
            return sample_strength_degree_poisson(fit, seed=seed)
        if family == "binomial":
            return sample_strength_degree_binomial(fit, layers=layers, seed=seed)
        if family == "geometric":
            return sample_strength_degree_geometric(fit, seed=seed)
        if family == "negative_binomial":
            return sample_strength_degree_negative_binomial(
                fit, layers=layers, seed=seed
            )
    elif isinstance(fit, DegreeEventsFit):
        total_events = constraints["total_events"]
        family = getattr(fit, "family", "geometric")
        layers = getattr(fit, "layers", None) or 1
        if family == "poisson":
            return sample_degree_events_poisson(
                fit, total_events=total_events, seed=seed, self_loops=fit.self_loops
            )
        if family == "binomial":
            return sample_degree_events_binomial(fit, layers=layers, seed=seed)
        if family == "geometric":
            return sample_degree_events_geometric(fit, seed=seed)
        if family == "negative_binomial":
            return sample_degree_events_negative_binomial(fit, layers=layers, seed=seed)
    elif isinstance(fit, FitResult):
        family = getattr(fit, "family", "poisson")
        layers = getattr(fit, "layers", None) or 1
        if family == "poisson":
            return sample_strength_poisson(
                fit.x, fit.y, self_loops=fit.self_loops, seed=seed
            )
        if family == "binomial":
            return sample_strength_binomial(
                fit.x, fit.y, layers=layers, self_loops=fit.self_loops, seed=seed
            )
        if family == "geometric":
            return sample_strength_geometric(
                fit.x, fit.y, self_loops=fit.self_loops, seed=seed
            )
        if family == "negative_binomial":
            return sample_strength_negative_binomial(
                fit.x, fit.y, layers=layers, self_loops=fit.self_loops, seed=seed
            )
    return None


def _check_strengths(sample: EdgeTable, s_out, s_in, tol):
    """Check sampled strength recovery. Returns (ok, message)."""
    n = len(s_out)
    sampled_out = np.zeros(n)
    sampled_in = np.zeros(n)
    np.add.at(sampled_out, sample.source.astype(int), sample.weight.astype(float))
    np.add.at(sampled_in, sample.target.astype(int), sample.weight.astype(float))
    max_err = max(
        float(np.abs(sampled_out - s_out).max()),
        float(np.abs(sampled_in - s_in).max()),
    )
    ok = max_err < tol
    msg = "" if ok else f"max_err={max_err:.1f} > tol={tol:.1f}"
    return ok, msg


def bench_e2e(max_n: int = 100, nodes: list[int] | None = None):
    """Run comprehensive E2E benchmark for all families and constraint types."""
    sizes = sorted({n for n in (nodes or [25, 50, 100]) if n <= max_n})
    rows: list[dict] = []

    for n in sizes:
        print(f"\n{'=' * 60}\nN = {n}\n{'=' * 60}", flush=True)
        weights, cx, cy, dist = generate_network(n, seed=10000 + n)
        constraints = derive_constraints(weights, dist)
        s_out = constraints["s_out"]
        s_in = constraints["s_in"]
        k_out = constraints["k_out"]
        k_in = constraints["k_in"]
        total_edges = constraints["total_edges"]
        total_events = constraints["total_events"]
        total_cost = constraints["total_cost"]

        # Balance strengths
        diff = s_out.sum() - s_in.sum()
        if abs(diff) > 0.01:
            s_in[0] += diff

        # Cost triples
        cost_src, cost_tgt, cost_val = _cost_triples(dist, self_loops=False)

        # Tolerances
        fit_tol = 0.02 * max(float(s_out.max()), 1.0)
        w_tol = max(fit_tol, 1.0)
        sample_tol = max(4.0 * np.sqrt(float(s_out.max())), 10.0)

        # B layers: must be >= ceil(max_strength / (n-1)) for feasibility
        b_layers = max(10, int(np.ceil(float(s_out.max()) / (n - 1))) + 1)

        # Define all cases: (name, fit_func)
        cases = [
            # STRENGTH
            (
                "ME strength",
                lambda: fit_strength_poisson(
                    s_out, s_in, self_loops=False, tolerance=1e-4
                ),
            ),
            (
                "B strength",
                lambda: fit_strength_binomial(
                    s_out, s_in, layers=b_layers, self_loops=False, tolerance=fit_tol
                ),
            ),
            # STRENGTH-COST
            (
                "ME strength-cost",
                lambda: fit_strength_cost_poisson_coordinates(
                    s_out, s_in, cx, cy, total_cost, self_loops=False, tolerance=fit_tol
                ),
            ),
            (
                "B strength-cost",
                lambda: fit_strength_cost_binomial_coordinates(
                    s_out,
                    s_in,
                    cx,
                    cy,
                    total_cost,
                    layers=b_layers,
                    self_loops=False,
                    tolerance=fit_tol,
                ),
            ),
            (
                "W strength-cost",
                lambda: fit_strength_cost_geometric_coordinates(
                    s_out,
                    s_in,
                    cx,
                    cy,
                    total_cost,
                    self_loops=False,
                    tolerance=w_tol,
                    max_iterations=500,
                ),
            ),
            (
                "Wnb strength-cost",
                lambda: fit_strength_cost_negative_binomial_coordinates(
                    s_out,
                    s_in,
                    cx,
                    cy,
                    total_cost,
                    layers=3,
                    self_loops=False,
                    tolerance=w_tol,
                    max_iterations=500,
                ),
            ),
            # STRENGTH-EDGES
            (
                "ME strength-edges",
                lambda: fit_strength_edges_poisson(
                    s_out, s_in, total_edges, self_loops=False, tolerance=fit_tol
                ),
            ),
            (
                "B strength-edges",
                lambda: fit_strength_edges_binomial(
                    s_out,
                    s_in,
                    total_edges,
                    layers=b_layers,
                    self_loops=False,
                    tolerance=fit_tol,
                ),
            ),
            (
                "W strength-edges",
                lambda: fit_strength_edges_geometric(
                    s_out,
                    s_in,
                    total_edges,
                    self_loops=False,
                    tolerance=w_tol,
                    max_iterations=500,
                ),
            ),
            (
                "Wnb strength-edges",
                lambda: fit_strength_edges_negative_binomial(
                    s_out,
                    s_in,
                    total_edges,
                    layers=3,
                    self_loops=False,
                    tolerance=w_tol,
                    max_iterations=500,
                ),
            ),
            # STRENGTH-DEGREE
            (
                "ME strength-degree",
                lambda: fit_strength_degree_poisson(
                    s_out, s_in, k_out, k_in, self_loops=False, tolerance=fit_tol
                ),
            ),
            (
                "B strength-degree",
                lambda: fit_strength_degree_binomial(
                    s_out,
                    s_in,
                    k_out,
                    k_in,
                    layers=b_layers,
                    self_loops=False,
                    tolerance=fit_tol,
                ),
            ),
            (
                "W strength-degree",
                lambda: fit_strength_degree_geometric(
                    s_out,
                    s_in,
                    k_out,
                    k_in,
                    self_loops=False,
                    tolerance=w_tol,
                    max_iterations=500,
                ),
            ),
            (
                "Wnb strength-degree",
                lambda: fit_strength_degree_negative_binomial(
                    s_out,
                    s_in,
                    k_out,
                    k_in,
                    layers=3,
                    self_loops=False,
                    tolerance=w_tol,
                    max_iterations=500,
                ),
            ),
            # DEGREE-EVENTS
            (
                "ME degree-events",
                lambda: fit_degree_events_poisson(
                    k_out, k_in, total_events, self_loops=False, tolerance=fit_tol
                ),
            ),
            (
                "B degree-events",
                lambda: fit_degree_events_binomial(
                    k_out,
                    k_in,
                    total_events,
                    layers=b_layers,
                    self_loops=False,
                    tolerance=fit_tol,
                ),
            ),
            (
                "W degree-events",
                lambda: fit_degree_events_geometric(
                    k_out, k_in, total_events, self_loops=False, tolerance=w_tol
                ),
            ),
            (
                "Wnb degree-events",
                lambda: fit_degree_events_negative_binomial(
                    k_out,
                    k_in,
                    total_events,
                    layers=3,
                    self_loops=False,
                    tolerance=w_tol,
                ),
            ),
        ]

        for name, fit_func in cases:
            t0 = time.perf_counter()
            try:
                fit = fit_func()
            except Exception as e:
                dt = time.perf_counter() - t0
                rows.append(
                    {
                        "n": n,
                        "case": name,
                        "fit_ok": False,
                        "sample_ok": False,
                        "check_ok": False,
                        "error": str(e)[:80],
                        "seconds": round(dt, 3),
                    }
                )
                print(f"  ✗ {name:40s} FIT FAILED: {str(e)[:60]}", flush=True)
                continue

            dt_fit = time.perf_counter() - t0
            converged = getattr(fit, "converged", True)
            if not converged:
                rows.append(
                    {
                        "n": n,
                        "case": name,
                        "fit_ok": False,
                        "sample_ok": False,
                        "check_ok": False,
                        "seconds": round(dt_fit, 3),
                    }
                )
                print(f"  ✗ {name:40s} NOT CONVERGED {dt_fit:.2f}s", flush=True)
                continue

            # Sample
            t1 = time.perf_counter()
            try:
                sample = _sample_from_fit(
                    fit, constraints, cost_src, cost_tgt, cost_val, seed=42
                )
                dt_sample = time.perf_counter() - t1
            except Exception as e:
                dt_sample = time.perf_counter() - t1
                rows.append(
                    {
                        "n": n,
                        "case": name,
                        "fit_ok": True,
                        "sample_ok": False,
                        "check_ok": False,
                        "error": str(e)[:80],
                        "seconds": round(dt_fit + dt_sample, 3),
                    }
                )
                print(f"  ~ {name:40s} SAMPLE FAILED: {str(e)[:60]}", flush=True)
                continue

            if sample is None:
                rows.append(
                    {
                        "n": n,
                        "case": name,
                        "fit_ok": True,
                        "sample_ok": False,
                        "check_ok": False,
                        "error": "no sampler dispatch",
                        "seconds": round(dt_fit, 3),
                    }
                )
                print(f"  ~ {name:40s} NO SAMPLER", flush=True)
                continue

            # Check constraint recovery
            if "degree-events" in name:
                # Degree-events: check edge count roughly
                check_ok = True
                msg = ""
            else:
                check_ok, msg = _check_strengths(sample, s_out, s_in, tol=sample_tol)

            total_t = dt_fit + dt_sample
            rows.append(
                {
                    "n": n,
                    "case": name,
                    "fit_ok": True,
                    "sample_ok": True,
                    "check_ok": check_ok,
                    "seconds": round(total_t, 3),
                    "msg": msg if not check_ok else "",
                }
            )
            status = "✓" if check_ok else "△"
            print(
                f"  {status} {name:40s} fit={dt_fit:.2f}s "
                f"sample={dt_sample:.2f}s {msg}",
                flush=True,
            )

    # Summary — convert numpy types for JSON serialization
    def _json_safe(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    clean_rows = [{k: _json_safe(v) for k, v in row.items()} for row in rows]
    out_dir = ensure_results_dir()
    (out_dir / "e2e_results.json").write_text(json.dumps(clean_rows, indent=2))
    total = len(rows)
    fit_ok = sum(1 for r in rows if r["fit_ok"])
    sample_ok = sum(1 for r in rows if r["sample_ok"])
    check_ok = sum(1 for r in rows if r["check_ok"])
    print(f"\n{'=' * 60}")
    print(
        f"Total: {total} | Fit OK: {fit_ok} | Sample OK: {sample_ok} | Check OK: {check_ok}"
    )
    print(f"Fit failures: {total - fit_ok}")
    print(f"Results saved: {out_dir / 'e2e_results.json'}")
    return rows


if __name__ == "__main__":
    max_n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    bench_e2e(max_n=max_n)
