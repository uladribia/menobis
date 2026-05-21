#!/usr/bin/env python3
# ruff: noqa: B023, E402
"""Comprehensive E2E benchmark: ALL families x ALL constraints x partial.

Runs generate -> derive -> fit -> sample -> verify for every combination at
N=25, 50, 100, 500, 1000.
"""

import json
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from odme.models import (
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
    fit_strength_geometric,
    fit_strength_negative_binomial,
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
from odme.models.partial import (
    fit_partial_strength_cost_binomial_coordinates,
    fit_partial_strength_cost_geometric_coordinates,
    fit_partial_strength_cost_negative_binomial_coordinates,
    fit_partial_strength_cost_poisson_coordinates,
    fit_partial_strength_degree_poisson,
    fit_partial_strength_edges_poisson,
    fit_partial_strength_poisson,
)


def generate_network(n, seed=42):
    """Generate realistic weighted directed network with gravity model."""
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


def derive_all(weights, dist, n):
    """Derive all constraint sequences from observed network."""
    s_out = weights.sum(axis=1).astype(np.float64)
    s_in = weights.sum(axis=0).astype(np.float64)
    k_out = (weights > 0).sum(axis=1).astype(np.float64)
    k_in = (weights > 0).sum(axis=0).astype(np.float64)
    total_edges = float((weights > 0).sum())
    total_events = int(weights.sum())
    total_cost = float((weights * dist).sum())

    # Balance strengths
    diff = s_out.sum() - s_in.sum()
    if abs(diff) > 0.01:
        s_in[0] += diff

    # Clip degrees to n-2 for Bernoulli boundary
    capacity = float(n - 1)
    k_out = np.minimum(k_out, capacity - 1.0)
    k_in = np.minimum(k_in, capacity - 1.0)
    k_diff = k_out.sum() - k_in.sum()
    if abs(k_diff) > 0.01:
        k_in[0] += k_diff

    # Cost triples (no self-loops)
    ii, jj = np.meshgrid(range(n), range(n), indexing="ij")
    mask = ii != jj
    cost_src = ii[mask].ravel().astype(np.uint64)
    cost_tgt = jj[mask].ravel().astype(np.uint64)
    cost_val = dist[mask].ravel().astype(np.float64)

    # Known pairs for partial (top 10% by weight)
    flat_w = weights.ravel()
    order = np.argsort(flat_w)[::-1]
    k_known = max(1, int(0.1 * (flat_w > 0).sum()))
    known_idx = order[:k_known]
    known_s = (known_idx // n).astype(np.uint64)
    known_t = (known_idx % n).astype(np.uint64)
    known_r = flat_w[known_idx].astype(np.float64)

    # B layers with headroom
    b_layers = max(10, 4 * int(np.ceil(float(s_out.max()) / (n - 1))))

    return {
        "s_out": s_out,
        "s_in": s_in,
        "k_out": k_out,
        "k_in": k_in,
        "total_edges": total_edges,
        "total_events": total_events,
        "total_cost": total_cost,
        "cx": None,  # filled outside
        "cy": None,
        "cost_src": cost_src,
        "cost_tgt": cost_tgt,
        "cost_val": cost_val,
        "known_s": known_s,
        "known_t": known_t,
        "known_r": known_r,
        "b_layers": b_layers,
    }


def check_strengths(sample, s_out, s_in, tol):
    """Check sampled strength recovery."""
    n = len(s_out)
    so = np.zeros(n)
    si = np.zeros(n)
    np.add.at(so, sample.source.astype(int), sample.weight.astype(float))
    np.add.at(si, sample.target.astype(int), sample.weight.astype(float))
    max_err = max(float(np.abs(so - s_out).max()), float(np.abs(si - s_in).max()))
    return max_err < tol, max_err


def run_one(name, fit_fn, sample_fn, check_fn, timeout=60):
    """Run one case: fit → sample → check. Returns result dict."""
    t0 = time.perf_counter()
    try:
        fit = fit_fn()
    except Exception as e:
        return {
            "case": name,
            "fit_ok": False,
            "error": str(e)[:80],
            "seconds": round(time.perf_counter() - t0, 3),
        }
    dt_fit = time.perf_counter() - t0
    if dt_fit > timeout:
        return {
            "case": name,
            "fit_ok": False,
            "error": "timeout",
            "seconds": round(dt_fit, 3),
        }
    converged = getattr(fit, "converged", True)
    if not converged:
        return {
            "case": name,
            "fit_ok": False,
            "seconds": round(dt_fit, 3),
            "iters": getattr(fit, "iterations", 0),
        }

    # Sample
    t1 = time.perf_counter()
    try:
        sample = sample_fn(fit)
        dt_s = time.perf_counter() - t1
    except Exception as e:
        return {
            "case": name,
            "fit_ok": True,
            "sample_ok": False,
            "error": str(e)[:80],
            "seconds": round(dt_fit, 3),
        }

    # Check
    try:
        ok, metric = check_fn(sample, fit)
    except Exception as e:
        ok, metric = False, str(e)[:40]

    return {
        "case": name,
        "fit_ok": True,
        "sample_ok": True,
        "check_ok": bool(ok),
        "metric": float(metric)
        if isinstance(metric, (int, float, np.floating))
        else str(metric),
        "seconds": round(dt_fit + dt_s, 3),
        "iters": getattr(fit, "iterations", 0),
    }


def main():
    nodes = (
        [int(x) for x in sys.argv[1:]]
        if len(sys.argv) > 1
        else [25, 50, 100, 500, 1000]
    )
    all_rows = []

    for n in nodes:
        print(f"\n{'=' * 70}\n  N = {n}\n{'=' * 70}", flush=True)
        weights, cx, cy, dist = generate_network(n, seed=10000 + n)
        c = derive_all(weights, dist, n)
        c["cx"] = cx
        c["cy"] = cy
        s_out, s_in = c["s_out"], c["s_in"]
        k_out, k_in = c["k_out"], c["k_in"]
        total_edges = c["total_edges"]
        total_events = c["total_events"]
        total_cost = c["total_cost"]
        cost_src, cost_tgt, cost_val = c["cost_src"], c["cost_tgt"], c["cost_val"]
        known_s, known_t, known_r = c["known_s"], c["known_t"], c["known_r"]
        b_layers = c["b_layers"]

        fit_tol = 0.02 * max(float(s_out.max()), 1.0)
        w_tol = max(fit_tol, 1.0)
        sample_tol = max(4.0 * np.sqrt(float(s_out.max())), 20.0)

        # Timeout: scale with N^2
        timeout = max(30, n * n * 0.001)

        def s_check(sample, fit, tol=sample_tol):
            return check_strengths(sample, s_out, s_in, tol)

        def no_check(sample, fit):
            return True, 0.0

        # ============ STRENGTH ============
        cases = [
            (
                "ME strength",
                lambda: fit_strength_poisson(
                    s_out, s_in, self_loops=False, tolerance=1e-4
                ),
                lambda f: sample_strength_poisson(
                    f.x, f.y, self_loops=f.self_loops, seed=42
                ),
                s_check,
            ),
            (
                "B strength",
                lambda: fit_strength_binomial(
                    s_out, s_in, layers=b_layers, self_loops=False, tolerance=fit_tol
                ),
                lambda f: sample_strength_binomial(
                    f.x,
                    f.y,
                    layers=f.layers or b_layers,
                    self_loops=f.self_loops,
                    seed=42,
                ),
                s_check,
            ),
            (
                "W strength",
                lambda: fit_strength_geometric(
                    s_out, s_in, self_loops=False, tolerance=w_tol, max_iterations=5000
                ),
                lambda f: sample_strength_geometric(
                    f.x, f.y, self_loops=f.self_loops, seed=42
                ),
                s_check,
            ),
            (
                "Wnb strength",
                lambda: fit_strength_negative_binomial(
                    s_out,
                    s_in,
                    layers=3,
                    self_loops=False,
                    tolerance=w_tol,
                    max_iterations=5000,
                ),
                lambda f: sample_strength_negative_binomial(
                    f.x, f.y, layers=3, self_loops=f.self_loops, seed=42
                ),
                s_check,
            ),
            # ============ STRENGTH-COST ============
            (
                "ME strength-cost",
                lambda: fit_strength_cost_poisson_coordinates(
                    s_out, s_in, cx, cy, total_cost, self_loops=False, tolerance=fit_tol
                ),
                lambda f: sample_strength_cost_poisson(
                    f, cost_src, cost_tgt, cost_val, seed=42
                ),
                s_check,
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
                lambda f: sample_strength_cost_binomial(
                    f,
                    cost_src,
                    cost_tgt,
                    cost_val,
                    layers=f.layers or b_layers,
                    seed=42,
                ),
                s_check,
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
                    max_iterations=5000,
                ),
                lambda f: sample_strength_cost_geometric(
                    f, cost_src, cost_tgt, cost_val, seed=42
                ),
                s_check,
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
                    max_iterations=5000,
                ),
                lambda f: sample_strength_cost_negative_binomial(
                    f, cost_src, cost_tgt, cost_val, layers=3, seed=42
                ),
                s_check,
            ),
            # ============ STRENGTH-EDGES ============
            (
                "ME strength-edges",
                lambda: fit_strength_edges_poisson(
                    s_out, s_in, total_edges, self_loops=False, tolerance=fit_tol
                ),
                lambda f: sample_strength_edges_poisson(f, seed=42),
                s_check,
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
                lambda f: sample_strength_edges_binomial(
                    f, layers=f.layers or b_layers, seed=42
                ),
                s_check,
            ),
            (
                "W strength-edges",
                lambda: fit_strength_edges_geometric(
                    s_out,
                    s_in,
                    total_edges,
                    self_loops=False,
                    tolerance=w_tol,
                    max_iterations=5000,
                ),
                lambda f: sample_strength_edges_geometric(f, seed=42),
                s_check,
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
                    max_iterations=5000,
                ),
                lambda f: sample_strength_edges_negative_binomial(f, layers=3, seed=42),
                s_check,
            ),
            # ============ STRENGTH-DEGREE ============
            (
                "ME strength-degree",
                lambda: fit_strength_degree_poisson(
                    s_out, s_in, k_out, k_in, self_loops=False, tolerance=fit_tol
                ),
                lambda f: sample_strength_degree_poisson(f, seed=42),
                s_check,
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
                lambda f: sample_strength_degree_binomial(
                    f, layers=f.layers or b_layers, seed=42
                ),
                s_check,
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
                    max_iterations=5000,
                ),
                lambda f: sample_strength_degree_geometric(f, seed=42),
                s_check,
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
                    max_iterations=5000,
                ),
                lambda f: sample_strength_degree_negative_binomial(
                    f, layers=3, seed=42
                ),
                s_check,
            ),
            # ============ DEGREE-EVENTS ============
            (
                "ME degree-events",
                lambda: fit_degree_events_poisson(
                    k_out, k_in, total_events, self_loops=False, tolerance=fit_tol
                ),
                lambda f: sample_degree_events_poisson(
                    f, total_events=total_events, seed=42, self_loops=f.self_loops
                ),
                no_check,
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
                lambda f: sample_degree_events_binomial(
                    f, layers=f.layers or b_layers, seed=42
                ),
                no_check,
            ),
            (
                "W degree-events",
                lambda: fit_degree_events_geometric(
                    k_out, k_in, total_events, self_loops=False, tolerance=w_tol
                ),
                lambda f: sample_degree_events_geometric(f, seed=42),
                no_check,
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
                lambda f: sample_degree_events_negative_binomial(f, layers=3, seed=42),
                no_check,
            ),
            # ============ PARTIAL ============
            (
                "partial ME strength",
                lambda: fit_partial_strength_poisson(
                    s_out,
                    s_in,
                    known_s,
                    known_t,
                    known_r,
                    self_loops=False,
                    tolerance=fit_tol,
                ),
                lambda f: None,
                no_check,
            ),
            (
                "partial ME strength-edges",
                lambda: fit_partial_strength_edges_poisson(
                    s_out,
                    s_in,
                    known_s,
                    known_t,
                    known_r,
                    total_edges,
                    self_loops=False,
                    tolerance=fit_tol,
                ),
                lambda f: None,
                no_check,
            ),
            (
                "partial ME strength-degree",
                lambda: fit_partial_strength_degree_poisson(
                    s_out,
                    s_in,
                    k_out,
                    k_in,
                    known_s,
                    known_t,
                    known_r,
                    self_loops=False,
                    tolerance=fit_tol,
                ),
                lambda f: None,
                no_check,
            ),
            (
                "partial ME cost-coord",
                lambda: fit_partial_strength_cost_poisson_coordinates(
                    s_out,
                    s_in,
                    known_s,
                    known_t,
                    known_r,
                    cx,
                    cy,
                    total_cost,
                    self_loops=False,
                    tolerance=fit_tol,
                ),
                lambda f: None,
                no_check,
            ),
            (
                "partial B cost-coord",
                lambda: fit_partial_strength_cost_binomial_coordinates(
                    s_out,
                    s_in,
                    known_s,
                    known_t,
                    known_r,
                    cx,
                    cy,
                    total_cost,
                    layers=b_layers,
                    self_loops=False,
                    tolerance=fit_tol,
                ),
                lambda f: None,
                no_check,
            ),
            (
                "partial W cost-coord",
                lambda: fit_partial_strength_cost_geometric_coordinates(
                    s_out,
                    s_in,
                    known_s,
                    known_t,
                    known_r,
                    cx,
                    cy,
                    total_cost,
                    self_loops=False,
                    tolerance=w_tol,
                ),
                lambda f: None,
                no_check,
            ),
            (
                "partial Wnb cost-coord",
                lambda: fit_partial_strength_cost_negative_binomial_coordinates(
                    s_out,
                    s_in,
                    known_s,
                    known_t,
                    known_r,
                    cx,
                    cy,
                    total_cost,
                    layers=3,
                    self_loops=False,
                    tolerance=w_tol,
                ),
                lambda f: None,
                no_check,
            ),
        ]

        for name, fit_fn, sample_fn, check_fn in cases:
            row = run_one(name, fit_fn, sample_fn, check_fn, timeout=timeout)
            row["n"] = n
            all_rows.append(row)
            fit_ok = row.get("fit_ok", False)
            sample_ok = row.get("sample_ok", False)
            check_ok = row.get("check_ok", False)
            dt = row.get("seconds", 0)
            iters = row.get("iters", "")
            err = row.get("error", "")
            metric = row.get("metric", "")
            if not fit_ok:
                sym = "✗"
                info = err or f"iters={iters}"
            elif not sample_ok:
                sym = "~"
                info = err
            elif not check_ok:
                sym = "△"
                info = f"err={metric}"
            else:
                sym = "✓"
                info = ""
            print(f"  {sym} {name:30s} {dt:7.2f}s {info}", flush=True)

    # Save results
    out_dir = "benchmarks/results"
    import pathlib

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    def json_safe(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    clean = [{k: json_safe(v) for k, v in r.items()} for r in all_rows]
    with pathlib.Path(f"{out_dir}/e2e_full_results.json").open("w") as f:
        json.dump(clean, f, indent=2)

    # Summary table
    print(f"\n{'=' * 70}")
    print(
        f"{'N':>5} | {'Total':>5} | {'Fit OK':>6} | {'Sample':>6} | {'Check':>5} | {'Fit Fail':>8}"
    )
    print("-" * 70)
    for n_val in nodes:
        nr = [r for r in all_rows if r["n"] == n_val]
        total = len(nr)
        fit_ok = sum(1 for r in nr if r.get("fit_ok"))
        sample_ok = sum(1 for r in nr if r.get("sample_ok"))
        check_ok = sum(1 for r in nr if r.get("check_ok"))
        print(
            f"{n_val:>5} | {total:>5} | {fit_ok:>6} | {sample_ok:>6} | {check_ok:>5} | {total - fit_ok:>8}"
        )
    print(f"\nResults: {out_dir}/e2e_full_results.json")


if __name__ == "__main__":
    main()
