"""End-to-end fitting benchmark with verbose convergence diagnostics.

Reports multiplier convergence and constraint residuals at each solver level,
similar to the original thesis-era fitter codes.

Usage:
    uv run python benchmarks/bench_e2e_fitting.py

Memory-safe sizing: uses N up to 200 for conic and 500 for coordinate solvers.
"""

import sys
import time

import numpy as np

from odme.models import (
    fit_degree_events_geometric,
    fit_degree_events_negative_binomial,
    fit_degree_events_poisson,
    fit_strength_cost_geometric,
    fit_strength_cost_poisson,
    fit_strength_degree_geometric,
    fit_strength_degree_poisson,
    fit_strength_edges_geometric,
    fit_strength_edges_poisson,
    fit_strength_geometric,
    fit_strength_poisson,
)

LAYERS = 3


def _pareto_strengths(n: int, total: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    raw = rng.pareto(2.3, n) + 1.0
    raw = np.clip(raw, 0.0, np.quantile(raw, 0.95))
    s_out = raw / raw.sum() * total
    raw_in = np.roll(raw[::-1], n // 5)
    s_in = raw_in / raw_in.sum() * total
    return s_out, s_in


def _costs(n: int, seed: int = 99):
    rng = np.random.default_rng(seed)
    src, tgt = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    val = rng.lognormal(0.0, 0.35, (n, n))
    return (
        src.ravel().astype(np.uint64),
        tgt.ravel().astype(np.uint64),
        val.ravel().astype(np.float64),
    )


def _degrees(s_out, s_in, frac: float = 0.3):
    n = len(s_out)
    k_total = n * frac
    k_out = s_out / s_out.sum() * k_total
    k_in = s_in / s_in.sum() * k_total
    scale = min((s_out / k_out).min(), (s_in / k_in).min())
    if scale < 1.0:
        k_out = k_out * scale * 0.9
        k_in = k_in * (k_out.sum() / k_in.sum())
    return k_out, k_in


def _report(name: str, n: int, fit, elapsed: float):
    status = (
        fit.status
        if hasattr(fit, "status")
        else ("solved" if fit.converged else "failed")
    )
    diag = fit.diagnostics if hasattr(fit, "diagnostics") else None
    max_s_resid = None
    if diag and diag.max_strength_residual is not None:
        max_s_resid = diag.max_strength_residual
    max_q = None
    if diag and diag.conic and diag.conic.max_q is not None:
        max_q = diag.conic.max_q
    min_margin = None
    if diag and diag.conic and diag.conic.min_margin is not None:
        min_margin = diag.conic.min_margin

    print(
        f"  {name:<35} N={n:<4} {elapsed:>7.2f}s "
        f"status={status:<10} iters={fit.iterations:<5} "
        f"max_s_resid={_fmt(max_s_resid)} max_q={_fmt(max_q)} "
        f"min_margin={_fmt(min_margin)}"
    )
    sys.stdout.flush()


def _fmt(val):
    if val is None:
        return "N/A"
    return f"{val:.3e}"


def bench_strength(n: int, total: float):
    s_out, s_in = _pareto_strengths(n, total)
    print(f"\n--- Strength fitting N={n} T={total:.0f} ---")

    t0 = time.perf_counter()
    fit = fit_strength_poisson(s_out, s_in, tolerance=1e-8, verbose=2)
    _report("ME poisson", n, fit, time.perf_counter() - t0)

    t0 = time.perf_counter()
    fit = fit_strength_geometric(s_out, s_in, tolerance=1e-8, verbose=2)
    _report("W geometric (Clarabel)", n, fit, time.perf_counter() - t0)


def bench_strength_cost(n: int, total: float):
    s_out, s_in = _pareto_strengths(n, total)
    c_src, c_tgt, c_val = _costs(n)
    base = fit_strength_geometric(s_out, s_in, tolerance=1e-7)
    target_cost = float(
        sum(
            c_val.reshape(n, n)[i, j] * base.x[i] * base.y[j]
            for i in range(n)
            for j in range(n)
        )
    )
    print(f"\n--- Strength-cost fitting N={n} T={total:.0f} C={target_cost:.1f} ---")

    t0 = time.perf_counter()
    fit = fit_strength_cost_poisson(
        s_out, s_in, c_src, c_tgt, c_val, target_cost, tolerance=1e-6, verbose=2
    )
    _report("ME poisson (IPF+scalar)", n, fit, time.perf_counter() - t0)

    t0 = time.perf_counter()
    fit = fit_strength_cost_geometric(
        s_out, s_in, c_src, c_tgt, c_val, target_cost, tolerance=1e-8, verbose=2
    )
    _report("W geometric (Clarabel)", n, fit, time.perf_counter() - t0)


def bench_strength_edges(n: int, total: float):
    s_out, s_in = _pareto_strengths(n, total)
    target_edges = n * 1.8
    print(f"\n--- Strength-edges fitting N={n} T={total:.0f} E={target_edges:.0f} ---")

    t0 = time.perf_counter()
    fit = fit_strength_edges_poisson(
        s_out, s_in, target_edges, tolerance=1e-10, verbose=2
    )
    _report("ME poisson (IPF+bisect)", n, fit, time.perf_counter() - t0)

    t0 = time.perf_counter()
    fit = fit_strength_edges_geometric(
        s_out, s_in, target_edges, tolerance=1e-7, max_iterations=500, verbose=2
    )
    _report("W geometric (coord-bisect)", n, fit, time.perf_counter() - t0)


def bench_strength_degree(n: int, total: float):
    s_out, s_in = _pareto_strengths(n, total)
    k_out, k_in = _degrees(s_out, s_in)
    print(f"\n--- Strength-degree fitting N={n} T={total:.0f} E={k_out.sum():.1f} ---")

    t0 = time.perf_counter()
    fit = fit_strength_degree_poisson(
        s_out, s_in, k_out, k_in, tolerance=1e-10, verbose=2
    )
    _report("ME poisson (IPF)", n, fit, time.perf_counter() - t0)

    t0 = time.perf_counter()
    fit = fit_strength_degree_geometric(
        s_out, s_in, k_out, k_in, tolerance=1e-7, max_iterations=500, verbose=2
    )
    _report("W geometric (coord-bisect)", n, fit, time.perf_counter() - t0)


def bench_degree_events(n: int, total: float):
    s_out, s_in = _pareto_strengths(n, total)
    k_out, k_in = _degrees(s_out, s_in)
    total_events = int(total)
    print(f"\n--- Degree-events fitting N={n} T={total_events} E={k_out.sum():.1f} ---")

    t0 = time.perf_counter()
    fit = fit_degree_events_poisson(
        k_out, k_in, total_events, tolerance=1e-10, verbose=2
    )
    _report("ME poisson (Bernoulli IPF)", n, fit, time.perf_counter() - t0)

    t0 = time.perf_counter()
    fit = fit_degree_events_geometric(
        k_out, k_in, total_events, tolerance=1e-8, verbose=2
    )
    _report("W geometric (root+IPF)", n, fit, time.perf_counter() - t0)

    t0 = time.perf_counter()
    fit = fit_degree_events_negative_binomial(
        k_out, k_in, total_events, layers=LAYERS, tolerance=1e-8, verbose=2
    )
    _report("W NB(3) (root+IPF)", n, fit, time.perf_counter() - t0)


def main():
    print("=" * 90)
    print("END-TO-END FITTING BENCHMARK")
    print("=" * 90)

    # Tier 1: small (fast, all solvers)
    for n in [10, 25, 50]:
        total = n * 6.0
        bench_strength(n, total)
        bench_strength_cost(n, total)
        bench_strength_edges(n, total)
        bench_strength_degree(n, total)
        bench_degree_events(n, total)

    # Tier 2: medium (Clarabel up to N=100, coordinate up to N=200)
    for n in [100]:
        total = n * 6.0
        bench_strength(n, total)
        bench_strength_cost(n, total)
        bench_strength_edges(n, total)
        bench_strength_degree(n, total)
        bench_degree_events(n, total)

    # Tier 3: large (only coordinate and IPF solvers, skip Clarabel)
    for n in [200, 500]:
        total = n * 6.0
        s_out, s_in = _pareto_strengths(n, total)
        target_edges = n * 1.8
        k_out, k_in = _degrees(s_out, s_in)

        print(f"\n--- [LARGE] Strength-edges N={n} ---")
        t0 = time.perf_counter()
        fit = fit_strength_edges_geometric(
            s_out, s_in, target_edges, tolerance=1e-7, max_iterations=500, verbose=2
        )
        _report("W geometric (coord-bisect)", n, fit, time.perf_counter() - t0)

        print(f"\n--- [LARGE] Strength-degree N={n} ---")
        t0 = time.perf_counter()
        fit = fit_strength_degree_geometric(
            s_out, s_in, k_out, k_in, tolerance=1e-7, max_iterations=500, verbose=2
        )
        _report("W geometric (coord-bisect)", n, fit, time.perf_counter() - t0)

        print(f"\n--- [LARGE] ME Strength-edges N={n} ---")
        t0 = time.perf_counter()
        fit = fit_strength_edges_poisson(
            s_out, s_in, target_edges, tolerance=1e-10, verbose=2
        )
        _report("ME poisson (IPF+bisect)", n, fit, time.perf_counter() - t0)

        print(f"\n--- [LARGE] ME Strength-degree N={n} ---")
        t0 = time.perf_counter()
        fit = fit_strength_degree_poisson(
            s_out, s_in, k_out, k_in, tolerance=1e-10, verbose=2
        )
        _report("ME poisson (IPF)", n, fit, time.perf_counter() - t0)

        print(f"\n--- [LARGE] Degree-events N={n} ---")
        total_events = int(total)
        t0 = time.perf_counter()
        fit = fit_degree_events_geometric(
            k_out, k_in, total_events, tolerance=1e-8, verbose=2
        )
        _report("W geometric (root+IPF)", n, fit, time.perf_counter() - t0)

    print("\n" + "=" * 90)
    print("BENCHMARK COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
