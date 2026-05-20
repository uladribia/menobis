"""Benchmark all 10 W fitting APIs at N={10, 25, 50, 100}."""

import time

import numpy as np

from odme.models import (
    fit_degree_events_geometric,
    fit_degree_events_negative_binomial,
    fit_strength_cost_geometric,
    fit_strength_cost_negative_binomial,
    fit_strength_degree_geometric,
    fit_strength_degree_negative_binomial,
    fit_strength_edges_geometric,
    fit_strength_edges_negative_binomial,
    fit_strength_geometric,
    fit_strength_negative_binomial,
)

LAYERS = 3
SIZES = [10, 25, 50, 100]


def _pareto_strengths(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    raw = rng.pareto(2.3, n) + 1.0
    raw = np.clip(raw, 0.0, np.quantile(raw, 0.95))
    total = n * 6.0
    s_out = raw / raw.sum() * total
    raw_in = np.roll(raw[::-1], n // 5)
    s_in = raw_in / raw_in.sum() * total
    return s_out, s_in


def _costs(n: int, seed: int = 99) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    src, tgt = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    val = rng.lognormal(0.0, 0.35, (n, n))
    return (
        src.ravel().astype(np.uint64),
        tgt.ravel().astype(np.uint64),
        val.ravel().astype(np.float64),
    )


def bench_strength(n: int) -> list[dict]:
    s_out, s_in = _pareto_strengths(n)
    results = []
    for name, func, kwargs in [
        ("strength_geo", fit_strength_geometric, {}),
        ("strength_nb3", fit_strength_negative_binomial, {"layers": LAYERS}),
    ]:
        t0 = time.perf_counter()
        fit = func(
            s_out, s_in, self_loops=True, tolerance=1e-7, max_iterations=500, **kwargs
        )
        dt = time.perf_counter() - t0
        results.append(
            {
                "model": name,
                "n": n,
                "time": dt,
                "converged": fit.converged,
                "iterations": fit.iterations,
                "max_resid": fit.max_strength_residual,
            }
        )
    return results


def bench_strength_cost(n: int) -> list[dict]:
    s_out, s_in = _pareto_strengths(n)
    c_src, c_tgt, c_val = _costs(n)
    base = fit_strength_geometric(s_out, s_in, self_loops=True, tolerance=1e-7)
    target_cost = float(
        sum(
            c_val.reshape(n, n)[i, j] * base.x[i] * base.y[j]
            for i in range(n)
            for j in range(n)
        )
    )
    results = []
    for name, func, kwargs in [
        ("cost_geo", fit_strength_cost_geometric, {}),
        ("cost_nb3", fit_strength_cost_negative_binomial, {"layers": LAYERS}),
    ]:
        t0 = time.perf_counter()
        fit = func(
            s_out,
            s_in,
            c_src,
            c_tgt,
            c_val,
            target_cost,
            self_loops=True,
            tolerance=1e-7,
            max_iterations=500,
            **kwargs,
        )
        dt = time.perf_counter() - t0
        results.append(
            {
                "model": name,
                "n": n,
                "time": dt,
                "converged": fit.converged,
                "iterations": fit.iterations,
                "max_resid": fit.max_strength_residual,
            }
        )
    return results


def bench_strength_edges(n: int) -> list[dict]:
    s_out, s_in = _pareto_strengths(n)
    target_edges = n * 1.8
    results = []
    for name, func, kwargs in [
        ("edges_geo", fit_strength_edges_geometric, {}),
        ("edges_nb3", fit_strength_edges_negative_binomial, {"layers": LAYERS}),
    ]:
        t0 = time.perf_counter()
        fit = func(
            s_out,
            s_in,
            target_edges,
            self_loops=True,
            tolerance=1e-7,
            max_iterations=500,
            **kwargs,
        )
        dt = time.perf_counter() - t0
        results.append(
            {
                "model": name,
                "n": n,
                "time": dt,
                "converged": fit.converged,
                "iterations": fit.iterations,
                "max_resid": fit.max_strength_residual,
            }
        )
    return results


def bench_strength_degree(n: int) -> list[dict]:
    s_out, s_in = _pareto_strengths(n)
    k_total = n * 0.3
    k_out = s_out / s_out.sum() * k_total
    k_in = s_in / s_in.sum() * k_total
    scale = min((s_out / k_out).min(), (s_in / k_in).min())
    if scale < 1.0:
        k_out = k_out * scale * 0.9
        k_in = k_in * (k_out.sum() / k_in.sum())
    results = []
    for name, func, kwargs in [
        ("degree_geo", fit_strength_degree_geometric, {}),
        ("degree_nb3", fit_strength_degree_negative_binomial, {"layers": LAYERS}),
    ]:
        t0 = time.perf_counter()
        fit = func(
            s_out,
            s_in,
            k_out,
            k_in,
            self_loops=True,
            tolerance=1e-7,
            max_iterations=500,
            **kwargs,
        )
        dt = time.perf_counter() - t0
        results.append(
            {
                "model": name,
                "n": n,
                "time": dt,
                "converged": fit.converged,
                "iterations": fit.iterations,
                "max_resid": getattr(fit, "max_strength_residual", None)
                or (fit.diagnostics.max_strength_residual if fit.diagnostics else None),
            }
        )
    return results


def bench_degree_events(n: int) -> list[dict]:
    s_out, s_in = _pareto_strengths(n)
    k_total = n * 0.3
    k_out = s_out / s_out.sum() * k_total
    k_in = s_in / s_in.sum() * k_total
    total_events = int(n * 6.0)
    results = []
    for name, func, kwargs in [
        ("devents_geo", fit_degree_events_geometric, {}),
        ("devents_nb3", fit_degree_events_negative_binomial, {"layers": LAYERS}),
    ]:
        t0 = time.perf_counter()
        fit = func(
            k_out,
            k_in,
            total_events,
            self_loops=True,
            tolerance=1e-7,
            max_iterations=500,
            **kwargs,
        )
        dt = time.perf_counter() - t0
        results.append(
            {
                "model": name,
                "n": n,
                "time": dt,
                "converged": fit.converged,
                "iterations": fit.iterations,
                "max_resid": None,
            }
        )
    return results


def main() -> None:
    all_results = []
    for n in SIZES:
        all_results.extend(bench_strength(n))
        all_results.extend(bench_strength_cost(n))
        all_results.extend(bench_strength_edges(n))
        all_results.extend(bench_strength_degree(n))
        all_results.extend(bench_degree_events(n))

    print(
        f"{'model':<14} {'N':>4} {'time':>8} {'conv':>5} {'iters':>6} {'max_resid':>12}"
    )
    print("-" * 60)
    for r in all_results:
        resid = f"{r['max_resid']:.2e}" if r["max_resid"] is not None else "N/A"
        print(
            f"{r['model']:<14} {r['n']:>4} {r['time']:>7.3f}s "
            f"{'Y' if r['converged'] else 'N':>5} {r['iterations']:>6} {resid:>12}"
        )


if __name__ == "__main__":
    main()
