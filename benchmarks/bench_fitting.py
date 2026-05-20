"""Unified fitting benchmark CLI.

Usage:
    uv run python benchmarks/bench_fitting.py [--max-n N] [--output DIR] [--plot]

Memory-safe defaults:
    Clarabel conic: N up to 150
    Monotone coordinate: N up to 1000
    ME/B IPF: N up to 1000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


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


def run_fit(name, func, kwargs, n):
    t0 = time.perf_counter()
    fit = func(**kwargs)
    elapsed = time.perf_counter() - t0
    diag = getattr(fit, "diagnostics", None)
    max_s = diag.max_strength_residual if diag else None
    max_q = diag.conic.max_q if diag and diag.conic else None
    result = {
        "name": name,
        "n": n,
        "time": elapsed,
        "converged": fit.converged,
        "iterations": fit.iterations,
        "max_strength_residual": max_s,
        "max_q": max_q,
    }
    status = "✓" if fit.converged else "✗"
    resid = f"{max_s:.2e}" if max_s is not None else "N/A"
    q = f"{max_q:.4f}" if max_q is not None else "N/A"
    print(
        f"  {status} {name:<40} N={n:<5} {elapsed:>7.2f}s "
        f"iter={fit.iterations:<5} resid={resid:<10} q={q}"
    )
    sys.stdout.flush()
    return result


def bench_all(max_n: int = 1000):
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

    # Safe N ranges per solver type
    conic_sizes = [n for n in [10, 25, 50, 100, 150] if n <= max_n]
    coord_sizes = [n for n in [10, 25, 50, 100, 200, 500, 1000] if n <= max_n]
    ipf_sizes = [n for n in [10, 25, 50, 100, 200, 500, 1000] if n <= max_n]

    results = []

    for n in ipf_sizes:
        total = n * 6.0
        s_out, s_in = _pareto_strengths(n, total)
        k_out, k_in = _degrees(s_out, s_in)
        target_edges = n * 1.8
        total_events = int(total)

        print(f"\n{'=' * 70}\nN = {n}  (T = {total:.0f})\n{'=' * 70}")

        # ME IPF solvers (all N)
        results.append(
            run_fit(
                "ME strength (IPF)",
                fit_strength_poisson,
                {"strength_out": s_out, "strength_in": s_in},
                n,
            )
        )
        results.append(
            run_fit(
                "ME strength-edges (IPF+bisect)",
                fit_strength_edges_poisson,
                {
                    "strength_out": s_out,
                    "strength_in": s_in,
                    "target_edges": target_edges,
                },
                n,
            )
        )
        results.append(
            run_fit(
                "ME strength-degree (IPF)",
                fit_strength_degree_poisson,
                {
                    "strength_out": s_out,
                    "strength_in": s_in,
                    "degree_out": k_out,
                    "degree_in": k_in,
                },
                n,
            )
        )
        results.append(
            run_fit(
                "ME degree-events (Bernoulli IPF)",
                fit_degree_events_poisson,
                {"degree_out": k_out, "degree_in": k_in, "total_events": total_events},
                n,
            )
        )

        # W degree-events (scalar+IPF, fast at all N)
        results.append(
            run_fit(
                "W degree-events geometric",
                fit_degree_events_geometric,
                {"degree_out": k_out, "degree_in": k_in, "total_events": total_events},
                n,
            )
        )
        results.append(
            run_fit(
                "W degree-events NB(3)",
                fit_degree_events_negative_binomial,
                {
                    "degree_out": k_out,
                    "degree_in": k_in,
                    "total_events": total_events,
                    "layers": 3,
                },
                n,
            )
        )

        # ME strength-cost (IPF+scalar)
        if n <= 200:  # cost matrix is N^2 entries
            c_src, c_tgt, c_val = _costs(n)
            base = fit_strength_poisson(s_out, s_in)
            target_cost = float(np.sum(c_val.reshape(n, n) * np.outer(base.x, base.y)))
            results.append(
                run_fit(
                    "ME strength-cost (IPF+scalar)",
                    fit_strength_cost_poisson,
                    {
                        "strength_out": s_out,
                        "strength_in": s_in,
                        "cost_sources": c_src,
                        "cost_targets": c_tgt,
                        "cost_values": c_val,
                        "target_cost": target_cost,
                    },
                    n,
                )
            )

        # Clarabel conic (only at safe N)
        if n in conic_sizes:
            results.append(
                run_fit(
                    "W strength geometric (Clarabel)",
                    fit_strength_geometric,
                    {"strength_out": s_out, "strength_in": s_in},
                    n,
                )
            )
            if n <= 100:
                c_src, c_tgt, c_val = _costs(n)
                base = fit_strength_geometric(s_out, s_in)
                target_cost = float(
                    np.sum(c_val.reshape(n, n) * np.outer(base.x, base.y))
                )
                results.append(
                    run_fit(
                        "W strength-cost geometric (Clarabel)",
                        fit_strength_cost_geometric,
                        {
                            "strength_out": s_out,
                            "strength_in": s_in,
                            "cost_sources": c_src,
                            "cost_targets": c_tgt,
                            "cost_values": c_val,
                            "target_cost": target_cost,
                        },
                        n,
                    )
                )

        # Monotone coordinate (safe up to coord_sizes)
        if n in coord_sizes:
            results.append(
                run_fit(
                    "W strength-edges geometric (coord)",
                    fit_strength_edges_geometric,
                    {
                        "strength_out": s_out,
                        "strength_in": s_in,
                        "target_edges": target_edges,
                        "tolerance": 1e-7,
                        "max_iterations": 500,
                    },
                    n,
                )
            )
            results.append(
                run_fit(
                    "W strength-degree geometric (coord)",
                    fit_strength_degree_geometric,
                    {
                        "strength_out": s_out,
                        "strength_in": s_in,
                        "degree_out": k_out,
                        "degree_in": k_in,
                        "tolerance": 1e-7,
                        "max_iterations": 500,
                    },
                    n,
                )
            )

    return results


def plot_results(results, output_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by solver name
    solvers = {}
    for r in results:
        name = r["name"]
        if name not in solvers:
            solvers[name] = {"n": [], "time": []}
        solvers[name]["n"].append(r["n"])
        solvers[name]["time"].append(r["time"])

    # Scaling plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in sorted(solvers.items()):
        ax.plot(data["n"], data["time"], "o-", label=name, markersize=4)
    ax.set_xlabel("N (nodes)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Fitting time vs network size")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fitting_scaling.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {output_dir / 'fitting_scaling.png'}")

    # Residual plot
    solvers_resid = {}
    for r in results:
        if r["max_strength_residual"] is not None:
            name = r["name"]
            if name not in solvers_resid:
                solvers_resid[name] = {"n": [], "resid": []}
            solvers_resid[name]["n"].append(r["n"])
            solvers_resid[name]["resid"].append(r["max_strength_residual"])

    if solvers_resid:
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, data in sorted(solvers_resid.items()):
            ax.plot(data["n"], data["resid"], "o-", label=name, markersize=4)
        ax.set_xlabel("N (nodes)")
        ax.set_ylabel("Max strength residual")
        ax.set_title("Constraint recovery accuracy vs network size")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "fitting_residuals.png", dpi=150)
        plt.close(fig)
        print(f"Plot saved: {output_dir / 'fitting_residuals.png'}")


def main():
    parser = argparse.ArgumentParser(description="Fitting benchmark")
    parser.add_argument("--max-n", type=int, default=1000, help="Maximum N")
    parser.add_argument(
        "--output", type=str, default="benchmarks/results", help="Output dir"
    )
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()

    print("=" * 70)
    print("ODME FITTING BENCHMARK")
    print(f"Max N: {args.max_n}")
    print("=" * 70)

    results = bench_all(max_n=args.max_n)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "fitting_benchmark.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {output_dir / 'fitting_benchmark.json'}")

    if args.plot:
        plot_results(results, output_dir)

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Solver':<45} {'N':>5} {'Time':>8} {'Converged':>10} {'Residual':>12}")
    print("-" * 85)
    for r in results:
        resid = (
            f"{r['max_strength_residual']:.2e}" if r["max_strength_residual"] else "N/A"
        )
        print(
            f"{r['name']:<45} {r['n']:>5} {r['time']:>7.2f}s "
            f"{'✓' if r['converged'] else '✗':>10} {resid:>12}"
        )


if __name__ == "__main__":
    main()
