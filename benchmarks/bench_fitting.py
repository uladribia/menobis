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
    ensure_results_dir,
    pareto_strengths,
    target_cost_from_fit,
)












def run_fit(
    name, func, kwargs, n, s_out=None, s_in=None, k_out=None, k_in=None, verbose=0
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
    }
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


def bench_all(max_n=1000, tolerance=1e-4, verbose=0):
    from odme.models import (
        fit_degree_events_binomial,
        fit_degree_events_geometric,
        fit_degree_events_poisson,
        fit_strength_binomial,
        fit_strength_cost_binomial,
        fit_strength_cost_geometric,
        fit_strength_cost_poisson,
        fit_strength_degree_binomial,
        fit_strength_degree_geometric,
        fit_strength_degree_poisson,
        fit_strength_edges_binomial,
        fit_strength_edges_geometric,
        fit_strength_edges_poisson,
        fit_strength_geometric,
        fit_strength_poisson,
    )

    conic_max = min(max_n, 150)
    coord_max = max_n
    all_sizes = sorted({n for n in [10, 25, 50, 100, 200, 500, 1000] if n <= max_n})
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

        # --- W (Geometric) ---
        if n <= conic_max:
            results.append(
                run_fit(
                    "W strength (Clarabel)",
                    fit_strength_geometric,
                    {
                        "strength_out": s_out,
                        "strength_in": s_in,
                        "tolerance": tolerance,
                    },
                    n,
                    s_out,
                    s_in,
                    verbose=verbose,
                )
            )

            if n >= 25 and n <= 100:
                c_src, c_tgt, c_val = complete_costs(n)
                base = fit_strength_geometric(s_out, s_in, tolerance=tolerance)
                target_cost = float(
                    np.sum(c_val.reshape(n, n) * np.outer(base.x, base.y))
                )
                results.append(
                    run_fit(
                        "W strength-cost (Clarabel)",
                        fit_strength_cost_geometric,
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

        if n <= coord_max:
            results.append(
                run_fit(
                    "W strength-edges (coord)",
                    fit_strength_edges_geometric,
                    {
                        "strength_out": s_out,
                        "strength_in": s_in,
                        "target_edges": target_edges,
                        "tolerance": tolerance,
                        "max_iterations": 500,
                    },
                    n,
                    s_out,
                    s_in,
                    verbose=verbose,
                )
            )

            results.append(
                run_fit(
                    "W strength-degree (coord)",
                    fit_strength_degree_geometric,
                    {
                        "strength_out": s_out,
                        "strength_in": s_in,
                        "degree_out": k_out,
                        "degree_in": k_in,
                        "tolerance": tolerance,
                        "max_iterations": 500,
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
                "W degree-events geometric",
                fit_degree_events_geometric,
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

    return results


def bench_partial(max_n=1000, tolerance=1e-4, verbose=0):
    """Benchmark partial-constraint fitting with 30% known pairs."""
    from odme.models.partial import (
        fit_partial_strength_cost_poisson,
        fit_partial_strength_degree_poisson,
        fit_partial_strength_edges_poisson,
        fit_partial_strength_poisson,
    )

    sizes = sorted({n for n in [10, 25, 50, 100, 200] if n <= max_n})
    results = []

    for n in sizes:
        total = n * 6.0
        s_out, s_in = pareto_strengths(n, total)
        k_out, k_in = degrees_from_strengths(s_out, s_in)

        # Generate 30% known pairs with rates from the Poisson fit
        from odme.models import fit_strength_poisson

        base = fit_strength_poisson(s_out, s_in)
        capacity = n * n
        n_known = int(capacity * 0.3)
        # Pick top-rate pairs from the fitted model
        rates_all = np.outer(base.x, base.y).ravel()
        pair_order = np.argsort(rates_all)[::-1][:n_known]
        k_src = (pair_order // n).astype(np.uint64)
        k_tgt = (pair_order % n).astype(np.uint64)
        k_rate = rates_all[pair_order]

        print(
            f"\n--- Partial fitting N={n} ({n_known} known pairs, {n_known / capacity * 100:.0f}%) ---"
        )

        t0 = time.perf_counter()
        fit = fit_partial_strength_poisson(
            s_out,
            s_in,
            k_src,
            k_tgt,
            k_rate,
            tolerance=tolerance,
        )
        dt = time.perf_counter() - t0
        results.append(
            {
                "name": "Partial ME strength",
                "n": n,
                "time": dt,
                "converged": fit.converged,
                "iterations": fit.iterations,
                "max_strength_residual": None,
                "max_degree_residual": None,
                "max_q": None,
            }
        )
        print(
            f"  ✓ Partial ME strength                  N={n:<4} {dt:>7.2f}s iter={fit.iterations}"
        )

        target_edges = n * 1.8
        t0 = time.perf_counter()
        fit = fit_partial_strength_edges_poisson(
            s_out,
            s_in,
            k_src,
            k_tgt,
            k_rate,
            target_edges,
            tolerance=tolerance,
        )
        dt = time.perf_counter() - t0
        results.append(
            {
                "name": "Partial ME strength-edges",
                "n": n,
                "time": dt,
                "converged": fit.converged,
                "iterations": fit.iterations,
                "max_strength_residual": None,
                "max_degree_residual": None,
                "max_q": None,
            }
        )
        print(
            f"  ✓ Partial ME strength-edges            N={n:<4} {dt:>7.2f}s iter={fit.iterations}"
        )

        t0 = time.perf_counter()
        fit = fit_partial_strength_degree_poisson(
            s_out,
            s_in,
            k_out,
            k_in,
            k_src,
            k_tgt,
            k_rate,
            tolerance=tolerance,
        )
        dt = time.perf_counter() - t0
        results.append(
            {
                "name": "Partial ME strength-degree",
                "n": n,
                "time": dt,
                "converged": fit.converged,
                "iterations": fit.iterations,
                "max_strength_residual": None,
                "max_degree_residual": None,
                "max_q": None,
            }
        )
        print(
            f"  ✓ Partial ME strength-degree           N={n:<4} {dt:>7.2f}s iter={fit.iterations}"
        )

        if n <= 100:
            c_src, c_tgt, c_val = complete_costs(n)
            target_cost = float(np.sum(c_val.reshape(n, n) * np.outer(base.x, base.y)))
            t0 = time.perf_counter()
            fit = fit_partial_strength_cost_poisson(
                s_out,
                s_in,
                k_src,
                k_tgt,
                k_rate,
                c_src.astype(np.int64),
                c_tgt.astype(np.int64),
                c_val,
                target_cost,
                tolerance=tolerance,
            )
            dt = time.perf_counter() - t0
            results.append(
                {
                    "name": "Partial ME strength-cost",
                    "n": n,
                    "time": dt,
                    "converged": fit.converged,
                    "iterations": fit.iterations,
                    "max_strength_residual": None,
                    "max_degree_residual": None,
                    "max_q": None,
                }
            )
            print(
                f"  ✓ Partial ME strength-cost             N={n:<4} {dt:>7.2f}s iter={fit.iterations}"
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
    print(f"{'Solver':<40} {'N':>4} {'Time':>7} {'Conv':>4} {'s_resid':>10} {'k_resid':>10}")
    print("-" * 80)
    for r in results:
        sr = f"{r['max_strength_residual']:.1e}" if r["max_strength_residual"] else "\u2014"
        kr = f"{r['max_degree_residual']:.1e}" if r.get("max_degree_residual") else "\u2014"
        print(
            f"{r['name']:<40} {r['n']:>4} {r['time']:>6.2f}s "
            f"{'\u2713' if r['converged'] else '\u2717':>4} {sr:>10} {kr:>10}"
        )
