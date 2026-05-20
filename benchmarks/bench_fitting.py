"""Unified fitting benchmark CLI.

Usage:
    uv run python benchmarks/bench_fitting.py [--max-n N] [--tolerance T] [--verbose V] [--plot] [--output DIR]
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
    return src.ravel().astype(np.uint64), tgt.ravel().astype(np.uint64), val.ravel()


def _degrees(s_out, s_in, frac=0.3):
    n = len(s_out)
    k_out = s_out / s_out.sum() * (n * frac)
    k_in = s_in / s_in.sum() * (n * frac)
    scale = min((s_out / k_out).min(), (s_in / k_in).min())
    if scale < 1.0:
        k_out *= scale * 0.9
        k_in *= k_out.sum() / k_in.sum()
    return k_out, k_in


def _compute_strength_residual(
    fit, s_out, s_in, self_loops=True, family=None, layers=None
):
    x, y = np.asarray(fit.x), np.asarray(fit.y)
    n = len(x)
    lam = getattr(fit, "lam", None)
    gamma = getattr(fit, "gamma", None)
    z = getattr(fit, "z", None)
    w = getattr(fit, "w", None)
    fam = family or getattr(fit, "family", "poisson")
    m = layers or getattr(fit, "layers", None) or 1
    pred_out, pred_in = np.zeros(n), np.zeros(n)
    for i in range(n):
        for j in range(n):
            if not self_loops and i == j:
                continue
            if z is not None and w is not None:
                # strength-degree: v*u/(exp(-u)+v*(1-exp(-u)))
                u = x[i] * y[j]
                v = z[i] * w[j]
                e_neg = np.exp(-u)
                den = e_neg + v * (1.0 - e_neg)
                mean = v * u / den if den > 0 else 0.0
            elif lam is not None:
                # strength-edges: lam*u/(exp(-u)+lam*(1-exp(-u)))
                u = x[i] * y[j]
                e_neg = np.exp(-u)
                den = e_neg + lam * (1.0 - e_neg)
                mean = lam * u / den if den > 0 else 0.0
            elif gamma is not None:
                # strength-cost
                mean = x[i] * y[j] * np.exp(-gamma * 0.0)
            elif fam == "binomial":
                # binomial: M * xy / (1 + xy)
                xy = x[i] * y[j]
                mean = m * xy / (1.0 + xy)
            else:
                # poisson: x*y
                mean = x[i] * y[j]
            pred_out[i] += mean
            pred_in[j] += mean
    return float(max(np.max(np.abs(pred_out - s_out)), np.max(np.abs(pred_in - s_in))))


def _compute_degree_residual(fit, k_out, k_in, self_loops=True):
    x, y = np.asarray(fit.x), np.asarray(fit.y)
    n = len(x)
    z = getattr(fit, "z", None)
    w = getattr(fit, "w", None)
    pred_out, pred_in = np.zeros(n), np.zeros(n)
    for i in range(n):
        for j in range(n):
            if not self_loops and i == j:
                continue
            if z is not None and w is not None:
                # strength-degree: v*(1-exp(-u))/(exp(-u)+v*(1-exp(-u)))
                u = x[i] * y[j]
                v = z[i] * w[j]
                e_neg = np.exp(-u)
                den = e_neg + v * (1.0 - e_neg)
                occ = v * (1.0 - e_neg) / den if den > 0 else 0.0
            else:
                # Bernoulli: xy/(1+xy)
                xy = x[i] * y[j]
                occ = xy / (1.0 + xy)
            pred_out[i] += occ
            pred_in[j] += occ
    return float(max(np.max(np.abs(pred_out - k_out)), np.max(np.abs(pred_in - k_in))))


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
        max_s = _compute_strength_residual(fit, s_out, s_in)
    max_k = None
    if k_out is not None:
        max_k = _compute_degree_residual(fit, k_out, k_in)

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
        s_out, s_in = _pareto_strengths(n, total)
        k_out, k_in = _degrees(s_out, s_in)
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
            c_src, c_tgt, c_val = _costs(n)
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
            c_src, c_tgt, c_val = _costs(n)
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
                c_src, c_tgt, c_val = _costs(n)
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


def main():
    parser = argparse.ArgumentParser(description="ODME fitting benchmark")
    parser.add_argument("--max-n", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--verbose", type=int, default=0, help="0=quiet, 1=warn, 2=convergence info"
    )
    parser.add_argument("--output", type=str, default="benchmarks/results")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print(
        f"ODME FITTING BENCHMARK  max_n={args.max_n}  tol={args.tolerance}  verbose={args.verbose}"
    )
    print("=" * 80)

    results = bench_all(
        max_n=args.max_n, tolerance=args.tolerance, verbose=args.verbose
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "fitting_benchmark.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults: {output_dir / 'fitting_benchmark.json'}")

    if args.plot:
        plot_results(results, output_dir)

    print(f"\n{'=' * 80}\nSUMMARY\n{'=' * 80}")
    print(
        f"{'Solver':<40} {'N':>4} {'Time':>7} {'Conv':>4} {'s_resid':>10} {'k_resid':>10}"
    )
    print("-" * 80)
    for r in results:
        sr = f"{r['max_strength_residual']:.1e}" if r["max_strength_residual"] else "—"
        kr = f"{r['max_degree_residual']:.1e}" if r.get("max_degree_residual") else "—"
        print(
            f"{r['name']:<40} {r['n']:>4} {r['time']:>6.2f}s {'✓' if r['converged'] else '✗':>4} {sr:>10} {kr:>10}"
        )


if __name__ == "__main__":
    main()
