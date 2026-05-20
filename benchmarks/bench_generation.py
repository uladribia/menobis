"""Streaming generation benchmark."""

import sys
import time

import numpy as np

from benchmarks.common import ensure_results_dir, max_rss_mb, pareto_strengths


def bench_generation(max_n=10000, repeats=3):
    """Benchmark sampling speed and memory across N."""
    from odme.models import (
        fit_strength_poisson,
        sample_strength_multinomial,
        sample_strength_poisson,
        sample_strength_stub_matching,
    )

    sizes = sorted({n for n in [100, 500, 1000, 5000, 10000] if n <= max_n})
    results = []

    for n in sizes:
        s_out, s_in = pareto_strengths(n)
        fit = fit_strength_poisson(s_out, s_in)
        total = int(s_out.sum())

        cases = [
            ("Poisson", lambda: sample_strength_poisson(fit.x, fit.y, seed=0)),
            ("Multinomial", lambda: sample_strength_multinomial(fit.x, fit.y, total_events=total, seed=0)),
            ("Stub-matching", lambda: sample_strength_stub_matching(s_out.astype(np.uint64), s_in.astype(np.uint64), seed=0)),
        ]

        for name, func in cases:
            times = []
            for r in range(repeats):
                t0 = time.perf_counter()
                func()
                times.append(time.perf_counter() - t0)
            avg = np.mean(times)
            rss = max_rss_mb()
            results.append({"name": name, "n": n, "time": avg, "rss_mb": rss})
            print(f"  {name:<20} N={n:<6} {avg:.3f}s  RSS={rss:.0f}MB")
            sys.stdout.flush()

    return results


def plot_generation(results):
    """Plot generation scaling."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    from benchmarks.common import ensure_figures_dir
    fig_dir = ensure_figures_dir()

    solvers = {}
    for r in results:
        name = r["name"]
        if name not in solvers:
            solvers[name] = {"n": [], "time": []}
        solvers[name]["n"].append(r["n"])
        solvers[name]["time"].append(r["time"])

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in sorted(solvers.items()):
        ax.plot(data["n"], data["time"], "o-", label=name)
    ax.set_xlabel("N")
    ax.set_ylabel("Time (s)")
    ax.set_title("Sampling time vs N")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "generation_scaling.png", dpi=150)
    plt.close(fig)
    print(f"Plot: {fig_dir / 'generation_scaling.png'}")
