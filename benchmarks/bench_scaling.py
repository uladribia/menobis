"""Performance benchmarks for ODME: scaling with N nodes.

Generates figures in docs/figures/ showing time and memory scaling.
"""

import gc
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from odme.analysis import directed_degrees, directed_strengths
from odme.analysis.stats import compute_all_stats
from odme.data.frames import EdgeTable
from odme.models import (
    fit_fixed_degree_binary,
    fit_fixed_strength_me,
    fit_strength_cost_me,
    fit_strength_degree_me,
    fit_strength_edges_me,
    sample_microcanonical,
    sample_multinomial,
    sample_poisson,
    sample_strength_degree_me,
    sample_strength_edges_me,
)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_VALUES = [50, 100, 500, 1000, 2000, 5000, 10000]
AVERAGE_STRENGTH = 100  # average s_out per node


def _generate_pareto_network(n: int, avg_s: int, seed: int = 42) -> EdgeTable:
    """Generate a Pareto-like directed weighted network with N nodes."""
    rng = np.random.default_rng(seed)
    # Pareto-like strength profile.
    raw = rng.pareto(1.5, size=n) + 1.0
    p_out = raw / raw.sum()
    raw = rng.pareto(1.5, size=n) + 1.0
    p_in = raw / raw.sum()
    total = n * avg_s
    s_out = np.round(p_out * total).astype(np.uint64)
    s_in = np.round(p_in * total).astype(np.uint64)
    # Balance.
    diff = int(s_out.sum()) - int(s_in.sum())
    if diff > 0:
        s_in[np.argmax(s_in)] += abs(diff)
    elif diff < 0:
        s_out[np.argmax(s_out)] += abs(diff)
    # Sample a network from fixed-strength ME.
    fit = fit_fixed_strength_me(s_out, s_in)
    return sample_poisson(fit.x, fit.y, seed=seed)


def _time_fn(fn, *args, **kwargs) -> float:
    gc.collect()
    start = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - start


def benchmark_analysis(edges: EdgeTable) -> dict[str, float]:
    """Benchmark analysis operations."""
    return {
        "directed_strengths": _time_fn(directed_strengths, edges),
        "directed_degrees": _time_fn(directed_degrees, edges),
        "compute_all_stats": _time_fn(compute_all_stats, edges),
    }


def benchmark_fitting(edges: EdgeTable) -> dict[str, float]:
    """Benchmark fitting operations."""
    s = directed_strengths(edges)
    k = directed_degrees(edges)
    results = {}
    results["fit_fixed_strength_me"] = _time_fn(
        fit_fixed_strength_me, s.out, s.incoming
    )
    results["fit_fixed_degree_binary"] = _time_fn(
        fit_fixed_degree_binary,
        k.out.astype(float),
        k.incoming.astype(float),
    )
    # Only run expensive fitters for small N.
    n = int(max(edges.source.max(), edges.target.max())) + 1
    if n <= 1000:
        results["fit_strength_edges_me"] = _time_fn(
            fit_strength_edges_me,
            s.out.astype(float),
            s.incoming.astype(float),
            float(edges.num_edges),
        )
    if n <= 200:
        results["fit_strength_degree_me"] = _time_fn(
            fit_strength_degree_me,
            s.out.astype(float),
            s.incoming.astype(float),
            k.out.astype(float),
            k.incoming.astype(float),
        )
    return results


def benchmark_generation(edges: EdgeTable) -> dict[str, float]:
    """Benchmark generation operations."""
    s = directed_strengths(edges)
    fit = fit_fixed_strength_me(s.out, s.incoming)
    total = edges.total_events
    results = {}
    results["sample_poisson"] = _time_fn(sample_poisson, fit.x, fit.y, seed=0)
    results["sample_multinomial"] = _time_fn(
        sample_multinomial, fit.x, fit.y, total_events=total, seed=0
    )
    results["sample_microcanonical"] = _time_fn(
        sample_microcanonical, s.out, s.incoming, seed=0
    )
    return results


def run_benchmarks() -> dict[str, dict[int, dict[str, float]]]:
    """Run all benchmarks for increasing N."""
    all_results: dict[str, dict[int, dict[str, float]]] = {
        "analysis": {},
        "fitting": {},
        "generation": {},
    }
    for n in N_VALUES:
        print(f"  N={n}...", end=" ", flush=True)
        edges = _generate_pareto_network(n, AVERAGE_STRENGTH)
        print(f"E={edges.num_edges}, T={edges.total_events}", end=" ", flush=True)
        all_results["analysis"][n] = benchmark_analysis(edges)
        all_results["fitting"][n] = benchmark_fitting(edges)
        all_results["generation"][n] = benchmark_generation(edges)
        print("done")
    return all_results


def plot_benchmarks(
    results: dict[str, dict[int, dict[str, float]]],
) -> None:
    """Generate scaling plots."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, category in zip(axes, ["analysis", "fitting", "generation"], strict=True):
        data = results[category]
        all_ops = set()
        for ops in data.values():
            all_ops.update(ops.keys())
        for op in sorted(all_ops):
            ns = []
            times = []
            for n in N_VALUES:
                if n in data and op in data[n]:
                    ns.append(n)
                    times.append(data[n][op])
            if ns:
                ax.loglog(ns, times, "o-", label=op.replace("_", " "), markersize=4)
        ax.set_xlabel("N (nodes)")
        ax.set_ylabel("Time (seconds)")
        ax.set_title(category.capitalize())
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "benchmark_scaling.png", dpi=150)
    plt.close(fig)

    # Summary table figure.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    headers = ["Operation", "N=100", "N=1000", "N=5000", "N=10000"]
    rows = []
    for category in ["analysis", "fitting", "generation"]:
        for op in sorted(results[category].get(100, {}).keys()):
            row = [f"{category}: {op}"]
            for n in [100, 1000, 5000, 10000]:
                t = results[category].get(n, {}).get(op)
                row.append(f"{t:.4f}s" if t is not None else "—")
            rows.append(row)
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.4)
    fig.savefig(FIGURES_DIR / "benchmark_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("Running ODME benchmarks...")
    results = run_benchmarks()
    plot_benchmarks(results)
    print(f"Figures saved to {FIGURES_DIR}")
