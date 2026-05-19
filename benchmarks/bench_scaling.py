"""Performance benchmarks for ODME: scaling with N nodes.

Generates figures in docs/figures/ showing time and memory scaling.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.common import (
    FIGURES_DIR,
    complete_euclidean_costs,
    pareto_strength_network,
    time_call,
)
from odme.analysis import directed_degrees, directed_strengths
from odme.analysis.stats import compute_all_stats
from odme.data.frames import EdgeTable
from odme.models import (
    fit_degree_bernoulli,
    fit_strength_cost_poisson,
    fit_strength_degree_poisson,
    fit_strength_edges_poisson,
    fit_strength_poisson,
    sample_strength_multinomial,
    sample_strength_poisson,
    sample_strength_stub_matching,
)

N_VALUES = [50, 100, 500, 1000, 2000, 5000, 10000]
AVERAGE_STRENGTH = 100  # average s_out per node


def benchmark_analysis(edges: EdgeTable) -> dict[str, float]:
    """Benchmark analysis operations."""
    return {
        "directed_strengths": time_call(directed_strengths, edges),
        "directed_degrees": time_call(directed_degrees, edges),
        "compute_all_stats": time_call(compute_all_stats, edges),
    }


def benchmark_fitting(edges: EdgeTable) -> dict[str, float]:
    """Benchmark fitting operations."""
    s = directed_strengths(edges)
    k = directed_degrees(edges)
    n = int(max(edges.source.max(), edges.target.max())) + 1
    results = {}
    results["fit_strength_poisson"] = time_call(fit_strength_poisson, s.out, s.incoming)
    if n <= 1000:
        results["fit_degree_bernoulli"] = time_call(
            fit_degree_bernoulli,
            k.out.astype(float),
            k.incoming.astype(float),
        )
    # Only run expensive fitters for small N.
    if n <= 1000:
        results["fit_strength_edges_poisson"] = time_call(
            fit_strength_edges_poisson,
            s.out.astype(float),
            s.incoming.astype(float),
            float(edges.num_edges),
        )
    if n <= 100:
        results["fit_strength_degree_poisson"] = time_call(
            fit_strength_degree_poisson,
            s.out.astype(float),
            s.incoming.astype(float),
            k.out.astype(float),
            k.incoming.astype(float),
        )
    if n <= 1000:
        costs = complete_euclidean_costs(edges, n)
        results["fit_strength_cost_poisson"] = time_call(
            fit_strength_cost_poisson,
            s.out.astype(float),
            s.incoming.astype(float),
            costs.source,
            costs.target,
            costs.value,
            costs.target_cost,
        )
    return results


def benchmark_generation(edges: EdgeTable) -> dict[str, float]:
    """Benchmark generation operations."""
    s = directed_strengths(edges)
    fit = fit_strength_poisson(s.out, s.incoming)
    total = edges.total_events
    results = {}
    results["sample_strength_poisson"] = time_call(
        sample_strength_poisson, fit.x, fit.y, seed=0
    )
    results["sample_strength_multinomial"] = time_call(
        sample_strength_multinomial, fit.x, fit.y, total_events=total, seed=0
    )
    results["sample_strength_stub_matching"] = time_call(
        sample_strength_stub_matching, s.out, s.incoming, seed=0
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
        edges = pareto_strength_network(n, AVERAGE_STRENGTH)
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
