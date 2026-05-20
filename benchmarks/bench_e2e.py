"""End-to-end benchmark: fit → sample null ensemble → compare constraints."""

import sys

import numpy as np

from benchmarks.bench_fitting import _benchmark_sizes
from benchmarks.common import pareto_strengths


def bench_e2e(max_n=100, ensemble=300, tolerance=1e-4, nodes=None):
    """Fit, sample ensemble, compute z-scores on constraint recovery."""
    from odme.models import (
        fit_strength_geometric,
        fit_strength_poisson,
        sample_strength_poisson,
    )

    sizes = (
        _benchmark_sizes(max_n, nodes)
        if nodes is not None
        else sorted({n for n in [10, 25, 50, 100] if n <= max_n})
    )
    results = []

    for n in sizes:
        s_out, s_in = pareto_strengths(n)
        print(f"\n{'=' * 60}\nE2E N={n} ensemble={ensemble}\n{'=' * 60}")

        # ME strength
        fit = fit_strength_poisson(s_out, s_in, tolerance=tolerance)
        sum_out = np.zeros(n)
        sum_in = np.zeros(n)
        sq_out = np.zeros(n)
        sq_in = np.zeros(n)
        for idx in range(ensemble):
            sample = sample_strength_poisson(fit.x, fit.y, seed=1000 + idx)
            out = np.zeros(n)
            inc = np.zeros(n)
            np.add.at(out, sample.source.astype(int), sample.weight.astype(float))
            np.add.at(inc, sample.target.astype(int), sample.weight.astype(float))
            sum_out += out
            sum_in += inc
            sq_out += out * out
            sq_in += inc * inc
        mean_out = sum_out / ensemble
        se_out = np.sqrt(np.maximum(sq_out / ensemble - mean_out**2, 0) / ensemble)
        z_out = np.divide(mean_out - s_out, se_out, out=np.zeros(n), where=se_out > 0)
        max_z = float(np.max(np.abs(z_out)))
        results.append(
            {"name": "ME strength", "n": n, "max_z": max_z, "converged": fit.converged}
        )
        print(f"  ME strength: max_z={max_z:.2f}")

        # W geometric strength
        fit_w = fit_strength_geometric(s_out, s_in, tolerance=tolerance)
        from odme.models import sample_strength_geometric

        sum_out = np.zeros(n)
        sq_out = np.zeros(n)
        for idx in range(ensemble):
            sample = sample_strength_geometric(fit_w.x, fit_w.y, seed=2000 + idx)
            out = np.zeros(n)
            np.add.at(out, sample.source.astype(int), sample.weight.astype(float))
            sum_out += out
            sq_out += out * out
        mean_out = sum_out / ensemble
        se_out = np.sqrt(np.maximum(sq_out / ensemble - mean_out**2, 0) / ensemble)
        z_out = np.divide(mean_out - s_out, se_out, out=np.zeros(n), where=se_out > 0)
        max_z = float(np.max(np.abs(z_out)))
        results.append(
            {
                "name": "W strength geometric",
                "n": n,
                "max_z": max_z,
                "converged": fit_w.converged,
            }
        )
        print(f"  W strength geometric: max_z={max_z:.2f}")

        sys.stdout.flush()

    return results


def plot_e2e(results):
    """Plot z-scores vs N."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    from benchmarks.common import ensure_figures_dir

    fig_dir = ensure_figures_dir()

    solvers = {}
    for r in results:
        name = r["name"]
        if name not in solvers:
            solvers[name] = {"n": [], "z": []}
        solvers[name]["n"].append(r["n"])
        solvers[name]["z"].append(r["max_z"])

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in sorted(solvers.items()):
        ax.plot(data["n"], data["z"], "o-", label=name)
    ax.axhline(3.0, color="red", linestyle="--", alpha=0.5, label="3-sigma threshold")
    ax.set_xlabel("N")
    ax.set_ylabel("Max |z-score|")
    ax.set_title("Ensemble constraint recovery (z-scores)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "e2e_zscore.png", dpi=150)
    plt.close(fig)
    print(f"Plot: {fig_dir / 'e2e_zscore.png'}")
