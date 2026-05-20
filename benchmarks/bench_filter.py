"""Filter benchmark: fit null → sample → filter → verify FPR/FNR."""

import sys

import numpy as np

from benchmarks.common import pareto_strengths


def bench_filter(max_n=100, ensemble=100):
    """Benchmark false-positive rates across alpha levels."""
    from odme.filtering import filter_strength_poisson
    from odme.models import fit_strength_poisson, sample_strength_poisson

    sizes = sorted({n for n in [10, 25, 50, 100] if n <= max_n})
    alphas = [0.01, 0.05, 0.1]
    results = []

    for n in sizes:
        s_out, s_in = pareto_strengths(n)
        print(f"\n--- Filter FPR N={n} ensemble={ensemble} ---")

        for alpha in alphas:
            upper_count = 0
            lower_count = 0
            total_edges = 0
            for idx in range(ensemble):
                sample = sample_strength_poisson(
                    s_out / np.sqrt(s_out.sum()),
                    s_in / np.sqrt(s_in.sum()),
                    seed=5000 + idx,
                )
                if sample.num_edges == 0:
                    continue
                result = filter_strength_poisson(sample, alpha=alpha)
                total_edges += sample.num_edges
                upper_count += len(result.upper.edges.source) if result.upper.edges else 0
                lower_count += len(result.lower.edges.source) if result.lower.edges else 0

            fpr_upper = upper_count / total_edges if total_edges > 0 else 0.0
            fpr_lower = lower_count / total_edges if total_edges > 0 else 0.0
            results.append({
                "n": n, "alpha": alpha,
                "fpr_upper": fpr_upper, "fpr_lower": fpr_lower,
                "total_edges": total_edges,
            })
            print(f"  alpha={alpha:.2f}: FPR_upper={fpr_upper:.4f} FPR_lower={fpr_lower:.4f}")
            sys.stdout.flush()

    return results


def plot_filter(results):
    """Plot FPR vs alpha."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    from benchmarks.common import ensure_figures_dir
    fig_dir = ensure_figures_dir()

    fig, ax = plt.subplots(figsize=(8, 6))
    ns = sorted(set(r["n"] for r in results))
    for n in ns:
        sub = [r for r in results if r["n"] == n]
        alphas = [r["alpha"] for r in sub]
        fprs = [r["fpr_upper"] for r in sub]
        ax.plot(alphas, fprs, "o-", label=f"N={n}")
    ax.plot([0, 0.15], [0, 0.15], "k--", alpha=0.3, label="ideal")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("FPR (upper tail)")
    ax.set_title("Filter false positive rate calibration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "filter_fpr.png", dpi=150)
    plt.close(fig)
    print(f"Plot: {fig_dir / 'filter_fpr.png'}")
