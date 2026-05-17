"""Ensemble equivalence validation: micro/canonical/grand-canonical convergence.

Generates figures in docs/figures/ and validates that the three ensembles
converge at large T for all higher-order graph statistics.
"""

from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from odme.analysis import directed_degrees, directed_strengths
from odme.analysis.graph_algorithms import clustering_coefficient
from odme.analysis.stats import compute_all_stats
from odme.data.frames import EdgeTable
from odme.models import (
    fit_strength_poisson,
    sample_strength_microcanonical,
    sample_strength_multinomial,
    sample_strength_poisson,
)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N = 5
REPETITIONS = 200
T_VALUES = [100, 500, 2000, 10000]

# Relative strength profile (Pareto-like, asymmetric).
P_OUT = np.array([0.35, 0.25, 0.20, 0.12, 0.08])
P_IN = np.array([0.30, 0.22, 0.20, 0.15, 0.13])
# Normalize to sum to 1.
P_OUT = P_OUT / P_OUT.sum()
P_IN = P_IN / P_IN.sum()


def _balanced_integer_strengths(
    p_out: np.ndarray, p_in: np.ndarray, total: int
) -> tuple[np.ndarray, np.ndarray]:
    s_out = np.round(p_out * total).astype(np.uint64)
    s_in = np.round(p_in * total).astype(np.uint64)
    diff = int(s_out.sum()) - int(s_in.sum())
    if diff > 0:
        s_in[np.argmax(s_in)] += abs(diff)
    elif diff < 0:
        s_out[np.argmax(s_out)] += abs(diff)
    return s_out, s_in


def _stats_vector(edges: EdgeTable) -> np.ndarray:
    """Extract a flat vector of higher-order statistics from a sample."""
    s = directed_strengths(edges)
    k = directed_degrees(edges)
    all_stats = compute_all_stats(edges)
    clust = clustering_coefficient(edges).values

    parts = [
        s.out.astype(float),
        s.incoming.astype(float),
        k.out.astype(float),
        k.incoming.astype(float),
        all_stats.y2_out,
        all_stats.y2_in,
        all_stats.k_nn_out,
        all_stats.k_nn_in,
        all_stats.s_nn_out,
        all_stats.s_nn_in,
        clust.astype(float),
    ]
    return np.concatenate(parts)


def _ensemble_stats(
    sampler: Callable[[int], EdgeTable], repetitions: int
) -> tuple[np.ndarray, np.ndarray]:
    vectors = []
    for seed in range(repetitions):
        sample = sampler(seed)
        vectors.append(_stats_vector(sample))
    stacked = np.vstack(vectors)
    return stacked.mean(axis=0), stacked.std(axis=0, ddof=1)


def _run_convergence() -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    results: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        "microcanonical": {},
        "canonical": {},
        "grand_canonical": {},
    }

    for total in T_VALUES:
        s_out, s_in = _balanced_integer_strengths(P_OUT, P_IN, total)
        fit = fit_strength_poisson(s_out, s_in)

        _s_out, _s_in, _fit, _total = s_out, s_in, fit, total
        results["microcanonical"][total] = _ensemble_stats(
            lambda seed, so=_s_out, si=_s_in: sample_strength_microcanonical(
                so, si, seed=seed
            ),
            REPETITIONS,
        )
        results["canonical"][total] = _ensemble_stats(
            lambda seed, f=_fit, t=_total: sample_strength_multinomial(
                f.x, f.y, total_events=t, seed=seed
            ),
            REPETITIONS,
        )
        results["grand_canonical"][total] = _ensemble_stats(
            lambda seed, f=_fit: sample_strength_poisson(f.x, f.y, seed=seed),
            REPETITIONS,
        )

    return results


def _collect_per_node_stats(
    total: int, repetitions: int
) -> dict[str, dict[str, np.ndarray]]:
    """Collect per-node ensemble-averaged Y2_out and s_nn_out for each ensemble."""
    s_out, s_in = _balanced_integer_strengths(P_OUT, P_IN, total)
    fit = fit_strength_poisson(s_out, s_in)

    ensemble_y2: dict[str, list[np.ndarray]] = {
        "microcanonical": [],
        "canonical": [],
        "grand_canonical": [],
    }
    ensemble_snn: dict[str, list[np.ndarray]] = {
        "microcanonical": [],
        "canonical": [],
        "grand_canonical": [],
    }

    samplers: dict[str, Callable[[int], EdgeTable]] = {
        "microcanonical": lambda seed, so=s_out, si=s_in: (
            sample_strength_microcanonical(so, si, seed=seed)
        ),
        "canonical": lambda seed, f=fit, t=total: sample_strength_multinomial(
            f.x, f.y, total_events=t, seed=seed
        ),
        "grand_canonical": lambda seed, f=fit: sample_strength_poisson(
            f.x, f.y, seed=seed
        ),
    }

    for name, sampler in samplers.items():
        for seed in range(repetitions):
            sample = sampler(seed)
            stats = compute_all_stats(sample)
            ensemble_y2[name].append(stats.y2_out)
            ensemble_snn[name].append(stats.s_nn_out)

    result: dict[str, dict[str, np.ndarray]] = {}
    for name in ensemble_y2:
        result[name] = {
            "y2_out": np.vstack(ensemble_y2[name]).mean(axis=0),
            "y2_out_std": np.vstack(ensemble_y2[name]).std(axis=0, ddof=1),
            "snn_out": np.vstack(ensemble_snn[name]).mean(axis=0),
            "snn_out_std": np.vstack(ensemble_snn[name]).std(axis=0, ddof=1),
        }
    return result


def _plot_per_node_vs_strength() -> None:
    """Plot Y2 and s_nn vs relative strength p_s for each ensemble and T."""
    ensemble_names = ["microcanonical", "canonical", "grand_canonical"]
    ensemble_labels = ["microcanonical", "canonical", "grand-canonical"]
    markers = ["o", "s", "^"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(
        2, len(T_VALUES), figsize=(4 * len(T_VALUES), 8), sharey="row"
    )
    if len(T_VALUES) == 1:
        axes = axes.reshape(2, 1)

    for col, total in enumerate(T_VALUES):
        per_node = _collect_per_node_stats(total, REPETITIONS)
        p_out = P_OUT

        # Y2 vs p_s
        ax = axes[0, col]
        for idx, name in enumerate(ensemble_names):
            ax.errorbar(
                p_out,
                per_node[name]["y2_out"],
                yerr=per_node[name]["y2_out_std"] / np.sqrt(REPETITIONS),
                fmt=markers[idx],
                color=colors[idx],
                label=ensemble_labels[idx],
                markersize=6,
                capsize=3,
                alpha=0.8,
            )
        ax.set_xlabel("$p_s = s^{out} / T$")
        if col == 0:
            ax.set_ylabel("$\\langle Y_2^{out} \\rangle$")
        ax.set_title(f"T = {total}")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7)

        # s_nn / T vs p_s
        ax = axes[1, col]
        for idx, name in enumerate(ensemble_names):
            ax.errorbar(
                p_out,
                per_node[name]["snn_out"] / total,
                yerr=per_node[name]["snn_out_std"] / total / np.sqrt(REPETITIONS),
                fmt=markers[idx],
                color=colors[idx],
                label=ensemble_labels[idx],
                markersize=6,
                capsize=3,
                alpha=0.8,
            )
        ax.set_xlabel("$p_s = s^{out} / T$")
        if col == 0:
            ax.set_ylabel("$\\langle s^{w,out}_{nn} \\rangle / T$")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Higher-order statistics vs relative strength across ensembles",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "ensemble_y2_snn_vs_ps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_convergence(
    results: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
) -> None:
    stat_labels = (
        [f"s_out_{i}" for i in range(N)]
        + [f"s_in_{i}" for i in range(N)]
        + [f"k_out_{i}" for i in range(N)]
        + [f"k_in_{i}" for i in range(N)]
        + [f"y2_out_{i}" for i in range(N)]
        + [f"y2_in_{i}" for i in range(N)]
        + [f"knn_out_{i}" for i in range(N)]
        + [f"knn_in_{i}" for i in range(N)]
        + [f"snn_out_{i}" for i in range(N)]
        + [f"snn_in_{i}" for i in range(N)]
        + [f"clust_{i}" for i in range(N)]
    )

    # 1. Mean convergence: max absolute difference between ensembles vs T.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Max |mean_micro - mean_canonical| and |mean_micro - mean_gc| per T.
    mc_vs_can = []
    mc_vs_gc = []
    can_vs_gc = []
    for total in T_VALUES:
        m_micro = results["microcanonical"][total][0]
        m_can = results["canonical"][total][0]
        m_gc = results["grand_canonical"][total][0]
        # Normalize by T to compare across scales.
        norm = max(1.0, total)
        mc_vs_can.append(np.max(np.abs(m_micro - m_can)) / norm)
        mc_vs_gc.append(np.max(np.abs(m_micro - m_gc)) / norm)
        can_vs_gc.append(np.max(np.abs(m_can - m_gc)) / norm)

    ax = axes[0]
    ax.loglog(T_VALUES, mc_vs_can, "o-", label="micro vs canonical")
    ax.loglog(T_VALUES, mc_vs_gc, "s-", label="micro vs grand-canonical")
    ax.loglog(T_VALUES, can_vs_gc, "^-", label="canonical vs grand-canonical")
    ax.set_xlabel("T (total events)")
    ax.set_ylabel("max |Δmean| / T")
    ax.set_title("Mean convergence across ensembles")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Variance decrease: mean std across statistics vs T.
    ax = axes[1]
    for name, marker in [
        ("microcanonical", "o-"),
        ("canonical", "s-"),
        ("grand_canonical", "^-"),
    ]:
        mean_stds = []
        for total in T_VALUES:
            std = results[name][total][1]
            mean_stds.append(np.mean(std) / total)
        ax.loglog(T_VALUES, mean_stds, marker, label=name)
    ax.set_xlabel("T (total events)")
    ax.set_ylabel("mean(std) / T")
    ax.set_title("Variance decrease with T")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "ensemble_equivalence.png", dpi=150)
    plt.close(fig)

    # 3. Per-statistic comparison at largest T.
    largest_t = T_VALUES[-1]
    m_micro = results["microcanonical"][largest_t][0]
    m_can = results["canonical"][largest_t][0]
    m_gc = results["grand_canonical"][largest_t][0]

    fig, ax = plt.subplots(figsize=(14, 5))
    x_pos = np.arange(len(stat_labels))
    width = 0.25
    ax.bar(x_pos - width, m_micro, width, label="microcanonical", alpha=0.8)
    ax.bar(x_pos, m_can, width, label="canonical", alpha=0.8)
    ax.bar(x_pos + width, m_gc, width, label="grand-canonical", alpha=0.8)
    ax.set_xticks(x_pos[::5])
    ax.set_xticklabels(
        [stat_labels[i] for i in range(0, len(stat_labels), 5)],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel("Mean value")
    ax.set_title(f"Per-statistic ensemble means at T={largest_t}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "ensemble_per_statistic.png", dpi=150)
    plt.close(fig)


def test_ensemble_equivalence_convergence() -> None:
    """Three ensembles converge at large T for all higher-order statistics."""
    results = _run_convergence()
    _plot_convergence(results)
    _plot_per_node_vs_strength()

    # At largest T, assert convergence.
    largest_t = T_VALUES[-1]
    m_micro = results["microcanonical"][largest_t][0]
    m_can = results["canonical"][largest_t][0]
    m_gc = results["grand_canonical"][largest_t][0]

    # Relative difference should be small at large T.
    scale = np.maximum(np.abs(m_micro), 1.0)
    np.testing.assert_allclose(m_micro / scale, m_can / scale, atol=0.15)
    np.testing.assert_allclose(m_micro / scale, m_gc / scale, atol=0.15)

    # Variances should decrease with T for all ensembles.
    for name in ["microcanonical", "canonical", "grand_canonical"]:
        std_small = np.mean(results[name][T_VALUES[0]][1])
        std_large = np.mean(results[name][T_VALUES[-1]][1])
        # Variance per unit T should decrease.
        assert std_large / T_VALUES[-1] < std_small / T_VALUES[0], (
            f"{name}: variance/T did not decrease"
        )


def test_microcanonical_preserves_exact_strengths() -> None:
    """Microcanonical samples preserve exact strength sequences."""
    for total in [100, 1000]:
        s_out, s_in = _balanced_integer_strengths(P_OUT, P_IN, total)
        for seed in range(50):
            sample = sample_strength_microcanonical(s_out, s_in, seed=seed)
            actual_s = directed_strengths(sample)
            np.testing.assert_array_equal(actual_s.out, s_out)
            np.testing.assert_array_equal(actual_s.incoming, s_in)
            assert sample.total_events == total


def test_canonical_preserves_exact_total() -> None:
    """Canonical multinomial always preserves exact T."""
    for total in [100, 1000]:
        s_out, s_in = _balanced_integer_strengths(P_OUT, P_IN, total)
        fit = fit_strength_poisson(s_out, s_in)
        for seed in range(50):
            sample = sample_strength_multinomial(
                fit.x, fit.y, total_events=total, seed=seed
            )
            assert sample.total_events == total
