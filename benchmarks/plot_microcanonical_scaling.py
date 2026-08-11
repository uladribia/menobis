#!/usr/bin/env python3
"""Generate microcanonical benchmark figures from JSON result files.

Reads the 4 benchmark JSON files and produces 6 PNG figures into docs/figures/.
Does NOT rerun benchmarks — only reads existing results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("benchmarks/results")
FIGURES_DIR = Path("docs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

JSON_FILES = {
    "matrix": RESULTS_DIR / "microcanonical-bench-matrix.json",
    "n100": RESULTS_DIR / "microcanonical-n100-matrix.json",
    "edges": RESULTS_DIR / "microcanonical-edges-events.json",
    "degree": RESULTS_DIR / "microcanonical-degree-events.json",
}

# ---------------------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------------------
def _load_all() -> list[dict]:
    """Load all 4 JSON files into a flat list of dicts."""
    all_rows: list[dict] = []
    for key, path in JSON_FILES.items():
        with open(path) as f:
            data = json.load(f)
        for r in data:
            r["_source"] = key  # tag for debugging
            all_rows.append(r)
    return all_rows


ROWS = _load_all()
print(f"Loaded {len(ROWS)} rows from {len(JSON_FILES)} files.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _field(
    rows: list[dict], name: str, default: float = 0.0
) -> np.ndarray:
    """Extract a float field, replacing None with default."""
    return np.array([r.get(name, default) if r.get(name) is not None else default for r in rows], dtype=float)


def _unique_sorted(rows: list[dict], field: str) -> list:
    """Sorted unique values of a field."""
    return sorted(set(r[field] for r in rows))


def _filter(
    rows: list[dict], **kwargs
) -> list[dict]:
    """Filter rows where all kwargs match."""
    result = rows[:]
    for k, v in kwargs.items():
        result = [r for r in result if r.get(k) == v]
    return result


# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "strength": "#4C72B0",
    "strength-cost": "#DD8452",
    "edges-events": "#55A868",
    "degree-events": "#C44E52",
}
MARKERS = {
    "me": "o",
    "b": "s",
    "w": "^",
}
CONSTRAINT_LABELS = {
    "strength": "Strength",
    "strength-cost": "Strength-Cost",
    "edges-events": "Edges-Events",
    "degree-events": "Degree-Events",
}
FAMILY_LABELS = {
    "me": "ME",
    "b": "B",
    "w": "W",
}
REGIME_LABELS = {
    "sparse": "Sparse",
    "dense": "Dense",
}


def _save(fig: plt.Figure, name: str) -> None:
    """Save figure to docs/figures/."""
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# ===================================================================
# Figure 1 — Microcanonical scaling (wall time vs N, log-log)
# ===================================================================
def figure_scaling(all_rows: list[dict]) -> plt.Figure:
    """Total wall time vs N, one line per (constraint, regime) averaged over families.

    Uses matrix data (N=100,500,1000).  Only strength and strength-cost constraints.
    """
    # Only matrix files (strength, strength-cost) — the edges/degree files have
    # a different pipeline and don't match the matrix N coverage.
    matrix = [r for r in all_rows if r["_source"] in ("matrix", "n100")]
    constraints = ["strength", "strength-cost"]
    regimes = ["sparse", "dense"]
    nodes = [100, 500, 1000]

    fig, ax = plt.subplots(figsize=(8, 5))

    for constraint in constraints:
        for regime in regimes:
            means = []
            for n in nodes:
                subset = _filter(
                    matrix, node_count=n, constraint=constraint, regime=regime
                )
                if subset:
                    means.append(np.mean(_field(subset, "wall_seconds")))
                else:
                    means.append(np.nan)
            label = f"{CONSTRAINT_LABELS[constraint]}-{REGIME_LABELS[regime]}"
            ax.loglog(
                nodes,
                means,
                marker="o",
                linestyle="-" if regime == "dense" else "--",
                color=COLORS[constraint],
                label=label,
                linewidth=1.8,
                markersize=8,
            )

    ax.set_xlabel("Number of nodes $N$")
    ax.set_ylabel("Total wall time (s)")
    ax.set_title("Microcanonical Sampling Scaling")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(50, 2000)
    return fig


# ===================================================================
# Figure 2 — Stage breakdown stacked bar (N=1000)
# ===================================================================
def figure_stage_breakdown(all_rows: list[dict]) -> plt.Figure:
    """Stacked bar chart per (constraint, family) at N=1000.

    Uses matrix data only.  Colors: construction=green, repair=yellow,
    MCMC=blue, gamma_fit=red.
    """
    n1000 = _filter(all_rows, node_count=1000)
    n1000 = [r for r in n1000 if r["_source"] in ("matrix", "n100")]
    constraints = ["strength", "strength-cost"]
    families = ["me", "b", "w"]

    # Build data: for each constraint, for each family + average
    stage_names = ["construction_time_s", "repair_time_s", "mcmc_time_s", "gamma_fit_time_s"]
    stage_labels = ["Construction", "Repair", "MCMC", "Gamma fit"]
    stage_colors = ["#2ca02c", "#ffb74d", "#1f77b4", "#d62728"]

    groups = []
    group_labels = []

    for constraint in constraints:
        for fam in families + ["__avg__"]:
            if fam == "__avg__":
                subset = _filter(n1000, constraint=constraint)
                label = f"{CONSTRAINT_LABELS[constraint]} avg"
            else:
                subset = _filter(n1000, constraint=constraint, family=fam)
                label = f"{CONSTRAINT_LABELS[constraint]} {FAMILY_LABELS[fam]}"
            stages = []
            for s in stage_names:
                vals = _field(subset, s)
                val = float(np.nanmean(vals)) if len(vals) > 0 else 0.0
                stages.append(max(val, 0.0))
            groups.append(stages)
            group_labels.append(label)

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(groups))
    bar_width = 0.6
    bottoms = np.zeros(len(groups))

    for i, (sname, color) in enumerate(zip(stage_labels, stage_colors)):
        heights = [g[i] for g in groups]
        ax.bar(
            x,
            heights,
            bar_width,
            bottom=bottoms,
            label=sname,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms += np.array(heights)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Time (s)")
    ax.set_title("Microcanonical Stage Breakdown (N=1000)")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0.001)

    return fig


# ===================================================================
# Figure 3 — Throughput scatter (proposals/s vs occupied pairs)
# ===================================================================
def figure_throughput(all_rows: list[dict]) -> plt.Figure:
    """Proposal rate vs occupied pairs scatter.

    All 4 data files.  Only rows with non-zero throughput.
    Add theoretical max reference line at 160k proposals/s.
    """
    # Filter to rows with non-zero throughput
    valid = [
        r
        for r in all_rows
        if r.get("structurally_valid_proposals_per_sec", 0) is not None
        and r["structurally_valid_proposals_per_sec"] > 0
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    constraints = sorted(set(r["constraint"] for r in valid))
    families = sorted(set(r["family"] for r in valid))

    for constraint in constraints:
        for family in families:
            subset = [r for r in valid if r["constraint"] == constraint and r["family"] == family]
            if not subset:
                continue
            occ = _field(subset, "occupied_pairs")
            tput = _field(subset, "structurally_valid_proposals_per_sec")
            ax.scatter(
                occ,
                tput,
                marker=MARKERS[family],
                color=COLORS.get(constraint, "#333333"),
                label=f"{CONSTRAINT_LABELS.get(constraint, constraint)}-{FAMILY_LABELS[family]}",
                s=60,
                alpha=0.8,
                edgecolors="black",
                linewidth=0.4,
            )

    # Theoretical max reference line
    ax.axhline(y=160_000, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(
        ax.get_xlim()[1] * 0.6,
        160_000 * 1.05,
        "Theoretical max (160k/s)",
        fontsize=8,
        color="gray",
        va="bottom",
    )

    ax.set_xlabel("Occupied pairs")
    ax.set_ylabel("Structurally valid proposals per second")
    ax.set_title("Microcanonical MCMC Throughput")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    return fig


# ===================================================================
# Figure 4 — Cost ESS vs N
# ===================================================================
def figure_cost_ess(all_rows: list[dict]) -> plt.Figure:
    """Cost ESS vs N for strength-cost constraints.

    Two lines: sparse and dense.  Horizontal dashed line at n_samples (=80).
    """
    sc_rows = [r for r in all_rows if r["constraint"] == "strength-cost"]
    nodes = [100, 500, 1000]
    n_samples = 80  # default fit_estimation_sweeps

    fig, ax = plt.subplots(figsize=(8, 5))

    for regime, marker, ls in [("sparse", "o", "-"), ("dense", "s", "-")]:
        means = []
        for n in nodes:
            subset = [r for r in sc_rows if r["node_count"] == n and r["regime"] == regime]
            vals = [r.get("cost_ess", 0) or 0 for r in subset]
            means.append(np.mean(vals) if vals else np.nan)
        ax.semilogx(
            nodes,
            means,
            marker=marker,
            linestyle=ls,
            label=f"{REGIME_LABELS[regime]}",
            linewidth=1.8,
            markersize=8,
        )

    ax.axhline(y=n_samples, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(
        ax.get_xlim()[1] * 0.6,
        n_samples * 1.05,
        f"n_samples = {n_samples}",
        fontsize=9,
        color="gray",
        va="bottom",
    )

    ax.set_xlabel("Number of nodes $N$")
    ax.set_ylabel("Cost ESS")
    ax.set_title("Microcanonical Cost Effective Sample Size")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    return fig


# ===================================================================
# Figure 5 — Repair steps vs N for B dense
# ===================================================================
def figure_repair_steps(all_rows: list[dict]) -> plt.Figure:
    """Repair steps vs N for B dense (the only case with significant repair).

    Bar chart for B at N=100,500,1000 dense.  Annotate ME/W with <10 steps.
    """
    matrix = [r for r in all_rows if r["_source"] in ("matrix", "n100")]
    nodes = [100, 500, 1000]

    b_dense = []
    for n in nodes:
        subset = _filter(matrix, node_count=n, family="b", regime="dense")
        if subset:
            b_dense.append(max(r.get("repair_steps", 0) or 0 for r in subset))
        else:
            b_dense.append(0)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(nodes))
    ax.bar(x, b_dense, width=0.5, color="#C44E52", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in nodes])
    ax.set_ylabel("Repair steps")
    ax.set_title("Repair Steps for B (Dense)")
    ax.grid(True, axis="y", alpha=0.3)

    # Annotations for ME/W
    for n in nodes:
        me_steps = max(
            (r.get("repair_steps", 0) or 0)
            for r in _filter(matrix, node_count=n, family="me", regime="dense")
        )
        w_steps = max(
            (r.get("repair_steps", 0) or 0)
            for r in _filter(matrix, node_count=n, family="w", regime="dense")
        )
        ax.annotate(
            f"ME={me_steps}  W={w_steps}",
            xy=(nodes.index(n), b_dense[nodes.index(n)]),
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=8,
            ha="center",
            color="#555555",
        )

    return fig


# ===================================================================
# Figure 6 — Peak RSS memory vs N
# ===================================================================
def figure_memory(all_rows: list[dict]) -> plt.Figure:
    """Peak RSS memory vs N, lines for sparse and dense averaged across families and constraints.

    Uses matrix data only (strength, strength-cost).
    """
    matrix = [r for r in all_rows if r["_source"] in ("matrix", "n100")]
    nodes = [100, 500, 1000]

    fig, ax = plt.subplots(figsize=(8, 5))

    for regime, marker, ls in [("sparse", "o", "-"), ("dense", "s", "-")]:
        means = []
        for n in nodes:
            subset = _filter(matrix, node_count=n, regime=regime)
            vals = _field(subset, "memory_rss_peak_mb")
            means.append(np.mean(vals) if len(vals) > 0 else np.nan)
        # Filter out zero values that might skew
        means_arr = np.array(means)
        ax.semilogy(
            nodes,
            means_arr,
            marker=marker,
            linestyle=ls,
            label=f"{REGIME_LABELS[regime]}",
            linewidth=1.8,
            markersize=8,
        )

    ax.set_xlabel("Number of nodes $N$")
    ax.set_ylabel("Peak RSS memory (MB)")
    ax.set_title("Microcanonical Peak Memory Usage")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # Show some annotations
    for n in nodes:
        sparse_vals = _field(_filter(matrix, node_count=n, regime="sparse"), "memory_rss_peak_mb")
        dense_vals = _field(_filter(matrix, node_count=n, regime="dense"), "memory_rss_peak_mb")
        sparse_mean = np.mean(sparse_vals) if len(sparse_vals) > 0 else 0
        dense_mean = np.mean(dense_vals) if len(dense_vals) > 0 else 0
        print(f"  N={n}: sparse mem={sparse_mean:.2f} MB, dense mem={dense_mean:.2f} MB")

    return fig


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    """Generate all 6 figures."""
    print(f"Loaded {len(ROWS)} rows total.")

    print("Figure 1: microcanonical_scaling.png")
    fig = figure_scaling(ROWS)
    _save(fig, "microcanonical_scaling.png")

    print("Figure 2: microcanonical_stage_breakdown.png")
    fig = figure_stage_breakdown(ROWS)
    _save(fig, "microcanonical_stage_breakdown.png")

    print("Figure 3: microcanonical_throughput.png")
    fig = figure_throughput(ROWS)
    _save(fig, "microcanonical_throughput.png")

    print("Figure 4: microcanonical_cost_ess.png")
    fig = figure_cost_ess(ROWS)
    _save(fig, "microcanonical_cost_ess.png")

    print("Figure 5: microcanonical_repair_steps.png")
    fig = figure_repair_steps(ROWS)
    _save(fig, "microcanonical_repair_steps.png")

    print("Figure 6: microcanonical_memory.png")
    fig = figure_memory(ROWS)
    _save(fig, "microcanonical_memory.png")

    print("\nAll figures saved to docs/figures/")


if __name__ == "__main__":
    main()