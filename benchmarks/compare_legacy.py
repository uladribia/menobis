"""Compare ODME modern implementation results against legacy thesis-era code.

This script is intentionally standalone (not part of the typer CLI) because it
requires the legacy code to be importable, which depends on the legacy Python 2
environment or manual adaptation.

Usage:
    uv run python benchmarks/compare_legacy.py

The script compares fitted multipliers and expected values between:
- Modern ODME Rust-backed fitting
- Legacy Python fitter_s.py, fitter_E.py, fitter_k.py, fitter_sk.py, fitter_grav.py

Only ME cases are compared (the legacy code supports ME and B; W uses CVXOPT
which we intentionally do not port).
"""

import sys
from pathlib import Path

import numpy as np

# Add legacy code to path
LEGACY_DIR = Path(__file__).resolve().parent.parent / "2. Model Fitting"
sys.path.insert(0, str(LEGACY_DIR))


def _pareto_strengths(n: int, total: float = None, seed: int = 42):
    if total is None:
        total = n * 6.0
    rng = np.random.default_rng(seed)
    raw = rng.pareto(2.3, n) + 1.0
    raw = np.clip(raw, 0.0, np.quantile(raw, 0.95))
    s_out = raw / raw.sum() * total
    raw_in = np.roll(raw[::-1], n // 5)
    s_in = raw_in / raw_in.sum() * total
    return s_out, s_in


def compare_strength_me(n: int = 10):
    """Compare ME fixed-strength fitting."""
    from odme.models import fit_strength_poisson

    s_out, s_in = _pareto_strengths(n)
    modern = fit_strength_poisson(s_out, s_in, self_loops=True)

    try:
        from multi_edge_fitter.fitter_s import balance_xy
        legacy_x, legacy_y = balance_xy(
            s_in, s_out, tol=1e-9, maxreps=10000, selfs=True, case="ME"
        )
        max_diff_x = float(np.max(np.abs(modern.x - legacy_x)))
        max_diff_y = float(np.max(np.abs(modern.y - legacy_y)))
        print(f"  ME strength N={n}: max|x_diff|={max_diff_x:.2e} max|y_diff|={max_diff_y:.2e}")
    except Exception as e:
        print(f"  ME strength N={n}: LEGACY IMPORT FAILED: {e}")


def compare_degree_bernoulli(n: int = 10):
    """Compare Bernoulli fixed-degree fitting."""
    from odme.models import fit_degree_bernoulli

    s_out, s_in = _pareto_strengths(n)
    k_out = s_out / s_out.sum() * (n * 0.3)
    k_in = s_in / s_in.sum() * (n * 0.3)
    modern = fit_degree_bernoulli(k_out, k_in, self_loops=True)

    try:
        from multi_edge_fitter.fitter_k import balance_xy
        legacy_x, legacy_y = balance_xy(
            k_in, k_out, tol=1e-9, maxreps=10000, selfs=True
        )
        max_diff_x = float(np.max(np.abs(modern.x - legacy_x)))
        max_diff_y = float(np.max(np.abs(modern.y - legacy_y)))
        print(f"  Degree Bernoulli N={n}: max|x_diff|={max_diff_x:.2e} max|y_diff|={max_diff_y:.2e}")
    except Exception as e:
        print(f"  Degree Bernoulli N={n}: LEGACY IMPORT FAILED: {e}")


def compare_strength_edges(n: int = 10):
    """Compare ME strength-edges fitting."""
    from odme.models import fit_strength_edges_poisson

    s_out, s_in = _pareto_strengths(n)
    target_edges = n * 1.8
    modern = fit_strength_edges_poisson(s_out, s_in, target_edges, self_loops=True)

    try:
        from multi_edge_fitter.fitter_E import fit_lambda
        legacy_x, legacy_y, legacy_lam = fit_lambda(
            s_in, s_out, target_edges, selfs=True, verbose=False
        )
        max_diff_x = float(np.max(np.abs(modern.x - legacy_x)))
        max_diff_y = float(np.max(np.abs(modern.y - legacy_y)))
        lam_diff = abs(modern.lam - legacy_lam)
        print(
            f"  Strength-edges N={n}: max|x_diff|={max_diff_x:.2e} "
            f"max|y_diff|={max_diff_y:.2e} |lam_diff|={lam_diff:.2e}"
        )
    except Exception as e:
        print(f"  Strength-edges N={n}: LEGACY IMPORT FAILED: {e}")


def compare_strength_cost(n: int = 10):
    """Compare ME strength-cost (gravity) fitting."""
    from odme.models import fit_strength_cost_poisson

    s_out, s_in = _pareto_strengths(n)
    rng = np.random.default_rng(99)
    costs = rng.lognormal(0.0, 0.35, (n, n))
    c_src = np.repeat(np.arange(n), n).astype(np.uint64)
    c_tgt = np.tile(np.arange(n), n).astype(np.uint64)
    c_val = costs.ravel()
    # Use modern fit to get a target cost
    from odme.models import fit_strength_poisson
    base = fit_strength_poisson(s_out, s_in)
    target_cost = float(np.sum(costs * np.outer(base.x, base.y)))

    modern = fit_strength_cost_poisson(
        s_out, s_in, c_src, c_tgt, c_val, target_cost, self_loops=True
    )

    try:
        from multi_edge_fitter.fitter_grav import fit_gamma
        legacy_x, legacy_y, legacy_gamma = fit_gamma(
            s_in, s_out, costs, target_cost, selfs=True, verbose=False
        )
        max_diff_x = float(np.max(np.abs(modern.x - legacy_x)))
        max_diff_y = float(np.max(np.abs(modern.y - legacy_y)))
        gamma_diff = abs(modern.gamma - legacy_gamma)
        print(
            f"  Strength-cost N={n}: max|x_diff|={max_diff_x:.2e} "
            f"max|y_diff|={max_diff_y:.2e} |gamma_diff|={gamma_diff:.2e}"
        )
    except Exception as e:
        print(f"  Strength-cost N={n}: LEGACY IMPORT FAILED: {e}")


def main():
    print("=" * 70)
    print("ODME vs Legacy Comparison")
    print("=" * 70)
    print("\nNote: Legacy imports will fail unless the thesis-era code is")
    print("adapted for Python 3. This script documents the comparison interface.\n")

    for n in [10, 25]:
        print(f"\n--- N = {n} ---")
        compare_strength_me(n)
        compare_degree_bernoulli(n)
        compare_strength_edges(n)
        compare_strength_cost(n)


if __name__ == "__main__":
    main()
