"""Shared helpers for ODME benchmark scripts."""

from __future__ import annotations

import gc
import resource
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def pareto_strengths(n: int, total: float = None, seed: int = 42):
    """Generate balanced Pareto-distributed strength sequences."""
    if total is None:
        total = n * 6.0
    rng = np.random.default_rng(seed)
    raw = rng.pareto(2.3, n) + 1.0
    raw = np.clip(raw, 0.0, np.quantile(raw, 0.95))
    s_out = raw / raw.sum() * total
    raw_in = np.roll(raw[::-1], n // 5)
    s_in = raw_in / raw_in.sum() * total
    return s_out, s_in


def degrees_from_strengths(s_out, s_in, frac: float = 0.3):
    """Generate non-saturating degree sequences proportional to strengths."""
    n = len(s_out)
    k_out = s_out / s_out.sum() * (n * frac)
    k_in = s_in / s_in.sum() * (n * frac)
    scale = min((s_out / k_out).min(), (s_in / k_in).min())
    if scale < 1.0:
        k_out *= scale * 0.9
        k_in *= k_out.sum() / k_in.sum()
    return k_out, k_in


def complete_costs(n: int, seed: int = 99):
    """Generate complete lognormal cost matrix as sparse triples."""
    rng = np.random.default_rng(seed)
    src, tgt = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    val = rng.lognormal(0.0, 0.35, (n, n))
    return (
        src.ravel().astype(np.uint64),
        tgt.ravel().astype(np.uint64),
        val.ravel().astype(np.float64),
    )


def target_cost_from_fit(fit, c_val, n: int):
    """Compute target cost from a fitted strength model."""
    return float(np.sum(c_val.reshape(n, n) * np.outer(fit.x, fit.y)))


def max_rss_mb():
    """Current peak RSS in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def time_call(func, *args, **kwargs):
    """Time a function call, return (result, elapsed_seconds)."""
    gc.collect()
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - t0


def compute_strength_residual(fit, s_out, s_in, self_loops=True):
    """Post-hoc max absolute strength residual from fit multipliers."""
    x, y = np.asarray(fit.x), np.asarray(fit.y)
    n = len(x)
    lam = getattr(fit, "lam", None)
    gamma = getattr(fit, "gamma", None)
    z = getattr(fit, "z", None)
    w = getattr(fit, "w", None)
    fam = getattr(fit, "family", "poisson")
    layers = getattr(fit, "layers", None) or 1

    pred_out, pred_in = np.zeros(n), np.zeros(n)
    for i in range(n):
        for j in range(n):
            if not self_loops and i == j:
                continue
            if z is not None and w is not None:
                u = x[i] * y[j]
                v = z[i] * w[j]
                e_neg = np.exp(-u)
                den = e_neg + v * (1.0 - e_neg)
                mean = v * u / den if den > 0 else 0.0
            elif lam is not None:
                u = x[i] * y[j]
                e_neg = np.exp(-u)
                den = e_neg + lam * (1.0 - e_neg)
                mean = lam * u / den if den > 0 else 0.0
            elif gamma is not None:
                mean = x[i] * y[j]
            elif fam == "binomial":
                xy = x[i] * y[j]
                mean = layers * xy / (1.0 + xy)
            else:
                mean = x[i] * y[j]
            pred_out[i] += mean
            pred_in[j] += mean
    return float(max(np.max(np.abs(pred_out - s_out)), np.max(np.abs(pred_in - s_in))))


def compute_degree_residual(fit, k_out, k_in, self_loops=True):
    """Post-hoc max absolute degree residual."""
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
                u = x[i] * y[j]
                v = z[i] * w[j]
                e_neg = np.exp(-u)
                den = e_neg + v * (1.0 - e_neg)
                occ = v * (1.0 - e_neg) / den if den > 0 else 0.0
            else:
                xy = x[i] * y[j]
                occ = xy / (1.0 + xy)
            pred_out[i] += occ
            pred_in[j] += occ
    return float(max(np.max(np.abs(pred_out - k_out)), np.max(np.abs(pred_in - k_in))))


def ensure_results_dir():
    """Create results directory if needed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def ensure_figures_dir():
    """Create figures directory if needed."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR
