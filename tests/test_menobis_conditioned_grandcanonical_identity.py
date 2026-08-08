"""Conditioned grand-canonical identity validation for ME, B, W.

This is the strongest finite-size validation available (thesis §"exact
conditioning identity"): for a grand-canonical law P_GC, conditioning on the
hard constraints C(t) = c gives exactly the microcanonical law:

    P_GC(t | C(t) = c) = P_MC(t | C = c).

We compare the *occupation histogram* distribution (a summary observable with
enough expected counts to run a chi-square test) between:

1. grand-canonical EDGES_EVENTS samples conditioned on (E, T);
2. the direct microcanonical fixed-(E,T) sampler.

The test is statistically expensive (many GC samples are discarded), so it is
kept separate from the fast unit/E2E suites and uses a conservative p-value
threshold (p > 0.01) with fixed seeds.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from menobis.models.fitting import _fit_edges_events
from menobis.models.generation import _sample_edges_events
from menobis.models.spec import Constraint, Ensemble, ModelFamily
from menobis.routing import sample_model

# Parameter points chosen for adequate GC conditional yield (> 1%).
CASES = [
    # (family, node_count, E, T, layers)
    ("me", 4, 5, 12, 1),
    ("b", 4, 5, 12, 3),
    ("w", 4, 5, 12, 3),
    ("w", 3, 4, 8, 2),  # second W point to confirm the first is not a bug
]

# GC family-name mapping
_GC_FAMILY = {"me": "poisson", "b": "binomial", "w": "negative_binomial"}

_GC_MAX_ATTEMPTS = 1_000_000
_GC_TARGET = 15_000
_MC_TARGET = 15_000


def _occ_histogram(s) -> tuple[int, ...]:
    """Occupation histogram as a fixed-length tuple of per-value counts."""
    return tuple(np.bincount(s.occ_num.astype(int)).tolist())


def _gc_conditioned(family: str, n: int, e: int, t: int, layers: int, seed: int):
    """Grand-canonical EDGES_EVENTS samples conditioned on (E, T)."""
    gc_fam = _GC_FAMILY[family]
    fit = _fit_edges_events(
        gc_fam, float(e), t, n, layers=layers, self_loops=True, max_iterations=100
    )
    results: list[tuple[int, ...]] = []
    attempts = 0
    while len(results) < _GC_TARGET and attempts < _GC_MAX_ATTEMPTS:
        attempts += 1
        s = _sample_edges_events(
            n,
            fit.q,
            fit.occupation,
            gc_fam,
            layers=layers,
            self_loops=True,
            seed=attempts * 7919 + seed,
        )
        if len(s) == e and s.occ_num.sum() == t:
            results.append(_occ_histogram(s))
    return results


def _mc_samples(family: str, n: int, e: int, t: int, layers: int, seed: int):
    """Direct microcanonical samples."""
    rng = np.random.default_rng(seed)
    results: list[tuple[int, ...]] = []
    for _ in range(_MC_TARGET):
        kw: dict = {
            "ensemble": Ensemble.MICROCANONICAL,
            "family": ModelFamily[family.upper()],
            "constraint": Constraint.EDGES_EVENTS,
            "node_count": n,
            "total_events": t,
            "target_edges": e,
            "self_loops": True,
            "seed": int(rng.integers(0, 2**31)),
        }
        if family in ("b", "w"):
            kw["layers"] = layers
        s = sample_model(**kw)
        results.append(_occ_histogram(s))
    return results


def _chi2_pvalue(chi2: float, dof: int) -> float:
    """Upper-tail p-value of a chi-square distribution (no scipy).

    Uses the Wilson-Hilferty normal approximation, accurate enough for
    dof >= 1 at the 1% significance level used here.
    """
    if dof <= 0:
        return 1.0
    z = ((chi2 / dof) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dof))) / np.sqrt(
        2.0 / (9.0 * dof)
    )
    # standard normal survival function (erfc approximation)
    p = 0.5 * _erfc(z / np.sqrt(2.0))
    return float(p)


def _erfc(x: float) -> float:
    # Abramowitz & Stegun 7.1.26
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    y = t * (
        0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    return y * np.exp(-x * x) if x >= 0 else 2.0 - y * np.exp(-x * x)


def _compare(family: str, n: int, e: int, t: int, layers: int, seed: int) -> dict:
    gc = _gc_conditioned(family, n, e, t, layers, seed)
    mc = _mc_samples(family, n, e, t, layers, seed + 1000)

    gcc, mcc = Counter(gc), Counter(mc)
    cats = sorted(set(gcc) | set(mcc))
    tg, tm = len(gc), len(mc)

    table_g, table_m, merged_g, merged_m = [], [], 0, 0
    for cat in cats:
        og, om = gcc.get(cat, 0), mcc.get(cat, 0)
        if (og + om) * tg / (tg + tm) >= 5:
            table_g.append(og)
            table_m.append(om)
        else:
            merged_g += og
            merged_m += om
    if merged_g + merged_m > 0:
        table_g.append(merged_g)
        table_m.append(merged_m)

    dof = len(table_g) - 1
    chi2 = 0.0
    for og, om in zip(table_g, table_m, strict=True):
        eg = (og + om) * tg / (tg + tm)
        em = (og + om) * tm / (tg + tm)
        chi2 += (og - eg) ** 2 / eg + (om - em) ** 2 / em

    tv = 0.5 * sum(
        abs(og / tg - om / tm) for og, om in zip(table_g, table_m, strict=True)
    )
    p = _chi2_pvalue(chi2, dof)
    return {
        "gc": len(gc),
        "mc": len(mc),
        "cats": len(cats),
        "used": len(table_g),
        "chi2": chi2,
        "dof": dof,
        "p": p,
        "tv": tv,
    }


@pytest.mark.heavy
@pytest.mark.parametrize("family,n,e,t,layers", CASES)
def test_conditioned_grandcanonical_identity(family, n, e, t, layers) -> None:
    """P_GC(obs | E,T) == P_MC(obs | E,T) within statistical tolerance.

    Reject only if p < 0.01 (conservative; the identity is exact at finite
    size, so a significant p-value indicates an implementation bug).
    """
    res = _compare(family, n, e, t, layers, seed=42)
    assert res["gc"] >= 1000, f"GC yield too low ({res['gc']})"
    assert res["p"] > 0.01, (
        f"{family}: chi2={res['chi2']:.1f} (dof={res['dof']}, p={res['p']:.4f}) "
        f"→ GC and MC distributions differ! TV={res['tv']:.4f}"
    )
