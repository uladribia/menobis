"""Unified dispatch registry for benchmark fit/sample/filter operations.

Maps (family, constraint) pairs to the appropriate menobis functions, eliminating
the repeated if/elif dispatch trees that previously lived in fit.py, sample.py,
and filter.py.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from menobis.data.frames import EdgeTable, ProbabilityTable
from menobis.filtering import (
    FilterResult,
    _solve_ztp_rate,
    filter_custom_poisson,
    filter_degree_events_binomial,
    filter_degree_events_geometric,
    filter_degree_events_negative_binomial,
    filter_degree_events_poisson,
    filter_strength_binomial,
    filter_strength_cost_binomial,
    filter_strength_cost_geometric,
    filter_strength_cost_negative_binomial,
    filter_strength_cost_poisson,
    filter_strength_degree_geometric,
    filter_strength_degree_negative_binomial,
    filter_strength_degree_poisson,
    filter_strength_edges_geometric,
    filter_strength_edges_negative_binomial,
    filter_strength_edges_poisson,
    filter_strength_geometric,
    filter_strength_negative_binomial,
)
from menobis.models import (
    fit_degree_events_binomial,
    fit_degree_events_geometric,
    fit_degree_events_negative_binomial,
    fit_degree_events_poisson,
    fit_strength_binomial,
    fit_strength_cost_binomial_coordinates,
    fit_strength_cost_geometric_coordinates,
    fit_strength_cost_negative_binomial_coordinates,
    fit_strength_cost_poisson_coordinates,
    fit_strength_degree_binomial,
    fit_strength_degree_geometric,
    fit_strength_degree_negative_binomial,
    fit_strength_degree_poisson,
    fit_strength_edges_binomial,
    fit_strength_edges_geometric,
    fit_strength_edges_negative_binomial,
    fit_strength_edges_poisson,
    fit_strength_geometric,
    fit_strength_negative_binomial,
    fit_strength_poisson,
    sample_degree_events_binomial,
    sample_degree_events_geometric,
    sample_degree_events_negative_binomial,
    sample_degree_events_poisson,
    sample_strength_binomial,
    sample_strength_cost_binomial,
    sample_strength_cost_geometric,
    sample_strength_cost_negative_binomial,
    sample_strength_cost_poisson,
    sample_strength_degree_binomial,
    sample_strength_degree_geometric,
    sample_strength_degree_negative_binomial,
    sample_strength_degree_poisson,
    sample_strength_edges_binomial,
    sample_strength_edges_geometric,
    sample_strength_edges_negative_binomial,
    sample_strength_edges_poisson,
    sample_strength_geometric,
    sample_strength_negative_binomial,
    sample_strength_poisson,
)
from menobis.utilities.synthetic import SyntheticConstraints, SyntheticNetwork

FAMILIES = ("me", "b", "w", "wnb")
CONSTRAINTS = (
    "strength",
    "strength-cost",
    "strength-edges",
    "strength-degree",
    "degree-events",
)


@dataclass(frozen=True)
class CaseDispatch:
    """Callable triplet for one (family, constraint) combination."""

    fit: Callable[..., Any]
    sample: Callable[..., EdgeTable]
    filter: Callable[..., FilterResult] | None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[tuple[str, str], CaseDispatch] = {
    # ME family
    ("me", "strength"): CaseDispatch(
        fit=fit_strength_poisson,
        sample=sample_strength_poisson,
        filter=None,  # uses custom_poisson path
    ),
    ("me", "strength-cost"): CaseDispatch(
        fit=fit_strength_cost_poisson_coordinates,
        sample=sample_strength_cost_poisson,
        filter=filter_strength_cost_poisson,
    ),
    ("me", "strength-edges"): CaseDispatch(
        fit=fit_strength_edges_poisson,
        sample=sample_strength_edges_poisson,
        filter=filter_strength_edges_poisson,
    ),
    ("me", "strength-degree"): CaseDispatch(
        fit=fit_strength_degree_poisson,
        sample=sample_strength_degree_poisson,
        filter=filter_strength_degree_poisson,
    ),
    ("me", "degree-events"): CaseDispatch(
        fit=fit_degree_events_poisson,
        sample=sample_degree_events_poisson,
        filter=filter_degree_events_poisson,
    ),
    # B family
    ("b", "strength"): CaseDispatch(
        fit=fit_strength_binomial,
        sample=sample_strength_binomial,
        filter=filter_strength_binomial,
    ),
    ("b", "strength-cost"): CaseDispatch(
        fit=fit_strength_cost_binomial_coordinates,
        sample=sample_strength_cost_binomial,
        filter=filter_strength_cost_binomial,
    ),
    ("b", "strength-edges"): CaseDispatch(
        fit=fit_strength_edges_binomial,
        sample=sample_strength_edges_binomial,
        filter=None,
    ),
    ("b", "strength-degree"): CaseDispatch(
        fit=fit_strength_degree_binomial,
        sample=sample_strength_degree_binomial,
        filter=None,
    ),
    ("b", "degree-events"): CaseDispatch(
        fit=fit_degree_events_binomial,
        sample=sample_degree_events_binomial,
        filter=filter_degree_events_binomial,
    ),
    # W (geometric) family
    ("w", "strength"): CaseDispatch(
        fit=fit_strength_geometric,
        sample=sample_strength_geometric,
        filter=filter_strength_geometric,
    ),
    ("w", "strength-cost"): CaseDispatch(
        fit=fit_strength_cost_geometric_coordinates,
        sample=sample_strength_cost_geometric,
        filter=filter_strength_cost_geometric,
    ),
    ("w", "strength-edges"): CaseDispatch(
        fit=fit_strength_edges_geometric,
        sample=sample_strength_edges_geometric,
        filter=filter_strength_edges_geometric,
    ),
    ("w", "strength-degree"): CaseDispatch(
        fit=fit_strength_degree_geometric,
        sample=sample_strength_degree_geometric,
        filter=filter_strength_degree_geometric,
    ),
    ("w", "degree-events"): CaseDispatch(
        fit=fit_degree_events_geometric,
        sample=sample_degree_events_geometric,
        filter=filter_degree_events_geometric,
    ),
    # WNB (negative binomial) family
    ("wnb", "strength"): CaseDispatch(
        fit=fit_strength_negative_binomial,
        sample=sample_strength_negative_binomial,
        filter=filter_strength_negative_binomial,
    ),
    ("wnb", "strength-cost"): CaseDispatch(
        fit=fit_strength_cost_negative_binomial_coordinates,
        sample=sample_strength_cost_negative_binomial,
        filter=filter_strength_cost_negative_binomial,
    ),
    ("wnb", "strength-edges"): CaseDispatch(
        fit=fit_strength_edges_negative_binomial,
        sample=sample_strength_edges_negative_binomial,
        filter=filter_strength_edges_negative_binomial,
    ),
    ("wnb", "strength-degree"): CaseDispatch(
        fit=fit_strength_degree_negative_binomial,
        sample=sample_strength_degree_negative_binomial,
        filter=filter_strength_degree_negative_binomial,
    ),
    ("wnb", "degree-events"): CaseDispatch(
        fit=fit_degree_events_negative_binomial,
        sample=sample_degree_events_negative_binomial,
        filter=filter_degree_events_negative_binomial,
    ),
}


# ---------------------------------------------------------------------------
# Argument builders
# ---------------------------------------------------------------------------


def build_fit_kwargs(
    family: str,
    constraint: str,
    constraints: SyntheticConstraints,
    network: SyntheticNetwork,
    *,
    self_loops: bool,
    tolerance: float,
    degree_tolerance: float,
    max_iterations: int,
    layers: int | None,
) -> dict[str, Any]:
    """Build keyword arguments for a fit function from constraint data."""
    common: dict[str, Any] = {
        "self_loops": self_loops,
        "tolerance": tolerance,
        "max_iterations": max_iterations,
    }
    degree_common: dict[str, Any] = {
        "self_loops": self_loops,
        "tolerance": degree_tolerance,
        "max_iterations": max_iterations,
    }

    if constraint == "strength":
        kwargs: dict[str, Any] = {
            "strength_out": constraints.strength_out,
            "strength_in": constraints.strength_in,
            **common,
        }
    elif constraint == "strength-cost":
        kwargs = {
            "strength_out": constraints.strength_out,
            "strength_in": constraints.strength_in,
            "x": network.x,
            "y": network.y,
            "target_cost": constraints.total_cost,
            **common,
        }
    elif constraint == "strength-edges":
        kwargs = {
            "strength_out": constraints.strength_out,
            "strength_in": constraints.strength_in,
            "target_edges": constraints.total_edges,
            **common,
        }
    elif constraint == "strength-degree":
        kwargs = {
            "strength_out": constraints.strength_out,
            "strength_in": constraints.strength_in,
            "degree_out": constraints.degree_out,
            "degree_in": constraints.degree_in,
            **common,
        }
    elif constraint == "degree-events":
        kwargs = {
            "degree_out": constraints.degree_out,
            "degree_in": constraints.degree_in,
            "total_events": constraints.total_events,
            **degree_common,
        }
    else:
        msg = f"unknown constraint: {constraint}"
        raise ValueError(msg)

    if layers is not None and family in {"b", "wnb"}:
        kwargs["layers"] = layers
    return kwargs


def build_sample_kwargs(
    family: str,
    constraint: str,
    fit: object,
    network: SyntheticNetwork,
    constraints: SyntheticConstraints,
    *,
    layers: int,
    seed: int,
) -> dict[str, Any]:
    """Build keyword arguments for a sample function from a fitted model."""
    if constraint == "strength":
        kwargs: dict[str, Any] = {
            "x": fit.x,
            "y": fit.y,
            "self_loops": fit.self_loops,
            "seed": seed,
        }
        if family in {"b", "wnb"}:
            kwargs["layers"] = layers
        return kwargs
    if constraint == "strength-cost":
        cost_src, cost_tgt, cost_val = network.complete_cost_triples()
        kwargs = {
            "fit": fit,
            "cost_sources": cost_src,
            "cost_targets": cost_tgt,
            "cost_values": cost_val,
            "seed": seed,
        }
        if family in {"b", "wnb"}:
            kwargs["layers"] = layers
        return kwargs
    if constraint == "strength-edges":
        kwargs = {"fit": fit, "seed": seed}
        if family in {"b", "wnb"}:
            kwargs["layers"] = layers
        return kwargs
    if constraint == "strength-degree":
        kwargs = {"fit": fit, "seed": seed}
        if family in {"b", "wnb"}:
            kwargs["layers"] = layers
        return kwargs
    if constraint == "degree-events":
        if family == "me":
            return {
                "fit": fit,
                "total_events": constraints.total_events,
                "seed": seed,
                "self_loops": fit.self_loops,
            }
        if family == "b":
            return {"fit": fit, "layers": layers, "seed": seed}
        if family == "wnb":
            return {"fit": fit, "layers": layers, "seed": seed}
        # w (geometric)
        return {"fit": fit, "seed": seed}
    msg = f"unknown constraint: {constraint}"
    raise ValueError(msg)


def filter_one(
    family: str,
    constraint: str,
    sample: EdgeTable,
    fit: object,
    network: SyntheticNetwork,
    *,
    layers: int,
    alpha: float,
) -> FilterResult:
    """Filter a single sample against its fitted null model."""
    # Special case: ME strength uses filter_custom_poisson
    if family == "me" and constraint == "strength":
        source = sample.source.astype(np.int64)
        target = sample.target.astype(np.int64)
        rates = ProbabilityTable(
            source=sample.source,
            target=sample.target,
            probability=fit.x[source] * fit.y[target],
        )
        return filter_custom_poisson(sample, rates, alpha=alpha, tail="upper")

    dispatch = REGISTRY[(family, constraint)]
    if dispatch.filter is None:
        msg = f"no filter for ({family}, {constraint})"
        raise ValueError(msg)

    kwargs = _build_filter_kwargs(
        family, constraint, sample, fit, network, layers=layers, alpha=alpha
    )
    return dispatch.filter(**kwargs)


def _build_filter_kwargs(
    family: str,
    constraint: str,
    sample: EdgeTable,
    fit: object,
    network: SyntheticNetwork,
    *,
    layers: int,
    alpha: float,
) -> dict[str, Any]:
    """Build keyword arguments for a filter function."""
    base: dict[str, Any] = {"alpha": alpha, "tail": "upper"}

    if constraint == "strength":
        # filter_strength_{binomial,geometric,negative_binomial}
        kwargs: dict[str, Any] = {
            "edges": sample,
            "x": fit.x,
            "y": fit.y,
            "self_loops": fit.self_loops,
            **base,
        }
        if family in {"b", "wnb"}:
            kwargs["layers"] = layers
        return kwargs
    if constraint == "strength-cost":
        cost_src, cost_tgt, cost_val = network.complete_cost_triples()
        kwargs = {
            "edges": sample,
            "fit": fit,
            "cost_sources": cost_src,
            "cost_targets": cost_tgt,
            "cost_values": cost_val,
            **base,
        }
        if family in {"b", "wnb"}:
            kwargs["layers"] = layers
        return kwargs
    if constraint == "strength-edges":
        kwargs = {"edges": sample, "fit": fit, **base}
        if family in {"b", "wnb"}:
            kwargs["layers"] = layers
        return kwargs
    if constraint == "strength-degree":
        kwargs = {"edges": sample, "fit": fit, **base}
        if family in {"b", "wnb"}:
            kwargs["layers"] = layers
        return kwargs
    if constraint == "degree-events":
        if family == "me":
            return {
                "edges": sample,
                "x": fit.x,
                "y": fit.y,
                "positive_weight_rate": _solve_ztp_rate(fit.positive_mean),
                "self_loops": fit.self_loops,
                **base,
            }
        if family == "b":
            return {
                "edges": sample,
                "x": fit.x,
                "y": fit.y,
                "positive_weight_rate": fit.q,
                "layers": layers,
                "self_loops": fit.self_loops,
                **base,
            }
        # w and wnb use fit object
        kwargs = {
            "edges": sample,
            "fit": fit,
            "positive_weight_rate": fit.q,
            "self_loops": fit.self_loops,
            **base,
        }
        if family == "wnb":
            kwargs["layers"] = layers
        return kwargs
    msg = f"unknown constraint: {constraint}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Case metadata for tests
# ---------------------------------------------------------------------------


def fitting_case_registry(include_partial: bool = True) -> list[dict[str, object]]:
    """Return fitting benchmark case metadata for coverage tests."""
    cases: list[dict[str, object]] = []
    specs = (
        ("ME", "poisson", None),
        ("B", "binomial", 3),
        ("W", "geometric", None),
        ("W", "negative-binomial", 3),
    )
    for ensemble, family_name, layers_val in specs:
        for constraint in CONSTRAINTS:
            cases.append(
                {
                    "ensemble": ensemble,
                    "family": family_name,
                    "constraint": constraint,
                    "layers": layers_val,
                    "partial": False,
                }
            )
    if include_partial:
        for ensemble, family_name, layers_val in specs:
            for constraint in CONSTRAINTS:
                cases.append(
                    {
                        "ensemble": ensemble,
                        "family": family_name,
                        "constraint": constraint,
                        "layers": layers_val,
                        "partial": True,
                    }
                )
    return cases
