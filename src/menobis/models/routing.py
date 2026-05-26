"""Unified model fitting and sampling router."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from menobis.filtering.types import FilterResult

import numpy as np
from numpy.typing import NDArray

from menobis.data.frames import EdgeTable
from menobis.models.fitting import (
    DegreeEventsFit,
    StrengthCostFit,
    StrengthDegreeFit,
    StrengthEdgesFit,
    StrengthFit,
    fit_degree_events_binomial,
    fit_degree_events_geometric,
    fit_degree_events_negative_binomial,
    fit_degree_events_poisson,
    fit_strength_binomial,
    fit_strength_cost_binomial,
    fit_strength_cost_binomial_coordinates,
    fit_strength_cost_geometric,
    fit_strength_cost_geometric_coordinates,
    fit_strength_cost_negative_binomial,
    fit_strength_cost_negative_binomial_coordinates,
    fit_strength_cost_poisson,
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
)
from menobis.models.generation import (
    sample_degree_events_binomial,
    sample_degree_events_geometric,
    sample_degree_events_negative_binomial,
    sample_degree_events_poisson,
    sample_strength_binomial,
    sample_strength_geometric,
    sample_strength_multinomial,
    sample_strength_negative_binomial,
    sample_strength_poisson,
    sample_strength_stub_matching,
)
from menobis.models.types import FitResult


class Ensemble(StrEnum):
    """Supported ensemble types."""

    GRAND_CANONICAL = "grandcanonical"
    CANONICAL = "canonical"
    MICROCANONICAL = "microcanonical"


class Family(StrEnum):
    """Supported distribution families."""

    ME = "me"
    BINOMIAL = "binomial"
    GEOMETRIC = "geometric"
    NEGATIVE_BINOMIAL = "negative_binomial"


class Constraint(StrEnum):
    """Supported constraint families."""

    STRENGTH = "strength"
    STRENGTH_COST = "strength_cost"
    STRENGTH_EDGES = "strength_edges"
    STRENGTH_DEGREE = "strength_degree"
    DEGREE_EVENTS = "degree_events"


class UnsupportedModelCaseError(ValueError):
    """Raised when a requested ensemble/family/constraint case is unsupported."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_model(
    *,
    ensemble: Ensemble = Ensemble.GRAND_CANONICAL,
    family: Family,
    constraint: Constraint,
    strength_out: NDArray[Any] | None = None,
    strength_in: NDArray[Any] | None = None,
    degree_out: NDArray[Any] | None = None,
    degree_in: NDArray[Any] | None = None,
    total_events: int | None = None,
    target_edges: float | None = None,
    target_cost: float | None = None,
    cost_sources: NDArray[Any] | None = None,
    cost_targets: NDArray[Any] | None = None,
    cost_values: NDArray[Any] | None = None,
    coord_x: NDArray[Any] | None = None,
    coord_y: NDArray[Any] | None = None,
    layers: int = 1,
    self_loops: bool = True,
    tolerance: float = 1e-8,
    max_iterations: int = 10000,
) -> (
    StrengthFit
    | StrengthCostFit
    | StrengthEdgesFit
    | StrengthDegreeFit
    | DegreeEventsFit
):
    """Fit a model selected by ensemble, family, and constraint.

    Raises:
        UnsupportedModelCaseError: If the combination is not implemented.
        ValueError: If required arrays or parameters are missing.
    """
    match ensemble:
        case Ensemble.MICROCANONICAL:
            msg = (
                "microcanonical has no fitted multipliers; "
                "use sample_model with strength sequences directly"
            )
            raise UnsupportedModelCaseError(msg)
        case Ensemble.CANONICAL if family != Family.ME:
            msg = "canonical ensemble supports only family=ME"
            raise UnsupportedModelCaseError(msg)
        case Ensemble.GRAND_CANONICAL | Ensemble.CANONICAL:
            pass
        case _:
            msg = f"invalid ensemble: {ensemble!r}"
            raise UnsupportedModelCaseError(msg)

    # Dispatch table: (constraint, family) -> fit function
    _fit_dispatch = {
        (Constraint.STRENGTH, Family.ME): fit_strength_poisson,
        (Constraint.STRENGTH, Family.BINOMIAL): fit_strength_binomial,
        (Constraint.STRENGTH, Family.GEOMETRIC): fit_strength_geometric,
        (Constraint.STRENGTH, Family.NEGATIVE_BINOMIAL): fit_strength_negative_binomial,
        (Constraint.STRENGTH_EDGES, Family.ME): fit_strength_edges_poisson,
        (Constraint.STRENGTH_EDGES, Family.BINOMIAL): fit_strength_edges_binomial,
        (Constraint.STRENGTH_EDGES, Family.GEOMETRIC): fit_strength_edges_geometric,
        (
            Constraint.STRENGTH_EDGES,
            Family.NEGATIVE_BINOMIAL,
        ): fit_strength_edges_negative_binomial,
        (Constraint.STRENGTH_DEGREE, Family.ME): fit_strength_degree_poisson,
        (Constraint.STRENGTH_DEGREE, Family.BINOMIAL): fit_strength_degree_binomial,
        (Constraint.STRENGTH_DEGREE, Family.GEOMETRIC): fit_strength_degree_geometric,
        (
            Constraint.STRENGTH_DEGREE,
            Family.NEGATIVE_BINOMIAL,
        ): fit_strength_degree_negative_binomial,
        (Constraint.STRENGTH_COST, Family.ME): fit_strength_cost_poisson,
        (Constraint.STRENGTH_COST, Family.BINOMIAL): fit_strength_cost_binomial,
        (Constraint.STRENGTH_COST, Family.GEOMETRIC): fit_strength_cost_geometric,
        (
            Constraint.STRENGTH_COST,
            Family.NEGATIVE_BINOMIAL,
        ): fit_strength_cost_negative_binomial,
        (Constraint.DEGREE_EVENTS, Family.ME): fit_degree_events_poisson,
        (Constraint.DEGREE_EVENTS, Family.BINOMIAL): fit_degree_events_binomial,
        (Constraint.DEGREE_EVENTS, Family.GEOMETRIC): fit_degree_events_geometric,
        (
            Constraint.DEGREE_EVENTS,
            Family.NEGATIVE_BINOMIAL,
        ): fit_degree_events_negative_binomial,
    }

    key = (constraint, family)
    if key not in _fit_dispatch:
        msg = f"unsupported (constraint, family): ({constraint!r}, {family!r})"
        raise UnsupportedModelCaseError(msg)

    # Build args per constraint
    common: dict[str, Any] = {
        "self_loops": self_loops,
        "tolerance": tolerance,
        "max_iterations": max_iterations,
    }
    if family in (Family.BINOMIAL, Family.NEGATIVE_BINOMIAL):
        common["layers"] = layers

    match constraint:
        case Constraint.STRENGTH:
            if strength_out is None or strength_in is None:
                msg = "strength requires strength_out, strength_in"
                raise ValueError(msg)
            return _fit_dispatch[key](
                np.asarray(strength_out, dtype=np.float64),
                np.asarray(strength_in, dtype=np.float64),
                **common,
            )
        case Constraint.STRENGTH_EDGES:
            if strength_out is None or strength_in is None:
                msg = "strength_edges requires strength_out, strength_in"
                raise ValueError(msg)
            if target_edges is None:
                msg = "strength_edges requires target_edges"
                raise ValueError(msg)
            return _fit_dispatch[key](
                np.asarray(strength_out, dtype=np.float64),
                np.asarray(strength_in, dtype=np.float64),
                target_edges,
                **common,
            )
        case Constraint.STRENGTH_DEGREE:
            if strength_out is None or strength_in is None:
                msg = "strength_degree requires strength sequences"
                raise ValueError(msg)
            if degree_out is None or degree_in is None:
                msg = "strength_degree requires degree sequences"
                raise ValueError(msg)
            return _fit_dispatch[key](
                np.asarray(strength_out, dtype=np.float64),
                np.asarray(strength_in, dtype=np.float64),
                np.asarray(degree_out, dtype=np.float64),
                np.asarray(degree_in, dtype=np.float64),
                **common,
            )
        case Constraint.STRENGTH_COST:
            if strength_out is None or strength_in is None:
                msg = "strength_cost requires strength sequences"
                raise ValueError(msg)
            if target_cost is None:
                msg = "strength_cost requires target_cost"
                raise ValueError(msg)
            s_out = np.asarray(strength_out, dtype=np.float64)
            s_in = np.asarray(strength_in, dtype=np.float64)
            if coord_x is not None and coord_y is not None:
                # Use coordinate variant
                coord_dispatch = {
                    Family.ME: fit_strength_cost_poisson_coordinates,
                    Family.BINOMIAL: fit_strength_cost_binomial_coordinates,
                    Family.GEOMETRIC: fit_strength_cost_geometric_coordinates,
                    Family.NEGATIVE_BINOMIAL: (
                        fit_strength_cost_negative_binomial_coordinates
                    ),
                }
                return coord_dispatch[family](
                    s_out,
                    s_in,
                    coord_x,
                    coord_y,
                    target_cost,
                    **common,
                )
            if cost_sources is None:
                msg = "strength_cost needs cost triples or coordinates"
                raise ValueError(msg)
            return _fit_dispatch[key](
                s_out,
                s_in,
                cost_sources,
                cost_targets,
                cost_values,
                target_cost,
                **common,
            )
        case Constraint.DEGREE_EVENTS:
            if ensemble != Ensemble.GRAND_CANONICAL:
                msg = "degree_events requires ensemble=GRAND_CANONICAL"
                raise UnsupportedModelCaseError(msg)
            if degree_out is None or degree_in is None:
                msg = "degree_events requires degree_out, degree_in"
                raise ValueError(msg)
            if total_events is None:
                msg = "degree_events requires total_events"
                raise ValueError(msg)
            return _fit_dispatch[key](
                np.asarray(degree_out, dtype=np.float64),
                np.asarray(degree_in, dtype=np.float64),
                total_events,
                **common,
            )
        case _:
            msg = f"invalid constraint: {constraint!r}"
            raise UnsupportedModelCaseError(msg)


def sample_model(
    *,
    ensemble: Ensemble = Ensemble.GRAND_CANONICAL,
    family: Family,
    constraint: Constraint,
    fit: FitResult | None = None,
    strength_out: NDArray[Any] | None = None,
    strength_in: NDArray[Any] | None = None,
    total_events: int | None = None,
    seed: int = 0,
) -> EdgeTable:
    """Sample a network from a fitted model or directly via stub matching.

    Raises:
        UnsupportedModelCaseError: If the combination is not implemented.
        ValueError: If required parameters are missing.
        TypeError: If ``fit`` has the wrong type for the requested route.
    """
    match ensemble:
        case Ensemble.MICROCANONICAL:
            return _sample_microcanonical(
                family, constraint, strength_out, strength_in, seed
            )
        case Ensemble.CANONICAL:
            return _sample_canonical(family, constraint, fit, total_events, seed)
        case Ensemble.GRAND_CANONICAL:
            return _sample_grand_canonical(family, constraint, fit, seed)
        case _:
            msg = f"invalid ensemble: {ensemble!r}"
            raise UnsupportedModelCaseError(msg)


# ---------------------------------------------------------------------------
# Private dispatch (immutable match logic)


def _sample_microcanonical(
    family: Family,
    constraint: Constraint,
    strength_out: NDArray[Any] | None,
    strength_in: NDArray[Any] | None,
    seed: int,
) -> EdgeTable:
    if family != Family.ME or constraint != Constraint.STRENGTH:
        msg = "microcanonical supports only family=ME, constraint=STRENGTH"
        raise UnsupportedModelCaseError(msg)
    if strength_out is None or strength_in is None:
        msg = "microcanonical requires strength_out and strength_in"
        raise ValueError(msg)
    return sample_strength_stub_matching(
        np.asarray(strength_out, dtype=np.uint64),
        np.asarray(strength_in, dtype=np.uint64),
        seed=seed,
    )


def _sample_canonical(
    family: Family,
    constraint: Constraint,
    fit: FitResult | None,
    total_events: int | None,
    seed: int,
) -> EdgeTable:
    if family != Family.ME or constraint != Constraint.STRENGTH:
        msg = "canonical supports only family=ME, constraint=STRENGTH"
        raise UnsupportedModelCaseError(msg)
    if fit is None:
        msg = "canonical sampling requires a fit result"
        raise ValueError(msg)
    if not isinstance(fit, StrengthFit):
        msg = (
            "canonical strength sampling requires StrengthFit,"
            f" got {type(fit).__name__}"
        )
        raise TypeError(msg)
    if total_events is None:
        msg = "canonical sampling requires total_events"
        raise ValueError(msg)
    return sample_strength_multinomial(
        fit.x, fit.y, total_events=total_events, self_loops=fit.self_loops, seed=seed
    )


def _sample_grand_canonical(
    family: Family,
    constraint: Constraint,
    fit: FitResult | None,
    seed: int,
) -> EdgeTable:
    if fit is None:
        msg = "grand-canonical sampling requires a fit result"
        raise ValueError(msg)

    match constraint:
        case Constraint.STRENGTH:
            if not isinstance(fit, StrengthFit):
                msg = (
                    f"strength sampling requires StrengthFit, got {type(fit).__name__}"
                )
                raise TypeError(msg)
            match family:
                case Family.ME:
                    return sample_strength_poisson(
                        fit.x, fit.y, self_loops=fit.self_loops, seed=seed
                    )
                case Family.BINOMIAL:
                    return sample_strength_binomial(
                        fit.x,
                        fit.y,
                        layers=fit.layers or 1,
                        self_loops=fit.self_loops,
                        seed=seed,
                    )
                case Family.GEOMETRIC:
                    return sample_strength_geometric(
                        fit.x, fit.y, self_loops=fit.self_loops, seed=seed
                    )
                case Family.NEGATIVE_BINOMIAL:
                    return sample_strength_negative_binomial(
                        fit.x,
                        fit.y,
                        layers=fit.layers or 1,
                        self_loops=fit.self_loops,
                        seed=seed,
                    )
        case Constraint.DEGREE_EVENTS:
            if not isinstance(fit, DegreeEventsFit):
                msg = (
                    "degree_events sampling requires DegreeEventsFit,"
                    f" got {type(fit).__name__}"
                )
                raise TypeError(msg)
            match family:
                case Family.ME:
                    return sample_degree_events_poisson(fit, seed=seed)
                case Family.BINOMIAL:
                    return sample_degree_events_binomial(fit, seed=seed)
                case Family.GEOMETRIC:
                    return sample_degree_events_geometric(fit, seed=seed)
                case Family.NEGATIVE_BINOMIAL:
                    return sample_degree_events_negative_binomial(fit, seed=seed)
        case _:
            msg = f"unsupported constraint: {constraint!r}"
            raise UnsupportedModelCaseError(msg)


# ---------------------------------------------------------------------------
# FILTER
# ---------------------------------------------------------------------------


def filter_model(
    edges: EdgeTable,
    *,
    family: Family,
    constraint: Constraint,
    fit: FitResult | None = None,
    layers: int = 1,
    self_loops: bool = True,
    alpha: float = 0.05,
    tail: str = "two-sided",
    correction: str = "none",
    detect_absent: bool = False,
    min_occupation: float = 0.5,
    min_expected: float = 0.0,
    max_absent: int | None = None,
) -> FilterResult:
    """Filter edges against a null model selected by family and constraint.

    If ``fit`` is not provided, calls ``fit_model`` to obtain it
    (deriving constraints from the edge table).

    All per-family filter functions share the signature
    ``(edges, fit, *, alpha, tail, ...)``.  This router selects the
    correct one based on (family, constraint).

    Raises:
        UnsupportedModelCaseError: If the combination is not implemented.
    """
    from menobis.filtering.models import (
        _node_count,
        _strengths,
        filter_degree_events_binomial,
        filter_degree_events_geometric,
        filter_degree_events_negative_binomial,
        filter_degree_events_poisson,
        filter_strength_binomial,
        filter_strength_cost_binomial,
        filter_strength_cost_geometric,
        filter_strength_cost_negative_binomial,
        filter_strength_cost_poisson,
        filter_strength_degree_binomial,
        filter_strength_degree_geometric,
        filter_strength_degree_negative_binomial,
        filter_strength_degree_poisson,
        filter_strength_edges_binomial,
        filter_strength_edges_geometric,
        filter_strength_edges_negative_binomial,
        filter_strength_edges_poisson,
        filter_strength_geometric,
        filter_strength_negative_binomial,
        filter_strength_poisson,
    )

    # If no fit provided, derive constraints from edges and fit
    if fit is None:
        nc = _node_count(edges)
        s_out, s_in = _strengths(edges, nc)
        fit = fit_model(
            family=family,
            constraint=constraint,
            strength_out=s_out,
            strength_in=s_in,
            layers=layers,
            self_loops=self_loops,
        )

    kwargs: dict[str, Any] = {
        "alpha": alpha,
        "tail": tail,
        "correction": correction,
        "detect_absent": detect_absent,
        "self_loops": self_loops,
        "min_occupation": min_occupation,
        "min_expected": min_expected,
        "max_absent": max_absent,
    }

    # Dispatch table: (constraint, family) -> filter function
    dispatch = {
        (Constraint.STRENGTH, Family.ME): filter_strength_poisson,
        (Constraint.STRENGTH, Family.BINOMIAL): filter_strength_binomial,
        (Constraint.STRENGTH, Family.GEOMETRIC): filter_strength_geometric,
        (
            Constraint.STRENGTH,
            Family.NEGATIVE_BINOMIAL,
        ): filter_strength_negative_binomial,
        (Constraint.STRENGTH_COST, Family.ME): filter_strength_cost_poisson,
        (Constraint.STRENGTH_COST, Family.BINOMIAL): filter_strength_cost_binomial,
        (Constraint.STRENGTH_COST, Family.GEOMETRIC): filter_strength_cost_geometric,
        (
            Constraint.STRENGTH_COST,
            Family.NEGATIVE_BINOMIAL,
        ): filter_strength_cost_negative_binomial,
        (Constraint.STRENGTH_EDGES, Family.ME): filter_strength_edges_poisson,
        (Constraint.STRENGTH_EDGES, Family.BINOMIAL): filter_strength_edges_binomial,
        (Constraint.STRENGTH_EDGES, Family.GEOMETRIC): filter_strength_edges_geometric,
        (
            Constraint.STRENGTH_EDGES,
            Family.NEGATIVE_BINOMIAL,
        ): filter_strength_edges_negative_binomial,
        (Constraint.STRENGTH_DEGREE, Family.ME): filter_strength_degree_poisson,
        (Constraint.STRENGTH_DEGREE, Family.BINOMIAL): filter_strength_degree_binomial,
        (
            Constraint.STRENGTH_DEGREE,
            Family.GEOMETRIC,
        ): filter_strength_degree_geometric,
        (
            Constraint.STRENGTH_DEGREE,
            Family.NEGATIVE_BINOMIAL,
        ): filter_strength_degree_negative_binomial,
        (Constraint.DEGREE_EVENTS, Family.ME): filter_degree_events_poisson,
        (Constraint.DEGREE_EVENTS, Family.BINOMIAL): filter_degree_events_binomial,
        (Constraint.DEGREE_EVENTS, Family.GEOMETRIC): filter_degree_events_geometric,
        (
            Constraint.DEGREE_EVENTS,
            Family.NEGATIVE_BINOMIAL,
        ): filter_degree_events_negative_binomial,
    }

    key = (constraint, family)
    if key not in dispatch:
        msg = f"unsupported (constraint, family): ({constraint!r}, {family!r})"
        raise UnsupportedModelCaseError(msg)

    return dispatch[key](edges, fit, **kwargs)


__all__ = [
    "Constraint",
    "Ensemble",
    "Family",
    "UnsupportedModelCaseError",
    "filter_model",
    "fit_model",
    "sample_model",
]
