"""Unified verb-based model router for MENoBiS workflows."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from menobis.models.spec import (
    Constraint,
    Ensemble,
    ModelFamily,
    UnsupportedModelCaseError,
    Verb,
)

if TYPE_CHECKING:
    from menobis.data.frames import EdgeTable
    from menobis.filtering.types import FilterResult
    from menobis.models.types import FitResult


def route_model(
    verb: Verb,
    *,
    ensemble: Ensemble = Ensemble.GRAND_CANONICAL,
    family: ModelFamily,
    constraint: Constraint,
    **kwargs: Any,  # noqa: ANN401 - unified verb router accepts verb-specific args.
) -> FitResult | EdgeTable | FilterResult:
    """Route a model workflow by verb, ensemble, family, and constraint."""
    match verb:
        case Verb.FIT:
            return _fit_model(
                ensemble=ensemble,
                family=family,
                constraint=constraint,
                **kwargs,
            )
        case Verb.SAMPLE:
            return _sample_model(
                ensemble=ensemble,
                family=family,
                constraint=constraint,
                **kwargs,
            )
        case Verb.FILTER:
            return _filter_model(
                ensemble=ensemble,
                family=family,
                constraint=constraint,
                **kwargs,
            )
        case _:
            msg = f"invalid verb: {verb!r}"
            raise UnsupportedModelCaseError(msg)


def fit_model(
    *,
    ensemble: Ensemble = Ensemble.GRAND_CANONICAL,
    family: ModelFamily,
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
) -> FitResult:
    """Fit a model selected by ensemble, family, and constraint."""
    return cast(
        "FitResult",
        route_model(
            Verb.FIT,
            ensemble=ensemble,
            family=family,
            constraint=constraint,
            strength_out=strength_out,
            strength_in=strength_in,
            degree_out=degree_out,
            degree_in=degree_in,
            total_events=total_events,
            target_edges=target_edges,
            target_cost=target_cost,
            cost_sources=cost_sources,
            cost_targets=cost_targets,
            cost_values=cost_values,
            coord_x=coord_x,
            coord_y=coord_y,
            layers=layers,
            self_loops=self_loops,
            tolerance=tolerance,
            max_iterations=max_iterations,
        ),
    )


def sample_model(
    *,
    ensemble: Ensemble = Ensemble.GRAND_CANONICAL,
    family: ModelFamily,
    constraint: Constraint,
    fit: FitResult | None = None,
    strength_out: NDArray[Any] | None = None,
    strength_in: NDArray[Any] | None = None,
    total_events: int | None = None,
    layers: int = 1,
    seed: int = 0,
) -> EdgeTable:
    """Sample a network from a fitted model or directly via stub matching."""
    return cast(
        "EdgeTable",
        route_model(
            Verb.SAMPLE,
            ensemble=ensemble,
            family=family,
            constraint=constraint,
            fit=fit,
            strength_out=strength_out,
            strength_in=strength_in,
            total_events=total_events,
            layers=layers,
            seed=seed,
        ),
    )


def filter_model(
    edges: EdgeTable,
    *,
    family: ModelFamily,
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
    """Filter edges against a null model selected by family and constraint."""
    return cast(
        "FilterResult",
        route_model(
            Verb.FILTER,
            family=family,
            constraint=constraint,
            edges=edges,
            fit=fit,
            layers=layers,
            self_loops=self_loops,
            alpha=alpha,
            tail=tail,
            correction=correction,
            detect_absent=detect_absent,
            min_occupation=min_occupation,
            min_expected=min_expected,
            max_absent=max_absent,
        ),
    )


def _fit_variant(family: ModelFamily, layers: int) -> str:
    if layers < 1:
        msg = "layers must be positive"
        raise ValueError(msg)
    match family:
        case ModelFamily.ME:
            return "poisson"
        case ModelFamily.B:
            return "binomial"
        case ModelFamily.W:
            return "geometric" if layers == 1 else "negative_binomial"
        case _:
            msg = f"invalid family: {family!r}"
            raise UnsupportedModelCaseError(msg)


def _family_common(family: ModelFamily, layers: int, common: dict[str, Any]) -> None:
    if (
        family in (ModelFamily.B, ModelFamily.W) and layers > 1
    ) or family == ModelFamily.B:
        common["layers"] = layers


def _fit_model(
    *,
    ensemble: Ensemble,
    family: ModelFamily,
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
) -> FitResult:
    from menobis.models import fitting

    match ensemble:
        case Ensemble.MICROCANONICAL:
            msg = (
                "microcanonical has no fitted multipliers; "
                "use sample_model with strength sequences directly"
            )
            raise UnsupportedModelCaseError(msg)
        case Ensemble.CANONICAL if family != ModelFamily.ME:
            msg = "canonical ensemble supports only family=ME"
            raise UnsupportedModelCaseError(msg)
        case Ensemble.GRAND_CANONICAL | Ensemble.CANONICAL:
            pass
        case _:
            msg = f"invalid ensemble: {ensemble!r}"
            raise UnsupportedModelCaseError(msg)

    variant = _fit_variant(family, layers)
    dispatch: dict[tuple[Constraint, str], Callable[..., FitResult]] = {
        (Constraint.STRENGTH, "poisson"): fitting.fit_strength_poisson,
        (Constraint.STRENGTH, "binomial"): fitting.fit_strength_binomial,
        (Constraint.STRENGTH, "geometric"): fitting.fit_strength_geometric,
        (
            Constraint.STRENGTH,
            "negative_binomial",
        ): fitting.fit_strength_negative_binomial,
        (Constraint.STRENGTH_EDGES, "poisson"): fitting.fit_strength_edges_poisson,
        (Constraint.STRENGTH_EDGES, "binomial"): fitting.fit_strength_edges_binomial,
        (Constraint.STRENGTH_EDGES, "geometric"): fitting.fit_strength_edges_geometric,
        (
            Constraint.STRENGTH_EDGES,
            "negative_binomial",
        ): fitting.fit_strength_edges_negative_binomial,
        (Constraint.STRENGTH_DEGREE, "poisson"): fitting.fit_strength_degree_poisson,
        (Constraint.STRENGTH_DEGREE, "binomial"): fitting.fit_strength_degree_binomial,
        (
            Constraint.STRENGTH_DEGREE,
            "geometric",
        ): fitting.fit_strength_degree_geometric,
        (
            Constraint.STRENGTH_DEGREE,
            "negative_binomial",
        ): fitting.fit_strength_degree_negative_binomial,
        (Constraint.STRENGTH_COST, "poisson"): fitting.fit_strength_cost_poisson,
        (Constraint.STRENGTH_COST, "binomial"): fitting.fit_strength_cost_binomial,
        (Constraint.STRENGTH_COST, "geometric"): fitting.fit_strength_cost_geometric,
        (
            Constraint.STRENGTH_COST,
            "negative_binomial",
        ): fitting.fit_strength_cost_negative_binomial,
        (Constraint.DEGREE_EVENTS, "poisson"): fitting.fit_degree_events_poisson,
        (Constraint.DEGREE_EVENTS, "binomial"): fitting.fit_degree_events_binomial,
        (Constraint.DEGREE_EVENTS, "geometric"): fitting.fit_degree_events_geometric,
        (
            Constraint.DEGREE_EVENTS,
            "negative_binomial",
        ): fitting.fit_degree_events_negative_binomial,
    }
    key = (constraint, variant)
    if key not in dispatch:
        msg = f"unsupported (constraint, family): ({constraint!r}, {family!r})"
        raise UnsupportedModelCaseError(msg)

    common: dict[str, Any] = {
        "self_loops": self_loops,
        "tolerance": tolerance,
        "max_iterations": max_iterations,
    }
    _family_common(family, layers, common)

    match constraint:
        case Constraint.STRENGTH:
            if strength_out is None or strength_in is None:
                msg = "strength requires strength_out, strength_in"
                raise ValueError(msg)
            return dispatch[key](
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
            return dispatch[key](
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
            return dispatch[key](
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
                coord_dispatch: dict[str, Callable[..., FitResult]] = {
                    "poisson": fitting.fit_strength_cost_poisson_coordinates,
                    "binomial": fitting.fit_strength_cost_binomial_coordinates,
                    "geometric": fitting.fit_strength_cost_geometric_coordinates,
                    "negative_binomial": (
                        fitting.fit_strength_cost_negative_binomial_coordinates
                    ),
                }
                return coord_dispatch[variant](
                    s_out,
                    s_in,
                    coord_x,
                    coord_y,
                    target_cost,
                    **common,
                )
            if cost_sources is None or cost_targets is None or cost_values is None:
                msg = "strength_cost needs cost triples or coordinates"
                raise ValueError(msg)
            return dispatch[key](
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
            return dispatch[key](
                np.asarray(degree_out, dtype=np.float64),
                np.asarray(degree_in, dtype=np.float64),
                total_events,
                **common,
            )
        case _:
            msg = f"invalid constraint: {constraint!r}"
            raise UnsupportedModelCaseError(msg)


def _sample_model(
    *,
    ensemble: Ensemble,
    family: ModelFamily,
    constraint: Constraint,
    fit: FitResult | None = None,
    strength_out: NDArray[Any] | None = None,
    strength_in: NDArray[Any] | None = None,
    total_events: int | None = None,
    layers: int = 1,
    seed: int = 0,
) -> EdgeTable:
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
    from menobis.models.types import DegreeEventsFit, StrengthFit

    match ensemble:
        case Ensemble.MICROCANONICAL:
            if family != ModelFamily.ME or constraint != Constraint.STRENGTH:
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
        case Ensemble.CANONICAL:
            if family != ModelFamily.ME or constraint != Constraint.STRENGTH:
                msg = "canonical supports only family=ME, constraint=STRENGTH"
                raise UnsupportedModelCaseError(msg)
            if fit is None:
                msg = "canonical sampling requires a fit result"
                raise ValueError(msg)
            if not isinstance(fit, StrengthFit):
                msg = (
                    "canonical strength sampling requires StrengthFit, got "
                    f"{type(fit).__name__}"
                )
                raise TypeError(msg)
            if total_events is None:
                msg = "canonical sampling requires total_events"
                raise ValueError(msg)
            return sample_strength_multinomial(
                fit.x,
                fit.y,
                total_events=total_events,
                self_loops=fit.self_loops,
                seed=seed,
            )
        case Ensemble.GRAND_CANONICAL:
            pass
        case _:
            msg = f"invalid ensemble: {ensemble!r}"
            raise UnsupportedModelCaseError(msg)

    if fit is None:
        msg = "grand-canonical sampling requires a fit result"
        raise ValueError(msg)
    fit_layers = getattr(fit, "layers", None) or layers
    variant = _fit_variant(family, fit_layers)
    if constraint == Constraint.STRENGTH:
        if not isinstance(fit, StrengthFit):
            msg = f"strength sampling requires StrengthFit, got {type(fit).__name__}"
            raise TypeError(msg)
        dispatch = {
            "poisson": lambda: sample_strength_poisson(
                fit.x, fit.y, self_loops=fit.self_loops, seed=seed
            ),
            "binomial": lambda: sample_strength_binomial(
                fit.x,
                fit.y,
                layers=fit.layers or 1,
                self_loops=fit.self_loops,
                seed=seed,
            ),
            "geometric": lambda: sample_strength_geometric(
                fit.x, fit.y, self_loops=fit.self_loops, seed=seed
            ),
            "negative_binomial": lambda: sample_strength_negative_binomial(
                fit.x,
                fit.y,
                layers=fit.layers or 1,
                self_loops=fit.self_loops,
                seed=seed,
            ),
        }
        return dispatch[variant]()
    if constraint == Constraint.DEGREE_EVENTS:
        if not isinstance(fit, DegreeEventsFit):
            msg = (
                "degree_events sampling requires DegreeEventsFit, got "
                f"{type(fit).__name__}"
            )
            raise TypeError(msg)
        dispatch = {
            "poisson": lambda: sample_degree_events_poisson(fit, seed=seed),
            "binomial": lambda: sample_degree_events_binomial(fit, seed=seed),
            "geometric": lambda: sample_degree_events_geometric(fit, seed=seed),
            "negative_binomial": lambda: sample_degree_events_negative_binomial(
                fit, seed=seed
            ),
        }
        return dispatch[variant]()
    msg = f"unsupported constraint: {constraint!r}"
    raise UnsupportedModelCaseError(msg)


def _node_count(edges: EdgeTable) -> int:
    if len(edges) == 0:
        return 0
    return int(max(edges.source.max(), edges.target.max())) + 1


def _strengths(edges: EdgeTable, node_count: int) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros(node_count, dtype=np.uint64)
    incoming = np.zeros(node_count, dtype=np.uint64)
    np.add.at(out, edges.source, edges.weight)
    np.add.at(incoming, edges.target, edges.weight)
    return out, incoming


def _filter_model(
    edges: EdgeTable,
    *,
    ensemble: Ensemble,
    family: ModelFamily,
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
    from menobis.filtering import models as filtering

    if ensemble != Ensemble.GRAND_CANONICAL:
        msg = "filtering requires ensemble=GRAND_CANONICAL"
        raise UnsupportedModelCaseError(msg)

    if fit is None:
        nc = _node_count(edges)
        s_out, s_in = _strengths(edges, nc)
        fit = _fit_model(
            ensemble=ensemble,
            family=family,
            constraint=constraint,
            strength_out=s_out,
            strength_in=s_in,
            layers=layers,
            self_loops=self_loops,
        )

    variant = _fit_variant(family, layers if layers is not None else 1)
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
    dispatch: dict[tuple[Constraint, str], Callable[..., FilterResult]] = {
        (Constraint.STRENGTH, "poisson"): filtering.filter_strength_poisson,
        (Constraint.STRENGTH, "binomial"): filtering.filter_strength_binomial,
        (Constraint.STRENGTH, "geometric"): filtering.filter_strength_geometric,
        (
            Constraint.STRENGTH,
            "negative_binomial",
        ): filtering.filter_strength_negative_binomial,
        (Constraint.STRENGTH_COST, "poisson"): filtering.filter_strength_cost_poisson,
        (Constraint.STRENGTH_COST, "binomial"): filtering.filter_strength_cost_binomial,
        (
            Constraint.STRENGTH_COST,
            "geometric",
        ): filtering.filter_strength_cost_geometric,
        (
            Constraint.STRENGTH_COST,
            "negative_binomial",
        ): filtering.filter_strength_cost_negative_binomial,
        (Constraint.STRENGTH_EDGES, "poisson"): filtering.filter_strength_edges_poisson,
        (
            Constraint.STRENGTH_EDGES,
            "binomial",
        ): filtering.filter_strength_edges_binomial,
        (
            Constraint.STRENGTH_EDGES,
            "geometric",
        ): filtering.filter_strength_edges_geometric,
        (
            Constraint.STRENGTH_EDGES,
            "negative_binomial",
        ): filtering.filter_strength_edges_negative_binomial,
        (
            Constraint.STRENGTH_DEGREE,
            "poisson",
        ): filtering.filter_strength_degree_poisson,
        (
            Constraint.STRENGTH_DEGREE,
            "binomial",
        ): filtering.filter_strength_degree_binomial,
        (
            Constraint.STRENGTH_DEGREE,
            "geometric",
        ): filtering.filter_strength_degree_geometric,
        (
            Constraint.STRENGTH_DEGREE,
            "negative_binomial",
        ): filtering.filter_strength_degree_negative_binomial,
        (Constraint.DEGREE_EVENTS, "poisson"): filtering.filter_degree_events_poisson,
        (Constraint.DEGREE_EVENTS, "binomial"): filtering.filter_degree_events_binomial,
        (
            Constraint.DEGREE_EVENTS,
            "geometric",
        ): filtering.filter_degree_events_geometric,
        (
            Constraint.DEGREE_EVENTS,
            "negative_binomial",
        ): filtering.filter_degree_events_negative_binomial,
    }
    key = (constraint, variant)
    if key not in dispatch:
        msg = f"unsupported (constraint, family): ({constraint!r}, {family!r})"
        raise UnsupportedModelCaseError(msg)
    return dispatch[key](edges, fit, **kwargs)


__all__ = [
    "Constraint",
    "Ensemble",
    "ModelFamily",
    "UnsupportedModelCaseError",
    "Verb",
    "filter_model",
    "fit_model",
    "route_model",
    "sample_model",
]
