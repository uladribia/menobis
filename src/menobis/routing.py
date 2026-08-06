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
    from menobis.models.types import FitResult, SamplingResult


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
    coord_x: NDArray[Any] | None = None,
    coord_y: NDArray[Any] | None = None,
    node_count: int | None = None,
    known_source: NDArray[Any] | None = None,
    known_target: NDArray[Any] | None = None,
    known_occnum: NDArray[Any] | None = None,
    layers: int = 1,
    self_loops: bool = True,
    tolerance: float = 1e-8,
    max_iterations: int = 10000,
) -> FitResult:
    """Fit a model selected by ensemble, family, and constraint.

    When known_source, known_target, and known_occnum are provided, performs
    partial fitting where those pairs are frozen and only the remaining
    pairs are fitted.

    `node_count` is required by the EDGES_EVENTS constraint, which has no
    per-node sequences.
    """
    if (
        known_source is not None
        and known_target is not None
        and known_occnum is not None
    ):
        return cast(
            "FitResult",
            _fit_partial(
                family=family,
                constraint=constraint,
                strength_out=strength_out,
                strength_in=strength_in,
                degree_out=degree_out,
                degree_in=degree_in,
                target_edges=target_edges,
                target_cost=target_cost,
                coord_x=coord_x,
                coord_y=coord_y,
                known_source=known_source,
                known_target=known_target,
                known_occnum=known_occnum,
                layers=layers,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            ),
        )
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
            coord_x=coord_x,
            coord_y=coord_y,
            node_count=node_count,
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
    degree_out: NDArray[Any] | None = None,
    degree_in: NDArray[Any] | None = None,
    total_events: int | None = None,
    target_edges: int | None = None,
    coord_x: NDArray[Any] | None = None,
    coord_y: NDArray[Any] | None = None,
    node_count: int | None = None,
    known_source: NDArray[Any] | None = None,
    known_target: NDArray[Any] | None = None,
    known_occnum: NDArray[Any] | None = None,
    layers: int = 1,
    self_loops: bool = True,
    seed: int = 0,
    burn_in_sweeps: int = 50,
    sweeps_per_sample: int = 10,
) -> EdgeTable:
    """Sample a network from a fitted model or directly via stub matching.

    Returns only the sampled occupied-pair table. Use
    :func:`sample_model_detailed` for the full result with metadata.

    `self_loops` is only consulted by the direct (microcanonical) sampler;
    grand-canonical and canonical sampling read it from the fitted model.
    """
    return sample_model_detailed(
        ensemble=ensemble,
        family=family,
        constraint=constraint,
        fit=fit,
        strength_out=strength_out,
        strength_in=strength_in,
        degree_out=degree_out,
        degree_in=degree_in,
        total_events=total_events,
        target_edges=target_edges,
        coord_x=coord_x,
        coord_y=coord_y,
        node_count=node_count,
        known_source=known_source,
        known_target=known_target,
        known_occnum=known_occnum,
        layers=layers,
        self_loops=self_loops,
        seed=seed,
        burn_in_sweeps=burn_in_sweeps,
        sweeps_per_sample=sweeps_per_sample,
    ).edges


def sample_model_detailed(
    *,
    ensemble: Ensemble = Ensemble.GRAND_CANONICAL,
    family: ModelFamily,
    constraint: Constraint,
    fit: FitResult | None = None,
    strength_out: NDArray[Any] | None = None,
    strength_in: NDArray[Any] | None = None,
    degree_out: NDArray[Any] | None = None,
    degree_in: NDArray[Any] | None = None,
    total_events: int | None = None,
    target_edges: int | None = None,
    coord_x: NDArray[Any] | None = None,
    coord_y: NDArray[Any] | None = None,
    node_count: int | None = None,
    known_source: NDArray[Any] | None = None,
    known_target: NDArray[Any] | None = None,
    known_occnum: NDArray[Any] | None = None,
    layers: int = 1,
    self_loops: bool = True,
    seed: int = 0,
    burn_in_sweeps: int = 50,
    sweeps_per_sample: int = 10,
) -> SamplingResult:
    """Sample a network and return the detailed :class:`SamplingResult`.

    The result carries the ensemble, family, constraint, generation method,
    exactness category, seed, and diagnostics.
    """
    from menobis.capabilities import SamplingExactness, capability
    from menobis.models.types import SamplingDiagnostics, SamplingResult

    cap = capability(Verb.SAMPLE, ensemble, family, constraint)
    if cap is None or not cap.supported:
        msg = (
            f"unsupported sampling case: verb=sample ensemble={ensemble!r} "
            f"family={family!r} constraint={constraint!r}"
        )
        raise UnsupportedModelCaseError(msg)

    edges = cast(
        "EdgeTable",
        route_model(
            Verb.SAMPLE,
            ensemble=ensemble,
            family=family,
            constraint=constraint,
            fit=fit,
            strength_out=strength_out,
            strength_in=strength_in,
            degree_out=degree_out,
            degree_in=degree_in,
            total_events=total_events,
            target_edges=target_edges,
            coord_x=coord_x,
            coord_y=coord_y,
            node_count=node_count,
            known_source=known_source,
            known_target=known_target,
            known_occnum=known_occnum,
            layers=layers,
            self_loops=self_loops,
            seed=seed,
            burn_in_sweeps=burn_in_sweeps,
            sweeps_per_sample=sweeps_per_sample,
        ),
    )

    if ensemble is Ensemble.MICROCANONICAL:
        if constraint is Constraint.EDGES_EVENTS:
            method = "microcanonical_fixed_et"
            exactness = SamplingExactness.EXACT_DIRECT
        elif constraint is Constraint.DEGREE_EVENTS:
            method = "microcanonical_fixed_kt"
            exactness = SamplingExactness.EXACT_STATIONARY_MCMC
        else:
            method = "stub_matching"
            exactness = SamplingExactness.EXACT_DIRECT
    elif ensemble is Ensemble.CANONICAL:
        method = "canonical_multinomial"
        exactness = SamplingExactness.EXACT_DIRECT
    else:
        method = "grandcanonical_independent"
        exactness = SamplingExactness.EXACT_INDEPENDENT

    return SamplingResult(
        edges=edges,
        ensemble=ensemble,
        family=family,
        constraint=constraint,
        method=method,
        exactness=exactness,
        seed=seed,
        diagnostics=SamplingDiagnostics(method=method, exactness=exactness),
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
    coord_x: NDArray[Any] | None = None,
    coord_y: NDArray[Any] | None = None,
    node_count: int | None = None,
    total_events: int | None = None,
    target_edges: float | None = None,
) -> FilterResult:
    """Filter edges against a null model selected by family and constraint.

    `node_count`, `total_events`, and `target_edges` are required by the
    EDGES_EVENTS constraint when no fitted model is supplied.
    """
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
            coord_x=coord_x,
            coord_y=coord_y,
            node_count=node_count,
            total_events=total_events,
            target_edges=target_edges,
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
    coord_x: NDArray[Any] | None = None,
    coord_y: NDArray[Any] | None = None,
    node_count: int | None = None,
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
        (Constraint.STRENGTH, "poisson"): fitting._fit_strength_poisson,
        (Constraint.STRENGTH, "binomial"): fitting._fit_strength_binomial,
        (Constraint.STRENGTH, "geometric"): fitting._fit_strength_geometric,
        (
            Constraint.STRENGTH,
            "negative_binomial",
        ): fitting._fit_strength_negative_binomial,
        (Constraint.STRENGTH_EDGES, "poisson"): fitting._fit_strength_edges_poisson,
        (Constraint.STRENGTH_EDGES, "binomial"): fitting._fit_strength_edges_binomial,
        (Constraint.STRENGTH_EDGES, "geometric"): fitting._fit_strength_edges_geometric,
        (
            Constraint.STRENGTH_EDGES,
            "negative_binomial",
        ): fitting._fit_strength_edges_negative_binomial,
        (Constraint.STRENGTH_DEGREE, "poisson"): fitting._fit_strength_degree_poisson,
        (Constraint.STRENGTH_DEGREE, "binomial"): fitting._fit_strength_degree_binomial,
        (
            Constraint.STRENGTH_DEGREE,
            "geometric",
        ): fitting._fit_strength_degree_geometric,
        (
            Constraint.STRENGTH_DEGREE,
            "negative_binomial",
        ): fitting._fit_strength_degree_negative_binomial,
        (Constraint.STRENGTH_COST, "poisson"): fitting._fit_strength_cost_poisson,
        (Constraint.STRENGTH_COST, "binomial"): fitting._fit_strength_cost_binomial,
        (Constraint.STRENGTH_COST, "geometric"): fitting._fit_strength_cost_geometric,
        (
            Constraint.STRENGTH_COST,
            "negative_binomial",
        ): fitting._fit_strength_cost_negative_binomial,
        (Constraint.DEGREE_EVENTS, "poisson"): fitting._fit_degree_events_poisson,
        (Constraint.DEGREE_EVENTS, "binomial"): fitting._fit_degree_events_binomial,
        (Constraint.DEGREE_EVENTS, "geometric"): fitting._fit_degree_events_geometric,
        (
            Constraint.DEGREE_EVENTS,
            "negative_binomial",
        ): fitting._fit_degree_events_negative_binomial,
        (Constraint.EDGES_EVENTS, "poisson"): fitting._fit_edges_events,
        (Constraint.EDGES_EVENTS, "binomial"): fitting._fit_edges_events,
        (Constraint.EDGES_EVENTS, "geometric"): fitting._fit_edges_events,
        (Constraint.EDGES_EVENTS, "negative_binomial"): fitting._fit_edges_events,
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
            if coord_x is None or coord_y is None:
                msg = "strength_cost requires projected coord_x and coord_y"
                raise ValueError(msg)
            coord_dispatch: dict[str, Callable[..., FitResult]] = {
                "poisson": fitting._fit_strength_cost_poisson,
                "binomial": fitting._fit_strength_cost_binomial,
                "geometric": fitting._fit_strength_cost_geometric,
                "negative_binomial": fitting._fit_strength_cost_negative_binomial,
            }
            return coord_dispatch[variant](
                s_out,
                s_in,
                coord_x,
                coord_y,
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
        case Constraint.EDGES_EVENTS:
            if ensemble != Ensemble.GRAND_CANONICAL:
                msg = "edges_events requires ensemble=GRAND_CANONICAL"
                raise UnsupportedModelCaseError(msg)
            if node_count is None:
                msg = "edges_events requires node_count"
                raise ValueError(msg)
            if total_events is None:
                msg = "edges_events requires total_events"
                raise ValueError(msg)
            if target_edges is None:
                msg = "edges_events requires target_edges"
                raise ValueError(msg)
            return dispatch[key](
                variant,
                target_edges,
                total_events,
                node_count,
                layers=layers,
                self_loops=self_loops,
                max_iterations=max_iterations,
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
    degree_out: NDArray[Any] | None = None,
    degree_in: NDArray[Any] | None = None,
    total_events: int | None = None,
    target_edges: int | None = None,
    coord_x: NDArray[Any] | None = None,
    coord_y: NDArray[Any] | None = None,
    node_count: int | None = None,
    known_source: NDArray[Any] | None = None,
    known_target: NDArray[Any] | None = None,
    known_occnum: NDArray[Any] | None = None,
    layers: int = 1,
    self_loops: bool = True,
    seed: int = 0,
    burn_in_sweeps: int = 50,
    sweeps_per_sample: int = 10,
) -> EdgeTable:
    from menobis.data.frames import EdgeTable
    from menobis.models.generation import (
        _sample_degree_events_binomial,
        _sample_degree_events_fixed_kt,
        _sample_degree_events_geometric,
        _sample_degree_events_negative_binomial,
        _sample_degree_events_poisson,
        _sample_edges_events,
        _sample_strength_binomial,
        _sample_strength_cost_binomial,
        _sample_strength_cost_geometric,
        _sample_strength_cost_negative_binomial,
        _sample_strength_cost_poisson,
        _sample_strength_degree_binomial,
        _sample_strength_degree_geometric,
        _sample_strength_degree_negative_binomial,
        _sample_strength_degree_poisson,
        _sample_strength_edges_binomial,
        _sample_strength_edges_geometric,
        _sample_strength_edges_negative_binomial,
        _sample_strength_edges_poisson,
        _sample_strength_fixed_strength_mcmc,
        _sample_strength_geometric,
        _sample_strength_multinomial,
        _sample_strength_negative_binomial,
        _sample_strength_poisson,
        _sample_strength_stub_matching,
    )
    from menobis.models.types import (
        DegreeEventsFit,
        EdgesEventsFit,
        StrengthCostFit,
        StrengthDegreeFit,
        StrengthEdgesFit,
        StrengthFit,
    )

    match ensemble:
        case Ensemble.MICROCANONICAL:
            if constraint is Constraint.EDGES_EVENTS:
                if family not in (ModelFamily.ME, ModelFamily.B, ModelFamily.W):
                    msg = (
                        f"microcanonical EDGES_EVENTS does not "
                        f"support family={family!r}"
                    )
                    raise UnsupportedModelCaseError(msg)
                # Exact ME fixed-(E,T) microcanonical sampler
                if target_edges is None:
                    msg = "microcanonical EDGES_EVENTS requires target_edges"
                    raise ValueError(msg)
                if total_events is None:
                    msg = "microcanonical EDGES_EVENTS requires total_events"
                    raise ValueError(msg)
                if node_count is None:
                    msg = "microcanonical EDGES_EVENTS requires node_count"
                    raise ValueError(msg)
                fam = (
                    "ME"
                    if family == ModelFamily.ME
                    else ("B" if family == ModelFamily.B else "W")
                )
                layers_val = int(layers) if fam in ("B", "W") else 1
                return _sample_fixed_et_edges_events(
                    family=fam,
                    node_count=int(node_count),
                    total_edges=int(target_edges),
                    total_events=int(total_events),
                    self_loops=bool(self_loops),
                    layers=layers_val,
                    known_source=known_source,
                    known_target=known_target,
                    known_occnum=known_occnum,
                    seed=seed,
                )
            if constraint is Constraint.STRENGTH:
                if strength_out is None or strength_in is None:
                    msg = (
                        "microcanonical strength requires strength_out and strength_in"
                    )
                    raise ValueError(msg)
                has_fixed = not (
                    known_source is None or known_target is None or known_occnum is None
                )
                # ME direct fast path: eligible when self-loops allowed and
                # no fixed pairs. The Rust backend handles the routing.
                if family is ModelFamily.ME and self_loops and not has_fixed:
                    return _sample_strength_stub_matching(
                        np.asarray(strength_out, dtype=np.uint64),
                        np.asarray(strength_in, dtype=np.uint64),
                        seed=seed,
                    )
                # Generic MCMC backend for all other cases.
                fam = (
                    "ME"
                    if family == ModelFamily.ME
                    else ("B" if family == ModelFamily.B else "W")
                )
                return _sample_strength_fixed_strength_mcmc(
                    family=fam,
                    strength_out=np.asarray(strength_out, dtype=np.uint64),
                    strength_in=np.asarray(strength_in, dtype=np.uint64),
                    self_loops=bool(self_loops),
                    known_source=known_source,
                    known_target=known_target,
                    known_occnum=known_occnum,
                    layers=int(layers),
                    seed=seed,
                    burn_in_sweeps=burn_in_sweeps,
                    sweeps_per_sample=sweeps_per_sample,
                )
            if constraint is Constraint.DEGREE_EVENTS:
                if degree_out is None or degree_in is None:
                    msg = (
                        "microcanonical DEGREE_EVENTS requires degree_out and degree_in"
                    )
                    raise ValueError(msg)
                if total_events is None:
                    msg = "microcanonical DEGREE_EVENTS requires total_events"
                    raise ValueError(msg)
                has_fixed = not (
                    known_source is None or known_target is None or known_occnum is None
                )
                fam = (
                    "ME"
                    if family == ModelFamily.ME
                    else ("B" if family == ModelFamily.B else "W")
                )
                if not has_fixed:
                    return _sample_degree_events_fixed_kt(
                        family=fam,
                        degree_out=np.asarray(degree_out, dtype=np.uint32).tolist(),
                        degree_in=np.asarray(degree_in, dtype=np.uint32).tolist(),
                        total_events=int(total_events),
                        layers=int(layers),
                        seed=seed,
                        self_loops=bool(self_loops),
                        burn_in_sweeps=burn_in_sweeps,
                        sweeps_per_sample=sweeps_per_sample,
                    )
                # ---- fixed-pair residualization ----
                k_src = np.asarray(known_source, dtype=np.uint64)
                k_tgt = np.asarray(known_target, dtype=np.uint64)
                k_occ = np.asarray(known_occnum, dtype=np.uint64)
                # Fixed contributions to out/in degrees and total T
                k_out_fix = np.zeros(len(degree_out), dtype=np.int64)
                k_in_fix = np.zeros(len(degree_in), dtype=np.int64)
                t_fix = 0
                for s, t, o in zip(k_src, k_tgt, k_occ, strict=True):
                    if o > 0:
                        k_out_fix[int(s)] += 1
                        k_in_fix[int(t)] += 1
                        t_fix += int(o)
                # Residual degrees (int64 to avoid uint64 promotion)
                k_out_res = np.asarray(degree_out, dtype=np.int64) - k_out_fix
                k_in_res = np.asarray(degree_in, dtype=np.int64) - k_in_fix
                if (k_out_res < 0).any() or (k_in_res < 0).any():
                    msg = "fixed pairs exceed target degree sequence"
                    raise ValueError(msg)
                # Cast back to uint32 for the native kernel
                k_out_res_u32 = k_out_res.astype(np.uint32)
                k_in_res_u32 = k_in_res.astype(np.uint32)
                t_res = int(total_events) - t_fix
                if t_res < 0:
                    msg = "fixed pairs exceed total events"
                    raise ValueError(msg)
                # Build admissible pair set (exclude fixed positive pairs)
                n = len(degree_out)
                fixed_keys = set()
                for s, t, o in zip(k_src, k_tgt, k_occ, strict=True):
                    if o > 0:
                        fixed_keys.add((int(s), int(t)))
                adm_src, adm_tgt = [], []
                for i in range(n):
                    for j in range(n):
                        if not self_loops and i == j:
                            continue
                        if (i, j) in fixed_keys:
                            continue
                        adm_src.append(i)
                        adm_tgt.append(j)
                # Sample residual
                residual = _sample_degree_events_fixed_kt(
                    family=fam,
                    degree_out=k_out_res_u32.tolist(),
                    degree_in=k_in_res_u32.tolist(),
                    total_events=t_res,
                    layers=int(layers),
                    seed=seed,
                    self_loops=bool(self_loops),
                    burn_in_sweeps=burn_in_sweeps,
                    sweeps_per_sample=sweeps_per_sample,
                )
                # Merge fixed pairs back
                pos_mask = k_occ > 0
                k_src_pos = k_src[pos_mask]
                k_tgt_pos = k_tgt[pos_mask]
                k_occ_pos = k_occ[pos_mask]
                if len(residual) == 0:
                    return EdgeTable(
                        source=k_src_pos, target=k_tgt_pos, occ_num=k_occ_pos
                    )
                return EdgeTable(
                    source=np.concatenate([residual.source, k_src_pos]),
                    target=np.concatenate([residual.target, k_tgt_pos]),
                    occ_num=np.concatenate([residual.occ_num, k_occ_pos]),
                )
            msg = (
                f"microcanonical does not support constraint={constraint!r}; "
                "supported: STRENGTH, EDGES_EVENTS, DEGREE_EVENTS"
            )
            raise UnsupportedModelCaseError(msg)
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
            return _sample_strength_multinomial(
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
            "poisson": lambda: _sample_strength_poisson(
                fit.x, fit.y, self_loops=fit.self_loops, seed=seed
            ),
            "binomial": lambda: _sample_strength_binomial(
                fit.x,
                fit.y,
                layers=fit.layers or 1,
                self_loops=fit.self_loops,
                seed=seed,
            ),
            "geometric": lambda: _sample_strength_geometric(
                fit.x, fit.y, self_loops=fit.self_loops, seed=seed
            ),
            "negative_binomial": lambda: _sample_strength_negative_binomial(
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
            "poisson": lambda: _sample_degree_events_poisson(fit, seed=seed),
            "binomial": lambda: _sample_degree_events_binomial(fit, seed=seed),
            "geometric": lambda: _sample_degree_events_geometric(fit, seed=seed),
            "negative_binomial": lambda: _sample_degree_events_negative_binomial(
                fit, seed=seed
            ),
        }
        return dispatch[variant]()
    if constraint == Constraint.EDGES_EVENTS:
        if not isinstance(fit, EdgesEventsFit):
            msg = (
                "edges_events sampling requires EdgesEventsFit, got "
                f"{type(fit).__name__}"
            )
            raise TypeError(msg)
        return _sample_edges_events(
            fit.node_count,
            fit.q,
            fit.occupation,
            fit.family,
            layers=fit.layers or 1,
            self_loops=fit.self_loops,
            seed=seed,
        )
    if constraint == Constraint.STRENGTH_EDGES:
        if not isinstance(fit, StrengthEdgesFit):
            msg = (
                "strength_edges sampling requires StrengthEdgesFit, got "
                f"{type(fit).__name__}"
            )
            raise TypeError(msg)
        edges_fit = fit
        dispatch = {
            "poisson": lambda: _sample_strength_edges_poisson(edges_fit, seed=seed),
            "binomial": lambda: _sample_strength_edges_binomial(
                edges_fit, layers=fit_layers, seed=seed
            ),
            "geometric": lambda: _sample_strength_edges_geometric(edges_fit, seed=seed),
            "negative_binomial": lambda: _sample_strength_edges_negative_binomial(
                edges_fit, layers=fit_layers, seed=seed
            ),
        }
        return dispatch[variant]()
    if constraint == Constraint.STRENGTH_DEGREE:
        if not isinstance(fit, StrengthDegreeFit):
            msg = (
                "strength_degree sampling requires StrengthDegreeFit, got "
                f"{type(fit).__name__}"
            )
            raise TypeError(msg)
        degree_fit = fit
        dispatch = {
            "poisson": lambda: _sample_strength_degree_poisson(degree_fit, seed=seed),
            "binomial": lambda: _sample_strength_degree_binomial(
                degree_fit, layers=fit_layers, seed=seed
            ),
            "geometric": lambda: _sample_strength_degree_geometric(
                degree_fit, seed=seed
            ),
            "negative_binomial": lambda: _sample_strength_degree_negative_binomial(
                degree_fit, layers=fit_layers, seed=seed
            ),
        }
        return dispatch[variant]()
    if constraint == Constraint.STRENGTH_COST:
        if coord_x is None or coord_y is None:
            msg = "strength_cost sampling requires coord_x and coord_y"
            raise ValueError(msg)
        if not isinstance(fit, StrengthCostFit):
            msg = (
                "strength_cost sampling requires StrengthCostFit, got "
                f"{type(fit).__name__}"
            )
            raise TypeError(msg)
        cost_fit = fit
        dispatch = {
            "poisson": lambda: _sample_strength_cost_poisson(
                cost_fit, coord_x, coord_y, seed=seed
            ),
            "binomial": lambda: _sample_strength_cost_binomial(
                cost_fit, coord_x, coord_y, layers=fit_layers, seed=seed
            ),
            "geometric": lambda: _sample_strength_cost_geometric(
                cost_fit, coord_x, coord_y, seed=seed
            ),
            "negative_binomial": lambda: _sample_strength_cost_negative_binomial(
                cost_fit, coord_x, coord_y, layers=fit_layers, seed=seed
            ),
        }
        return dispatch[variant]()
    msg = f"unsupported constraint: {constraint!r}"
    raise UnsupportedModelCaseError(msg)


def _sample_fixed_et_edges_events(
    *,
    family: str,
    node_count: int,
    total_edges: int,
    total_events: int,
    self_loops: bool,
    layers: int = 1,
    known_source: NDArray[Any] | None,
    known_target: NDArray[Any] | None,
    known_occnum: NDArray[Any] | None,
    seed: int,
) -> EdgeTable:
    """Sample the exact microcanonical (E,T) model for any family.

    Applies the shared preprocessing pipeline: fixed (known) pairs are
    subtracted from the constraints, the residual problem is sampled over
    the admissible pairs, and fixed pairs are merged back into the result.

    * No fixed pairs: uses the O(1) index-mapped fast path (no pair list).
    * With fixed pairs: builds the admissible pair set explicitly, computes
      residual E/T, and calls the explicit-pair Rust kernel.
    """
    from menobis.data.frames import EdgeTable
    from menobis.models.generation import (
        _sample_b_fixed_et,
        _sample_b_fixed_et_explicit,
        _sample_me_fixed_et,
        _sample_me_fixed_et_explicit,
        _sample_w_fixed_et,
        _sample_w_fixed_et_explicit,
    )

    n = int(node_count)
    e_total = int(total_edges)
    t_total = int(total_events)
    sl = bool(self_loops)

    has_fixed = not (
        known_source is None or known_target is None or known_occnum is None
    )
    if not has_fixed:
        # Fast path: all candidate pairs are admissible, no residualization.
        f_dispatch = {
            "ME": lambda: _sample_me_fixed_et(
                n,
                self_loops=sl,
                residual_edges=e_total,
                residual_total=t_total,
                seed=seed,
            ),
            "B": lambda: _sample_b_fixed_et(
                n,
                self_loops=sl,
                layers=layers,
                residual_edges=e_total,
                residual_total=t_total,
                seed=seed,
            ),
            "W": lambda: _sample_w_fixed_et(
                n,
                self_loops=sl,
                layers=layers,
                residual_edges=e_total,
                residual_total=t_total,
                seed=seed,
            ),
        }
        return f_dispatch[family]()

    # ---- fixed-pair preprocessing ----
    k_src = np.asarray(known_source, dtype=np.uint64)
    k_tgt = np.asarray(known_target, dtype=np.uint64)
    k_occ = np.asarray(known_occnum, dtype=np.uint64)
    if not (len(k_src) == len(k_tgt) == len(k_occ)):
        msg = "known_source, known_target, known_occnum must have same length"
        raise ValueError(msg)
    if family == "B" and (k_occ > layers).any():
        msg = f"B fixed occupation exceeds layer capacity M={layers}"
        raise ValueError(msg)

    # Fixed contributions to E and T
    e_fixed = int((k_occ > 0).sum())
    t_fixed = int(k_occ.sum())
    if e_fixed > e_total or t_fixed > t_total:
        msg = (
            f"fixed pairs contribute E={e_fixed}, T={t_fixed} which exceeds "
            f"requested E={e_total}, T={t_total}"
        )
        raise ValueError(msg)
    e_res = e_total - e_fixed
    t_res = t_total - t_fixed

    # Build the admissible (non-fixed) pair set.
    # Pairs are indexed as in the Rust fast path: row-major (i, j),
    # skipping the diagonal when self_loops=False.
    fixed_keys: set[tuple[int, int]] = {
        (int(s), int(t)) for s, t in zip(k_src, k_tgt, strict=True)
    }
    adm_src: list[int] = []
    adm_tgt: list[int] = []
    for i in range(n):
        for j in range(n):
            if not sl and i == j:
                continue
            if (i, j) in fixed_keys:
                continue
            adm_src.append(i)
            adm_tgt.append(j)

    if e_res > len(adm_src):
        msg = (
            f"residual edges ({e_res}) exceed admissible pairs ({len(adm_src)}) "
            "after removing fixed pairs"
        )
        raise ValueError(msg)
    if e_res > t_res:
        msg = f"residual total ({t_res}) < residual edges ({e_res})"
        raise ValueError(msg)

    # Sample the residual graph
    adm_src_arr = np.asarray(adm_src, dtype=np.uint64)
    adm_tgt_arr = np.asarray(adm_tgt, dtype=np.uint64)
    e_dispatch = {
        "ME": lambda: _sample_me_fixed_et_explicit(
            adm_src_arr,
            adm_tgt_arr,
            residual_edges=e_res,
            residual_total=t_res,
            seed=seed,
        ),
        "B": lambda: _sample_b_fixed_et_explicit(
            adm_src_arr,
            adm_tgt_arr,
            layers=layers,
            residual_edges=e_res,
            residual_total=t_res,
            seed=seed,
        ),
        "W": lambda: _sample_w_fixed_et_explicit(
            adm_src_arr,
            adm_tgt_arr,
            layers=layers,
            residual_edges=e_res,
            residual_total=t_res,
            seed=seed,
        ),
    }
    residual = e_dispatch[family]()

    # ---- merge fixed pairs back (drop fixed zeros: occ_num == 0) ----
    pos_mask = k_occ > 0
    k_src_pos = k_src[pos_mask]
    k_tgt_pos = k_tgt[pos_mask]
    k_occ_pos = k_occ[pos_mask]
    if len(residual) == 0:
        return EdgeTable(source=k_src_pos, target=k_tgt_pos, occ_num=k_occ_pos)
    merged = EdgeTable(
        source=np.concatenate([residual.source, k_src_pos]),
        target=np.concatenate([residual.target, k_tgt_pos]),
        occ_num=np.concatenate([residual.occ_num, k_occ_pos]),
    )
    # Final validation: total E and T must match the original request
    assert len(merged) == e_total, f"merged E={len(merged)} != requested E={e_total}"
    assert int(merged.occ_num.sum()) == t_total, (
        f"merged T={merged.occ_num.sum()} != requested T={t_total}"
    )
    return merged


def _node_count(edges: EdgeTable) -> int:
    if len(edges) == 0:
        return 0
    return int(max(edges.source.max(), edges.target.max())) + 1


def _strengths(edges: EdgeTable, node_count: int) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros(node_count, dtype=np.uint64)
    incoming = np.zeros(node_count, dtype=np.uint64)
    np.add.at(out, edges.source, edges.occ_num)
    np.add.at(incoming, edges.target, edges.occ_num)
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
    coord_x: NDArray[Any] | None = None,
    coord_y: NDArray[Any] | None = None,
    node_count: int | None = None,
    total_events: int | None = None,
    target_edges: float | None = None,
) -> FilterResult:
    from menobis.filtering import models as filtering

    if ensemble != Ensemble.GRAND_CANONICAL:
        msg = "filtering requires ensemble=GRAND_CANONICAL"
        raise UnsupportedModelCaseError(msg)

    if fit is None:
        if constraint == Constraint.EDGES_EVENTS:
            fit = _fit_model(
                ensemble=ensemble,
                family=family,
                constraint=constraint,
                node_count=node_count,
                total_events=total_events,
                target_edges=target_edges,
                layers=layers,
                self_loops=self_loops,
            )
        else:
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
        "min_occupation": min_occupation,
        "min_expected": min_expected,
        "max_absent": max_absent,
    }
    dispatch: dict[tuple[Constraint, str], Callable[..., FilterResult]] = {
        (Constraint.STRENGTH, "poisson"): filtering._filter_strength_poisson,
        (Constraint.STRENGTH, "binomial"): filtering._filter_strength_binomial,
        (Constraint.STRENGTH, "geometric"): filtering._filter_strength_geometric,
        (
            Constraint.STRENGTH,
            "negative_binomial",
        ): filtering._filter_strength_negative_binomial,
        (Constraint.STRENGTH_COST, "poisson"): filtering._filter_strength_cost_poisson,
        (
            Constraint.STRENGTH_COST,
            "binomial",
        ): filtering._filter_strength_cost_binomial,
        (
            Constraint.STRENGTH_COST,
            "geometric",
        ): filtering._filter_strength_cost_geometric,
        (
            Constraint.STRENGTH_COST,
            "negative_binomial",
        ): filtering._filter_strength_cost_negative_binomial,
        (
            Constraint.STRENGTH_EDGES,
            "poisson",
        ): filtering._filter_strength_edges_poisson,
        (
            Constraint.STRENGTH_EDGES,
            "binomial",
        ): filtering._filter_strength_edges_binomial,
        (
            Constraint.STRENGTH_EDGES,
            "geometric",
        ): filtering._filter_strength_edges_geometric,
        (
            Constraint.STRENGTH_EDGES,
            "negative_binomial",
        ): filtering._filter_strength_edges_negative_binomial,
        (
            Constraint.STRENGTH_DEGREE,
            "poisson",
        ): filtering._filter_strength_degree_poisson,
        (
            Constraint.STRENGTH_DEGREE,
            "binomial",
        ): filtering._filter_strength_degree_binomial,
        (
            Constraint.STRENGTH_DEGREE,
            "geometric",
        ): filtering._filter_strength_degree_geometric,
        (
            Constraint.STRENGTH_DEGREE,
            "negative_binomial",
        ): filtering._filter_strength_degree_negative_binomial,
        (Constraint.DEGREE_EVENTS, "poisson"): filtering._filter_degree_events_poisson,
        (
            Constraint.DEGREE_EVENTS,
            "binomial",
        ): filtering._filter_degree_events_binomial,
        (
            Constraint.DEGREE_EVENTS,
            "geometric",
        ): filtering._filter_degree_events_geometric,
        (
            Constraint.DEGREE_EVENTS,
            "negative_binomial",
        ): filtering._filter_degree_events_negative_binomial,
        (Constraint.EDGES_EVENTS, "poisson"): filtering._filter_edges_events,
        (Constraint.EDGES_EVENTS, "binomial"): filtering._filter_edges_events,
        (Constraint.EDGES_EVENTS, "geometric"): filtering._filter_edges_events,
        (Constraint.EDGES_EVENTS, "negative_binomial"): filtering._filter_edges_events,
    }
    if constraint == Constraint.STRENGTH_COST:
        if coord_x is None or coord_y is None:
            msg = "strength_cost filtering requires projected coord_x and coord_y"
            raise ValueError(msg)
        kwargs["coord_x"] = coord_x
        kwargs["coord_y"] = coord_y

    key = (constraint, variant)
    if key not in dispatch:
        msg = f"unsupported (constraint, family): ({constraint!r}, {family!r})"
        raise UnsupportedModelCaseError(msg)
    return dispatch[key](edges, fit, **kwargs)


def _fit_partial(
    *,
    family: ModelFamily,
    constraint: Constraint,
    strength_out: NDArray[Any] | None,
    strength_in: NDArray[Any] | None,
    degree_out: NDArray[Any] | None,
    degree_in: NDArray[Any] | None,
    target_edges: float | None,
    target_cost: float | None,
    coord_x: NDArray[Any] | None,
    coord_y: NDArray[Any] | None,
    known_source: NDArray[Any],
    known_target: NDArray[Any],
    known_occnum: NDArray[Any],
    layers: int,
    self_loops: bool,
    tolerance: float,
    max_iterations: int,
) -> Any:  # noqa: ANN401
    """Dispatch partial fitting by family and constraint."""
    from menobis.models.partial import (
        _fit_partial_strength_binomial,
        _fit_partial_strength_cost_binomial_coordinates,
        _fit_partial_strength_cost_geometric_coordinates,
        _fit_partial_strength_cost_negative_binomial_coordinates,
        _fit_partial_strength_cost_poisson_coordinates,
        _fit_partial_strength_degree_binomial,
        _fit_partial_strength_degree_geometric,
        _fit_partial_strength_degree_poisson,
        _fit_partial_strength_edges_binomial,
        _fit_partial_strength_edges_geometric,
        _fit_partial_strength_edges_poisson,
        _fit_partial_strength_geometric,
        _fit_partial_strength_poisson,
    )

    if strength_out is None or strength_in is None:
        msg = "partial fitting requires strength_out and strength_in"
        raise ValueError(msg)

    variant = _fit_variant(family, layers)

    if constraint == Constraint.STRENGTH:
        dispatch: dict[str, Callable[..., Any]] = {
            "poisson": lambda: _fit_partial_strength_poisson(
                strength_out,
                strength_in,
                known_source,
                known_target,
                known_occnum,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            ),
            "binomial": lambda: _fit_partial_strength_binomial(
                strength_out,
                strength_in,
                known_source,
                known_target,
                known_occnum,
                layers=layers,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            ),
            "geometric": lambda: _fit_partial_strength_geometric(
                strength_out,
                strength_in,
                known_source,
                known_target,
                known_occnum,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            ),
            "negative_binomial": lambda: _fit_partial_strength_geometric(
                strength_out,
                strength_in,
                known_source,
                known_target,
                known_occnum,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            ),
        }
        return dispatch[variant]()

    if constraint == Constraint.STRENGTH_EDGES:
        if target_edges is None:
            msg = "partial strength-edges fitting requires target_edges"
            raise ValueError(msg)
        dispatch = {
            "poisson": lambda: _fit_partial_strength_edges_poisson(
                strength_out,
                strength_in,
                known_source,
                known_target,
                known_occnum,
                target_edges,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            ),
            "binomial": lambda: _fit_partial_strength_edges_binomial(
                strength_out,
                strength_in,
                known_source,
                known_target,
                known_occnum,
                target_edges,
                layers=layers,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            ),
            "geometric": lambda: _fit_partial_strength_edges_geometric(
                strength_out,
                strength_in,
                known_source,
                known_target,
                known_occnum,
                target_edges,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            ),
            "negative_binomial": lambda: _fit_partial_strength_edges_geometric(
                strength_out,
                strength_in,
                known_source,
                known_target,
                known_occnum,
                target_edges,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            ),
        }
        return dispatch[variant]()

    if constraint == Constraint.STRENGTH_DEGREE:
        if degree_out is None or degree_in is None:
            msg = "partial strength-degree fitting requires degree_out and degree_in"
            raise ValueError(msg)
        dispatch = {
            "poisson": lambda: _fit_partial_strength_degree_poisson(
                strength_out,
                strength_in,
                degree_out,
                degree_in,
                known_source,
                known_target,
                known_occnum,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            ),
            "binomial": lambda: _fit_partial_strength_degree_binomial(
                strength_out,
                strength_in,
                degree_out,
                degree_in,
                known_source,
                known_target,
                known_occnum,
                layers=layers,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            ),
            "geometric": lambda: _fit_partial_strength_degree_geometric(
                strength_out,
                strength_in,
                degree_out,
                degree_in,
                known_source,
                known_target,
                known_occnum,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            ),
            "negative_binomial": lambda: _fit_partial_strength_degree_geometric(
                strength_out,
                strength_in,
                degree_out,
                degree_in,
                known_source,
                known_target,
                known_occnum,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            ),
        }
        return dispatch[variant]()

    if constraint == Constraint.STRENGTH_COST:
        if coord_x is None or coord_y is None or target_cost is None:
            msg = "partial strength-cost requires coord_x, coord_y, target_cost"
            raise ValueError(msg)
        dispatch = {
            "poisson": lambda: _fit_partial_strength_cost_poisson_coordinates(
                strength_out,
                strength_in,
                known_source,
                known_target,
                known_occnum,
                coord_x,
                coord_y,
                target_cost,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            ),
            "binomial": lambda: _fit_partial_strength_cost_binomial_coordinates(
                strength_out,
                strength_in,
                known_source,
                known_target,
                known_occnum,
                coord_x,
                coord_y,
                target_cost,
                layers=layers,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            ),
            "geometric": lambda: _fit_partial_strength_cost_geometric_coordinates(
                strength_out,
                strength_in,
                known_source,
                known_target,
                known_occnum,
                coord_x,
                coord_y,
                target_cost,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            ),
            "negative_binomial": lambda: (
                _fit_partial_strength_cost_negative_binomial_coordinates(
                    strength_out,
                    strength_in,
                    known_source,
                    known_target,
                    known_occnum,
                    coord_x,
                    coord_y,
                    target_cost,
                    layers=layers,
                    self_loops=self_loops,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )
            ),
        }
        return dispatch[variant]()

    msg = f"partial fitting not supported for constraint: {constraint!r}"
    raise UnsupportedModelCaseError(msg)


__all__ = [
    "Constraint",
    "Ensemble",
    "ModelFamily",
    "UnsupportedModelCaseError",
    "filter_model",
    "fit_model",
    "sample_model",
    "sample_model_detailed",
]
