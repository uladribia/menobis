"""Unified model fitting and sampling router."""

from collections.abc import Callable
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from menobis.data.frames import EdgeTable
from menobis.models.fitting import (
    DegreeEventsFit,
    StrengthFit,
    fit_degree_events_binomial,
    fit_degree_events_geometric,
    fit_degree_events_negative_binomial,
    fit_degree_events_poisson,
    fit_strength_binomial,
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
    """Supported routed constraint families."""

    STRENGTH = "strength"
    DEGREE_EVENTS = "degree_events"


class UnsupportedModelCaseError(ValueError):
    """Raised when a requested ensemble/family/constraint case is unsupported."""


# ---------------------------------------------------------------------------
# Dispatch tables
# ---------------------------------------------------------------------------

_FIT_STRENGTH: dict[Family, Callable[..., StrengthFit]] = {
    Family.ME: fit_strength_poisson,
    Family.BINOMIAL: fit_strength_binomial,
    Family.GEOMETRIC: fit_strength_geometric,
    Family.NEGATIVE_BINOMIAL: fit_strength_negative_binomial,
}

_FIT_DEGREE_EVENTS: dict[Family, Callable[..., DegreeEventsFit]] = {
    Family.ME: fit_degree_events_poisson,
    Family.BINOMIAL: fit_degree_events_binomial,
    Family.GEOMETRIC: fit_degree_events_geometric,
    Family.NEGATIVE_BINOMIAL: fit_degree_events_negative_binomial,
}

_SAMPLE_STRENGTH: dict[Family, Callable[..., EdgeTable]] = {
    Family.ME: lambda fit, seed: sample_strength_poisson(
        fit.x, fit.y, self_loops=fit.self_loops, seed=seed
    ),
    Family.BINOMIAL: lambda fit, seed: sample_strength_binomial(
        fit.x, fit.y, layers=fit.layers or 1, self_loops=fit.self_loops, seed=seed
    ),
    Family.GEOMETRIC: lambda fit, seed: sample_strength_geometric(
        fit.x, fit.y, self_loops=fit.self_loops, seed=seed
    ),
    Family.NEGATIVE_BINOMIAL: lambda fit, seed: sample_strength_negative_binomial(
        fit.x, fit.y, layers=fit.layers or 1, self_loops=fit.self_loops, seed=seed
    ),
}

_SAMPLE_DEGREE_EVENTS: dict[Family, Callable[..., EdgeTable]] = {
    Family.ME: lambda fit, seed: sample_degree_events_poisson(fit, seed=seed),
    Family.BINOMIAL: lambda fit, seed: sample_degree_events_binomial(fit, seed=seed),
    Family.GEOMETRIC: lambda fit, seed: sample_degree_events_geometric(fit, seed=seed),
    Family.NEGATIVE_BINOMIAL: lambda fit, seed: sample_degree_events_negative_binomial(
        fit, seed=seed
    ),
}


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
    layers: int = 1,
    self_loops: bool = True,
    tolerance: float = 1e-8,
    max_iterations: int = 10000,
) -> StrengthFit | DegreeEventsFit:
    """Fit a model selected by ensemble, family, and constraint.

    Args:
        ensemble: Statistical ensemble. Only ``GRAND_CANONICAL`` and
            ``CANONICAL`` support fitting. Microcanonical has no fitted
            multipliers (use ``sample_model`` directly).
        family: Distribution family (ME, binomial, geometric, negative_binomial).
        constraint: Constraint type (strength, degree_events).
        strength_out: Outgoing strength per node (for strength constraints).
        strength_in: Incoming strength per node (for strength constraints).
        degree_out: Outgoing degree per node (for degree_events constraints).
        degree_in: Incoming degree per node (for degree_events constraints).
        total_events: Total events T (required for degree_events).
        layers: Number of layers M (binomial/negative_binomial families).
        self_loops: Whether self-loops are allowed.
        tolerance: Convergence tolerance.
        max_iterations: Maximum solver iterations.

    Raises:
        UnsupportedModelCaseError: If the ensemble/family/constraint combination
            is not implemented.
        ValueError: If required constraint arrays or parameters are missing.
    """
    if ensemble == Ensemble.MICROCANONICAL:
        msg = (
            "microcanonical has no fitted multipliers; "
            "use sample_model with strength sequences directly"
        )
        raise UnsupportedModelCaseError(msg)

    if ensemble == Ensemble.CANONICAL and family != Family.ME:
        msg = "canonical ensemble supports only family=ME"
        raise UnsupportedModelCaseError(msg)

    if constraint == Constraint.STRENGTH:
        if strength_out is None or strength_in is None:
            msg = "strength constraint requires strength_out and strength_in"
            raise ValueError(msg)
        s_out = np.asarray(strength_out, dtype=np.float64)
        s_in = np.asarray(strength_in, dtype=np.float64)
        fitter = _FIT_STRENGTH.get(family)
        if fitter is None:
            msg = f"unsupported family for strength: {family}"
            raise UnsupportedModelCaseError(msg)
        kwargs: dict[str, Any] = {
            "self_loops": self_loops,
            "tolerance": tolerance,
            "max_iterations": max_iterations,
        }
        if family in {Family.BINOMIAL, Family.NEGATIVE_BINOMIAL}:
            kwargs["layers"] = layers
        return fitter(s_out, s_in, **kwargs)

    if constraint == Constraint.DEGREE_EVENTS:
        if ensemble != Ensemble.GRAND_CANONICAL:
            msg = "degree_events constraint requires ensemble=GRAND_CANONICAL"
            raise UnsupportedModelCaseError(msg)
        if degree_out is None or degree_in is None:
            msg = "degree_events constraint requires degree_out and degree_in"
            raise ValueError(msg)
        if total_events is None:
            msg = "degree_events constraint requires total_events"
            raise ValueError(msg)
        k_out = np.asarray(degree_out, dtype=np.float64)
        k_in = np.asarray(degree_in, dtype=np.float64)
        fitter_de = _FIT_DEGREE_EVENTS.get(family)
        if fitter_de is None:
            msg = f"unsupported family for degree_events: {family}"
            raise UnsupportedModelCaseError(msg)
        kwargs_de: dict[str, Any] = {
            "self_loops": self_loops,
            "tolerance": tolerance,
            "max_iterations": max_iterations,
        }
        if family in {Family.BINOMIAL, Family.NEGATIVE_BINOMIAL}:
            kwargs_de["layers"] = layers
        return fitter_de(k_out, k_in, total_events, **kwargs_de)

    msg = f"unsupported constraint: {constraint}"
    raise UnsupportedModelCaseError(msg)


def sample_model(
    *,
    ensemble: Ensemble = Ensemble.GRAND_CANONICAL,
    family: Family,
    constraint: Constraint,
    fit: StrengthFit | DegreeEventsFit | None = None,
    strength_out: NDArray[Any] | None = None,
    strength_in: NDArray[Any] | None = None,
    total_events: int | None = None,
    seed: int = 0,
) -> EdgeTable:
    """Sample a network from a fitted model or directly via stub matching.

    Args:
        ensemble: Statistical ensemble.
        family: Distribution family.
        constraint: Constraint type.
        fit: Fitted result (required for grand-canonical and canonical).
        strength_out: Outgoing strengths (required for microcanonical).
        strength_in: Incoming strengths (required for microcanonical).
        total_events: Fixed total events (required for canonical).
        seed: Random seed.

    Raises:
        UnsupportedModelCaseError: If the combination is not implemented.
        ValueError: If required parameters are missing.
        TypeError: If ``fit`` has the wrong type for the requested route.
    """
    if ensemble == Ensemble.MICROCANONICAL:
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

    if fit is None:
        msg = "grand-canonical and canonical sampling require a fit result"
        raise ValueError(msg)

    if ensemble == Ensemble.CANONICAL:
        if family != Family.ME or constraint != Constraint.STRENGTH:
            msg = "canonical supports only family=ME, constraint=STRENGTH"
            raise UnsupportedModelCaseError(msg)
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
            fit.x,
            fit.y,
            total_events=total_events,
            self_loops=fit.self_loops,
            seed=seed,
        )

    if ensemble != Ensemble.GRAND_CANONICAL:
        msg = f"unsupported ensemble: {ensemble}"
        raise UnsupportedModelCaseError(msg)

    if constraint == Constraint.STRENGTH:
        if not isinstance(fit, StrengthFit):
            msg = f"strength sampling requires StrengthFit, got {type(fit).__name__}"
            raise TypeError(msg)
        sampler = _SAMPLE_STRENGTH.get(family)
        if sampler is None:
            msg = f"unsupported family for strength sampling: {family}"
            raise UnsupportedModelCaseError(msg)
        return sampler(fit, seed)

    if constraint == Constraint.DEGREE_EVENTS:
        if not isinstance(fit, DegreeEventsFit):
            msg = (
                "degree_events sampling requires DegreeEventsFit,"
                f" got {type(fit).__name__}"
            )
            raise TypeError(msg)
        sampler_de = _SAMPLE_DEGREE_EVENTS.get(family)
        if sampler_de is None:
            msg = f"unsupported family for degree_events sampling: {family}"
            raise UnsupportedModelCaseError(msg)
        return sampler_de(fit, seed)

    msg = f"unsupported constraint: {constraint}"
    raise UnsupportedModelCaseError(msg)


__all__ = [
    "Constraint",
    "Ensemble",
    "Family",
    "UnsupportedModelCaseError",
    "fit_model",
    "sample_model",
]
