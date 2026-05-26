"""Unified model fitting and sampling router."""

from enum import StrEnum

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


def _normalize_ensemble(value: Ensemble | str) -> Ensemble:
    normalized = value.value if isinstance(value, Ensemble) else value
    normalized = normalized.replace("-", "_").lower()
    if normalized in {"grand_canonical", "grandcanonical"}:
        return Ensemble.GRAND_CANONICAL
    return Ensemble(normalized)


def _normalize_family(value: Family | str) -> Family:
    normalized = value.value if isinstance(value, Family) else value
    normalized = normalized.replace("-", "_").lower()
    if normalized == "poisson":
        return Family.ME
    return Family(normalized)


def _normalize_constraint(value: Constraint | str) -> Constraint:
    normalized = value.value if isinstance(value, Constraint) else value
    return Constraint(normalized.replace("-", "_").lower())


def _require_strengths(
    strength_out: NDArray[np.floating] | NDArray[np.integer] | None = None,
    strength_in: NDArray[np.floating] | NDArray[np.integer] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if strength_out is None or strength_in is None:
        msg = "strength constraints require strength_out and strength_in"
        raise ValueError(msg)
    return (
        np.asarray(strength_out, dtype=np.float64),
        np.asarray(strength_in, dtype=np.float64),
    )


def _require_degrees(
    degree_out: NDArray[np.floating] | None,
    degree_in: NDArray[np.floating] | None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    if degree_out is None or degree_in is None:
        msg = "degree-events constraints require degree_out and degree_in"
        raise ValueError(msg)
    return degree_out, degree_in


def fit_model(
    *,
    ensemble: Ensemble | str = Ensemble.GRAND_CANONICAL,
    family: Family | str,
    constraint: Constraint | str,
    strength_out: NDArray[np.floating] | NDArray[np.integer] | None = None,
    strength_in: NDArray[np.floating] | NDArray[np.integer] | None = None,
    degree_out: NDArray[np.floating] | None = None,
    degree_in: NDArray[np.floating] | None = None,
    total_events: int | None = None,
    layers: int = 1,
    self_loops: bool = True,
    tolerance: float = 1e-8,
    max_iterations: int = 10000,
) -> StrengthFit | DegreeEventsFit:
    """Fit a model from ensemble, family, and constraint names.

    Canonical fixed-strength ME uses the same fitted multipliers as the
    grand-canonical ME case and differs only at sampling time. Microcanonical ME
    fixed strengths are sampled by stub matching and therefore have no fitted
    multiplier object.
    """
    ens = _normalize_ensemble(ensemble)
    fam = _normalize_family(family)
    con = _normalize_constraint(constraint)

    if ens == Ensemble.MICROCANONICAL:
        msg = (
            "microcanonical ME fixed strength has no fitted multiplier"
            " object; use sample_model"
        )
        raise UnsupportedModelCaseError(msg)
    if ens == Ensemble.CANONICAL and fam != Family.ME:
        msg = "canonical ensemble is ME-only"
        raise UnsupportedModelCaseError(msg)
    if ens not in {Ensemble.GRAND_CANONICAL, Ensemble.CANONICAL}:
        msg = f"unsupported ensemble: {ens}"
        raise UnsupportedModelCaseError(msg)

    if con == Constraint.STRENGTH:
        s_out, s_in = _require_strengths(strength_out, strength_in)
        if fam == Family.ME:
            return fit_strength_poisson(
                s_out,
                s_in,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
        if ens == Ensemble.CANONICAL:
            msg = "canonical ensemble is ME-only"
            raise UnsupportedModelCaseError(msg)
        if fam == Family.BINOMIAL:
            return fit_strength_binomial(
                s_out,
                s_in,
                layers=layers,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
        if fam == Family.GEOMETRIC:
            return fit_strength_geometric(
                s_out,
                s_in,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
        if fam == Family.NEGATIVE_BINOMIAL:
            return fit_strength_negative_binomial(
                s_out,
                s_in,
                layers=layers,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )

    if con == Constraint.DEGREE_EVENTS:
        if ens != Ensemble.GRAND_CANONICAL:
            msg = "degree-events is supported only for grandcanonical ensembles"
            raise UnsupportedModelCaseError(msg)
        k_out, k_in = _require_degrees(degree_out, degree_in)
        if total_events is None:
            msg = "degree-events fitting requires total_events"
            raise ValueError(msg)
        if fam == Family.ME:
            return fit_degree_events_poisson(
                k_out,
                k_in,
                total_events,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
        if fam == Family.BINOMIAL:
            return fit_degree_events_binomial(
                k_out,
                k_in,
                total_events,
                layers=layers,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
        if fam == Family.GEOMETRIC:
            return fit_degree_events_geometric(
                k_out,
                k_in,
                total_events,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
        if fam == Family.NEGATIVE_BINOMIAL:
            return fit_degree_events_negative_binomial(
                k_out,
                k_in,
                total_events,
                layers=layers,
                self_loops=self_loops,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )

    msg = f"unsupported model case: ensemble={ens}, family={fam}, constraint={con}"
    raise UnsupportedModelCaseError(msg)


def sample_model(
    *,
    ensemble: Ensemble | str = Ensemble.GRAND_CANONICAL,
    family: Family | str,
    constraint: Constraint | str,
    fit: StrengthFit | DegreeEventsFit | None = None,
    strength_out: NDArray[np.integer] | None = None,
    strength_in: NDArray[np.integer] | None = None,
    total_events: int | None = None,
    seed: int = 0,
) -> EdgeTable:
    """Sample a model from ensemble, family, and constraint names."""
    ens = _normalize_ensemble(ensemble)
    fam = _normalize_family(family)
    con = _normalize_constraint(constraint)

    if ens == Ensemble.MICROCANONICAL:
        if fam != Family.ME or con != Constraint.STRENGTH:
            msg = "microcanonical ensemble supports only ME fixed strength"
            raise UnsupportedModelCaseError(msg)
        if strength_out is None or strength_in is None:
            msg = (
                "microcanonical strength sampling requires strength_out and strength_in"
            )
            raise ValueError(msg)
        return sample_strength_stub_matching(strength_out, strength_in, seed=seed)

    if fit is None:
        msg = "sampling requires a fit result except for microcanonical stub matching"
        raise ValueError(msg)

    if ens == Ensemble.CANONICAL:
        if fam != Family.ME or con != Constraint.STRENGTH:
            msg = "canonical ensemble supports only ME fixed strength"
            raise UnsupportedModelCaseError(msg)
        if not isinstance(fit, StrengthFit):
            msg = "canonical strength sampling requires StrengthFit"
            raise TypeError(msg)
        if total_events is None:
            msg = "canonical strength sampling requires total_events"
            raise ValueError(msg)
        return sample_strength_multinomial(
            fit.x,
            fit.y,
            total_events=total_events,
            self_loops=fit.self_loops,
            seed=seed,
        )

    if ens != Ensemble.GRAND_CANONICAL:
        msg = f"unsupported ensemble: {ens}"
        raise UnsupportedModelCaseError(msg)

    if con == Constraint.STRENGTH:
        if not isinstance(fit, StrengthFit):
            msg = "strength sampling requires StrengthFit"
            raise TypeError(msg)
        if fam == Family.ME:
            return sample_strength_poisson(
                fit.x, fit.y, self_loops=fit.self_loops, seed=seed
            )
        if fam == Family.BINOMIAL:
            return sample_strength_binomial(
                fit.x,
                fit.y,
                layers=fit.layers or 1,
                self_loops=fit.self_loops,
                seed=seed,
            )
        if fam == Family.GEOMETRIC:
            return sample_strength_geometric(
                fit.x, fit.y, self_loops=fit.self_loops, seed=seed
            )
        if fam == Family.NEGATIVE_BINOMIAL:
            return sample_strength_negative_binomial(
                fit.x,
                fit.y,
                layers=fit.layers or 1,
                self_loops=fit.self_loops,
                seed=seed,
            )

    if con == Constraint.DEGREE_EVENTS:
        if not isinstance(fit, DegreeEventsFit):
            msg = "degree-events sampling requires DegreeEventsFit"
            raise TypeError(msg)
        if fam == Family.ME:
            return sample_degree_events_poisson(fit, seed=seed)
        if fam == Family.BINOMIAL:
            return sample_degree_events_binomial(fit, seed=seed)
        if fam == Family.GEOMETRIC:
            return sample_degree_events_geometric(fit, seed=seed)
        if fam == Family.NEGATIVE_BINOMIAL:
            return sample_degree_events_negative_binomial(fit, seed=seed)

    msg = f"unsupported model case: ensemble={ens}, family={fam}, constraint={con}"
    raise UnsupportedModelCaseError(msg)


__all__ = [
    "Constraint",
    "Ensemble",
    "Family",
    "UnsupportedModelCaseError",
    "fit_model",
    "sample_model",
]
