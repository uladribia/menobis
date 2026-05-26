"""Model ontology and routing vocabulary for MENoBiS."""

from enum import StrEnum


class Verb(StrEnum):
    """Supported model workflow verbs."""

    FIT = "fit"
    SAMPLE = "sample"
    FILTER = "filter"


class Ensemble(StrEnum):
    """Supported ensemble types."""

    GRAND_CANONICAL = "grandcanonical"
    CANONICAL = "canonical"
    MICROCANONICAL = "microcanonical"


class ModelFamily(StrEnum):
    """MENoBiS model families from the thesis ontology."""

    ME = "me"
    B = "b"
    W = "w"


class Constraint(StrEnum):
    """Supported constraint families."""

    STRENGTH = "strength"
    STRENGTH_COST = "strength_cost"
    STRENGTH_EDGES = "strength_edges"
    STRENGTH_DEGREE = "strength_degree"
    DEGREE_EVENTS = "degree_events"


class UnsupportedModelCaseError(ValueError):
    """Raised when a requested verb/ensemble/family/constraint case is unsupported."""


__all__ = [
    "Constraint",
    "Ensemble",
    "ModelFamily",
    "UnsupportedModelCaseError",
    "Verb",
]
