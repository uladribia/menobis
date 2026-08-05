"""Capability registry for MENoBiS model routes.

One machine-readable source of truth keyed by ``(verb, ensemble, family,
constraint)``. Routing tests and end-to-end tests must agree with this
matrix; a supported entry without a fixture is a test failure.
"""

from __future__ import annotations

from dataclasses import dataclass

from menobis.models.spec import Constraint, Ensemble, ModelFamily, Verb

# ---------------------------------------------------------------------------
# Capability record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelCapability:
    """Capability of one (verb, ensemble, family, constraint) route."""

    supported: bool
    requires_fit: bool
    backend: str
    required_arguments: frozenset[str] = frozenset()
    optional_arguments: frozenset[str] = frozenset()
    supports_self_loops: bool = True
    supports_no_self_loops: bool = True
    result_kind: str = "fit"


# ---------------------------------------------------------------------------
# Sampling exactness
# ---------------------------------------------------------------------------


class SamplingExactness:
    """Exactness category of a generation method."""

    EXACT_INDEPENDENT = "exact_independent"
    EXACT_DIRECT = "exact_direct"
    EXACT_STATIONARY_MCMC = "exact_stationary_mcmc"
    EXACT_PSEUDO_MARGINAL = "exact_pseudo_marginal"
    APPROXIMATE = "approximate"
    HEURISTIC = "heuristic"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_FAMILIES = (ModelFamily.ME, ModelFamily.B, ModelFamily.W)
_CONSTRAINTS = (
    Constraint.STRENGTH,
    Constraint.STRENGTH_COST,
    Constraint.STRENGTH_EDGES,
    Constraint.STRENGTH_DEGREE,
    Constraint.DEGREE_EVENTS,
    Constraint.EDGES_EVENTS,
)

# Arguments each constraint consumes for fitting.
_FIT_ARGS: dict[Constraint, frozenset[str]] = {
    Constraint.STRENGTH: frozenset({"strength_out", "strength_in"}),
    Constraint.STRENGTH_COST: frozenset(
        {"strength_out", "strength_in", "coord_x", "coord_y", "target_cost"}
    ),
    Constraint.STRENGTH_EDGES: frozenset(
        {"strength_out", "strength_in", "target_edges"}
    ),
    Constraint.STRENGTH_DEGREE: frozenset(
        {"strength_out", "strength_in", "degree_out", "degree_in"}
    ),
    Constraint.DEGREE_EVENTS: frozenset({"degree_out", "degree_in", "total_events"}),
    Constraint.EDGES_EVENTS: frozenset({"target_edges", "total_events", "node_count"}),
}

# Arguments each constraint consumes for sampling.
_SAMPLE_ARGS: dict[Constraint, frozenset[str]] = {
    Constraint.STRENGTH: frozenset({"fit"}),
    Constraint.STRENGTH_COST: frozenset({"fit", "coord_x", "coord_y"}),
    Constraint.STRENGTH_EDGES: frozenset({"fit"}),
    Constraint.STRENGTH_DEGREE: frozenset({"fit"}),
    Constraint.DEGREE_EVENTS: frozenset({"fit"}),
    Constraint.EDGES_EVENTS: frozenset({"fit"}),
}

# Arguments each constraint consumes for filtering.
_FILTER_ARGS: dict[Constraint, frozenset[str]] = {
    Constraint.STRENGTH: frozenset({"fit"}),
    Constraint.STRENGTH_COST: frozenset({"fit", "coord_x", "coord_y"}),
    Constraint.STRENGTH_EDGES: frozenset({"fit"}),
    Constraint.STRENGTH_DEGREE: frozenset({"fit"}),
    Constraint.DEGREE_EVENTS: frozenset({"fit"}),
    Constraint.EDGES_EVENTS: frozenset({"fit"}),
}


def _build_registry() -> dict[
    tuple[Verb, Ensemble, ModelFamily, Constraint], ModelCapability
]:
    """Build the full (verb, ensemble, family, constraint) capability matrix."""
    registry: dict[tuple[Verb, Ensemble, ModelFamily, Constraint], ModelCapability] = {}

    # --- FIT ---
    for family in _FAMILIES:
        for constraint in _CONSTRAINTS:
            key = (Verb.FIT, Ensemble.GRAND_CANONICAL, family, constraint)
            registry[key] = ModelCapability(
                supported=True,
                requires_fit=False,
                backend="grandcanonical_fit",
                required_arguments=_FIT_ARGS[constraint],
                optional_arguments=frozenset({"layers", "self_loops"}),
                result_kind="fit",
            )
            # Canonical fitting reuses the grand-canonical solver (ME only).
            if family is ModelFamily.ME:
                key = (Verb.FIT, Ensemble.CANONICAL, family, constraint)
                registry[key] = ModelCapability(
                    supported=True,
                    requires_fit=False,
                    backend="canonical_fit",
                    required_arguments=_FIT_ARGS[constraint],
                    optional_arguments=frozenset({"self_loops"}),
                    result_kind="fit",
                )

    # --- SAMPLE ---
    for family in _FAMILIES:
        for constraint in _CONSTRAINTS:
            key = (Verb.SAMPLE, Ensemble.GRAND_CANONICAL, family, constraint)
            registry[key] = ModelCapability(
                supported=True,
                requires_fit=True,
                backend="grandcanonical_independent",
                required_arguments=_SAMPLE_ARGS[constraint],
                optional_arguments=frozenset({"seed"}),
                result_kind="sampled_network",
            )
    # Canonical ME strength: exact fixed-total multinomial.
    registry[(Verb.SAMPLE, Ensemble.CANONICAL, ModelFamily.ME, Constraint.STRENGTH)] = (
        ModelCapability(
            supported=True,
            requires_fit=True,
            backend="canonical_multinomial",
            required_arguments=frozenset({"fit", "total_events"}),
            optional_arguments=frozenset({"seed"}),
            result_kind="sampled_network",
        )
    )
    # Microcanonical ME strength: exact stub matching, no fit.
    registry[
        (Verb.SAMPLE, Ensemble.MICROCANONICAL, ModelFamily.ME, Constraint.STRENGTH)
    ] = ModelCapability(
        supported=True,
        requires_fit=False,
        backend="microcanonical_stub_matching",
        required_arguments=frozenset({"strength_out", "strength_in"}),
        optional_arguments=frozenset({"seed"}),
        supports_self_loops=True,
        supports_no_self_loops=False,  # no-self-loop stub matching unsupported
        result_kind="sampled_network",
    )
    # Microcanonical ME fixed (E,T): exact direct sampler, no fit.
    registry[
        (Verb.SAMPLE, Ensemble.MICROCANONICAL, ModelFamily.ME, Constraint.EDGES_EVENTS)
    ] = ModelCapability(
        supported=True,
        requires_fit=False,
        backend="microcanonical_fixed_et",
        required_arguments=frozenset({"node_count", "target_edges", "total_events"}),
        optional_arguments=frozenset({"seed", "self_loops"}),
        supports_self_loops=True,
        supports_no_self_loops=True,
        result_kind="sampled_network",
    )
    # Microcanonical B fixed (E,T): exact direct sampler, no fit.
    registry[
        (Verb.SAMPLE, Ensemble.MICROCANONICAL, ModelFamily.B, Constraint.EDGES_EVENTS)
    ] = ModelCapability(
        supported=True,
        requires_fit=False,
        backend="microcanonical_fixed_et",
        required_arguments=frozenset(
            {"node_count", "target_edges", "total_events", "layers"}
        ),
        optional_arguments=frozenset({"seed", "self_loops"}),
        supports_self_loops=True,
        supports_no_self_loops=True,
        result_kind="sampled_network",
    )
    # Microcanonical W fixed (E,T): exact direct sampler, no fit.
    registry[
        (Verb.SAMPLE, Ensemble.MICROCANONICAL, ModelFamily.W, Constraint.EDGES_EVENTS)
    ] = ModelCapability(
        supported=True,
        requires_fit=False,
        backend="microcanonical_fixed_et",
        required_arguments=frozenset(
            {"node_count", "target_edges", "total_events", "layers"}
        ),
        optional_arguments=frozenset({"seed", "self_loops"}),
        supports_self_loops=True,
        supports_no_self_loops=True,
        result_kind="sampled_network",
    )
    # Microcanonical ME/B/W fixed (k,T): MCMC support + occupation allocator, no fit.
    for fam in [ModelFamily.ME, ModelFamily.B, ModelFamily.W]:
        req_args = {"degree_out", "degree_in", "total_events"}
        if fam is not ModelFamily.ME:
            req_args.add("layers")
        registry[
            (Verb.SAMPLE, Ensemble.MICROCANONICAL, fam, Constraint.DEGREE_EVENTS)
        ] = ModelCapability(
            supported=True,
            requires_fit=False,
            backend="microcanonical_fixed_kt",
            required_arguments=frozenset(req_args),
            optional_arguments=frozenset(
                {"seed", "self_loops", "burn_in_sweeps", "sweeps_per_sample"}
            ),
            supports_self_loops=True,
            supports_no_self_loops=True,
            result_kind="sampled_network",
        )

    # --- FILTER ---
    for family in _FAMILIES:
        for constraint in _CONSTRAINTS:
            key = (Verb.FILTER, Ensemble.GRAND_CANONICAL, family, constraint)
            registry[key] = ModelCapability(
                supported=True,
                requires_fit=True,
                backend="grandcanonical_filter",
                required_arguments=_FILTER_ARGS[constraint],
                optional_arguments=frozenset(
                    {"alpha", "tail", "correction", "detect_absent"}
                ),
                result_kind="filter_result",
            )

    return registry


REGISTRY: dict[tuple[Verb, Ensemble, ModelFamily, Constraint], ModelCapability] = (
    _build_registry()
)


def capability(
    verb: Verb,
    ensemble: Ensemble,
    family: ModelFamily,
    constraint: Constraint,
) -> ModelCapability | None:
    """Look up a route capability; `None` when the combination is unknown."""
    return REGISTRY.get((verb, ensemble, family, constraint))


def unsupported_cases() -> list[tuple[Verb, Ensemble, ModelFamily, Constraint]]:
    """Explicitly unsupported combinations that callers may attempt."""
    _supported_microcanonical: set[tuple[ModelFamily, Constraint]] = {
        (ModelFamily.ME, Constraint.STRENGTH),
        (ModelFamily.ME, Constraint.EDGES_EVENTS),
        (ModelFamily.B, Constraint.EDGES_EVENTS),
        (ModelFamily.W, Constraint.EDGES_EVENTS),
        (ModelFamily.ME, Constraint.DEGREE_EVENTS),
        (ModelFamily.B, Constraint.DEGREE_EVENTS),
        (ModelFamily.W, Constraint.DEGREE_EVENTS),
    }
    cases: list[tuple[Verb, Ensemble, ModelFamily, Constraint]] = []
    for family in _FAMILIES:
        for constraint in _CONSTRAINTS:
            if (family, constraint) not in _supported_microcanonical:
                cases.append((Verb.SAMPLE, Ensemble.MICROCANONICAL, family, constraint))
            # Canonical sampling only supports ME strength.
            if not (family is ModelFamily.ME and constraint is Constraint.STRENGTH):
                cases.append((Verb.SAMPLE, Ensemble.CANONICAL, family, constraint))
            # No canonical/microcanonical filtering.
            cases.append((Verb.FILTER, Ensemble.CANONICAL, family, constraint))
            cases.append((Verb.FILTER, Ensemble.MICROCANONICAL, family, constraint))
    return sorted(cases, key=lambda c: (c[0].value, c[1].value, c[2].value, c[3].value))


__all__ = [
    "REGISTRY",
    "ModelCapability",
    "SamplingExactness",
    "capability",
    "unsupported_cases",
]
