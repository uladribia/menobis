---
description: What MENoBiS actually supports today — the generated capability matrix by ensemble, family, constraint, and verb.
---

# Supported models

## TL;DR

This page answers: **what is actually supported today?** The matrix below is
generated from MENoBiS' capability registry. If the code and another
documentation page disagree, this table is authoritative for public support.

> **Version note:** capabilities on this page are generated from the current
> source tree. Regenerate with
> `uv run python scripts/docs/generate_capabilities.py`; the docs CI checks
> it with the `--check` flag.

--8<-- "_generated/capabilities.md"

## Reading the table

- **Fit** — a grand-canonical fit exists for the route (the canonical fit
  column follows the same registry).
- **Sample** — a sampler exists for the route.
- **Filter** — statistical filtering exists for the route.
- **Exactness / semantics** — what a sampled network guarantees:

| Label | Meaning |
|---|---|
| exact independent draws | each pair drawn directly from its fitted law; an exact *direct* sampler |
| exact direct | one draw from the target distribution, up to ordinary pseudorandom error |
| exact stationary MCMC | kernel with the correct stationary distribution; finite runs still need burn-in/mixing |
| constraints matched in expectation | soft fitted quantities that fluctuate (grand canonical) |
| strengths exact, cost expected | the hybrid microcanonical strength+cost route |

## Canonical support

Canonical fitting reuses the grand-canonical solver for ME. Canonical
**sampling** is implemented for ME + STRENGTH only (fixed total occupation
\(T\), multinomial kernel). See [Ensembles](../science/ensembles.md).

## Microcanonical routes

The dedicated route table below is also generated from the registry:

--8<-- "_generated/microcanonical-routes.md"

Every microcanonical route shares the two-stage philosophy documented in
[Microcanonical sampling](microcanonical.md): construct one feasible state,
then sample the target measure on the constraint fiber.