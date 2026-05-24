---
description: Historical API consistency audit, superseded by the ontology audit.
---

# 0006 API consistency audit

## TL;DR

This decision recorded an earlier ME/W/B API audit. Most result-shape issues
were fixed. The remaining mathematical and routing gaps are now tracked in
[Ontology conformance audit](../development/ontology-conformance-audit.md),
which should be treated as the active checklist.

## Resolved outcomes

| Area | Current state |
|---|---|
| Result dataclasses | Fit types carry `family`, `layers`, `self_loops`, convergence fields, and diagnostics. |
| Missing wrappers | Python exposes ME/B/W fitting wrappers for the main constraint families. |
| W absent filtering | W zero-inflated absent-edge wrappers are exposed. |
| Diagnostics | Shared `OptimizationDiagnostics` is used, with W metrics nested under the historical `conic` field. |

## Still active elsewhere

| Gap | Active tracking |
|---|---|
| B strength-edges and strength-degree wrappers call ME kernels | [Ontology audit](../development/ontology-conformance-audit.md) |
| Degree-events positive-weight parameters are not fully family-specific | [Ontology audit](../development/ontology-conformance-audit.md) |
| Missing unified Python router | [Ontology audit](../development/ontology-conformance-audit.md) |
| Partial B/W coordinate fitting runs too much logic in Python | [Ontology audit](../development/ontology-conformance-audit.md) |

## Decision

Do not add new action items to this historical audit. Update the ontology audit
when code or public APIs change.
