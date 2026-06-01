---
description: Solver reliability, convergence regimes, and scaling guidance.
---

# Solvers and scaling

## TL;DR

Edge-list analysis is O(E). Independent-pair fitting with binary or cost
constraints is usually O(N² × iterations). MENoBiS streams generation and
filtering to avoid dense rate matrices.

!!! tip "Large N is memory-feasible"
    The hard part is usually **time**, not memory. Public workflows keep
    multipliers, coordinates, and masks sparse or O(N); generation and filtering
    stream pair distributions instead of storing dense `N x N` rate tables.

## Practical scaling

| Operation | Typical cost | Memory note |
|---|---:|---|
| Read/write edge list | O(E) | sparse edge table |
| MENoBiS core statistics | O(E) | single pass |
| ME strength fit | O(N) to cheap IPF | multiplier arrays |
| B/W strength fit | O(N² × iterations) | streamed all-pairs sweeps |
| strength-cost fit | O(N² × iterations) | coordinates are O(N) |
| strength-edges fit | O(N² × iterations) | zero-inflated |
| strength-degree fit | O(N² × iterations) | often the slowest common fit |
| generation | streamed over candidate pairs | output is sparse sampled edges |
| filtering | O(E) or O(N²) with absent scan | no dense rate matrix |

## Regimes

| Regime | Meaning | Solver behaviour |
|---|---|---|
| sparse | degree close to strength; many occupations equal 1 | zero-inflated models can be ill-conditioned |
| dense | moderate support and positive occupations | recommended for tests and benchmarks |
| saturated | degrees near pair capacity | boundary handling required |

## Current reliability summary

| Case | Status for practical use |
|---|---|
| ME strength, strength-cost | reliable and fast |
| ME strength-edges, strength-degree | reliable in dense regimes; sparse can be hard |
| B strength and cost | reliable with feasible `layers` |
| B zero-inflated | generally usable; sparse/saturated regimes need care |
| W strength | usable; check `q<1` diagnostics through fit status |
| W strength-cost | slower and sensitive at larger N |
| W zero-inflated | experimental; inspect convergence before scientific use |

## Memory rules

- Use `EdgeTable` sparse arrays for observed and sampled networks.
- Do not create dense `N x N` cost, rate, or probability matrices.
- Use projected coordinates for costs so Rust can compute pair distances on
  demand.
- Treat absent-edge filtering as an all-pairs scan unless using sparse custom
  support.

## Diagnostics to inspect

Every fit exposes at least:

| Field | Meaning |
|---|---|
| `converged` | solver reached the requested criterion |
| `status` | solver status string |
| `iterations` | number of iterations |
| residual fields | constraint mismatch when available |

For exploratory runs, prefer a moderate tolerance such as `1e-6` before asking
for stricter recovery.
