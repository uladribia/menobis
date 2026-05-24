# 0004 Keep W fitting separate from ME and B balancing

## TL;DR

W fitting must remain family-specific because its kernel has the
$q_{ij}<1$ boundary and mean $M q/(1-q)$. Earlier docs described a Clarabel
conic implementation; the current code uses Newton/coordinate-style solvers but
the separation decision still stands.

## Context

ME, B, and W share support masks, cost providers, residual checks, and scalar
search patterns. They do not share the same expected-weight equation:

| Family | Mean |
|---|---|
| ME | $q$ |
| B | $M q/(1+q)$ |
| W | $M q/(1-q)$ |

The W pole at `q=1` makes direct reuse of ME or B balancing invalid.

## Decision

Use this solver boundary:

| Problem type | Preferred method |
|---|---|
| ME fixed strengths | analytic/IPF balancing |
| B fixed strengths | B-specific IPF with saturation handling |
| ME/B strength-cost | family-specific IPF plus gamma search |
| ME/B strength-edges | family-specific zero-inflated solver plus edge residual |
| ME/B strength-degree | family-specific zero-inflated solver plus degree residual |
| W strengths/cost | W-specific Newton or future convex solver |
| W strength-edges/degree | W-specific zero-inflated coordinate/root solver |
| degree-events | Bernoulli occupation plus family-specific positive-weight parameter |

Conic solvers remain a possible future implementation detail, not part of the
public ontology.

## Consequences

- B and W cannot call ME solvers and relabel outputs.
- Solver diagnostics should be comparable across families even if internal
  methods differ.
- Benchmarks must report residuals, convergence, wall time, and memory before
  changing a solver family.

## Validation

Keep separate test paths for ME, B, and W. Cross-family tests should confirm
that fitted expectations differ on the same feasible input when formulas differ.
