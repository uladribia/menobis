# 0004 Keep W conic fitting separate from ME and B balancing

## TL;DR

Use Clarabel/CVXRust only for W strength-family fits that need conic modeling.
Keep ME and B fits on IPF/balancing plus scalar root searches unless benchmarks
prove a specific replacement is better.

## Context

ODME now depends on Clarabel and CVXRust for W ensembles. This raises a design
question: should the same solver stack replace non-balancing parts of ME and B
models, such as fixed-cost `gamma` or fixed-edge `lambda` searches?

ME and B models have simpler multiplier structure than W models:

| Family | Typical fitting structure | Solver need |
|--------|---------------------------|-------------|
| ME Poisson | multiplicative IPF plus scalar search | balancing/root finding |
| B binomial | bounded multiplicative IPF | balancing/root finding |
| W geometric / NB | pole at `q_ij = 1` for strength constraints | conic solver |

## Decision

Use this solver boundary:

| Problem type | Preferred method |
|--------------|------------------|
| ME/B fixed strengths | IPF/balancing |
| ME/B strength-cost | IPF with bracketed scalar search for `gamma` |
| ME/B strength-edges | IPF with bracketed scalar search for `lambda` |
| W degree-events | scalar `q` solve plus Bernoulli IPF |
| W strengths, cost | conic formulation (requires sparse rewrite for N>25) |
| W strength-edges, strength-degree | monotone coordinate solver |

Conic solvers may be considered for ME/B only after a benchmark or convergence
regression shows that current methods fail on a documented workload.

## Scaling limits

Current practical limits as of 2026-05-20 benchmarks:

| Solver | Constraint | Max N (comfortable) | Max N (feasible) | Bottleneck |
|--------|-----------|---:|---:|---|
| Clarabel conic (sparse assembly) | W strengths, cost | 100 | 200 | O(N² cones) interior-point |
| Monotone coordinate | W edges, degree | 200 | 500 | O(N³) time |
| IPF balancing | ME/B all | 1000+ | 10000+ | O(N²) per iteration |

## Consequences

- W fitting can use lifted exponential-cone problems without making all fitting
  code depend on solver abstractions.
- ME/B code remains faster, simpler, and easier to audit scientifically.
- Scalar ME/B improvements should focus on bracketing, Brent/bisection, warm
  starts, and diagnostics before introducing conic models.
- Benchmarks must report wall time, residuals, convergence, and memory before
  changing this boundary.

## Validation

Keep separate test paths:

- W conic tests check solver status, residuals, `max_q < 1`, and lifted metrics.
- ME/B tests check IPF convergence, scalar residuals, and constraint recovery.
- Cross-family benchmarks should compare methods only when a replacement is
  proposed.
