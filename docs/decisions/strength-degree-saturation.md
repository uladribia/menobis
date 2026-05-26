# Strength-degree saturation handling

TL;DR: ME/W/Wnb strength-degree solvers now keep degree-saturated occupation multipliers fixed at a large value while continuing to fit strength multipliers.

## Context

In coupled strength-degree models, a node with degree equal to its candidate-pair capacity must occupy every admissible pair. The binary channel saturates, but the weight channel still has to fit node strengths.

## Decision

For ME, geometric W, and negative-binomial W strength-degree fitting:

| Saturated constraint | Solver behavior |
|---|---|
| `k_out[i] = capacity` | Fix `z_i` to a large multiplier |
| `k_in[j] = capacity` | Fix `w_j` to a large multiplier |
| Strength sequence | Continue updating `x_i` and `y_j` |
| Non-saturated degrees | Continue coordinate updates |

This represents the limiting zero-inflated equations where occupation tends to one while positive-support weights remain family-specific.

## Consequences

- Boundary feasible cases no longer fail solely because a degree reaches capacity.
- The fitted multipliers can be very large for saturated nodes.
- The implementation preserves separate ME/W/Wnb expectation formulas.

## Verification

`tests/test_menobis_strength_degree_saturation.py` covers ME, W, and Wnb saturated degree cases.
