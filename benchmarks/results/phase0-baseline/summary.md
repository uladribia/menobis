# Phase 0 baseline benchmarks

Captured before any production code changes, per [microcanonical-phase-0.md §18.1](../../docs/development/agent-specifications/microcanonical-phase-0.md).

## Metadata

| Field | Value |
|---|---|
| Git commit | `c2a2a39cd989e5770914e7446613c1b87ebffdba` |
| Branch | `refactor/phase-0-foundation` |
| Python | 3.13.12 |
| Rust | 1.95.0 |
| OS | Linux 7.0.0-28-generic x86_64 |
| Node count | 50 |
| Families | ME, B, W |
| Constraints | strength, strength-cost, strength-edges, strength-degree |
| Known-pair fractions | 0.0, 0.02 |
| Self-loop policies | both |

## Self-loops (`--self-loops`)

- **111 rows** | **59 converged** | **0 errors** | **wall=112.5s**
- File: `self-loops.json`

## No self-loops (`--no-self-loops`)

- **111 rows** | **60 converged** | **0 errors** | **wall=76.9s**
- File: `no-self-loops.json`

## Known behaviour reproduced (matches §18.2)

| Case | Expected | Observed |
|---|---|---|
| ME strength (all regimes) | reliable and fast | all converged, sub-second |
| ME strength-cost | reliable and fast | all converged |
| ME strength-edges | reliable baseline | all converged |
| ME strength-degree | reliable baseline | all converged |
| B strength | reliable with feasible layers | all converged |
| B strength-cost | generally reliable | all converged |
| B strength-edges | generally usable | all converged |
| B strength-degree | generally usable | all converged (self-loops sparse 0% hit max-iter) |
| W strength | usable | all converged |
| W strength-cost | slower and sensitive | all converged, slower ~1-2s |
| W strength-edges | experimental | all non-convergent (max-iter) |
| W strength-degree | experimental | all non-convergent (max-iter) |