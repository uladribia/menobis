---
description: How to contribute and extend MENoBiS with new constraints.
---

# Contributing and extending

## TL;DR

Keep public APIs explicit, reuse shared model concepts, and put heavy numerical
work in Rust. Partial fitting is not a separate model family: it is full fitting
on a support with fixed known weighted-pair contributions.

## Code philosophy

| Principle | Practice |
|---|---|
| scientific names | use strength, degree, events, cost |
| one ontology | reuse `WeightFamily` and constraint concepts |
| thin Python | validate, call Rust, wrap typed results |
| Rust kernels | fitting, sampling, filtering loops |
| sparse first | edge lists and support triples |
| tested invariants | residuals and conservation laws |

## Module map

| Path | Role |
|---|---|
| `crates/menobis-core/src/distribution.rs` | pair distributions |
| `crates/menobis-core/src/pairs.rs` | provider abstractions |
| `crates/menobis-core/src/fitting/` | fitting solvers |
| `crates/menobis-core/src/generation.rs` | sampling |
| `crates/menobis-core/src/filter.rs` | p-values and filters |
| `crates/menobis-python/src/lib.rs` | PyO3 bridge |
| `src/menobis/analysis/` | Python analysis and ensemble helpers |
| `src/menobis/models/` | Python fitting, generation, and partial wrappers |
| `src/menobis/filtering/` | Python filtering wrappers and result types |
| `src/menobis/utilities/` | Synthetic fixtures and logging configuration |
| `benchmarks/` | Repository benchmark CLI, not installed as `menobis` |

## Add a new linear constraint

1. Define the sufficient statistic and Lagrange multiplier.
2. Add a Rust residual helper for expected statistic recovery.
3. Add a fitting routine or extend an existing support-aware routine.
4. Add a pair provider when generation/filtering need the constraint.
5. Add PyO3 wrapper and Python typed result fields.
6. Add tests: homogeneous fit, residual recovery, invalid inputs.
7. Add benchmark registry coverage.
8. Document thesis terminology and equations.

## Add a new binary constraint

Binary constraints affect occupation probabilities. Implement them through the
zero-inflated/provider path when possible:

1. Express occupation `pi_ij` and expected weight `mu_ij`.
2. Reuse Bernoulli balancing for pure degree constraints when possible.
3. For strength + binary constraints, add coordinate/root updates with residual
   checks.
4. Ensure absent-edge filtering uses the same occupation probability.
5. Add seeded generation and p-value tests.

## Partial support

For a new constraint, partial support should reuse the full solver concept:

```text
full fit = family + constraint + full support
partial fit = family + constraint + known-weight support
```

Known weights are subtracted from linear statistics and removed from free pair
support. Positive known weights also contribute occupation for edge/degree
constraints.

## Commit workflow

Use conventional commits and keep changes small:

```bash
git switch -c refactor/my-change
uv run pytest tests/my_focus.py -q
git add ...
git commit -m "feat(scope): concise summary"
```

Before handoff, report branch, changed files, checks run, checks not run, and
next red/green step.
