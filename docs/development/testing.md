---
description: Testing strategy and conventions for ODME.
---

# Testing

## TL;DR

ODME uses TDD red/green cycles with pytest (Python) and cargo test (Rust).
Property-based tests cover scientific invariants. All tests must pass before
merging.

## Running tests

```bash
# Rust
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings

# Python
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

## Test layers

| Layer | Purpose | Tools |
|-------|---------|-------|
| Unit | Individual functions and edge cases | pytest, cargo test |
| Integration | Full fit → generate → filter workflows | pytest |
| Property | Scientific invariants across random inputs | hypothesis, proptest |
| CLI smoke | Typer commands with CliRunner | pytest + typer.testing |
| Benchmark | Performance regression guards | pytest (time assertions) |

## Scientific invariants

Tests verify these properties across models:

| Invariant | Applies to |
|-----------|-----------|
| `sum(s_out) == sum(s_in) == T` | all directed graphs |
| Probability vectors sum to one | all probability tables |
| Generated weights are non-negative integers | all samplers |
| Self-loop removal is consistent | all models with `--no-self-loops` |
| Fixed-strength expectations recover strengths | Poisson models |
| Fixed-degree expectations recover degrees | degree models |
| Seeded generation is reproducible | all samplers |
| Filter partitions cover all edges | all filter functions |
| P-values are in [0, 1] | all filter functions |

## Tolerances

Floating-point comparisons use documented tolerances:

| Context | Tolerance |
|---------|-----------|
| IPF convergence | `1e-8` default |
| Ensemble mean recovery | `0.1 * sqrt(N)` typical |
| P-value bounds | exact [0, 1] |
| Strength balance | `sum(out) == sum(in)` exact for integers |

## File organization

One test file per module where practical:

```
tests/
├── test_odme_filtering_all_models.py
├── test_odme_cli_filter.py
├── test_odme_cli_convert.py
├── test_odme_generation.py
├── test_odme_fitting.py
├── test_odme_partial.py
└── ...
```
