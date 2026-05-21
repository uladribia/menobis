---
description: ODME testing strategy and commands.
---

# Testing

## TL;DR

Tests prioritize scientific invariants over legacy golden files. Run focused
checks first, then the full suite before release.

## Test layers

| Layer | Purpose |
|---|---|
| Rust unit tests | kernels, distributions, fitting invariants |
| Python unit tests | public wrappers, dataclasses, validation |
| CLI tests | command behavior and JSON output |
| Benchmark tests | performance guardrails and case coverage |
| Docs build | links, nav, API pages |

## Common commands

```bash
uv run pytest
cargo test --workspace
uv run ruff check .
uv run ruff format --check .
uv run ty check
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
uv run mkdocs build --strict
```

## Invariants

| Invariant | Where used |
|---|---|
| `sum(s_out) == sum(s_in)` | directed weighted networks |
| probabilities sum to one | pair distributions |
| weights are non-negative integers | generation |
| known partial pairs are preserved | partial fitting |
| residual constraints recover targets | fitting |
| seeded generation is reproducible | samplers |

## Partial fitting tests

Partial tests use known weighted pairs. For degree and edge constraints,
occupation is inferred from positive known weights. Occupation-only partial mode
is intentionally outside current benchmark scope.

## Benchmark coverage test

`tests/test_odme_benchmark_cases.py` ensures the fitting benchmark covers full
and partial ME/B/W combinations for the five constraint families.

## Warnings policy

Solver warnings should be actionable. Boundary cases may converge by residual
rather than by multiplier delta; tests should assert the scientifically relevant
residual behavior.
