---
description: MENoBiS testing strategy and commands.
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

## Canonical synthetic fixture

End-to-end tests and benchmarks must use `menobis.utilities.synthetic.generate_pa_geographic_network` unless they are pure math/API smoke tests. The fixture is deliberately not an MENoBiS null model; it supplies realistic constraints for fit/sample/filter workflows.

## Invariants

| Invariant | Where used |
|---|---|
| `sum(s_out) == sum(s_in)` | directed weighted networks |
| probabilities sum to one | pair distributions |
| weights are non-negative integers | generation |
| known partial pairs are preserved | partial fitting |
| residual constraints recover targets | fitting |
| seeded generation is reproducible | samplers |
| PA support edge count is controlled | synthetic fixtures |
| total events are controlled exactly | synthetic fixtures |

## Partial fitting tests

Partial tests use known weighted pairs. For degree and edge constraints,
occupation is inferred from positive known weights. Occupation-only partial mode
is intentionally outside current benchmark scope.

## Benchmark coverage test

`tests/test_menobis_benchmark_cases.py` ensures the fitting benchmark covers full
and partial ME/B/W combinations for the five constraint families.

## Warnings policy

Solver warnings should be actionable. Boundary cases may converge by residual
rather than by multiplier delta; tests should assert the scientifically relevant
residual behavior.

## Known solver limitations

| Model | Limitation |
|---|---|
| ME/W/Wnb **strength-degree** | Does NOT converge when any node has degree = capacity. Clip to `capacity - 1` or use degree-events. |
| ME/W/Wnb **strength-edges** | Rejects `target_edges ≥ capacity`. Use strength-only when all pairs are occupied. |
| B **strength-edges** / **strength-degree** | Known P5 bug: calls ME kernel. Do not use until fixed. |
| W **self_loops=False** realistic N≥50 | Newton solver may fail to converge; needs adaptive damping/projection. |
| B **strength self_loops=False** N≥200 | IPF is slow; needs convergence/performance investigation. |

Saturation peeling is implemented for B strength and all degree-events families
(Bernoulli IPF). The coupled strength-degree case requires a mixed-constraint
solver that is not yet implemented.
