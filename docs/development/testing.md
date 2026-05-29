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
| Rust unit tests (54 tests) | Kernels, gradients, overflow safety, mask logic |
| Python formula tests | Verify E[t], E[Theta] match thesis equations |
| Python validation tests | Input rejection at the boundary |
| Python E2E tests | PA-geographic generate -> fit -> sample -> check |
| Python sampling tests | Reproducibility, non-negativity, preservation |
| Python filtering tests | FPR under null model |
| Python saturation tests | Degree saturation edge cases |
| CLI tests | Command behavior and JSON output |
| Benchmark CLI tests | Smoke test the benchmark harness |
| Docs build | Links, nav, API pages |

## Test files (fitting/solver related)

| File | N | What it tests |
|---|---|---|
| `test_fitting_equations.py` | 5 | Pure formula verification (ME, B, W) |
| `test_fitting_validation.py` | 2-5 | Input rejection across families |
| `test_fitting_e2e.py` | 20 | Full pipeline: 12 combos + partial + regimes |
| `test_fitting_saturation.py` | 3 | Degree saturation multiplier clamping |
| `test_sampling.py` | 20 | Seeded reproducibility, structure checks |
| `test_filtering_e2e.py` | 20 | Null-model FPR bounds |

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

End-to-end tests and benchmarks use
`menobis.utilities.synthetic.generate_pa_geographic_network`. The fixture creates
networks outside the MENoBiS null family, supplying realistic constraints for
fit/sample/filter workflows.

Tests use seed `54320` which is known to produce convergent fits for ME and B
across all constraint types at N=20.

## Regimes tested

| Regime | Parameters | Character |
|---|---|---|
| Sparse | `average_degree=5.0, events_per_edge=4.0` | Moderate connectivity |
| Saturated | `average_degree=15.0, events_per_edge=5.0` | k near N-1 |

## Known solver limitations (xfail in tests)

| Model | Limitation |
|---|---|
| W strength-edges | Newton solver does not converge with heterogeneous inputs |
| W strength-degree | Newton solver does not converge with heterogeneous inputs |
| W/B saturation N=3 | Small-N saturation not converging |

## Partial fitting coverage

| Constraint | ME | B | W |
|---|---|---|---|
| strength | full | full | full |
| strength-cost (coord) | full | full | full |
| strength-edges | full | full | full |
| strength-degree | full | full | full |
