---
description: MENoBiS testing strategy and commands.
---

# Testing

## TL;DR

Tests prioritize scientific invariants over legacy golden files. Run focused
checks first, then the full suite before merge.

## Fast vs heavy split

Tests marked `@pytest.mark.heavy` are slow E2E / benchmark-level (>2 s).

The fast suite runs by default and completes in ≈11 s.

```bash
uv run pytest                      # fast (374 tests, ≈11 s)
uv run pytest --run-heavy          # full  (389 tests, ≈180 s)
uv run pytest -m heavy             # heavy only (11 tests)
```

## Test layers

| Layer | Count | Purpose |
|---|---|---|
| Rust unit tests | 286 | Kernels, gradients, overflow safety, mask logic |
| Rust oracle tests | 40 | Exact enumeration, legacy-comparison validation |
| Python formula tests | ≈30 | Verify E[t], E[Θ] match thesis equations |
| Python validation tests | ≈15 | Input rejection at the boundary |
| Python E2E tests | ≈40 | PA-geographic generate → fit → sample → check |
| Python sampling tests | ≈90 | Reproducibility, non-negativity, preservation |
| Python filtering tests | ≈60 | FPR under null model |
| Python saturation tests | ≈5 | Degree saturation edge cases |
| CLI tests | ≈30 | Command behavior and JSON output |
| Benchmark CLI tests | ≈5 | Smoke test the benchmark harness |
| Docs build | — | Links, nav, API pages |

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
uv run pytest                      # fast Python suite (≈11 s)
uv run pytest --run-heavy          # full Python suite (≈180 s)
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
uv run mkdocs build --strict
cargo bench -p menobis-core        # criterion benchmarks
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
| **Dense (default)** | **`average_degree=N/5, events_per_edge=8.0`** | **Optimal — exercises solvers realistically** |
| Saturated | `average_degree=15.0, events_per_edge=5.0` | k near N-1 |

All E2E tests use the **dense regime** by default to avoid pathological solver
behaviour seen in sparse (ill-posed degree constraints when `k ≈ s`) and
saturated (`k ≈ N`) regimes.

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
