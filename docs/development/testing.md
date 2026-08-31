---
description: MENoBiS testing strategy and commands.
---

# Testing

## TL;DR

Tests prioritize scientific invariants over legacy golden files. Run focused
checks first, then the full suite before merge.

## Fast vs heavy split

Tests marked `@pytest.mark.heavy` are slow E2E / benchmark-level (>2 s) and
are skipped by default.

```bash
uv run pytest                      # fast suite only
uv run pytest --run-heavy          # full suite (includes heavy)
uv run pytest -m heavy             # heavy only
```

## Test layers

- **Rust unit tests** — kernels, gradients, overflow safety, mask logic;
- **Rust oracle tests** — exact enumeration, legacy-comparison validation;
- **Python formula tests** — documented expectations (e.g. ME/B/W pair
  means, zero-inflated formulas) against code
  (`test_documentation_math_consistency.py`);
- **Python validation tests** — input rejection at the boundary;
- **Python E2E tests** — PA-geographic generate → fit → sample → check;
- **Python sampling tests** — reproducibility, non-negativity, preservation;
- **Python filtering tests** — false-positive rate under the null;
- **Python saturation tests** — degree-style saturation edge cases;
- **CLI and benchmark-CLI tests** — command behaviour and JSON output;
- **Documentation contract tests** — anti-drift checks on public pages
  (`test_public_docs_contract.py`) and executable doc workflows
  (`test_docs_examples.py`);
- **Docs build** — `uv run mkdocs build --strict` (links, nav, math,
  Mermaid, generated tables).

## Common commands

```bash
uv run pytest                      # fast Python suite
uv run pytest --run-heavy          # full Python suite
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run mkdocs build --strict
uv run python scripts/docs/generate_capabilities.py --check
cargo bench -p menobis-core        # criterion benchmarks
```

## Canonical synthetic fixture

End-to-end tests and benchmarks use
`menobis.utilities.synthetic.generate_pa_geographic_network`. The fixture
creates networks outside the MENoBiS null family, supplying realistic
constraints for fit/sample/filter workflows.

## Test fixture regimes

E2E tests use the **dense regime** by default to avoid pathological solver
behaviour in sparse (ill-posed degree constraints when \(k\approx s\)) and
saturated (\(k\approx N\)) settings:

| Regime | Fixture parameters | Character |
|---|---|---|
| sparse | `average_degree=4.0, events_per_edge=4.0` | moderate density, \(k < s\) |
| **dense (default)** | **`average_degree=N/5, events_per_edge=8.0`** | realistic unsaturated solvers |
| saturated | `average_degree=8.0, events_per_edge=5.0` | \(k\) near \(N-1\) |

These are **test-fixture** parameters. The benchmark presets are separate
and documented in [Practical scaling](../performance/scaling.md#sparsity-regimes).

## Known solver limitations

These are tracked with `xfail` markers in the fitting tests:

| Model | Limitation |
|---|---|
| W strength-edges | Newton solver does not converge with heterogeneous inputs |
| W strength-degree | Newton solver does not converge with heterogeneous inputs |
| W/B small-N saturation | small-\(N\) saturation not converging |

## Documentation contract

The public documentation must not drift from code:

- capability tables are generated from the registry and checked with
  `scripts/docs/generate_capabilities.py --check`;
- `test_public_docs_contract.py` denylists stale claims/patterns in public
  pages;
- `test_docs_examples.py` runs one deterministic end-to-end workflow per
  critical public route; a docs example change must update its test in the
  same commit.

See also the [contributor policy](extending-thesis-cases.md) for what a
change must ship with.