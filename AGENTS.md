# AGENTS.md

Guidance for coding agents and contributors working on the ODME modernization.

## Mission

Replace the thesis-era codebase with **ODME**, a modern Rust + Python project that is fast, memory efficient, well tested, and thoroughly documented. Preserve the thesis-level scientific concepts and equations, but do not preserve old CLI behavior, old module layout, or backward compatibility.

Primary scientific reference: <https://hdl.handle.net/10803/400560>.

## Pi skills

The following pi skills must be used when working on ODME:

- **`/skill:commit`** — Use for all git commits. Follows Conventional Commits format. Never commit without using this skill.
- **`/skill:create-cli`** — Use when designing or modifying CLI commands, flags, output formats, or error handling. Ensures consistent UX patterns (stdout for data, stderr for progress, `--json`, `--quiet`, `--output`, long options preferred, exit codes).
- **`/skill:write-docs`** — Use when writing or updating documentation. Ensures AI-scannable structure: max 150 lines per file, one concept per file, TL;DR at top, tables for structured data, concrete copy-pasteable examples, consistent naming (`{noun}.md` for reference, `{verb}-{noun}.md` for how-to).

For documentation writing, follow the documentation policy below and use `mkdocs build --strict` to validate.

## Mandatory workflow

1. Work on a dedicated branch for every relevant rewrite.
2. Keep changes small and targeted.
3. Use TDD red/green:
   - write a failing test first;
   - confirm it fails for the expected reason;
   - implement the minimum fix;
   - confirm tests pass;
   - refactor only after green.
4. Use property-based tests where invariants are more robust than fixed examples.
5. Update Markdown documentation with every public API, CLI, model, or architectural change.
6. Document architectural/scientific decisions in `docs/decisions/`.
7. Do not delete or move legacy code during early scaffolding, but treat full replacement as the end state once the new implementation has its own tests.

## Branch naming

Use descriptive branches, for example:

- `refactor/scaffold-rust-python`
- `refactor/graph-core`
- `refactor/io-edge-list`
- `refactor/stat-strength-degree`
- `refactor/model-fixed-strength`
- `refactor/model-gravity`
- `refactor/python-cli-analyze`
- `docs/mkdocs-site`
- `test/thesis-invariants`

Never perform substantial rewrite work directly on `master` or `main`.

## Tooling requirements

### Python

Use:

- `uv` for package management and task execution.
- Protect dependency resolution with `UV_EXCLUDE_NEWER` set to at least seven days before the current date. The Makefile computes this automatically for project tasks to reduce supply-chain risk from newly published packages.
- `ruff` for linting and formatting.
- `ty` for type checking.
- `pytest` for tests.
- `hypothesis` for property-based tests.
- `typer` for command-line interfaces.
- `loguru` for logging.
- `mkdocs` and `mkdocs-material` for documentation.

### Rust

Use:

- `cargo` workspace under `crates/`.
- `pyo3` and `maturin` for Python bindings.
- `proptest` for property-based tests.
- `criterion` for benchmarks.
- `thiserror` for library errors.
- `rayon` where parallelism is justified and tested.

## Expected checks

As the scaffold matures, agents should run the smallest relevant checks first, then broader checks before handoff.

Target commands:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
mkdocs build --strict
```

If a command is not available yet because the scaffold has not been created, state that explicitly in the handoff.

## Python style

ODME should use a clean, modern, Pythonic style. Required conventions:

- `uv`-managed `pyproject.toml` with explicit dependency groups.
- Strict Ruff linting, Ruff formatting, and `ty` type checking.
- Google-style docstrings for modules, classes, functions, and methods.
- Typer CLI with a central `app`, subcommands in separate modules, `--version`, and `python -m odme` support.
- Tests organized clearly, with one test file per module where practical.
- `typer.testing.CliRunner` for CLI tests.
- MkDocs Material documentation under `docs/`.
- numpy arrays for data exchange and pyarrow for file I/O.
- `rustworkx` for efficient standard graph algorithms in Python.
- Makefile targets that call `uv run --frozen` for reproducible checks once the lockfile exists.

Additional ODME defaults:

- explicit names over abbreviations;
- typed public APIs;
- small modules and functions;
- pure functions for model math when practical;
- dataclasses or typed result objects for structured returns;
- thin Typer CLI wrappers over tested library code;
- no print-based logging in library code;
- no hidden global mutable state except logging configuration;
- English for code, docs, tests, and logs.

## Model implementation policy

ODME implements three weight-distribution families: ME (Poisson), B (Binomial),
and W (Geometric / Negative Binomial). These share mathematical structure but
have distinct expectation equations.

### Mandatory separation and reuse rules

1. **Every family must have its own solver implementation.** Never implement B
   or W by calling the ME solver and relabeling the output. Each family has a
   different `E[t_ij]` formula; the code must reflect that.
2. **Shared infrastructure must be factored into reusable Rust abstractions.**
   Cost providers, IPF balancing loops, gamma search, mask handling, excess
   computation, and rate-table assembly are shared across families and must live
   in common factory functions or traits.
3. **Constraint variants (strength, strength-cost, strength-edges,
   strength-degree) add Lagrange multipliers to the same family-specific
   kernel.** Implement them as compositions: family kernel + constraint layer.
4. **Partial implementations must reuse their parent (non-partial) solver.**
   A partial fit computes excess sequences and then calls the corresponding
   full-fit solver on the free pairs. Never duplicate the inner solver logic
   inside the partial path.
5. **Cost providers are orthogonal to families.** The same cost abstraction
   (no cost, sparse triples, Euclidean coordinates) must work with ME, B, and W
   without per-family cost wrappers.

### Mathematical verification checks

When implementing or modifying a fitting kernel:

- Write down the thesis equation being implemented in a code comment or
  docstring.
- Add a unit test that verifies the implemented formula against a hand-computed
  example (small N=3 or N=4).
- For each family, verify that `E[t_ij]` matches the thesis definition:
  - ME/Poisson: `E[t_ij] = x_i * y_j * f_ij`
  - B/Binomial(M): `E[t_ij] = M * p_ij / (1 - p_ij)` where `p_ij = x_i * y_j * f_ij / (1 + x_i * y_j * f_ij)`
  - W/Geometric: `E[t_ij] = q_ij / (1 - q_ij)` conditioned on occupation
  - W/NegBin(M): `E[t_ij] = M * q_ij / (1 - q_ij)` conditioned on occupation
- Confirm that constraint recovery tests pass: fitted expectations must
  reproduce the input constraint sequences within documented tolerance.
- Add a comparative test showing ME ≠ B ≠ W results on the same input when
  implementing a new constraint variant.

### Current violations (to be fixed)

- `fit_strength_cost_binomial_coordinates` calls ME Poisson and relabels.
- `fit_strength_cost_geometric_coordinates` calls ME Poisson and relabels.
- `fit_strength_cost_negative_binomial_coordinates` calls ME Poisson and relabels.
- All partial coordinate B/W wrappers delegate to Poisson partial and relabel.
- These produce identical numerical results across families, which is
  scientifically wrong.

## Logging policy

Python entry points must configure Loguru with:

- console logging;
- rotating file logging capped at 5 MB;
- configurable log path;
- useful context fields such as model case, input file, seed, and run id.

Example:

```python
from loguru import logger

logger.add(log_path, rotation="5 MB", retention="30 days", enqueue=True)
```

Rust code should return structured errors and diagnostics. Rust library code should not print to stdout/stderr except in intentionally designed CLI/debug paths.

## Documentation policy

All new public elements require Markdown documentation.

Update or create:

- `docs/getting-started.md` for user-facing workflows.
- `docs/concepts/*.md` for model explanations.
- `docs/api/*.md` for public APIs.
- `docs/cli/*.md` for CLI commands.
- `docs/decisions/*.md` for architectural decisions.
- `docs/development/*.md` for testing, benchmarking, and release workflows.

The documentation should map implementation names back to the thesis terminology whenever possible.

## Testing policy

Prefer tests in this order:

1. Scientific invariants.
2. Fixture-based tests from existing `tests/` data.
3. End-to-end workflow tests for modern ODME APIs and CLI commands.
4. CLI smoke tests.
5. Performance regression tests after correctness is established.

Useful invariants:

- `sum(s_out) == sum(s_in) == T` for directed graphs.
- Probability vectors sum to one within tolerance.
- Generated weights are non-negative integers.
- Self-loop removal behaves consistently.
- Fixed-strength expectations recover requested strengths within tolerance.
- Fixed-degree expectations recover requested degrees within tolerance.
- Seeded generation is reproducible.

Use approximate comparisons for floating-point values and document tolerances.

## Memory, performance, and graph-library policy

- Prefer streaming I/O.
- Use numpy arrays as the canonical data exchange format between Rust and Python.
- Network readers must accept non-negative integer weights, ignore zero-weight edges, and reject negative or non-integer weights at the boundary.
- Prefer sparse or edge-list representations over dense `N x N` matrices.
- Allocate dense matrices only when scientifically or algorithmically required.
- Use efficient existing graph libraries for standard algorithms; do not reimplement common graph routines.
- Use `rustworkx` as the only Python graph library dependency for standard graph algorithms.
- All computation-heavy code must be implemented in Rust (`odme-core`), not in Python.
- Python modules are thin wrappers: validate inputs, call Rust via `_odme`, wrap results in numpy arrays and typed dataclasses.
- Never implement loops, numerical kernels, or graph algorithms in Python when they can run in Rust.
- Use rustworkx only when its algorithm is not already implemented in `odme-core`. Prefer Rust-native implementations for ODME-specific statistics.
- Always prefer rustworkx implementations over custom code for standard graph-theoretic computations that ODME does not already implement in Rust. Before implementing any graph algorithm, check if rustworkx or `odme-core` already provides it.
- Keep custom Rust graph kernels only for ODME-specific flow, fitting, and sampling work.
- Benchmark before optimizing.
- Record performance-sensitive design choices in `docs/decisions/`.

## Legacy replacement policy

- Treat the existing C/Python code as reference material, not as a compatibility target.
- Do not reformat or churn legacy code unnecessarily during early branches.
- Do not preserve historical CLI flags or module names unless they are still the best modern design.
- Prefer thesis equations, hand-checked examples, and property tests over golden-file compatibility.
- Fully replace the legacy code once ODME covers the selected scientific scope.
- When behavior differs from the old code, document whether the difference is intentional, numerical, stochastic, or a bug fix.

## Handoff checklist

At the end of each agent session, report:

- current branch;
- files changed;
- tests/checks run;
- tests/checks not run and why;
- next recommended red/green step.
