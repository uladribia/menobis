# AGENTS.md

Guidance for coding agents and contributors working on the ODME modernization.

## Mission

Replace the thesis-era codebase with **ODME**, a modern Rust + Python project that is fast, memory efficient, well tested, and thoroughly documented. Preserve the thesis-level scientific concepts and equations, but do not preserve old CLI behavior, old module layout, or backward compatibility.

**No backward compatibility is required.** ODME is an experimental package undergoing a full rewrite. Agents must never introduce shims, re-export facades, deprecated aliases, or compatibility layers. When code moves, all call sites update in the same change. When APIs change shape, old signatures are deleted, not preserved.

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
- Rust-native kernels for ODME graph statistics; external graph libraries such as NetworkX or rustworkx must remain optional user-side adapters, not runtime dependencies.
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
have distinct expectation equations. As such, they should share the maximum number of common abstractions via factory methods.

### Mandatory separation and reuse rules

1. **Every family must have its own solver implementation.** Never implement B
   or W by calling the ME solver and relabeling the output. Each family has a
   different `E[t_ij]` formula; the code must reflect that. Also the public PYTHON API should call separate method per family (specially for fitting).
2. **Unified entry point in python**: A single function should route the choice ensemble (grand canonical, canonical (ME only), microcanonical (ME with strengths only)), plus type ME/W/B, plus constraint to the different public API python endpoints that defer to rust for each case. Failing with custom errors if the choice is not supported. 
3. **Shared infrastructure must be factored into reusable Rust abstractions.**
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
  example (small N=5 or N=10).
- The formulation follows a clear ontology:
  - There are three cases that induce different statistics: ME, B and W.
  - B and W cases only implement the "grand canonical" ensemble, for which all node-pair `ij` statistics are independent.
  - ME implements additionally the "canonical" ensemble, based on multinomial statistics, and, exceptionally, the "microcanonical ensemble" only for fixed strength via stub matching.
  - All "grand canonical" ensembles contemplate two kinds of constraints, some of which are implemented in ODME (others can be added in the future)
    - Those that depend linearly on the occupation number $E[t_ij]$, like strength sequence, strength plus average cost. In this case, the statistics are independent.
    - Those that additionally depend on the occupation probability $E[\Theta(t_ij>0)] $, like fixed total binary edges E, fixed degree sequence. In this case, the statistics are always Zero Inflated.
  - All cases contemplate different constraints at three levels. When a constraint reaches saturation level (like total degree=N or N-1), it must be deducted from the problem.
    - Global level: Total events (always), total binary edges, total cost.
    - Node level: Strength sequence, degree sequence.
    - Node pair level: Some node pair statistics might be frozen (so the network is partially frozen).
- For the "grand canonical" ensemble, the mapping family to statistics are as follows:
  - ME maps to Poisson and Zero Inflated Poisson.
  - B maps to Binomial on M layers and zero inflated Binomial. The special case M=1 is a Bernoulli (binary network). This special case has total occupation per node pair bounded by M layers.
  - W maps on M layers to Negative Binomial and zero inflated Negative Binomial. The special case M=1 is geometric.
- Each grand-canonical family depends on `q_ij = x_i y_j f_ij`, where `f_ij` only applies when cost constraints are considered. Zero-inflated constraints additionally use a binary multiplier `l_ij` (`l_ij = w_i z_j` for degree constraints, or a scalar for global edge constraints). The zero-inflated occupation is not `l_ij / (1 + l_ij)` unless `l_ij` has already absorbed the positive-support partition factor.
  - Non zero inflated:
    - ME/Poisson: `E[t_ij] = q_ij \in (0,\infty)`.
    - B/Binomial(M): `E[t_ij] = M q_ij / (1+q_ij)` with `q_ij \in (0,\infty)`.
    - W/NegBin(M): `E[t_ij] = M q_ij / (1 - q_ij)` with `q_ij \in (0,1)`.
  - Zero inflated:
    - Define the positive-support partition factors `G_ME(q)=exp(q)-1`, `G_B(q)=(1+q)^M-1`, and `G_W(q)=(1-q)^(-M)-1`.
    - Binary occupation is `E[\Theta(t_ij>0)] = l_ij G_F(q_ij) / (1 + l_ij G_F(q_ij))`.
    - Expected weight is `E[t_ij] = l_ij q_ij G'_F(q_ij) / (1 + l_ij G_F(q_ij))`.
    - Equivalently, `E[t_ij | t_ij > 0] = q_ij G'_F(q_ij) / G_F(q_ij)`.
    - ME/Poisson: `E[t_ij | t_ij > 0] = q_ij exp(q_ij)/(exp(q_ij)-1) = q_ij/(1-exp(-q_ij))` with `q_ij, l_ij \in (0,\infty)`.
    - B/Binomial(M): `E[t_ij | t_ij > 0] = M q_ij (1+q_ij)^(M-1) / ((1+q_ij)^M - 1)` with `q_ij, l_ij \in (0,\infty)`.
    - W/NegBin(M): `E[t_ij | t_ij > 0] = M q_ij (1-q_ij)^(-M-1) / ((1-q_ij)^(-M) - 1)` with `l_ij \in (0,\infty)` and `q_ij \in (0,1)`.
- Confirm that constraint recovery tests pass: fitted expectations must
  reproduce the input constraint sequences within documented tolerance.
- Add a comparative test showing ME ≠ B ≠ W results on the same input when
  implementing a new constraint variant.

### Current violations (to be fixed)

- W Newton solver does not converge with `self_loops=False` at N≥50 for
  realistic gravity-model inputs. Needs adaptive damping or better
  feasibility projection.
- B fixed-strength `self_loops=False` is very slow at N≥200 (28s).
  The IPF convergence needs investigation.

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

### Mandatory E2E pipeline

All fitting, generation, and filtering tests MUST follow this pipeline:

1. **Generate** a realistic weighted directed network with projected XY
   coordinates using a gravity-like model (present in the synthethic module).
2. **Derive** constraints from the generated network: strengths, degrees,
   edge counts, total cost. These are guaranteed feasible because they come
   from an actual network.
3. **Fit** the model using derived constraints.
4. **Sample** from the fitted model.
5. **Verify** that the sampled network recovers the original constraints
   within documented stochastic tolerance.

Tests that skip steps 1–2 and use arbitrary hand-picked values are permitted
ONLY for:
- Pure mathematical function unit tests (e.g., verifying formulas).
- API contract tests (return type, field names, error messages).
- CLI smoke tests (exit codes, output format).

Never write a fitting test with arbitrary `s_out = [10, 20, 30]` unless you
first prove those values are feasible for the given model and constraint type.

### Test ordering

Prefer tests in this order:

1. E2E pipeline tests (generate → fit → sample → check).
2. Scientific invariants (sum conservation, non-negativity, reproducibility).
3. API contract tests.
4. CLI smoke tests.
5. Performance regression tests.

### Tolerances

Use **relative** tolerances for constraint recovery:
- Fitting tolerance: `0.01 * max(constraint_sequence)`
- Sampling check: allow 5× the fitting tolerance (stochastic noise)
- W models may need looser tolerance: `max(rel_tol, 1.0)`

Document all tolerance choices in test docstrings.

## Memory, performance, and graph-library policy

- Prefer streaming I/O.
- Use numpy arrays as the canonical data exchange format between Rust and Python.
- Network readers must accept non-negative integer weights, ignore zero-weight edges, and reject negative or non-integer weights at the boundary.
- Never use dense `N x N` matrices, unless it is absolutely unavoidable.
- All computation-heavy code must be implemented in Rust (`odme-core`), not in Python.
- Python modules are thin wrappers: validate inputs, call Rust via `_odme`, wrap results in numpy arrays and typed dataclasses.
- Never implement loops, numerical kernels, or graph algorithms in Python when they can run in Rust.
- Use Rust-native implementations for ODME-specific statistics.
- Before implementing any graph algorithm, check if `odme-core` already provides it.
- Benchmark before optimizing.
- Record performance-sensitive design choices in `docs/decisions/`.

## Legacy replacement policy

- Treat the existing C/Python code as reference material, not as a compatibility target.
- Do not reformat or churn legacy code unnecessarily during early branches.
- Do not preserve historical CLI flags or module names unless they are still the best modern design.
- Prefer thesis equations, hand-checked examples, and property tests over golden-file compatibility.
- Fully replace the legacy code once ODME covers the selected scientific scope.
- When behavior differs from the old code, document whether the difference is intentional, numerical, stochastic, or a bug fix.
- **Never add backward-compatibility shims, re-export wrappers, or deprecation aliases.** Move code cleanly and update all call sites in the same commit. Old paths are simply deleted.

## Handoff checklist

At the end of each agent session, report:

- current branch;
- files changed;
- tests/checks run;
- tests/checks not run and why;
- next recommended red/green step.
