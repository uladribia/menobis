# ODME Modernization Plan

This document defines the refactor plan for replacing the thesis-era codebase with **ODME**, a modern Rust + Python library focused on speed, memory efficiency, reproducibility, and maintainability.

The scientific reference for the project is the doctoral thesis available at <https://hdl.handle.net/10803/400560>, together with the papers cited in the existing READMEs.

## Goals

- Build a clean new package named **ODME**.
- Preserve the scientific concepts and equations from the thesis, not the old module boundaries or CLI behavior.
- Expose stable public endpoints in both Rust and Python.
- Rewrite performance-critical fitting, sampling, and model kernels in Rust.
- Use Python for ergonomic APIs, CLIs, documentation workflows, and data-science integration.
- Use numpy arrays as the canonical array interchange format.
- Support major weighted network formats with non-negative integer weights, ignoring zero-weight edges.
- Reuse efficient graph libraries for standard graph algorithms instead of reimplementing them.
- Improve memory efficiency by avoiding dense `N x N` matrices unless explicitly required.
- Make all changes in small branches with TDD red/green cycles.
- Use property-based testing where mathematical invariants are clearer than example fixtures.
- Produce thorough Markdown documentation and a static documentation site.
- Record architectural and scientific decisions explicitly.

## Non-goals

- Preserving the historical command-line interface.
- Preserving the old folder/module layout.
- Maintaining backward compatibility for consumers; there are no current external users.
- Reimplementing standard graph algorithms already available in efficient libraries.
- Optimizing before correctness is proven with tests and benchmarks.
- Changing thesis model definitions without documented scientific justification.

## Current package summary

The original repository contains three modules:

1. `1. Network analysis/`: C tool `MultiEdgeAnalyzer` for weighted network statistics.
2. `2. Model Fitting/`: Python 2 package `multi_edge_fitter` for Lagrange multiplier fitting.
3. `3. Model Generation/`: C tool `GenNetGen` for null-model network generation.

The refactor should replace these with a single coherent ODME project. The legacy code can be used as scientific reference and for temporary exploratory comparison, but it is not a compatibility target.

## Proposed project scaffold

```text
.
├── AGENTS.md
├── PLAN.md
├── README.md
├── CHANGELOG.md
├── LICENSE
├── pyproject.toml
├── Cargo.toml
├── uv.lock
├── mkdocs.yml
├── docs/
│   ├── index.md
│   ├── thesis-context.md
│   ├── getting-started.md
│   ├── concepts/
│   │   ├── multi-edge-networks.md
│   │   ├── maximum-entropy-models.md
│   │   ├── spatial-costs.md
│   │   └── ensembles.md
│   ├── api/
│   │   ├── python.md
│   │   └── rust.md
│   ├── cli/
│   │   ├── analyze.md
│   │   ├── fit.md
│   │   ├── generate.md
│   │   └── convert.md
│   ├── decisions/
│   │   └── 0001-rust-python-architecture.md
│   └── development/
│       ├── testing.md
│       ├── benchmarking.md
│       └── release-process.md
├── crates/
│   ├── odme-core/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── graph.rs
│   │       ├── io.rs
│   │       ├── stats.rs
│   │       ├── distances.rs
│   │       ├── fitting.rs
│   │       ├── generation.rs
│   │       ├── models/
│   │       │   ├── mod.rs
│   │       │   ├── fixed_strength.rs
│   │       │   ├── fixed_degree.rs
│   │       │   ├── fixed_strength_degree.rs
│   │       │   ├── cost.rs
│   │       │   ├── radiation.rs
│   │       │   └── custom_pij.rs
│   │       └── errors.rs
│   ├── odme-python/
│   │   ├── Cargo.toml
│   │   └── src/lib.rs
│   └── odme-cli-rs/                 # optional later: pure Rust CLI helpers
│       ├── Cargo.toml
│       └── src/main.rs
├── src/
│   └── odme/                         # distribution name: ODME; import name: odme
│       ├── __init__.py
│       ├── __main__.py
│       ├── py.typed
│       ├── logging.py
│       ├── analysis/
│       │   ├── __init__.py
│       │   └── summary.py
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   ├── analyze.py
│       │   ├── fit.py
│       │   └── generate.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── frames.py                # EdgeTable dataclass and numpy-based normalization
│       │   └── io.py
│       ├── interop/
│       │   ├── __init__.py
│       │   └── rustworkx.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── fixed_strength.py
│       └── config.py
├── tests/
│   ├── fixtures/                    # migrated from existing tests where scientifically useful
│   ├── test_odme_*.py               # Python tests colocated by feature
│   ├── rust/
│   ├── integration/
│   └── thesis_cases/                # compact examples tied to documented model equations
├── benches/
│   ├── rust/
│   └── python/
├── scripts/
│   ├── extract-thesis-fixtures.py
│   └── benchmark-legacy-reference.py # temporary, removed with legacy code
└── legacy/                          # temporary only, removed when replacement is complete
    ├── network-analysis/
    ├── model-fitting/
    └── model-generation/
```

The historical folders should remain untouched during early planning/scaffolding branches. Once ODME has tests for the scientific behavior it needs, a later branch should remove or archive the old code so the repository contains only the new implementation.

## Main technology choices

### Rust

- `pyo3` + `maturin`: Python bindings and wheel builds.
- `ndarray`: dense numerical arrays when needed.
- `sprs`: sparse matrices for cost/probability/network representations.
- `petgraph`: standard Rust graph algorithms when they fit the data model.
- Custom CSR/edge-list types only for ODME-specific flow/network kernels where external graph libraries do not provide the needed memory layout or multi-edge semantics.
- `rayon`: safe data parallelism.
- `rand`, `rand_distr`, `statrs`: random sampling and distributions.
- `thiserror`, `anyhow`: typed library errors and application-level errors.
- `serde`, `csv`: structured configuration and input/output.
- `approx`: floating-point comparisons in tests.
- `proptest`: Rust property-based testing.
- `criterion`: Rust benchmarking.
- `tracing` internally is acceptable, but Python-facing logging should use Loguru.

### Python

- `uv`: package management, virtualenvs, lockfile, task execution, with `UV_EXCLUDE_NEWER` set to at least seven days before the current date for dependency resolution.
- `ty`: type checking.
- `ruff`: linting and formatting.
- `typer`: CLI implementation.
- `loguru`: logging, including rotating file logs with a maximum file size of 5 MB.
- `pytest`: unit and integration tests.
- `hypothesis`: property-based tests.
- Removed: Polars was dropped in favour of numpy + pyarrow for lighter dependencies.
- `pyarrow`: Parquet and Arrow IPC file I/O.
- `numpy`: Python array interoperability, especially for PyO3 bindings and scientific users.
- `rustworkx`: the only Python graph library dependency, used for efficient standard graph algorithms.
- `mkdocs` + `mkdocs-material`: static documentation site.
- `mkdocstrings[python]`: API documentation.

### Python graph-library recommendation

Use **rustworkx** as the only Python-side graph library dependency. It is Rust-backed, fast, pip-installable, supports directed weighted graphs and multigraph-style use cases, and aligns well with ODME's Rust core. ODME should provide conversion helpers between ODME EdgeTable objects, Rust core graph structures, and `rustworkx.PyGraph` / `rustworkx.PyDiGraph`.

Do not add additional Python graph libraries unless a documented decision proves that `rustworkx` and the Rust ecosystem cannot cover an essential use case.

Rule of thumb: if `rustworkx`, or the Rust ecosystem already implements a standard graph/data operation efficiently, ODME should call that implementation rather than reimplementing it.

## Python style guide

ODME should follow a clean, modern, Pythonic style while remaining an independent project. Required conventions:

- `uv`-managed project metadata in `pyproject.toml`, with dependency groups for `dev`, `ci`, `docs`, `lint`, and `test`.
- `maturin` build integration for Rust extension modules; use `uv_build` only for pure-Python helper packaging if it remains useful.
- Python version policy expressed explicitly in `pyproject.toml`; target `>=3.11` initially unless a later decision requires newer syntax.
- Strict Ruff configuration with Google-style docstrings, import sorting, pyupgrade, pathlib, naming, annotation, bugbear, simplify, return, raise, and logging rules.
- `ty` configured through `[tool.ty.src]` to check both package and tests.
- Typer CLI with a central `app`, a `--version` callback, and subcommands implemented in separate modules.
- `python -m odme` support through `__main__.py`.
- One test file per module where practical, using `typer.testing.CliRunner` for CLI tests.
- MkDocs Material documentation with navigation configured in `mkdocs.yml` and Markdown guides under `docs/`.
- Makefile targets wrapping `uv run --frozen` for linting, formatting, and tests once `uv.lock` exists.

Default ODME style principles:

- Small modules with explicit responsibilities.
- Type annotations on public APIs.
- Google-style docstrings for modules, functions, classes, and methods.
- Dataclasses or typed models for configuration and results.
- No hidden global state except configured logging.
- Prefer pure functions for scientific transformations.
- CLI functions should be thin wrappers around tested library functions.
- Avoid premature abstraction; document all non-obvious choices.
- Use English in code, docs, log messages, and tests.

## Logging standard

Python entry points must configure Loguru with:

- stderr console logging for human-friendly CLI output.
- rotating file logging with `rotation="5 MB"`.
- log directory configurable via environment variable and CLI option.
- structured context fields for model case, input path, seed, and run id where possible.

Example target behavior:

```python
logger.add(log_path, rotation="5 MB", retention="30 days", enqueue=True)
```

Rust kernels should return structured errors and optional diagnostics to Python. They should not print directly except in explicitly marked debugging tools.

## Testing strategy: TDD red/green

Every implementation branch should follow this loop:

1. Write or update a failing test that defines the intended behavior.
2. Run the smallest relevant test command and confirm red.
3. Implement the minimum code to pass.
4. Run tests and confirm green.
5. Refactor only after tests are green.
6. Document decisions or edge cases discovered.

### Test layers

- Rust unit tests for low-level graph and sampling kernels.
- Python unit tests for public API and CLI behavior.
- Integration tests exercising complete modern ODME workflows.
- Fixture-based tests using compact, documented thesis examples.
- Property-based tests for invariants.
- Benchmark tests for performance regressions where appropriate.

### Useful property-based tests

- Reading and writing edge lists preserves total event count `T` and binary edge count `E`.
- Directed strength sums satisfy `sum(s_out) == sum(s_in) == T`.
- Undirected strength sum satisfies `sum(s) == 2T` except explicitly documented self-loop conventions.
- Generated probabilities are finite and non-negative.
- Normalized probability vectors sum to one within tolerance.
- Sampling with a fixed seed is reproducible.
- Removing self-loops never increases total weight.
- Fixed-strength expected matrices satisfy target strengths within tolerance.
- Histograms conserve counts before normalization.

## Documentation strategy

All new code should be documented in Markdown-first form.

Required docs:

- `docs/thesis-context.md`: mapping from thesis/model equations to implementation modules.
- `docs/concepts/*.md`: model explanations in researcher-friendly language.
- `docs/api/*.md`: public Python and Rust APIs.
- `docs/cli/*.md`: command usage and examples.
- `docs/decisions/*.md`: ADR-style decision records.
- `docs/development/*.md`: test, benchmark, release, and contribution workflows.

Every branch with an architectural consequence should add or update a decision document.

## Branching policy

- Never perform relevant rewrites on `master`/`main`.
- Use focused branches:
  - `refactor/scaffold-rust-python`
  - `refactor/graph-core`
  - `refactor/io-edge-list`
  - `refactor/stat-strength-degree`
  - `refactor/model-fixed-strength`
  - `refactor/python-cli-analyze`
  - `docs/mkdocs-site`
- Keep branches small enough to review and test independently.
- Each branch should include tests, documentation, and a short decision note if behavior changes.

## Implementation status (last updated: 2026-05-17)

| Milestone | Status | Tests |
|-----------|--------|-------|
| 0. Planning | ✅ Complete | — |
| 1. Scaffold | ✅ Complete | 3 (import, CLI, Rust extension) |
| 2. Graph/data | ✅ Complete | 6 (EdgeTable, rustworkx) |
| 3. I/O | ✅ Complete | 8 (CSV, TSV, Parquet, GraphML, MTX, Pajek) |
| 4. Analysis | ✅ Complete | 12 (strengths, degrees, Y2, k_nn, s_nn, P(w), clustering) |
| 5. Fixed-strength | ✅ Complete | 13 (fitting, generation, ensemble) |
| 6. Remaining models | ✅ Complete | All thesis cases 1–5 + strength-cost + partial constraints |
| 7. Additional kernels | ❌ Not started | W/AB/AW variants (geometric, binomial, NB) |
| 7b. Ensemble equivalence | ✅ Complete | Microcanonical sampler + convergence validation + figures |
| 8. CLI | ✅ Complete | All models exposed: analyze, fit, generate |
| 9. Docs site | ✅ Complete | Full LaTeX math, all models, partial constraints, ensembles |
| 10. Benchmarks | ✅ Complete | Parallel streaming generation to N=30000, regression tests, figures |
| 11. Statistical filtering | ❌ Not started | Flag edges incompatible with null model |

**Current validated subset:** streaming-generation Python tests, relevant generation tests, all Rust workspace tests, Ruff checks on touched files, Clippy, Rust formatting, and MkDocs strict build are green.

**Architecture:** All computation in Rust (`odme-core`). Python is thin wrappers + CLI + I/O. No Polars. numpy + pyarrow + rustworkx only. Generation uses streaming pair providers and Rayon parallel chunks instead of dense $N^2$ probability matrices.

**Next steps:** Start Milestone 11 statistical filtering, using the streaming pair-distribution providers to avoid dense null-model matrices.

## Proposed implementation milestones

### Milestone 0: Planning and conventions — ✅ COMPLETE

- Add `PLAN.md` and `AGENTS.md`.
- Decide package name and module names.
- Finalize ODME-specific style, tooling, and repository conventions.

### Milestone 1: Build scaffold — ✅ COMPLETE

- Add `pyproject.toml` managed by `uv` for the Python distribution named `ODME`, exposing the import package `odme`.
- Add Rust workspace with `odme-core` and `odme-python`.
- Add `maturin` build integration.
- Add `ruff`, `ty`, `pytest`, `hypothesis`, `rustworkx`, and `mkdocs` configuration.
- Add minimal `odme` Python package and Typer CLI with `--version`.
- Add public Rust crate exports and Python import smoke tests.
- UV supply-chain protection via `UV_EXCLUDE_NEWER` (7-day cutoff).

### Milestone 2: Core graph/data representation — ✅ COMPLETE

- `EdgeTable` frozen dataclass with numpy arrays.
- `WeightedEdge` Rust struct in `odme-core`.
- Rustworkx conversion via `edges_to_rustworkx`.
- Directed/undirected and self-loop support.

### Milestone 3: Modern I/O — ✅ COMPLETE

- Readers for CSV, TSV, Parquet, Arrow IPC, GraphML, Matrix Market, Pajek.
- Writers for CSV, TSV, Parquet, Arrow IPC.
- pyarrow-based file I/O (Polars was evaluated and dropped).
- Non-integer/negative weights rejected; zero weights ignored.

### Milestone 4: Analysis statistics — ✅ COMPLETE

- All kernels implemented in Rust (`odme-core`):
  - `graph.rs`: directed strengths, directed degrees.
  - `stats.rs`: single-pass `compute_all_node_stats` (Y2, s_nn, k_nn), `weight_distribution`.
  - `clustering.rs`: binary and weighted clustering coefficients.
- Python modules are thin wrappers returning typed dataclasses with numpy arrays.
- No Polars, no Python loops in any computation path.

### Milestone 5: Fixed-strength fitting and generation — ✅ COMPLETE

- Analytical solution `x = s_out / sqrt(T)` for ME with self-loops.
- Iterative proportional fitting (IPF) in Rust for no-self-loops case.
- Poisson and multinomial samplers in Rust, node-factorized, O(E) memory.
- No dense N² path.
- Large supports use deterministic Rayon chunks by default.
- Ensemble averaging utilities (`ensemble_average`, `ensemble_scalar_average`).

### Milestone 6: Remaining maximum-entropy models — ✅ COMPLETE

- ✅ Implement directed binary fixed-degree fitting with `p_ij = x_i y_j / (1 + x_i y_j)`.
- ✅ Implement custom-`p_ij` thesis Case 1 generation.
- ✅ Implement fixed-strength-and-total-edges thesis Case 3 fitting.
- ✅ Implement exact fixed-strength-and-degree thesis Case 4 fitting/generation.
- ✅ Implement fixed-degree thesis Case 5 weighted generation.
- ✅ Implement strength-cost/distance-constrained thesis Case 2 fitting and generation.
- Strength-cost generation streams `E[t_ij] = x_i * y_j * exp(-gamma d_ij)` in Rust; current sparse cost entries treat missing pairs as zero cost, matching the documented API semantics.
- Future strength-cost fitting should accept metric functions or row-streamed cost access to avoid dense distance storage during fitting as well as generation.
- Keep each model in its own small branch and expose both Rust and Python endpoints.
- TDD: expected degree, strength, probability, and cost constraints hold within documented tolerances.

### Milestone 7: Additional generation kernels — ❌ NOT STARTED

- Implement deterministic seeded samplers needed by the selected maximum-entropy models: Poisson, multinomial, geometric, binomial, and zero-inflated variants.
- Add radiation and sequential-gravity mobility models only after the maximum-entropy core is stable.
- TDD: distribution-level property tests, seed reproducibility, invariants on generated graphs.

### Milestone 7b: Ensemble equivalence validation — ✅ COMPLETE

One can prove that the maximum-entropy ensembles for the multi-edge case are equivalent in the limit of large `T`, irrespective of the number of nodes. ODME must include a validation step that demonstrates this convergence numerically.

Implement:

1. **Microcanonical fixed-strength sampler**: a stub-matching algorithm that preserves the strength sequence exactly at every step. Each event (unit of weight) is an individual stub. Assign outgoing stubs uniformly at random to incoming stubs, producing integer weights that preserve `s_out` and `s_in` exactly. This is equivalent to a random bipartite matching of individual events to source/destination stubs. **Note**: the analytical formula `E[t_ij] = s_out_i * s_in_j / T` is only exact when self-loops are allowed. The ensemble equivalence validation should therefore be performed with self-loops enabled.
2. **Canonical fixed-strength sampler**: multinomial allocation of `T` events with probabilities `p_ij = s_out_i * s_in_j / T^2`.
3. **Grand-canonical fixed-strength sampler**: independent Poisson processes with rate `lambda_ij = s_out_i * s_in_j / T`.

Then compare statistics (weight distribution, degree sequence, `Y2`, nearest-neighbor correlations) across the three ensembles for increasing `T` at fixed node count. In the large-`T` limit, all three must converge to the same marginals.

This serves as a scientific correctness test for the entire model pipeline, confirming that:

- the analytical expectations are correct,
- the samplers produce the right distributions,
- ensemble equivalence holds as predicted by the thesis.

TDD:

- For small `T`, the microcanonical, canonical, and grand-canonical samplers may differ visibly.
- For large `T` (e.g., `T >> N^2`), all ensemble-averaged statistics must agree within documented tolerances.
- Property tests: strength sequences are exactly preserved in microcanonical samples; total event count is exactly `T` in canonical samples.

### Milestone 8: Modern CLI — ✅ COMPLETE (initial)

- Implemented: `odme analyze strengths`, `odme fit strengths`, `odme fit degrees`, `odme fit strength-cost-me`, `odme fit strength-degree-me`, `odme fit strength-edges-me`, `odme generate poisson`, `odme generate multinomial`, `odme generate poisson-multinomial`, `odme generate degree-events-me`, `odme generate strength-cost-me`, `odme generate strength-degree-me`, `odme generate strength-edges-me`, `odme generate custom-pij`.
- Standard universal arguments: `--output`/`-o`, `--json`, `--quiet`, `--seed`/`-s`.
- stdout for data, stderr for progress.
- Remaining: `odme convert`, additional subcommands as models are added.

### Milestone 9: Documentation site — ✅ BUILDS

- MkDocs Material site with concepts, API, CLI, decisions, scalability docs.
- `mkdocs build --strict` passes.
- Remaining: tutorials, thesis-equation mapping, fuller API docs.

### Milestone 10: Performance and memory benchmarks — ✅ COMPLETE

- Added verbose Typer benchmark scripts under `benchmarks/`.
- Added streaming-generation benchmark data, figures, and docs.
- Benchmarked all generation cases with five repeats through `N = 30000` on a 14-core machine.
- Peak RSS remains below 270 MiB for the benchmark setup.
- Parallel generation uses Rayon row/sparse-entry chunks with deterministic per-chunk seeds.
- Dense $N^2$ probability matrices are not used for generation.
- Remaining future work: add Criterion microbenchmarks for Rust kernels and CI-friendly regression thresholds.

### Milestone 11: Statistical filtering module — ❌ NOT STARTED

Given an observed weighted network and a chosen ME null model, the filtering
module identifies pairs whose observed weight is statistically incompatible
with the fitted null distribution. It must reuse the streaming pair-distribution
machinery from generation rather than building dense probability matrices.

#### Scientific scope

| Filter model | Null distribution |
|--------------|-------------------|
| fixed-strength ME | Poisson rate $x_i y_j$ |
| strength-cost ME | Poisson rate $x_i y_j e^{-\gamma d_{ij}}$ |
| custom $p_{ij}$ ME | Poisson or multinomial-compatible rate $T p_{ij}$ |
| degree-events ME | Bernoulli occupation + conditional positive weight |
| strength-edges ME | ZIP with occupation $p_{ij}$ and ZTP rate $x_i y_j$ |
| strength-degree ME | ZIP with occupation $p_{ij}$ and ZTP rate $x_i y_j$ |
| partial constraints | known pairs bypassed; free pairs filtered by fitted model |

Initial implementation should support Poisson and ZIP/ZTP filters first, then
add canonical multinomial p-values only if the exact coupled test is required.

#### Reuse-first architecture

Extend the generation abstraction into a public Rust-internal pair-distribution
provider:

| Reusable component | Used by generation | Used by filtering |
|--------------------|-------------------|-------------------|
| pair provider | sample each candidate pair | compute p-values per pair |
| `PairDistribution` | draw Poisson or ZIP weight | choose CDF/survival formula |
| Rayon chunk runner | parallel sampling | parallel p-value computation |
| sparse support provider | custom $p_{ij}$ generation | custom $p_{ij}$ filtering |
| no-self-loop mask | skip diagonal sampling | skip diagonal filtering |
| known-pair mask | partial generation support | partial filtering support |

The filter should add sinks, not duplicate model math. Desired Rust shape:

```text
provider -> filter sink -> significant/compatible sparse outputs
```

The sink receives `(source, target, distribution, observed_weight)` and emits
classified results. This keeps future models reusable: a new model implements a
provider once and gets generation plus filtering.

#### Procedure

1. Extract observed sparse weights and constraints from the network.
2. Fit or accept a fitted null model.
3. Build the model's pair-distribution provider.
4. For observed edges, compute upper/lower p-values:
   - upper: $P(T \ge t_{ij}\mid null)$;
   - lower: $P(T \le t_{ij}\mid null)$.
5. Classify observed edges as upper, lower, or compatible.
6. Optionally stream candidate pairs for absent-edge detection.
7. Emit sparse result tables with p-values and expected weights.

Use strict, documented tail conventions:

| Tail | Suggested default |
|------|-------------------|
| upper | $P(T \ge t)$, inclusive observed count |
| lower | $P(T \le t)$, inclusive observed count |
| two-sided | `2 * min(upper, lower)` capped at 1 unless user chooses split-alpha |

#### Exact distributions

Poisson:

$$
P(T \le t)=F_\lambda(t),\qquad P(T \ge t)=1-F_\lambda(t-1).
$$

ZIP/ZTP with occupation probability $p$ and positive-weight rate $\lambda$:

$$
P(T=0)=1-p,
$$

$$
P(T=k>0)=p\,\frac{e^{-\lambda}\lambda^k/k!}{1-e^{-\lambda}}.
$$

Therefore:

$$
P(T \le 0)=1-p,
$$

$$
P(T \le t>0)=1-p+p\,F_{ZTP,\lambda}(t),
$$

$$
P(T \ge t>0)=p\,P_{ZTP,\lambda}(T\ge t).
$$

The zero-truncated Poisson boundary rules from generation apply: conditional
mean below one is infeasible, mean near one is deterministic positive weight 1,
and tiny rates use stable direct CDF/PMF logic.

#### Absent-edge detection

Observed-edge filtering is $O(E_{obs})$. Absent-edge detection can require
streaming all candidate pairs, so it must be opt-in:

| Option | Meaning |
|--------|---------|
| `detect_absent=True` | consider pairs with observed weight zero |
| `min_expected` | skip pairs with expected weight below threshold |
| `min_occupation` | skip ZIP pairs with small occupation probability |
| `max_absent` | optional cap to prevent huge outputs |

The implementation should reuse the same Rayon row/sparse chunks used for
generation. It should build an observed-pair hash set once, then stream only
candidate pairs not in that set and passing thresholds.

#### Rust modules

Add `crates/odme-core/src/filter.rs`:

- `FilterTailPValues { upper, lower }`.
- `FilteredEdge { source, target, weight, expected, upper_p, lower_p }`.
- `FilterClassified { upper, lower, compatible, absent_lower }`.
- Poisson p-value helpers using numerically stable CDF/survival routines.
- ZIP/ZTP p-value helpers shared with generation's ZTP functions.
- Parallel filtering sinks over provider chunks.
- Unit tests for exact PMF/CDF identities and boundary cases.

Refactor `generation.rs` only as needed to share provider/distribution types;
avoid a large rewrite. If needed, create `crates/odme-core/src/distribution.rs`
for reusable Poisson and ZIP/ZTP math.

#### Python API

Add `src/odme/filtering.py`:

- `FilterResult` dataclass with upper, lower, compatible, and optional absent
  edge tables plus p-value arrays.
- `filter_network(edges, model, alpha=0.05, tail="two-sided", ...)`.
- Typed convenience wrappers:
  - `filter_fixed_strength_me`;
  - `filter_strength_cost_me`;
  - `filter_strength_edges_me`;
  - `filter_strength_degree_me`;
  - `filter_degree_events_me`;
  - `filter_custom_pij_me`.

Python should validate inputs, call fitters if needed, pass arrays to Rust, and
wrap outputs. No Python loops over pairs.

#### CLI

Add a Typer subcommand module `src/odme/cli/filter.py` and register it in the
main app. Use `/skill:create-cli` before implementation.

Proposed commands:

```bash
odme filter fixed-strength edges.csv --alpha 0.05 --output-prefix filtered/
odme filter strength-cost edges.csv --costs costs.csv --detect-absent
odme filter strength-edges edges.csv --target-edges 500 --tail upper
```

Options:

| Option | Meaning |
|--------|---------|
| `--alpha` | significance level |
| `--tail` | `upper`, `lower`, `two-sided` |
| `--output-prefix` | write upper/lower/compatible/absent files |
| `--json` | emit summary JSON to stdout |
| `--detect-absent` | enable absent-edge streaming |
| `--min-expected` | absent-edge expected-weight threshold |
| `--min-occupation` | ZIP absent-edge occupation threshold |
| `--max-absent` | cap absent output size |
| `--self-loops/--no-self-loops` | diagonal handling |

#### Testing strategy

1. Red tests for Poisson p-values from hand-computed small examples; avoid
   adding SciPy as a dependency.
2. Red tests for ZIP/ZTP PMF normalization and CDF/survival consistency.
3. Seeded end-to-end tests: generate from a fitted null, filter at alpha, and
   assert flagged fractions are calibrated within documented tolerances.
4. Absent-edge tests with tiny graphs where missing high-rate pairs are flagged.
5. CLI smoke tests with `CliRunner` and JSON summaries.
6. Performance smoke test: filter observed edges at `N=10000` without dense
   allocation; absent detection tested with thresholds to keep output bounded.

#### Documentation and decisions

- Add `docs/concepts/statistical-filtering.md` with equations and tail
  conventions.
- Add `docs/api/filtering.md` for Python API.
- Add `docs/cli/filter.md` for CLI examples.
- Add a decision note if two-sided p-values, multiple-testing correction, or
  absent-edge defaults require scientific choices.

#### Open questions for the milestone

1. Should the default test be one-sided upper, one-sided lower, or two-sided?
2. Should two-sided filtering use `2 * min(p_upper, p_lower)` or split alpha
   (`upper < alpha/2` or `lower < alpha/2`)?
3. Should ODME include multiple-testing correction (Bonferroni/FDR), and if so
   should it be default or opt-in?
4. For absent-edge detection, should the default threshold be expected weight,
   occupation probability, p-value only, or a combination?
5. Should lower-significant observed edges include only positive observed edges,
   with absent edges reported separately, or should they be merged?
6. For canonical multinomial models, is a marginal approximation acceptable for
   filtering, or do we require exact coupled multinomial tests?
7. Should filtering accept already-fitted model objects/files to avoid refitting
   in repeated workflows?

## Replacement plan

- Treat the legacy implementation as disposable reference material, not as a compatibility contract.
- Use legacy code only to understand formulas, edge cases, and historical assumptions.
- Prefer thesis equations, small hand-checked examples, and property tests over golden-file compatibility.
- Design modern public APIs independently for Rust and Python.
- Replace the old C/Python code fully once the new ODME implementation covers the selected scientific scope.
- Prefer modern structured outputs: numpy arrays and typed dataclasses, Parquet, Arrow IPC, and explicit typed result objects.
- Do not preserve old CLI flags unless they are still the clearest user experience.

## Decision log seed

Initial decisions to formalize in `docs/decisions/`:

1. Rust + Python architecture with PyO3 and Maturin.
2. Sparse-first graph and cost matrix representation.
3. Typer CLI replacing ad-hoc C argument parsing.
4. UV, Ruff, and Ty as mandatory Python tooling.
5. TDD red/green and property-based testing policy.
6. Loguru logging with 5 MB rotating files.

## Resolved planning decisions

### Package naming

- Published Python distribution name: `ODME`.
- Python import package name: `odme`.
- Rust crates remain lowercase and hyphenated, e.g. `odme-core` and `odme-python`.

Rationale: `ODME` preserves the project/research identity, while `import odme` follows Python naming conventions.

### First release scope

The first complete release should proceed in staged scope:

1. Build analysis + fixed-strength fitting/generation first.
2. Add the remaining maximum-entropy models next.
3. Add radiation and sequential-gravity mobility models after the max-entropy core is stable.

The first release should not attempt to preserve legacy CLI behavior. It should expose clean modern Rust and Python APIs plus a Typer CLI designed around the new architecture.

### Numerical tolerances

Default deterministic floating-point comparisons:

- `rtol = 1e-8`
- `atol = 1e-10`

Default solver convergence for fitted constraints:

- relative residual `< 1e-8`
- user-configurable tolerance in public APIs and CLI options

Stochastic tests should use both:

- exact seeded reproducibility tests for small examples;
- distributional property checks for larger/randomized examples.

All tolerance choices must be documented beside tests and exposed configuration defaults.

### rustworkx versus custom Rust kernels

Use `rustworkx` for standard graph algorithms, including:

- connected components;
- shortest paths where needed;
- clustering where applicable;
- centrality algorithms;
- simple graph traversals;
- generic graph metrics.

Implement ODME-specific kernels in Rust, including:

- strength and degree computation over weighted OD flow tables;
- multi-edge-specific statistics;
- null-model expectations;
- Lagrange multiplier fitting kernels;
- random graph generation;
- distance/cost-constrained ensemble calculations;
- sparse probability matrix operations;
- model-specific likelihood, surprise, and entropy functions.

Rule: standard graph algorithms belong to `rustworkx`; thesis/model-specific weighted flow operations belong to ODME Rust.

### Ruff strictness

Start with a strict-but-practical Ruff rule set during scaffolding. Enable the core strict style immediately, but allow temporary exceptions for areas that slow down early Rust/PyO3 integration, especially:

- exhaustive annotations in tests;
- docstring completeness during initial scaffolding;
- complexity thresholds in numerical kernels;
- selected try/raise rules where scientific error handling is still being designed.

Before the first complete release, tighten the rule set and remove temporary exceptions unless a documented decision justifies keeping them.
