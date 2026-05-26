# MENoBiS modernization plan

Scientific reference: <https://hdl.handle.net/10803/400560>.

## Completed

| Step | Branch | Summary |
|---|---|---|
| Solver architecture unification | `refactor/sparse-mask` | Non-partial functions delegate to sparse-masked partials with null known-pair set |
| Dense mask replacement | `refactor/sparse-mask` | `PairMask` struct O(N+K); all dense masked functions removed from public API |
| Unified Python routing API | `refactor/pyo3-minimal-api` | `fit_model`, `sample_model`, `filter_model` with dispatch tables; `FitResult` base class; uniform filter signatures |

## Infrastructure status

### Benchmark CLI

```bash
uv run python -m benchmarks all --nodes 100,500
uv run python -m benchmarks all --nodes 100 --self-loops
uv run python -m benchmarks fit --nodes 500 --families me,w
```

Pipeline: PA geographic generate → fit → ensemble sample-check → null-filter FPR.

### Checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
mkdocs build --strict
```

### Known solver limitations (documented)

| Model | Limitation |
|---|---|
| ME/W/Wnb strength-degree | Degree-saturated nodes use fixed high occupation multipliers while strength multipliers remain active |
| ME/W/Wnb strength-edges | Rejects target_edges ≥ capacity (use strength-only) |


## Next steps (priority order)

1. ~~Solver architecture unification~~ ✅ Done (`refactor/sparse-mask`)
2. ~~Dense Rust mask replacement~~ ✅ Done (`refactor/sparse-mask`)
3. ~~Unified Python routing API~~ ✅ Done (`refactor/pyo3-minimal-api`)
4. **PyO3 binding split** — Split `crates/menobis-python/src/lib.rs` (3500 lines)
   into domain modules (fitting.rs, generation.rs, filter.rs, stats.rs).
   Add macro helpers to reduce #[pyfunction] boilerplate.
5. **Sparse matrix/cost handling cleanup** — Consolidate cost and probability
   providers across fitting, generation, and filtering.
6. Run an extensive benchmark, provide an estimate on how long it would take to do a N=1000,5000,10000,20000 separating by case (ME, B , W).
7. Write tutorials and notebooks with real-world examples + HOWTO on two+one cases: (1) filter a network with a null model (2) assess if some network statistic is relevant under a nullmodel (3) just fit and generate null model instances to do whatever the user needs. Use the PA synth model as realistic example. Do a complete audit of the docs to minimize overlap, maximize clarity and alignment, and refer to the thesis when needed. The docs should read cleary in the README using a mermaid diagram as follows (the notebook should follow these ideas).
    1. Choose a case based on your data:
        - Aggregated binary networks: Binomial. E.g. airlines connecting airports.
        - Integer weights are distinguishable events: ME case. E.g. trips.
        - Integer wegihts are undistinguishable events: W/NB case. E.g. total euros in milions of trade networks, global or per commodity (different layers).
    2. Select if you want to support self loops or not.
    3. Choose a constraint you want to check against. The constraint must be expressed as a linear combination of occupatin numbers or binary occupation events.
        - Is it supported by the package? (fixed strength, fixed strength plus cost, fixed total events and degrees, fixed total binary edges and strengths, fixed degrees and strengthsl)? Use appropriate fitter.
        - Is it not?
            -- Check the thesis, it might be solved there analytically or computationally. If so, implement it as a new fitter.
    4. Choose an ensemble:
        - Microcanonical is only supported for ifxed strengths and ME with self loops.
        - Canonical is only supported for ME (all cases).
        - Grandcanonical covers the rest of cases.
    Now your case is set: family, constraint, self-loop, ensemble.
    5. Choose use-case:
        1. Filter network: use the filter CLI, after having fitted the model.
        2. Generate networks: use the sampling CLI, after having fitted the model.
        3. Assess statistical relevance of high order network features: Use the sampling CLI to generate null models and ensemble the expectations. You might need to impemnet the magnitude you are interested in on the analysis package (or connect with existing packages like networkx or rustworkx).
8. Publish MkDocs site with updated docs.

