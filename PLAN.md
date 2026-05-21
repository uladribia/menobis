# ODME modernization plan

Scientific reference: <https://hdl.handle.net/10803/400560>.

## TL;DR

ODME is now a Rust + Python maximum-entropy toolkit for weighted directed
origin-destination networks. The modern implementation covers fitting,
generation, filtering, partial known-weight constraints, benchmark tooling, and
MkDocs documentation. Legacy thesis folders remain as reference material until
final archival.

## Current status

| Area | Status |
|---|---|
| Python/Rust scaffold | Done |
| Edge-list data model and analysis | Done |
| ME, B, and W fitting | Done for supported full constraints |
| Generation and filtering | Done for ME/B/W supported constraints |
| Partial known-weight benchmarks | N=100 validation green for ME/B/W |
| Benchmark CLI | Unified `python -m benchmarks` CLI |
| Docs site | MkDocs builds strictly |
| Legacy mobility models | Not ported; benchmark before archiving |
| Final rename to MENoBiS | Not started |

## Latest benchmark facts

N=100 validation command:

```bash
uv run python -m benchmarks fit --nodes 100 --max-n 100 \
  --known-fractions 0.05,0.40 --tolerance 1e-4 \
  --output /tmp/odme-bench-n100-all-partial-3
```

Result: 60 rows, 40 partial rows, 0 failures. Partial benchmarks use known
weighted pairs only; occupation contributions are inferred from positive known
weights.

Large release run attempted:

```bash
uv run python -m benchmarks fit --nodes 25,50,100,500,1000,5000 \
  --max-n 5000 --known-fractions 0.05,0.40 --tolerance 1e-4 \
  --verbose 2 --plot --output benchmarks/results/release-fit-25-5000
```

It timed out during full non-partial N=5000 after B strength-degree. No JSON was
written because results are saved at the end.

## Remaining steps

1. Fix W fixed-strength no-self-loop fitting before further large W benchmarks.
   The N=1000 no-self-loop fixed-strength benchmark panicked inside the Clarabel
   AMD sparse ordering path (`attempt to add with overflow`). This is separate
   from coordinate strength-cost and must be diagnosed directly: reproduce in a
   Rust/Python regression test, determine whether the conic formulation is too
   large/ill-structured or whether a non-conic IPF/shared kernel should replace
   it for fixed-strength W, and ensure Rust returns structured errors rather
   than panicking across PyO3.
2. Refactor fixed-strength and strength-cost fitting around shared Rust kernels.
   The current coordinate B/W wrappers are **not real implementations**: they
   call the ME Poisson coordinate solver and relabel output as binomial,
   geometric, or negative_binomial. This is why all families show identical
   benchmark timings. The next design must implement family-specific expectation
   equations (`E[t_ij]` differs per family) and share pair-potential/cost-provider
   logic across `NoCost`, sparse costs, and projected XY costs.
3. Verify `w_mean` against thesis `E[T]` for the W exponential family. The
   function `w_mean(r, M) = M*exp(-r)/(1-exp(-r))` may differ from the true
   expected value `E[T] = -d/dr ln(G_M(r))` by a factor `A_M/G_M`. The Newton
   coordinate-descent prototype (`w_lbfgs.rs`) uses `w_mean` as the predicted
   strength, which may be incorrect. Must reconcile with the thesis before the
   Newton solver can replace the conic solver. The conic solver remains correct
   because it solves the full coupled system via exponential cones without
   relying on `w_mean` for internal gradient computation.
3. Make benchmarks save incrementally after each case or node size.
4. Extend coordinate-based strength-cost coverage after the shared-kernel
   refactor: generation, filtering, true B/W coordinate fitting, and partial
   variants without dense cost triples.
5. Rerun large release benchmarks in chunks, not one all-or-nothing process.
6. Decide practical published limits for N=5000 dense strength-edge and
   strength-degree fits.
5. Benchmark legacy radiation and sequential gravity models before archiving.
6. Archive/remove legacy thesis-era folders after benchmark capture.
7. Run full project checks before release:

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

8. Final project rename decision: ODME -> MENoBiS.
9. Publish MkDocs site to GitHub Pages.
