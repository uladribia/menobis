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

1. Make benchmarks save incrementally after each case or node size.
2. Rerun large release benchmarks in chunks, not one all-or-nothing process.
3. Decide practical published limits for N=5000 dense strength-edge and
   strength-degree fits.
4. Benchmark legacy radiation and sequential gravity models before archiving.
5. Archive/remove legacy thesis-era folders after benchmark capture.
6. Run full project checks before release:

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

7. Final project rename decision: ODME -> MENoBiS.
8. Publish MkDocs site to GitHub Pages.
