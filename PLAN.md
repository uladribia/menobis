# ODME modernization plan

Scientific reference: <https://hdl.handle.net/10803/400560>.

## Done

- [x] Rust + Python scaffold with PyO3/maturin
- [x] Edge-list data model, I/O, analysis
- [x] ME fitting: strength, strength-cost (sparse + coordinate), strength-edges, strength-degree
- [x] B fitting: strength, strength-cost (sparse + coordinate)
- [x] W fitting: strength, strength-cost (coordinate Newton), strength-edges, strength-degree, degree-events
- [x] Generation and filtering for ME/B/W
- [x] Partial known-weight fitting (ME coordinate, B/W coordinate with family-specific formulas)
- [x] Unified `coord_distance` in `support.rs`
- [x] All gamma searches use bisection
- [x] Eliminated dense N² `f_mat` allocations in ME/B/partial sparse paths
- [x] Removed Clarabel/cvxrust dependency — all W uses Newton solver
- [x] Benchmark CLI, N=100 validation, coordinate strength-cost benchmarks
- [x] MkDocs documentation site builds strictly

## Remaining steps

1. **Improve W Newton robustness at N≥500.** The projected Newton coordinate-descent
   converges reliably at N≤100 but can stall for some inputs at N=500+. Needs
   adaptive damping, backtracking line search per variable, or Anderson
   acceleration. This is the top priority for W scalability.

2. **Incremental benchmark saving.** Long benchmark runs lose results on timeout.

3. **Verify `w_mean` usage across codebase.** The W expected weight
   `E[t_ij] = M·q/(1-q)` is confirmed correct. Ensure all generation, filtering,
   and residual-checking code uses it consistently.

4. **Archive/remove legacy thesis-era folders** after benchmark capture.

5. **Final project rename decision: ODME → MENoBiS.**

6. **Publish MkDocs site to GitHub Pages.**

## Checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
mkdocs build --strict
```
