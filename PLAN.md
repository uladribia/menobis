# PLAN.md — Outstanding Work

## 1. Solver Convergence Issues

Documented in `docs/decisions/convergence-issues.md`. Summary:

| Case | Status | Proposed fix |
|---|---|---|
| W strength-edges | never converges | barrier + adaptive damping |
| W strength-degree | never converges | barrier + adaptive damping |
| W strength-cost partial | diverges at all N | conic solver boundary with q→1 |
| B strength-cost saturated N≥1000 | fails to converge within 10k iters | IPF plateau — needs convergence relaxation |
| B strength-degree sparse N=5000 | overflow in z multipliers (z~1e+262) | numerical scaling for ZI solver |

These are algorithm improvements, not missing plumbing.

## 2. Benchmark Results

Comprehensive benchmarks completed for families ME, B, and W (see
`docs/development/benchmark-results.md`).

### Cell coverage

| N | ME | B | W |
|---:|---:|---:|---:|
| 10 | ✅ all | ✅ all | ✅ all (but ✗ for edges/degree) |
| 100 | ✅ all | ✅ all | ⚠️ edges/degree fail |
| 1000 | ✅ all | ✅ all (dense) | ❌ only strength works |
| 5000 | ✅ all (deg sparse 28min) | ⚠️ only strength/cost/edges sparse | ❌ not run |

### B gaps at N=5000

| Cell | Problem |
|---|---|
| strength-degree sparse | Overflow in ZI multiplier (z ~ 1e+262) |
| saturated (all except strength) | Unfinished (timed out) |
| strength-cost saturated 0%/5% | Did not converge (10k iters) |

### W gaps at N=1000+
- strength-edges and strength-degree never converge at any N
- strength-cost partial fits diverge
- N=5000 not run (no point until solvers are fixed)

### Key observations

- **IPF solvers** (strength, strength-cost): single-threaded O(N²), but
  strength is trivial (<1s at N=5000). Strength-cost is the bottleneck at
  large N (45s at N=5000).
- **L-BFGS ZI solvers** (strength-edges, strength-degree): 12× parallel,
  but sparse regime can be 10× slower than saturated.
- **ME works perfectly.** B works well except for known edge cases. W needs
  solver redesign for ZI constraints.
- **The dense regime (`average_degree = N/5`) is the optimal benchmark
  regime** — it exercises solvers realistically without triggering
  pathological edge cases from sparse (k ≈ s) or saturated (k ≈ N).
  All ME and B fits converge at 100% across constraints at N ≤ 1000.

## 3. Dense regime — implemented

[✓] Added to `benchmarks/cli.py` as the default regime
[✓] Documented in `docs/development/benchmarking.md` and `benchmark-results.md`
[✓] Verified: ME + B converge at 100% at N=1000 across all constraints
[✓] Verified: 2% known pairs, self-loops, and filtering all work

## 4. Missing: GitHub Pages site and comprehensive documentation

The project has MkDocs material configuration (`mkdocs.yml`) with all the
content pages, but has **not been published to GitHub Pages**. This is the
next deployment step.

**Steps:**
1. Configure `mkdocs.yml` with `site_url: https://<org>.github.io/MENoBiS/`
2. Add GitHub Actions workflow (`.github/workflows/docs.yml`) to build and
   deploy MkDocs to `gh-pages` branch on push to main
3. Verify: `uv run mkdocs build --strict` passes cleanly
4. Push to GitHub and verify the published site

## 5. Missing: Jupyter notebook with two main use cases

The package has two primary use cases that should be demonstrated in a
single comprehensive notebook (`docs/examples/main-use-cases.ipynb`):

### Use case A: Filtering a network according to a null model

1. Load or generate a weighted directed network
2. Fit a MENoBiS null model (ME strength-cost or strength-degree)
3. Filter the network: identify edges that are statistically significant
   (upper-tail) under the null
4. Visualise: original vs filtered network, edge-weight distribution

### Use case B: Generating null-model instances for ensemble statistics

1. Fit a MENoBiS null model to constraints derived from an observed network
2. Sample N null networks from the fitted model (e.g., N=1000)
3. For each sampled network, compute a network-level magnitude:
   - Average nearest-neighbour strength (ANNS)
   - Clustering coefficient (if applicable)
   - Y₂ or other higher-order moments
4. Compute ensemble mean, variance, and percentiles
5. Compare observed network's value against the null ensemble
6. Visualise: histogram of ensemble statistics + observed value

**Requirements:**
- Rendering must be clean in both Jupyter Lab and VSCode
- Use `pandas`, `numpy`, `matplotlib` (or `seaborn`)
- Show intermediate results with rich display (DataFrames, LaTeX equations)
- Include markdown cells explaining the scientific context
- Include summary statistics and interpretation at the end

## 6. Missing: Update tests to use dense regime

All E2E tests (`test_fitting_e2e.py`, `test_filtering_e2e.py`,
`test_sampling.py`, benchmark CLI smoke tests) should use the **dense
regime** as the default test configuration. Sparse and saturated can remain
as additional parametrised cases for regime-comparison tests.

**Changes needed:**
- Update test fixture parameters to `average_degree = N/5, events_per_edge = 8.0`
- Update `docs/development/testing.md` regime table
- Keep sparse/saturated as additional parametrised cases where relevant
