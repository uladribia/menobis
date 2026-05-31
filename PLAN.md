# PLAN.md — Outstanding Work

## Simplify package to intended use case target users

- Do not expose analysis functions in CLI. Are convert tool needed as CLI?
- Where are analysis functions used in the code? And the convert functions?
- THe internal modules are not using the routers properly. Minimize public python API exposure.
- Partial and full solvers should share the same public signature, one should only be called with the empty mask. For "known_" features.
- Filter and FIT CLI is not using the routing tools. Only the main fit_model, sample_model and filter_model should be used. SImplify signatures

## Missing: Update tests to use dense regime

All E2E tests (`test_fitting_e2e.py`, `test_filtering_e2e.py`,
`test_sampling.py`, benchmark CLI smoke tests) should use the **dense
regime** as the default test configuration. Sparse and saturated can remain
as additional parametrised cases for regime-comparison tests.

**Changes needed:**
- Update test fixture parameters to `average_degree = N/5, events_per_edge = 8.0`
- Update `docs/development/testing.md` regime table
- Keep sparse/saturated as additional parametrised cases where relevant


## Missing: GitHub Pages site and comprehensive documentation

The project has MkDocs material configuration (`mkdocs.yml`) with all the
content pages, but has **not been published to GitHub Pages**. This is the
next deployment step.

**Steps:**
1. Configure `mkdocs.yml` with `site_url: https://<org>.github.io/MENoBiS/`
2. Add GitHub Actions workflow (`.github/workflows/docs.yml`) to build and
   deploy MkDocs to `gh-pages` branch on push to main
3. Verify: `uv run mkdocs build --strict` passes cleanly
4. Push to GitHub and verify the published site

## Missing: Jupyter notebook with two main use cases

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

