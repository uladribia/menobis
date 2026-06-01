# PLAN.md — Outstanding Work

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

