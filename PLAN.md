# ODME Modernization Plan

Scientific reference: <https://hdl.handle.net/10803/400560>.

## Completed milestones

| # | Milestone | Tests |
|---|-----------|-------|
| 0 | Planning and conventions | — |
| 1 | Build scaffold | 3 |
| 2 | Core graph/data representation | 6 |
| 3 | Modern I/O | 8 |
| 4 | Analysis statistics | 12 |
| 5 | Fixed-strength fitting and generation | 13 |
| 6 | Remaining maximum-entropy models (Cases 1–5 + cost + partial) | all |
| 7b | Ensemble equivalence validation | microcanonical + convergence |
| 8 | Modern CLI (analyze, fit, generate, filter, convert) | all |
| 9 | Documentation site | mkdocs strict |
| 10 | Performance and memory benchmarks | regression baselines |
| 11 | Statistical filtering (all models + absent edges) | 16 filter tests |

Total: 145 Python tests, 27 Rust tests, all checks green.

## Remaining work

### Milestone 7: Additional distribution families — IN PROGRESS

Steps 7a (distributions/providers/generation/filtering) and 7b (binomial IPF)
are complete. Steps 7c (geometric/NB fitting) and 7d (mobility models) remain.

The current ODME implements only the **Poisson** (ME) and **ZIP/ZTP**
distribution families. The thesis and legacy code support three additional
families that share the same constraint sets but use different weight
distributions. Each family requires fitting, generation, filtering, and
provider support.

#### Distribution families

| Family | Weight distribution | Legacy case | Expected value |
|--------|-------------------|-------------|----------------|
| Poisson (ME) | $\text{Poi}(x_i y_j)$ | `ME` | $x_i y_j$ |
| Geometric | $\text{Geom}(x_i y_j)$ | `W` | $x_i y_j / (1 - x_i y_j)$ |
| Binomial (M layers) | $\text{Bin}(M, x_i y_j / (1+x_i y_j))$ | `B` | $M x_i y_j / (1+x_i y_j)$ |
| Negative binomial | $\text{NB}(M, x_i y_j)$ | via layers | $M x_i y_j / (1-x_i y_j)$ |

The Poisson family is fully implemented. The other three share the same IPF
structure but differ in the expected-value formula and the pair-level sampler.

#### Fitting complexity

**Binomial family** fitting uses the same iterative-proportional-fitting (IPF)
structure as Poisson. The only difference is the correction factor in each
iteration step:

- Poisson: `E[t_ij] = x_i * y_j` (no correction)
- Binomial: `E[t_ij] = M * x_i*y_j / (1 + x_i*y_j)` (correction `1/(1+x_i*y_j)`)

The existing Rust IPF solver can be parameterized to support binomial fitting
with minimal changes. This applies to all constraint sets (fixed-strength,
strength-edges, strength-degree).

**Geometric / negative binomial family** fitting does NOT use IPF. The legacy
code (`fitter_s.py` case `W`) uses bounded scipy optimization (`TNC`) with:

- Explicit log-likelihood function
- Analytical gradient vector
- Sparse Hessian matrix (scipy COO format)
- Domain constraint: $x_i y_j < 1$ for all pairs
- Initial conditions: $x_i = y_i = 0.9$ (not strength-derived)
- Preconditioning via `TNC` before the main solver

This is fundamentally different from IPF and requires a new Rust solver.

#### Implementation plan

**Step 7a: Distribution and provider layer (all families)**

1. Add `PairDistribution::Geometric { rate }` and
   `PairDistribution::Binomial { n_layers, prob }` and
   `PairDistribution::NegBinomial { n_layers, rate }` to `distribution.rs`.
2. Implement `sample()`, `expected()`, `occupation_probability()`,
   `lower_pvalue()`, `upper_pvalue()` for each.
3. Add corresponding providers to `pairs.rs` for each constraint set.
4. Wire generation through `sample_provider` — automatic.
5. Wire filtering through `filter_observed_provider` / `detect_absent_provider`
   — automatic.
6. Add Python wrappers and CLI `--distribution` / `--layers` options.
7. Add partial-constraint support via `filter_custom_rates_poisson` pattern
   (partial fitters already produce rate tables).
8. TDD: PMF normalization, CDF monotonicity, mean/variance match formulas,
   seed reproducibility, filter partition coverage.

**Step 7b: Binomial family fitting**

1. Parameterize the existing Rust IPF solver to accept a distribution-family
   enum that selects the correction factor:
   - `Poisson`: identity
   - `Binomial(M)`: `1 / (1 + x_i*y_j)`
2. This applies to `balance_strengths`, `balance_strength_edges_me`,
   `balance_strength_degree_me`, and the masked/partial variants.
3. Expose `layers` parameter in Python fitting functions and CLI.
4. TDD: recovered constraints within tolerance for binomial family at
   several M values.

**Step 7c: Geometric / negative binomial fitting (plan only)**

This requires a new Rust optimization solver, not IPF. The plan:

1. Implement the log-likelihood, gradient, and Hessian for the geometric
   family in Rust (`crates/odme-core/src/fitting_geometric.rs` or similar).
   The formulas from the legacy code:
   - $\mathcal{L} = M \sum_{ij} \log(1 - x_i y_j) + \sum_i s^{\text{out}}_i \log x_i + \sum_j s^{\text{in}}_j \log y_j$
   - Gradient: $\partial\mathcal{L}/\partial x_i = s^{\text{out}}_i / x_i - M \sum_j y_j / (1 - x_i y_j)$
   - Hessian: diagonal and off-diagonal blocks with $1/(1-x_i y_j)^2$ terms
2. Implement a bounded optimizer in Rust. Options:
   - L-BFGS-B style box-constrained solver (domain $0 < x_i y_j < 1$)
   - TNC (truncated Newton with bounds), matching the legacy scipy approach
   - Or use `argmin` crate if it provides suitable bounded solvers
3. The negative binomial case is the geometric case with $M > 1$ layers
   (same log-likelihood structure, different interpretation).
4. Add preconditioning strategy (legacy uses TNC preconditioning pass).
5. Support all constraint sets: fixed-strength, strength-edges,
   strength-degree, and partial variants.
6. TDD: convergence tests, domain-violation detection, comparison with
   legacy scipy results on small examples.

Do not implement Step 7c without first completing Steps 7a and 7b, because
the distribution/provider/generation/filtering infrastructure must be in
place before the fitting solver is useful.

**Step 7d: Mobility models (separate, after 7a–7c)**

| Model | Status |
|-------|--------|
| Sequential gravity (Bernoulli + multinomial) | ❌ |
| Radiation model (stochastic + multinomial) | ❌ |

These are standalone models, not ME distribution variants. Implement after
the distribution families are complete.

### Future: naming cleanup

The codebase uses inconsistent naming. The key insight is that **sampling and
filtering are distribution-level operations** — they don't care what constraint
produced the multipliers. The constraint only determines how multipliers
combine into the `xy` product (handled by the provider). Once you have a
`PairDistribution`, sampling and filtering are the same regardless of
constraint.

However, the public API exposes constraint-specific functions because users
pass constraint-specific parameters (e.g. `x, y` for strength vs `x, y, lam`
for strength-edges). The naming should reflect:

- **Fitting**: `fit_{constraint}_{distribution}` — needs both.
- **Sampling/filtering with constraint-specific params**: include constraint
  only when the function signature requires constraint-specific parameters
  (e.g. `lam`, `gamma`, `z`, `w`). When only `x, y` are needed, the
  constraint is implicit.
- **Generic provider-level**: `sample_provider`, `filter_observed_provider`,
  `detect_absent_provider` — distribution-agnostic.

The distribution name implies the ensemble type:

| Ensemble | Legacy case | Distributions | Fitting method |
|----------|-------------|---------------|----------------|
| ME (multi-edge) | `ME` | Poisson, Multinomial, Microcanonical | IPF (analytical for self-loops) |
| W (weighted) | `W` | Geometric, Negative binomial | Bounded optimization (TNC/L-BFGS-B) |
| B (binary layers) | `B` | Binomial, Bernoulli | IPF with correction factor |

The `_me` suffix is **removed** from function names. ME stands for multi-edge
(the project name), not maximum entropy.

The `_zip` suffix is also removed — ZIP describes the inflation structure,
not the distribution.

#### Naming convention

```text
fit_{constraint}_{distribution}           # fitting
sample_{distribution}                     # simple (x, y only)
sample_{constraint}_{distribution}        # when extra params needed
filter_{distribution}                     # simple (x, y only)
filter_{constraint}_{distribution}        # when extra params needed
```

#### Rust renames (crates/odme-core)

Generation:

| Current | Proposed | Reason |
|---------|----------|--------|
| `sample_poisson` | `sample_poisson` | x,y only — constraint implicit |
| `sample_geometric` | `sample_geometric` | x,y only |
| `sample_binomial` | `sample_binomial` | x,y,layers only |
| `sample_neg_binomial` | `sample_neg_binomial` | x,y,layers only |
| `sample_microcanonical` | `sample_microcanonical` | strengths only |
| `sample_multinomial` | `sample_multinomial` | x,y,T — constraint implicit |
| `sample_poisson_multinomial` | `sample_poisson_multinomial` | unchanged |
| `sample_strength_edges_me` | `sample_strength_edges_poisson` | constraint params needed |
| `sample_strength_degree_me` | `sample_strength_degree_poisson` | constraint params needed |
| `sample_strength_cost_me` | `sample_strength_cost_poisson` | constraint params needed |
| `sample_fixed_degree_events_me` | `sample_degree_events_poisson` | drop `fixed_` |
| `sample_custom_pij_events_poisson` | `sample_custom_poisson` | simplify |
| `sample_custom_pij_events_multinomial` | `sample_custom_multinomial` | simplify |

Filtering:

| Current | Proposed | Reason |
|---------|----------|--------|
| `filter_fixed_strength_poisson` | `filter_poisson` | x,y only |
| `filter_geometric` | `filter_geometric` | unchanged |
| `filter_binomial` | `filter_binomial` | unchanged |
| `filter_neg_binomial` | `filter_neg_binomial` | unchanged |
| `filter_strength_edges_zip` | `filter_strength_edges_poisson` | constraint params |
| `filter_strength_degree_zip` | `filter_strength_degree_poisson` | constraint params |
| `filter_strength_cost_poisson` | `filter_strength_cost_poisson` | unchanged |
| `filter_degree_events_zip` | `filter_degree_events_poisson` | constraint params |
| `filter_custom_poisson_rates` | `filter_custom_poisson` | simplify |
| `absent_*` | same pattern as `filter_*` | |

Fitting:

| Current | Proposed | Reason |
|---------|----------|--------|
| `balance_strength_edges_me` | `balance_strength_edges_poisson` | drop `_me` |
| `balance_strength_degree_me` | `balance_strength_degree_poisson` | drop `_me` |
| `balance_masked_strength_degree_me` | `balance_masked_strength_degree_poisson` | drop `_me` |
| `balance_binomial_strength` | `balance_strength_binomial` | reorder |
| `balance_masked_binomial_strength` | `balance_masked_strength_binomial` | reorder |
| `balance_no_self_loops` | `balance_strength_poisson_no_self_loops` | add constraint+dist |
| `balance_binary_degrees` | `balance_degree_bernoulli` | rename |
| `balance_masked_binary_degrees` | `balance_masked_degree_bernoulli` | rename |
| `balance_masked_strength` | `balance_masked_strength_poisson` | add distribution |

#### Python renames (src/odme)

Fitting:

| Current | Proposed |
|---------|----------|
| `fit_fixed_strength_me` | `fit_strength_poisson` |
| `fit_fixed_degree_binary` | `fit_degree_bernoulli` |
| `fit_strength_edges_me` | `fit_strength_edges_poisson` |
| `fit_strength_degree_me` | `fit_strength_degree_poisson` |
| `fit_strength_cost_me` | `fit_strength_cost_poisson` |
| `fit_binomial_strength` | `fit_strength_binomial` |
| `StrengthCostMEFit` | `StrengthCostFit` |
| `StrengthEdgesMEFit` | `StrengthEdgesFit` |
| `StrengthDegreeMEFit` | `StrengthDegreeFit` |

Generation (same logic as Rust — no constraint when params are simple):

| Current | Proposed |
|---------|----------|
| `sample_poisson` | `sample_poisson` |
| `sample_geometric` | `sample_geometric` |
| `sample_binomial` | `sample_binomial` |
| `sample_neg_binomial` | `sample_neg_binomial` |
| `sample_microcanonical` | `sample_microcanonical` |
| `sample_multinomial` | `sample_multinomial` |
| `sample_poisson_multinomial` | `sample_poisson_multinomial` |
| `sample_strength_edges_me` | `sample_strength_edges_poisson` |
| `sample_strength_degree_me` | `sample_strength_degree_poisson` |
| `sample_strength_cost_me` | `sample_strength_cost_poisson` |
| `sample_fixed_degree_events_me` | `sample_degree_events_poisson` |
| `sample_custom_pij_events_poisson` | `sample_custom_poisson` |
| `sample_custom_pij_events_multinomial` | `sample_custom_multinomial` |

Filtering:

| Current | Proposed |
|---------|----------|
| `filter_fixed_strength_me` | `filter_poisson` |
| `filter_geometric` | `filter_geometric` |
| `filter_binomial` | `filter_binomial` |
| `filter_neg_binomial` | `filter_neg_binomial` |
| `filter_strength_edges_me` | `filter_strength_edges_poisson` |
| `filter_strength_degree_me` | `filter_strength_degree_poisson` |
| `filter_strength_cost_me` | `filter_strength_cost_poisson` |
| `filter_degree_events_me` | `filter_degree_events_poisson` |
| `filter_custom_rates_poisson` | `filter_custom_poisson` |

Partial fitters:

| Current | Proposed |
|---------|----------|
| `fit_partial_strength_me` | `fit_partial_strength_poisson` |
| `fit_partial_strength_edges_me` | `fit_partial_strength_edges_poisson` |
| `fit_partial_strength_degree_me` | `fit_partial_strength_degree_poisson` |
| `fit_partial_strength_cost_me` | `fit_partial_strength_cost_poisson` |
| `fit_partial_degree_me` | `fit_partial_degree_poisson` |

#### CLI renames

Fitting always needs constraint + distribution:

| Current | Proposed |
|---------|----------|
| `odme fit strengths` | `odme fit strength-poisson` |
| `odme fit degrees` | `odme fit degree-bernoulli` |
| `odme fit strength-cost-me` | `odme fit strength-cost-poisson` |
| `odme fit strength-edges-me` | `odme fit strength-edges-poisson` |
| `odme fit strength-degree-me` | `odme fit strength-degree-poisson` |

Generation — include constraint only when CLI needs extra params:

| Current | Proposed |
|---------|----------|
| `odme generate poisson` | `odme generate poisson` |
| `odme generate multinomial` | `odme generate multinomial` |
| `odme generate poisson-multinomial` | `odme generate poisson-multinomial` |
| `odme generate degree-events-me` | `odme generate degree-events-poisson` |
| `odme generate strength-cost-me` | `odme generate strength-cost-poisson` |
| `odme generate strength-edges-me` | `odme generate strength-edges-poisson` |
| `odme generate strength-degree-me` | `odme generate strength-degree-poisson` |
| `odme generate custom-pij` | `odme generate custom-poisson` |

Filtering — same logic:

| Current | Proposed |
|---------|----------|
| `odme filter fixed-strength` | `odme filter poisson` |
| `odme filter strength-edges` | `odme filter strength-edges-poisson` |
| `odme filter strength-degree` | `odme filter strength-degree-poisson` |
| `odme filter strength-cost` | `odme filter strength-cost-poisson` |
| `odme filter degree-events` | `odme filter degree-events-poisson` |
| `odme filter custom-rates` | `odme filter custom-poisson` |

#### Docs

- Replace "maximum entropy" with "multi-edge" where ME abbreviation is
  expanded incorrectly. The models ARE derived from maximum entropy
  principles, but ME in ODME stands for multi-edge.
- Document the three ensemble types (ME/W/B) and which distributions
  belong to each in `docs/concepts/thesis-cases.md`.
- Update all function references in concept/API/CLI docs.

#### Scope

Mechanical rename across ~40 files. One dedicated branch, one commit.
Full test suite before and after.

### Future: thesis-equation mapping

Create `docs/thesis-context.md` mapping thesis equations to ODME code paths:

| Thesis ref | Equation / concept | ODME code path |
|------------|-------------------|----------------|
| §3.2 | Fixed-strength ME: $\langle t_{ij}\rangle = x_i y_j$ | `fit_fixed_strength_me` → `FixedStrengthPoissonProvider` |
| §3.3 | Strength-cost ME: $\langle t_{ij}\rangle = x_i y_j e^{-\gamma d_{ij}}$ | `fit_strength_cost_me` → `StrengthCostPoissonProvider` |
| §3.4 | Strength-edges ZIP | `fit_strength_edges_me` → `StrengthEdgesZipProvider` |
| §3.5 | Strength-degree ZIP | `fit_strength_degree_me` → `StrengthDegreeZipProvider` |
| ... | ... | ... |

Requires careful verification against the thesis PDF. Do not guess equation
numbers.

### Future: tutorials

1. `docs/tutorials/filter-workflow.md` — end-to-end: load → fit → generate →
   filter → export. Copy-pasteable Python + CLI equivalents. 80–120 lines.

2. `docs/tutorials/partial-constraints.md` — cutoff → partial fit → filter →
   compare with/without partial constraints.

### Future: legacy code removal

#### Inventory

**`1. Network analysis/`** — C tool `MultiEdgeAnalyzer`. **Fully replaced.**
Safe to remove now.

**`2. Model Fitting/`** — Python 2 `multi_edge_fitter`. **Partially replaced.**

| Legacy fitter | ODME equivalent | Status |
|---------------|-----------------|--------|
| `fitter_s.py` case `ME` | `fit_fixed_strength_me` | ✅ |
| `fitter_s.py` case `B` (binomial, M layers) | — | ❌ M7 |
| `fitter_s.py` case `W` (geometric) | — | ❌ M7 |
| `fitter_k.py` | `fit_fixed_degree_binary` | ✅ |
| `fitter_sk.py` (ME mode) | `fit_strength_degree_me` | ✅ |
| `fitter_sk.py` (`agg=True`, M layers) | — | ❌ M7 |
| `fitter_E.py` (ME mode) | `fit_strength_edges_me` | ✅ |
| `fitter_E.py` (`agg=True`, M layers) | — | ❌ M7 |
| `fitter_grav.py` | `fit_strength_cost_me` | ✅ |
| `fitter_pij.py` | partial fitting (`fit_partial_*`) | ✅ |
| `fitter_s_CVXOPT.py` | superseded by IPF | ✅ |

**`3. Model Generation/`** — C tool `GenNetGen`. **Partially replaced.**

| Legacy generator | ODME equivalent | Status |
|------------------|-----------------|--------|
| `fixeds_poisson_*` | `sample_poisson` | ✅ |
| `fixeds_computational_*` | `sample_microcanonical` | ✅ |
| `multinomial_*` | `sample_multinomial` | ✅ |
| `poisson_multinomial_*` | `sample_poisson_multinomial` | ✅ |
| `custompij_poisson_*` | `sample_custom_pij_events_poisson` | ✅ |
| `custompij_ZIP_*` | `sample_strength_edges_me` | ✅ |
| `custompij_ZIG_*` | — | ❌ M7 |
| `fixeds_geometric_*` | — | ❌ M7 |
| `fixeds_binomial_*` | — | ❌ M7 |
| `fixeds_negbinomial_*` | — | ❌ M7 |
| `fixedk_poisson_*` | `sample_fixed_degree_events_me` | ✅ |
| `fixedk_geometric_*` | — | ❌ M7 |
| `fixedk_negbinom_*` | — | ❌ M7 |
| `fixedk_binom_*` | — | ❌ M7 |
| `fixedk_bernouilli_*` | `fit_fixed_degree_binary` | ✅ |
| `fixedEs_poisson_*` | `sample_strength_edges_me` | ✅ |
| `fixedks_poisson_*` | `sample_strength_degree_me` | ✅ |
| `fixedks_binomial_*` | — | ❌ M7 |
| `gravity_poisson_*` | `sample_strength_cost_me` | ✅ |
| `w_graph_seq_gravity_*` | — | ❌ M7 |
| `w_graph_radiation_*` | — | ❌ M7 |

#### Removal plan

**Phase 1 (safe now):** Remove `1. Network analysis/` only.

**Phase 2 (after Milestone 7):** Remove `2. Model Fitting/` and
`3. Model Generation/`. Do not remove until Milestone 7 tests prove
equivalence for all distribution families.
