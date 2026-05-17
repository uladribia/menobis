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

### Milestone 7: Additional distribution families — ❌ NOT STARTED

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
