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

#### Required work by constraint set

**Fixed strength (thesis Case 1):**

| Item | Poisson | Geometric | Binomial | Neg. binomial |
|------|---------|-----------|----------|---------------|
| Fitting | ✅ | ❌ | ❌ | ❌ |
| Generation | ✅ | ❌ | ❌ | ❌ |
| Filtering | ✅ | ❌ | ❌ | ❌ |
| Provider | ✅ | ❌ | ❌ | ❌ |

**Fixed strength + edges (thesis Case 3):**

| Item | ZIP/Poisson | ZIP/Geometric | ZIP/Binomial |
|------|-------------|---------------|--------------|
| Fitting | ✅ | ❌ | ❌ |
| Generation | ✅ | ❌ | ❌ |
| Filtering | ✅ | ❌ | ❌ |

**Fixed strength + degree (thesis Case 4):**

| Item | ZIP/Poisson | ZIP/Binomial |
|------|-------------|--------------|
| Fitting | ✅ | ❌ |
| Generation | ✅ | ❌ |
| Filtering | ✅ | ❌ |

**Fixed degree (thesis Case 5):**

| Item | ZIP/Poisson | ZIP/Geometric | ZIP/Binomial | ZIP/Neg. binomial |
|------|-------------|---------------|--------------|-------------------|
| Generation | ✅ | ❌ | ❌ | ❌ |
| Filtering | ✅ | ❌ | ❌ | ❌ |

**Mobility models (separate from ME framework):**

| Model | Status |
|-------|--------|
| Sequential gravity (Bernoulli + multinomial) | ❌ |
| Radiation model (stochastic + multinomial) | ❌ |

#### Implementation strategy

1. Add `PairDistribution::Geometric`, `::Binomial`, `::NegBinomial` variants
   to `crates/odme-core/src/distribution.rs` with `sample()`, `expected()`,
   `occupation_probability()`, `lower_pvalue()`, `upper_pvalue()`.
2. Add corresponding providers to `crates/odme-core/src/pairs.rs`.
3. Fitting: modify Rust IPF to accept a distribution-family parameter that
   changes the expected-value formula. The legacy code shows the pattern:
   - `ME`: `aux = x_i * y_j`
   - `B`: `aux = M * x_i*y_j / (1 + x_i*y_j)`
   - `W`: `aux = M * x_i*y_j / (1 - x_i*y_j)`
4. Generation and filtering follow automatically from providers.
5. Add `--distribution` / `--layers` CLI options to existing fit/generate/filter
   commands.
6. Radiation and sequential-gravity are standalone models, not ME variants.
   Implement after the distribution families are done.

#### Legacy reference files

| File | Contains |
|------|----------|
| `2. Model Fitting/multi_edge_fitter/fitter_s.py` | cases `B`, `W` IPF |
| `2. Model Fitting/multi_edge_fitter/fitter_sk.py` | `agg=True` binomial variant |
| `2. Model Fitting/multi_edge_fitter/fitter_E.py` | `agg=True` binomial variant |
| `3. Model Generation/src/ula_null_models.c` | geometric/binomial/NB samplers |
| `3. Model Generation/src/others_null_models.c` | radiation, sequential gravity |

#### TDD plan

1. Distribution-level property tests: PMF sums to 1, CDF monotone, mean
   matches formula, variance matches formula.
2. Fitting convergence tests: recovered constraints within tolerance for each
   family.
3. Seed reproducibility tests for each sampler.
4. Filter partition tests for each family.
5. Ensemble equivalence: geometric/binomial ensembles converge in large-T
   limit like Poisson does.

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
