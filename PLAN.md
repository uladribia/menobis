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

### Milestone 7: Additional generation kernels — ❌ NOT STARTED

Implement deterministic seeded samplers for distribution variants not yet
covered: geometric, binomial, negative binomial, and zero-inflated variants
(W/AB/AW cases from the thesis).

Add radiation and sequential-gravity mobility models only after the
maximum-entropy core is stable.

TDD: distribution-level property tests, seed reproducibility, invariants on
generated graphs.

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

The three legacy directories can be removed once Milestone 7 is complete.
Until then, they serve as reference for unported distribution variants.

#### Inventory: what is ported vs what remains

**`1. Network analysis/`** — C tool `MultiEdgeAnalyzer`. **Fully replaced.**
All statistics (strengths, degrees, Y2, k_nn, s_nn, P(w), clustering) are
implemented in Rust `odme-core` with Python wrappers. Safe to remove.

**`2. Model Fitting/`** — Python 2 `multi_edge_fitter`. **Partially replaced.**

| Legacy fitter | ODME equivalent | Status |
|---------------|-----------------|--------|
| `fitter_s.py` case `ME` | `fit_fixed_strength_me` | ✅ ported |
| `fitter_s.py` case `B` (binomial, M layers) | — | ❌ Milestone 7 |
| `fitter_s.py` case `W` (geometric) | — | ❌ Milestone 7 |
| `fitter_k.py` | `fit_fixed_degree_binary` | ✅ ported |
| `fitter_sk.py` (ME mode) | `fit_strength_degree_me` | ✅ ported |
| `fitter_sk.py` (`agg=True`, M layers) | — | ❌ Milestone 7 |
| `fitter_E.py` (ME mode) | `fit_strength_edges_me` | ✅ ported |
| `fitter_E.py` (`agg=True`, M layers) | — | ❌ Milestone 7 |
| `fitter_grav.py` | `fit_strength_cost_me` | ✅ ported |
| `fitter_pij.py` | partial fitting (`fit_partial_*`) | ✅ ported |
| `fitter_s_CVXOPT.py` | not needed (IPF solver used) | ✅ superseded |

Keep until Milestone 7 ports the binomial (`B`) and geometric (`W`) fitting
variants, including the aggregated-layers (`agg=True`, `M`) parameter.

**`3. Model Generation/`** — C tool `GenNetGen`. **Partially replaced.**

| Legacy generator | ODME equivalent | Status |
|------------------|-----------------|--------|
| `fixeds_poisson_*` | `sample_poisson` | ✅ ported |
| `fixeds_computational_*` | `sample_microcanonical` | ✅ ported |
| `multinomial_*` | `sample_multinomial` | ✅ ported |
| `poisson_multinomial_*` | `sample_poisson_multinomial` | ✅ ported |
| `custompij_poisson_*` | `sample_custom_pij_events_poisson` | ✅ ported |
| `custompij_ZIP_*` | `sample_strength_edges_me` | ✅ ported |
| `custompij_ZIG_*` | — | ❌ Milestone 7 (geometric ZIP) |
| `fixeds_geometric_*` | — | ❌ Milestone 7 |
| `fixeds_binomial_*` | — | ❌ Milestone 7 |
| `fixeds_negbinomial_*` | — | ❌ Milestone 7 |
| `fixedk_poisson_*` | `sample_fixed_degree_events_me` | ✅ ported |
| `fixedk_geometric_*` | — | ❌ Milestone 7 |
| `fixedk_negbinom_*` | — | ❌ Milestone 7 |
| `fixedk_binom_*` | — | ❌ Milestone 7 |
| `fixedk_bernouilli_*` | `fit_fixed_degree_binary` (topology only) | ✅ ported |
| `fixedEs_poisson_*` | `sample_strength_edges_me` | ✅ ported |
| `fixedks_poisson_*` | `sample_strength_degree_me` | ✅ ported |
| `fixedks_binomial_*` | — | ❌ Milestone 7 |
| `gravity_poisson_*` | `sample_strength_cost_me` | ✅ ported |
| `w_graph_seq_gravity_*` | — | ❌ Milestone 7 (sequential gravity) |
| `w_graph_radiation_*` | — | ❌ Milestone 7 (radiation model) |

#### Removal plan

**Phase 1 (safe now):** Remove `1. Network analysis/` only. Fully superseded.

**Phase 2 (after Milestone 7):** Remove `2. Model Fitting/` and
`3. Model Generation/`. Both contain unported distribution variants:

- `fitter_s.py` cases `B` and `W`: binomial/geometric IPF with layers parameter
- `fitter_sk.py` and `fitter_E.py` with `agg=True`: aggregated-layer variants
- `ula_null_models.c` lines 30–70: geometric/binomial/NB pair-level samplers
- `others_null_models.c`: radiation and sequential-gravity models

Do not remove Phase 2 files until Milestone 7 tests prove equivalence.
