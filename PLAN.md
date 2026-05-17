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

Replace the old C/Python code (`1. Network analysis/`, `2. Model Fitting/`,
`3. Model Generation/`) once ODME covers the full scientific scope. The legacy
code is reference material, not a compatibility target.
