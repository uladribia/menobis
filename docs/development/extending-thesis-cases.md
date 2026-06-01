---
description: How to extend MENoBiS with new thesis cases.
---

# Extending thesis cases

## TL;DR

Add new cases as family kernels plus constraint layers in Rust. Python should
validate inputs, call Rust, and return typed numpy/dataclass results.

## Required design rules

| Rule | Consequence |
|---|---|
| One ontology | route by `Ensemble`, `ModelFamily`, `Constraint` |
| Family separation | ME, B, and W solvers must implement their own expectation equations |
| Shared infrastructure | reuse masks, cost providers, residual checks, providers |
| Sparse first | do not introduce dense `N x N` public inputs |
| Partial is not a family | subtract known pairs, then call the matching full solver |
| Rust owns kernels | no heavy graph or solver loops in Python |

## Add a new family/constraint route

1. Write the thesis equation in a Rust code comment.
2. Add or reuse a family kernel for `E[t_ij]` and, if needed,
   `E[Theta(t_ij>0)]`.
3. Add a fitting solver with diagnostics and residual checks.
4. Add a `PairDistributionProvider` for sampling and filtering.
5. Add PyO3 bindings in the domain file.
6. Add Python router dispatch in `menobis.routing`.
7. Add tests using the generate → derive constraints → fit → sample → check
   pipeline.
8. Add docs that map public names back to thesis terminology.

## Add a cost provider

Cost providers are Rust traits that return a pair cost on demand. To support a
new metric:

1. Add a provider struct in `menobis-core` that stores only O(N) or sparse state.
2. Implement the pair-cost method for `(source, target)`.
3. Wire it into fitting, generation, and filtering providers.
4. Expose only typed Python inputs; do not expose dense `N x N` matrices.
5. Add tests comparing a small reference case with hand-computed costs.

!!! warning "No dense public costs"
    Pair metrics can be expensive, but public APIs should not require users to
    allocate all pair costs in Python.

## Tests to add

| Test | Purpose |
|---|---|
| formula unit test | check implemented expectation against dense reference |
| feasibility validation | reject impossible constraints early |
| constraint recovery | fitted expectations reproduce inputs |
| family comparison | ME, B, W differ when formulas differ |
| sampling invariant | sampled weights are non-negative integers |
| CLI/API smoke | public route works and errors are useful |

## Agent workflow

Use a dedicated branch and keep the red/green step small. If changing CLI output
or flags, apply the CLI guidelines. If changing docs, keep pages brief and run
`uv run mkdocs build --strict`.
