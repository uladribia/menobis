---
description: How to extend MENoBiS — grand-canonical soft-constraint routes and microcanonical hard-constraint routes.
---

# Extending MENoBiS

## TL;DR

New cases are family kernels plus constraint layers in Rust: grand-canonical
(GC) routes need fitted multipliers; microcanonical (MC) routes need a
constructor, an initialization repair that stays on the constraint fiber,
and a sampling kernel with the correct stationary law. Python validates
inputs, calls Rust, and returns typed numpy/dataclass results.

## Required design rules

| Rule | Consequence |
|---|---|
| One ontology | route by `Ensemble`, `ModelFamily`, `Constraint` |
| Family separation | ME, B, and W solvers must implement their own expectation equations |
| Shared infrastructure | reuse masks, cost providers, residual checks, providers |
| Sparse first | do not introduce dense `N x N` public inputs |
| Partial is not a family | subtract known pairs, then call the matching full solver |
| Rust owns kernels | no heavy graph or solver loops in Python |
| Exactness is a sampler property | for MC, the sampling kernel defines exactness, not the constructor |
| MC construction needs no reversibility | only the sampler must be correct |

## Add a grand-canonical route

GC routes match constraints **in expectation** through fitted multipliers:

1. Write the thesis equation in a Rust code comment.
2. Add or reuse a family kernel for `E[t_ij]` and, if needed,
   `E[Theta(t_ij>0)]` (the zero-inflated support layer).
3. Add a fitting solver with diagnostics and residual checks.
4. Add a `PairDistributionProvider` for sampling and filtering.
5. Add PyO3 bindings in the domain file.
6. Add Python router dispatch in `menobis.routing`.
7. Add tests using the generate → derive constraints → fit → sample → check
   pipeline (expected-value recovery within tolerance).
8. Add docs that map public names back to thesis terminology.

Sampling a fitted GC model is exact independent per-pair drawing; the
fitted expectations are the constraint.

## Add a microcanonical route

MC routes fix constraints **exactly** on a fiber. Every route follows the
same three-part pattern:

1. **Generate** — construct one feasible state on the constraint fiber.
2. **Repair** — where the constructor cannot land exactly, drive the state
   back onto the fiber (e.g. the exact-\(E\) repair). Repair is biased and
   **initialization-only**; it never enters the stationary kernel.
3. **Iterate** — run a kernel whose stationary law is exactly the target
   constrained measure; thin and diagnose as with any MCMC
   ([MCMC diagnostics](../performance/mcmc-diagnostics.md)).

Construction need not be reversible, Markovian, or part of the sampling
law at all: sampling exactness is a property of the kernel
([Validation](../performance/validation.md)). Router overview:
[Microcanonical algorithms](microcanonical-algorithms.md).

### Route families in the code

| Plan | Routes | Construction | Sampler |
|---|---|---|---|
| factorized | `(E,T)`, `(k,T)` | exact-degree/edge support + occupation allocation | support stage + fixed-total pair-Gibbs chain |
| direct conditional | `(E,T)` | — | direct combinatorial draws (ME/B/W) |
| coupled occupation MCMC | `s`, `(s,E)`, `(s,k)` | compressed fixed-strength state; extras-first for `(s,k)` | occupied-cell 4-cycle chain; + censored bridge for `(s,E)`; capped first-return degree trace for `(s,k)` |

Dispatch happens in `SamplingPlan::classify` and
`generation/microcanonical/route.rs`.

### Steps

1. **Write the target law in a comment.** For the fiber(s) you fix:

   \[
   \pi(t)\propto \left[\prod_{ij}d_F(t_{ij})\right]
   \mathbf 1[C(t)=C^\star],
   \]

   with the same family degeneracies used everywhere else
   ([Event families](../science/event-families.md)).

2. **Classify the plan.** Choose factorized vs coupled vs direct conditional
   and add the dispatch branch in `route.rs`.

3. **Constructor.** Produce one feasible state on the fiber (compressed,
   extras-first, or a direct conditional allocation). Keep `O(N)` or sparse
   memory, never dense `N x N`; validate feasibility before allocating and
   report exhaustion as a structured error, distinct from genuine
   infeasibility ([Feasibility](../science/constraints.md#feasibility)).

4. **Repair.** If the constructor cannot land exactly (e.g. exact-\(E\)),
   add a biased initialization-only repair with bounded retries/restarts.
   Keep the bias out of the stationary kernel and document exhaustion
   semantics.

5. **Sampler.** Provide a kernel with the target as its stationary law:
   occupied-cell 4-cycle chain, local exact-\(E\) kernel + censored bridge,
   capped first-return trace, or fixed-total pair-Gibbs chain. Prove
   reversibility/detailed balance (Metropolis kernels) or the
   conditioning/tracing identity (trace kernels).

6. **Exactness label.** Classify as
   *exact direct* (one draw from the target law, up to ordinary
   pseudorandom error), *exact stationary MCMC* (kernel law exact; finite-run
   burn-in and mixing still matter), or *hybrid* (some quantities exact,
   others expected — e.g. strength+cost). The label goes in the capability
   registry and on `sample_model_detailed(...).diagnostics.exactness`.

7. **Diagnostics.** Expose what mixing you measure: acceptance,
   effective movement, support change rate, ESS of reported statistics,
   repeated-chain agreement; cost ESS for cost-influenced routes.

8. **Wire and document.** PyO3 binding + type stub, Python router dispatch,
   capability registry entry, CLI if appropriate, one tested user example
   (constraints derived from a witness network), smoke test, and docs.

### Feasibility for MC constraints

- \(s_i^{\mathrm{out}}\ge k_i^{\mathrm{out}}\), \(s_i^{\mathrm{in}}\ge k_i^{\mathrm{in}}\);
- B capacity \(s_i\le M k_i\); for B \(M=1\) (Bernoulli) \(s=k\);
- \((E,T)\): \(E\le T\), and for B \(T\le ME\);
- no-self-loop bounds \(k_i\le N-1\).

These are necessary; sparse-domain instances may need additional
conditions — derive constraints from a witness network in tests and examples.

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
| constraint recovery | fitted expectations reproduce inputs (GC) |
| tiny-fiber oracle | MC enumerated target: row sums, detailed balance, \(\pi P=\pi\) |
| transition-matrix oracle | MC trace kernels: exact \(Q/R\) on enumerated tiny fibers |
| exact recovery | MC constraints reproduced realization by realization (E2E) |
| hybrid cost check | strength+cost: strengths exact, expected cost within tolerance |
| family comparison | ME, B, W differ when formulas differ |
| sampling invariant | sampled weights are non-negative integers |
| CLI/API smoke | public route works and errors are useful |

## Agent workflow

Use a dedicated branch and keep the red/green step small. If changing CLI output
or flags, apply the CLI guidelines. If changing docs, keep pages brief and run
`uv run mkdocs build --strict`.

## Contributor documentation policy

Keep the public documentation and the capability registry in sync:

- **Changing the capability registry** — regenerate the support tables and
  run the drift check:

  ```bash
  uv run python scripts/docs/generate_capabilities.py
  uv run python scripts/docs/generate_capabilities.py --check
  ```

- **Changing a public signature** — update the documentation examples and
  their smoke tests (`tests/test_docs_examples.py`) in the same commit.
- **Adding a family/constraint** — update the scientific model page
  (`science/event-families.md`, `science/constraints.md`), the generated
  capabilities update automatically, add one tested user example, add
  validation evidence, and update performance context.
- **Adding a sampler** — document the exactness classification
  (exact direct / exact stationary MCMC / hybrid), feasibility,
  diagnostics, and tested scales.

`tests/test_public_docs_contract.py` enforces that public pages never claim
stale capabilities; include it in your red/green loop when touching docs.