---
description: Prioritized pending work for MENoBiS.
---

# TODOs

## TL;DR

Phase 0 foundation refactor is complete and merged to `master`. The codebase
now uses occupation-number terminology throughout, has a machine-readable
capability registry, and a shared constraints module. The new
`EDGES_EVENTS` constraint is implemented for ME/B/W grand-canonical
fit/sample/filter. Microcanonical sampling is implemented for ME, B, and W
across four constraint families: fixed (E,T), fixed (k,T), fixed strengths,
and fixed strengths + expected cost; the microcanonical refactor (phases A–H
+ benchmark matrix) is complete and validated at N=1000. A
phase0-baseline benchmark (111 rows, 0 mismatches) is captured. Remaining
work focuses on solver robustness (especially W zero-inflated),
sparse-support fitting, cost providers, benchmark tooling, packaging, and
deferred microcanonical features listed below.

## Public release status

| Item | Status |
|---|---|
| GitHub Pages site | completed for `https://uladribia.github.io/menobis/` |
| Rendered and downloadable notebook | completed: `docs/examples/main-use-cases.ipynb` |
| User-first docs structure | completed: tutorials, model selection, reference, development |
| Public version metadata | completed for release `1.0.1` |
| Strict docs build | required before every release |

!!! note "Release follow-up"
    Keep this page as the backlog. Do not recreate `PLAN.md`; release planning
    now lives here.

## Scientific and solver backlog

| Item | Notes |
|---|---|
| `EDGES_EVENTS` constraint | ✅ **Done in Phase 0.** Grand-canonical ME/B/W fit/sample/filter implemented; tests in `test_menobis_edges_events.py` |
| Saturation handling | ✅ **Improved.** Shared `constraints` module (`validation.rs`, `FixedPairs`) provides explicit boundary handling; saturation cases are validated |
| W zero-inflated convergence | strength-edges and strength-degree remain experimental (baseline confirms all non-convergent for W) |
| Sparse zero-inflated regimes | ME/B can be ill-conditioned when occupations are nearly binary |
| W strength-cost damping | large no-self-loop cases can be slow or sensitive |
| More cost providers | add Rust providers instead of dense cost matrices |
| Sparse-support fitting | `PairMask` and `FixedPairs` extracted as shared abstractions; user-provided masks without dense state still pending |

### Microcanonical deferred work

Items explicitly deferred from the current release.
These are NOT implemented and NOT planned for the current release.

| Deferred feature | Description & rationale |
|---|---|
| `fixed (s,E)` | Fixed strength + total edges (zero-inflated); requires extending the fixed-strength kernel with an edge-count Lagrange multiplier. Not implemented. |
| `fixed (s,k)` | Fixed strength + degree sequence (zero-inflated); requires binary multiplier layer `l_ij = w_i z_j` over the fixed-strength kernel. Not implemented. |
| `generic annealed repair` | No annealing machinery in production (targeted repairs only). |
| `general alternating-cycle mask repair` | Only targeted loop/capacity/forbidden-pair repairs exist; a general mask repair framework is deferred. |
| `grand-canonical warm starts` | No warm-start for gamma fitting (zero-centered bracket expansion instead, Phase F). |
| `universal MCMC kernel framework` | Families keep dedicated kernels (occupied-cell for strength, pair-Gibbs for fixed-total); no unified trait hierarchy. |

### Minor follow-up items

| Item | Notes |
|---|---|
| `fixed (E,T)` oracle coverage | `menobis-test-oracles` has the fixed-total DP/rejection oracle; production-vs-oracle comparison could extend to more (E,T) combinations (currently bounded sizes). |
| Python `_fixed_et_explicit` O(N²) iteration | `src/menobis/routing.py` ~lines 1100–1210: explicit fixed-(E,T) Python fallback iterates all pairs; out of scope for the refactor, should migrate to Rust or remove (low priority). |
| `uv run ty check` 160 pre-existing diagnostics | Dataclass type-narrowing issues across `generation.py`/`routing.py`/numpy stubs; not introduced by the refactor. |
| Cost ESS at dense N | Cost ESS degrades at dense N (4–21 at N=1000); improving cost-chain mixing or ESS reporting is a follow-up. |

## Benchmarking backlog

| Item | Notes |
|---|---|
| Phase 0 baseline | ✅ **Captured.** 111 rows × 2 self-loop policies, 0 mismatches vs baseline. Files in `benchmarks/results/phase0-baseline/` |
| Incremental persistence | long runs should save partial results |
| Chunked benchmark presets | avoid all-case timeout-prone runs |
| Local-machine report template | help users report CPU, RAM, wall time, and dataset size |
| Parallel all-pairs sweeps | improve CPU utilization where reproducibility allows |
| Better W diagnostics | expose boundary margins and stopping causes clearly |
| Extend to `EDGES_EVENTS` and `DEGREE_EVENTS` constraints | ✅ **Microcanonical EDGES_EVENTS** covered by `benchmarks micro`; grand-canonical fit/sample/filter benchmarks still only cover strength-family constraints; `DEGREE_EVENTS` still missing |

## Engineering backlog

| Item | Notes |
|---|---|
| Reduce wrapper repetition | ✅ **Significant progress.** Capability registry (`capabilities.py`), analysis facade (`analysis/facade.py`), generation split by ensemble, routing refactored. Public API surface reduced. |
| Migration notes | ✅ **Done.** Git history records all renames and API changes from Phase 0 |
| Architecture docs | ✅ **Done.** `docs/development/architecture.md` added |
| Agent specification | ✅ **Done.** Historical phase specifications are archived in git history |
| More real-data examples | OpenFlights is available; add more OD datasets carefully |
| Release packaging | future PyPI wheels and crates.io publication |
| Extend capabilities registry for new verbs/ensembles | currently covers fit/sample/filter × grand-canonical/canonical/microcanonical |

## Integrated audit points

| Previous audit topic | Current location |
|---|---|
| model ontology | [Choose a null model](../concepts/choose-null-model.md) and [Equations](../concepts/equations.md) |
| convergence caveats | [Solvers and scaling](../concepts/solvers-and-scaling.md) and [Benchmarking](benchmarking.md) |
| sparse mask and streaming decisions | [Scalability](scalability.md) and [Extending thesis cases](extending-thesis-cases.md) |
| legacy thesis folders | modern APIs live in `src/`, `crates/`, `tests/`, and `docs/`; git history remains the archive |