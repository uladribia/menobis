---
description: Prioritized pending work for MENoBiS.
---

# TODOs

## TL;DR

Phase 0 foundation refactor is complete and merged to `master`. The codebase
now uses occupation-number terminology throughout, has a machine-readable
capability registry, and a shared constraints module. The new
`EDGES_EVENTS` constraint is implemented for ME/B/W grand-canonical
fit/sample/filter. Microcanonical fixed-(E,T) samplers for ME, B, and W are
implemented (shared `FixedETOccupancy` architecture, rejection + DP
backends, fixed-pair preprocessing, conditioned grand-canonical validation,
`benchmarks micro` command). A phase0-baseline benchmark (111 rows,
0 mismatches) is captured. Remaining work focuses on solver robustness
(especially W zero-inflated), sparse-support fitting, cost providers,
benchmark tooling, packaging, and the next microcanonical phases.

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

## Microcanonical backlog

### Done (fixed-(E,T))

| Item | Notes |
|---|---|
| ME fixed-(E,T) | ✅ exact sampler: multinomial rejection + Stirling surjection fallback |
| B fixed-(E,T) | ✅ cell-subset rejection (complement mode) + bounded composition DP |
| W fixed-(E,T) | ✅ weak-composition (stars-and-bars) rejection + unbounded composition DP |
| Shared architecture | ✅ `FixedETOccupancy` trait drives one generic orchestrator; family files under `generation/microcanonical/fixed_et/` |
| Fixed-pair preprocessing | ✅ residualisation + merge for all three families; B fixed occupations validated against M |
| Validation | ✅ exact enumeration, E2E dense/sparse with/without fixed pairs, conditioned grand-canonical identity `P_GC(t|E,T)=P_MC(t|E,T)` |
| Benchmarks | ✅ `python -m benchmarks micro` (ME/B/W × sparse/dense × known pairs) |
| Docs | ✅ `docs/concepts/microcanonical.md`, API reference, spec docs in `docs/development/agent-specifications/` |

### Pending — reasonable-effort follow-ups

| Item | Notes |
|---|---|
| Backend diagnostics | expose selected backend, rejection attempts, estimated rejection (spec 02 §4, 03 §24, 04 §26) — currently hidden behind the public `sample_model` API |
| DP table caching | reuse `(E,T,M)` partition tables across calls (spec 03 §30, 04 §35) — optional, bounded |
| Estimated vs observed rejection | benchmark should compare estimate to actual rejection rate to decide whether the selector threshold needs tuning |
| W fallback performance | the exact DP is O(ET²); optimise via convolution, W-specific recurrences, or saddle-point proposals for larger systems (spec 04 §23.2) |
| W large-dense hard regime | e.g. E=2000, T=16000, M=8: p_acc ≈ 4×10⁻⁴ exceeds the scaled-rejection work budget → clean error today; needs a better backend |
| Conditioned-GC validation breadth | add larger systems and near-boundary regimes to `tests/test_menobis_conditioned_grandcanonical_identity.py` |
| Observable convergence across families/constraints | test the convergence of network observables — y2 (second occupation moment), average weighted neighbour strength, leading entropy per event, and related quantities — across all families (ME/B/W) and all constraints (strength, strength-cost, strength-edges, strength-degree, degree-events, edges-events), for grand-canonical vs canonical vs microcanonical ensembles. Validate ensemble equivalence where the theory predicts it and document deviations (sparse limits, W convergence boundary, B saturation). See `docs/development/agent-specifications/00_intro.md` §13 |

### Pending — next phases (roadmap, `00_intro.md` §2)

| Phase | Constraint | Notes |
|---|---|---|
| 3 | fixed (k,T) | degree sequence + total events; needs degree-residual machinery from `constraints/` |
| 4 | fixed strengths | exact strength sequences (microcanonical) |
| 5 | fixed strengths + expected cost | strengths hard, cost expected |
| 6 | fixed (s,E) | strengths + occupied-pair count |
| 7 | fixed (s,k) | strengths + degrees |
| 8 | advanced backends | backbone samplers, pseudo-marginal methods, MCMC kernels |

Phase 3 (fixed k,T) is the natural next implementation; see
`docs/development/agent-specifications/05_microcanonical_sampling_framework_fixed_se_plan.md`
for the general framework.

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
| Migration notes | ✅ **Done.** `docs/development/migration-notes.md` documents all renames and API changes from Phase 0 |
| Architecture docs | ✅ **Done.** `docs/development/architecture.md` added |
| Agent specification | ✅ **Done.** `docs/development/agent-specifications/microcanonical-phase-0.md` added as comprehensive specification |
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
