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
+ §35 benchmark matrix) is complete and validated at N=1000. A
phase0-baseline benchmark (111 rows, 0 mismatches) is captured. Remaining
work focuses on solver robustness (especially W zero-inflated),
sparse-support fitting, cost providers, benchmark tooling, and packaging.

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
