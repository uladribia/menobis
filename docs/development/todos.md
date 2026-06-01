---
description: Prioritized pending work for MENoBiS.
---

# TODOs

## TL;DR

The public documentation release is complete for MENoBiS `1.0.1`. Remaining
work focuses on solver robustness, benchmark reporting, packaging, and extension
walkthroughs.

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
| W zero-inflated convergence | strength-edges and strength-degree remain experimental |
| Sparse zero-inflated regimes | ME/B can be ill-conditioned when occupations are nearly binary |
| W strength-cost damping | large no-self-loop cases can be slow or sensitive |
| Saturation handling audits | keep boundary cases explicit and tested |
| More cost providers | add Rust providers instead of dense cost matrices |
| Sparse-support fitting | support user-provided masks without dense state |

## Benchmarking backlog

| Item | Notes |
|---|---|
| Incremental persistence | long runs should save partial results |
| Chunked benchmark presets | avoid all-case timeout-prone runs |
| Local-machine report template | help users report CPU, RAM, wall time, and dataset size |
| Parallel all-pairs sweeps | improve CPU utilization where reproducibility allows |
| Better W diagnostics | expose boundary margins and stopping causes clearly |

## Engineering backlog

| Item | Notes |
|---|---|
| Reduce wrapper repetition | keep public API small; use internal factories where safe |
| More real-data examples | OpenFlights is available; add more OD datasets carefully |
| Release packaging | future PyPI wheels and crates.io publication |
| Agent extension workflow | keep docs concise and code-source-of-truth oriented |

## Integrated audit points

| Previous audit topic | Current location |
|---|---|
| model ontology | [Choose a null model](../concepts/choose-null-model.md) and [Equations](../concepts/equations.md) |
| convergence caveats | [Solvers and scaling](../concepts/solvers-and-scaling.md) and [Benchmarking](benchmarking.md) |
| sparse mask and streaming decisions | [Scalability](scalability.md) and [Extending thesis cases](extending-thesis-cases.md) |
| legacy thesis folders | modern APIs live in `src/`, `crates/`, `tests/`, and `docs/`; git history remains the archive |
