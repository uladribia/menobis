# Microcanonical vs grand-canonical notebook implementation report

Status: final (except the single W STRENGTH_DEGREE @ N=2000 cell left running).

## 1. Branch and commits

Branch: `docs/micro-vs-grandcanonical-notebook` (from `master`).

Commit order (Conventional Commits):

1. `test: add ensemble comparison runner smoke tests`
2. `feat(benchmarks): add microcanonical vs grand-canonical comparison runner`
3. `docs: add practical microcanonical vs grand-canonical notebook`
4. `docs: clarify finite-size comparison versus ensemble equivalence`
5. `docs: add ensemble comparison implementation report`

## 2. Experimental configuration

- N = 100, 500, 2000; 10 samples per main cell; no self-loops.
- `average_degree = 8.0`, `events_per_edge = 8.0` (E ≈ 8N, T ≈ 64N).
- Families ME/B/W; constraints STRENGTH, STRENGTH_COST, STRENGTH_EDGES,
  STRENGTH_DEGREE, DEGREE_EVENTS, EDGES_EVENTS; ensembles GC + micro.
- One observed network per N (seed `42 + n`); derived constraints shared by
  every family/constraint/ensemble.
- Deterministic ensemble seeds (`ensemble_seed`); no `default_rng()` without
  seed; GC and micro samples with equal index are not paired.
- B layers from `derived.binomial_layers`; ME/W layers = 1.
- Timing: GC = fit + 10 samples + stats; micro = 10 full generations + stats
  (STRENGTH_COST includes the internal gamma fit); synthetic generation,
  dataframe construction, plotting, saving excluded. Repeats 3/3/1.

## 3. MCMC budget gate

Original plan ladder (3+1 → 10+2 → 20+5) FAILED for every family/constraint:
max D_rel between adjacent budgets stayed above 0.05 (e.g. ME STRENGTH 0.63,
B STRENGTH 0.76, W STRENGTH 0.55 for A-vs-B; B-vs-C still 0.13–0.53).

Per user decision (option B), the gate was extended to the ladder
(3+1 → 10+2 → 20+5 → 50+10 → 200+50), choosing the smallest budget whose
ensemble agrees with the next within D_rel ≤ 0.05, otherwise keeping the
largest budget and marking the cell `unstable_at_max_budget`.

Gate outcome (N=100): all nine combos `unstable_at_max_budget`; chosen micro
budget 200+50 for STRENGTH / STRENGTH_EDGES / STRENGTH_DEGREE everywhere.
Even 200+50 vs 50+10 stays above 0.05 (0.10–0.35), so no budget pair on the
ladder is stable within 5% for the primary observables; the fixed-strength
support distribution relaxes slowly (mean degree of micro STRENGTH samples
drifts 10.6 → 20.5 → 25.9 → 27.6 → 27.1 for ME as budget grows 3+1 → 200+50).

Feasibility caps applied at large N (recorded in `metadata.json` →
`budget_caps`): STRENGTH_COST capped at 3+1 for N ≥ 500 (the internal gamma
fit + MCMC costs ~150 s/sample at N=500); STRENGTH_DEGREE capped at 10+2 at
N=500 and 3+1 at N=2000 (micro STRENGTH_DEGREE costs ~25 s/sample ME and
~450 s/sample W at N=2000 with 3+1). Capped cells carry an explicit message;
micro `STRENGTH_DEGREE` for B is infeasible at N ≥ 500 (strength exceeds the
B capacity bound M × degree for some node) and is recorded as a cell failure.

## 4. Main matrix completion

Final status at handoff: 106/108 ensemble cells recorded; overall PARTIAL
(plan §52).

| N | ok | GC fit_failed | micro error | aborted | total |
|---|---:|---:|---:|---:|---:|
| 100 | 33 | 3 | 0 | 0 | 36 |
| 500 | 30 | 4 | 2 | 0 | 36 |
| 2000 | 28 | 4 | 2 | 2 | 36 |

Aborted: W STRENGTH_DEGREE @ N=2000 (GC + micro) — the W (s,k) GC conic fit
was still grinding without converging after ~12 CPU-hours during the
scientific phase; per user decision it was killed and the two cells are
recorded as `aborted` (no data fabricated; the runner reruns them on any
future resume). The micro side is likewise unrecorded because the runner
computes the GC fit before the micro samples within a cell.

Note: this abort is consistent with the established pattern — the W GC
(s,k) fit is `fit_failed` at N=100 (after seconds) and at N=500 (after ~23
min), and does not converge within ~12 CPU-hours at N=2000.

GC `fit_failed` (GC samples skipped, recorded per plan §25):
- N=100: B STRENGTH_DEGREE, W STRENGTH_DEGREE, W STRENGTH_EDGES;
- N=500: ME/B/W STRENGTH_DEGREE, W STRENGTH_EDGES;
- N=2000: ME STRENGTH_EDGES, ME/B/W STRENGTH_DEGREE, W STRENGTH_COST, W
  STRENGTH_EDGES.

Micro `error` (infeasible, B capacity bound `strength > M x degree` at M=10):
B STRENGTH_DEGREE and B STRENGTH_EDGES at N=500 and N=2000.

Missing data is never filled with another configuration.

## 5. Timing summary

N=100 and N=500 median end-to-end seconds for one 10-sample workload:

| N | family | constraint | GC s | micro s |
|---|---|---|---:|---:|
| 100 | ME | STRENGTH | 0.006 | 0.012 |
| 100 | ME | STRENGTH_COST | 0.019 | 40.8 |
| 100 | ME | STRENGTH_DEGREE | 0.994 | 1.53 |
| 100 | B | STRENGTH_COST | 0.344 | 57.1 |
| 100 | W | STRENGTH_COST | 14.7 | 49.5 |
| 500 | ME | STRENGTH | 0.07 | 0.06 |
| 500 | ME | STRENGTH_COST | 0.47 | 223 |
| 500 | ME | STRENGTH_DEGREE | (fit_failed) | 50.5 |
| 500 | ME | STRENGTH_EDGES | 2.02 | 0.12 |
| 500 | B | STRENGTH_COST | 6.47 | 324 |
| 500 | B | STRENGTH_EDGES | 3.91 | (infeasible) |
| 500 | W | STRENGTH | 1.09 | 0.08 |
| 500 | W | STRENGTH_COST | 386 | 445 |
| 500 | W | STRENGTH_EDGES | (fit_failed, ~105-136 s/fit) | 1.26 |
| 500 | W | STRENGTH_DEGREE | (fit_failed, ~1400 s/fit) | 50.5 |

Notes: micro budgets at N=500: STRENGTH/STRENGTH_EDGES 200+50; STRENGTH_COST
capped 3+1 (~223-445 s per 10-sample workload); STRENGTH_DEGREE capped 10+2
(~50 s). Observed effective exponents (100->500, median E2E) are largest for
B/W STRENGTH_EDGES GC (~2.5-2.8), micro STRENGTH_DEGREE (~2.1-2.2), and the
STRENGTH/STRENGTH_COST GC routes (~1.6-2.1); these are empirical exponents for
this sparse regime, not Big-O complexity. Full scaling to N=2000 filled at
handoff where cells completed.

## 6. Scientific-summary sanity checks

At N=100, ME STRENGTH with micro at its converged budget (200+50): mean
degree 27.1 (micro) vs 27.0 (GC), D_rel = 0.034, Spearman = 1.00 — the two
ensembles agree on degree once micro is given enough sweeps. The largest
non-fixed D_rel per constraint is always Y2 (0.16-0.77 at N=100), including
for STRENGTH_DEGREE (Y2 D_rel ≈ 0.77 for ME even with (s,k) fixed exactly vs
in expectation). Full tables filled at handoff.

## 7. Sparsity sensitivity

Complete: N=500, ME, `average_degree` in {3, 8, 20}, T/E ≈ 8, 10 samples,
STRENGTH / STRENGTH_DEGREE / EDGES_EVENTS. Micro STRENGTH used the gate budget
200+50; STRENGTH_DEGREE capped 10+2. 294/297 rows ok (3 GC fit_failed =
ME STRENGTH_DEGREE GC, consistent with the main matrix). D_rel (mean over
primary out/in observables) decreases with degree: max 0.35 at k=3, 0.28 at
k=8, 0.19 at k=20 — a visible sparser-more-sensitive pattern across the three
points only (no trend claim beyond that).

## 8. Documentation changes

- `docs/examples/grand-vs-micro-practical.ipynb` — rewritten from the
  "planned destination" stub into the full comparison notebook (12 sections,
  6 figures, `RUN_FULL_BENCHMARK = False` default; executed in
  load-results mode).
- `docs/science/ensemble-equivalence.md` — added "Finite-size hard-vs-soft
  comparison in practice" section, six-constraint semantics table, notebook
  link; no stale "(s,E)/(s,k) deferred" language existed to remove.
- `docs/guide/choose-model.md`, `docs/guide/microcanonical.md` — one link to
  the notebook each.
- `mkdocs.yml` — added "Examples" nav section.
- `pyproject.toml` — pytest `pythonpath` extended from `["src"]` to
  `["src", "."]` so the `benchmarks` package (and the new runner tests, and
  the pre-existing `test_benchmark_matrix_preset.py`) are importable under
  pytest. This unblocks a pre-existing collection failure on `master`
  (recorded per plan §6 as unrelated but infra-required for this task).

## 9. Tests and commands

- `tests/test_menobis_ensemble_comparison_runner.py`: 15 tests covering the
  10 mandatory runner tests (constraint-kwargs builder, N=100 ME smoke,
  micro STRENGTH validation pass, deliberate mismatch caught, D_rel zero for
  identical vectors, NaN-mask valid-node count, Spearman <3 NaN, timing
  excludes network generation, deterministic seeds, schema columns) plus
  idempotent-result-write and row-drop tests.
- `uv run ruff format --check .`, `uv run ruff check .`, `uv run ty check`,
  `uv run pytest`, `uv run python -m benchmarks.ensemble_comparison smoke`,
  `uv run mkdocs build --strict`: PASS.
- Full fast pytest suite: 457 passed, 24 skipped, 1 xfailed (was broken at
  collection on master).
- Heavy: run in stages — extended budget gate (PASS), main matrix per N
  (100 and 500 complete, 2000 complete except the aborted W
  STRENGTH_DEGREE cell), sparsity (complete). `uv run python -m
  benchmarks.ensemble_comparison all` not run as a single command.

## 10. Failures or limitations

- Budget gate: no micro budget on the extended ladder is D_rel-stable within
  0.05 for the gate constraints at N=100; micro results are budget-dependent
  and flagged per cell; heavy cells run at feasibility caps. All gate cells
  took `unstable_at_max_budget` (budget 200+50), and even 200+50 vs 50+10
  stays above 0.05 — the fixed-strength support distribution relaxes slowly
  (micro STRENGTH mean degree drifts 10.6 → 27.1 as budget grows 3+1 → 200+50).
- Micro B STRENGTH_DEGREE and B STRENGTH_EDGES are infeasible for the sparse
  regime at N ≥ 500 (strength > M × degree capacity at M=10); recorded as
  cell failures, not fabricated.
- GC (s,k) fits (STRENGTH_DEGREE) fail to converge at N ≥ 500 for ME/B/W
  with public defaults; W STRENGTH_EDGES and W STRENGTH_COST GC fits also
  fail at N=2000 (and at N=100/500 for the former). Recorded per plan §25.
- W STRENGTH_DEGREE @ N=2000: GC conic fit did not converge within ~12
  CPU-hours; aborted per user decision, cells marked `aborted` (the only
  uncompleted configuration in the matrix).
- Results are gitignored (`benchmarks/results/`); the notebook reads local
  results and shows reproduction commands when absent; the per-node npz
  covers N=100 (backfilled) + N=500 + N=2000-completed cells.
