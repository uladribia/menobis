# ODME modernization plan

Scientific reference: <https://hdl.handle.net/10803/400560>.

## Done

- [x] Rust + Python scaffold with PyO3/maturin
- [x] Edge-list data model, I/O, analysis
- [x] ME/B/W fitting for all constraint types
- [x] Generation and filtering for ME/B/W
- [x] Partial known-weight fitting with family-specific formulas
- [x] Unified `coord_distance`, bisection gamma, O(K) sparse IPF
- [x] Removed Clarabel/cvxrust — all W uses Newton solver
- [x] MkDocs documentation site
- [x] E2E pipeline tests (270 pass, 0 fail)
- [x] B feasibility validation (`max_s <= M*(N-1)`)
- [x] W Newton solver rewrite (bisection+adaptive damping)
- [x] Degree-events boundary fix (clip to `n-2`)
- [x] W diagnostics bug fix (probability→log-space conversion)

## Remaining detected problems

### P1. W/Wnb Newton solver fails at large N (strength AND strength-cost)

**Symptom:** W/Wnb fitting exhausts iterations at N≥500 for strength-only
and at N≥100 for strength-cost on realistic gravity-model inputs.

**Metrics (strength-only, self_loops=False):**

| N | W converged | iters | fit_residual | W tol |
|----:|:-----------:|------:|-------------:|------:|
| 25 | ✓ | 10 | 11.5 | 12.2 |
| 50 | ✓ | 12 | 13.2 | 18.0 |
| 100 | ✓ | 9 | 73.2 | 163.7 |
| 500 | ✗ | 50000 | — | 2720 |

**Metrics (strength-cost coordinates, self_loops=False):**

| N | W converged | iters | time |
|----:|:-----------:|------:|-----:|
| 25 | ✓ | 4510 | 0.12s |
| 50 | ✓ | 9824 | 1.7s |
| 100 | ✗ | 24009 | 8.5s |
| 500 | ✗ | 70000 | 98s |

**Note:** W single-sample errors (100–5000) are NOT bugs. The geometric
distribution has variance q/(1-q)² which is enormous when q→1. Ensemble
verification confirms correct mean recovery for converged fits.

**Ideas to fix:**
1. **Anderson/SQUAREM acceleration** on the (a,b) fixed-point iteration.
   Store last 3-5 iterates, extrapolate. Standard technique for Sinkhorn.
2. **L-BFGS on the dual**: reformulate as minimization of the W dual
   function and use scipy-style L-BFGS. The dual is convex and smooth.
3. **Block Newton**: update all `a` simultaneously using a dense Jacobian
   solve (O(N³) per step but converges in ~10 steps instead of thousands).
4. **Relaxed tolerance strategy**: start with loose tolerance (100×), then
   tighten using the previous solution as warm start. Avoids stalling.

### P2. B strength-cost coordinate fitting at N≥50

**Symptom:** `fit_strength_cost_binomial_coordinates` hits `max_iterations`
at N≥50 without converging.

**Metrics:**
- N=25: converges (10 iters)
- N=50: does NOT converge (5000 iters)
- N=100+: does NOT converge

**Root cause:** The B cost-coordinate solver uses the same IPF as B strength
but with a gamma-modulated rate. The same near-saturation conditioning
applies: when some pairs have `x_i * y_j * exp(-gamma*d)` close to 1,
the IPF oscillates.

**Note on B strength-only:** B strength DOES converge at all sizes tested
(N=25–1000) with adequate layers (`4*ceil(max_s/(N-1))`):

| N | B converged | iters | layers used |
|----:|:-----------:|------:|------------:|
| 25 | ✓ | 32 | 104 |
| 50 | ✓ | 9 | 76 |
| 100 | ✓ | 6 | 332 |
| 500 | ✓ | 12 | 1092 |
| 1000 | ✓ | 10 | 2476 |

**Ideas to fix:**
1. Same as P1: Anderson acceleration on the inner IPF.
2. Use the B strength-edges solver as a reference implementation (it
   converges via bisection over lambda). Adapt the same bisection strategy
   for gamma.
3. Log-space parameterization of the B IPF multipliers.

### P3. Partial W/Wnb cost-coord never converges

**Symptom:** `fit_partial_strength_cost_geometric_coordinates` and the Wnb
variant fail at all sizes tested (N=25–500).

**Metrics:** Reports 50-80 iterations but converged=False.

**Root cause:** The partial solver computes excess constraints and calls the
W cost-coord solver on the excess. But:
- The excess problem is poorly conditioned (some nodes have near-zero excess)
- The W solver itself (P1) fails on the resulting ill-conditioned problem
- The 50-80 iterations are bisection steps, each containing many inner Newton
  iterations that don't converge

**Ideas to fix:**
1. Fix P1 first — the partial solver inherits the underlying W convergence.
2. Regularize the excess: set a minimum excess per node (e.g. 1.0) to avoid
   near-zero targets that destabilize the solver.
3. For the Wnb case specifically: check if the layers parameter creates
   additional saturation issues on the reduced problem.

### P4. Sample check failures (△) are stochastic, not fitting bugs

**Observation:** W/Wnb models show large single-sample errors even when
fitting converges correctly. This is expected behavior:
- W geometric per-pair variance = q/(1-q)² (diverges as q→1)
- For high-strength nodes, q is close to 1, producing extreme variance
- Ensemble verification (50 samples) confirms correct mean recovery

**Recommendation:** Replace single-sample checks with ensemble z-scores.
For benchmarks, report fit residual (`max_strength_residual`) not sample error.

### P5. B strength-edges and B strength-degree use WRONG KERNEL (critical)

**Symptom:** `fit_strength_edges_binomial` and `fit_strength_degree_binomial`
fit correctly (ME) but sampling produces completely wrong strengths
(err=2638 at N=50 for B edges, err=2888 for B degree).

**Root cause:** Both Python functions call the **ME Poisson** Rust kernel:
```python
# In fit_strength_edges_binomial:
x_list, y_list, lam, converged, iters = _odme.fit_strength_edges_poisson(...)  # BUG!

# In fit_strength_degree_binomial:
_odme.fit_strength_degree_poisson(...)  # BUG!
```

No B-specific Rust kernel exists for these constraint types. The fit
"converges" because it runs the ME solver, but then the B sampler applies
`Binomial(M, p/(1+p))` to ME-fitted parameters → nonsensical results.

**Impact:** Any user calling `fit_strength_edges_binomial` or
`fit_strength_degree_binomial` gets silently wrong results.

**Fix required:**
1. Implement B-specific Rust kernels for strength-edges and strength-degree.
   These should use the same bisection-over-lambda/IPF structure as
   `fit_strength_edges_geometric` but with the B formula:
   `E[t_ij] = M * x_i*y_j*lam / (1 + x_i*y_j*lam)` (ZIP-Binomial).
2. The W implementations (`fit_strength_edges_geometric`,
   `fit_strength_degree_geometric`) in `crates/odme-core/src/fitting/w.rs`
   ARE correctly implemented with their own kernels and can serve as
   templates for the B versions.
3. Until fixed: either remove the B strength-edges/degree functions or
   clearly document them as unimplemented / raise NotImplementedError.

## Immediate next steps

### 8. Consolidate benchmark folder into single CLI tool

**Problem:** The `benchmarks/` folder has 7 scripts (3k lines) with
overlapping functionality, inconsistent APIs, and no unified CLI. Results
are scattered and hard to reproduce.

**Target:** One canonical benchmark CLI (`odme bench`) following the E2E
testing pattern, with stages that can run independently or together, and
structured logging throughout.

#### Design

**CLI structure** (Typer, following `/skill:create-cli` patterns):

```
odme bench [OPTIONS] [STAGE]

Stages (run in order, or specify one):
  generate    Generate test networks and derive constraints
  fit         Fit all models on derived constraints
  sample      Sample from fitted models
  check       Verify sampled networks against constraints
  filter      Apply significance filtering to samples

Options:
  --nodes TEXT       Comma-separated sizes [default: 25,50,100,500,1000]
  --families TEXT    Comma-separated families [default: me,b,w,wnb]
  --constraints TEXT Comma-separated types [default: strength,cost,edges,degree,degree-events]
  --partial / --no-partial  Include partial fitting [default: true]
  --known-fraction FLOAT    Fraction of pairs known for partial [default: 0.15]
  --seed INT         Base seed for generation [default: 10000]
  --tolerance FLOAT  Fitting tolerance factor [default: 0.02]
  --max-iterations INT  Max fitting iterations [default: 50000]
  --output DIR       Results directory [default: benchmarks/results]
  --json / --no-json JSON output to stdout [default: false]
  --quiet            Suppress progress, only errors
  --verbose          Extra detail per case
```

#### Stage design

**Stage 1: `generate`**
- For each N in `--nodes`:
  - Generate gravity-model network with coordinates
  - Derive all constraint sequences (strength, degree, cost, edges)
  - Validate feasibility: B layers computed, degrees clipped to n-2
  - Save to `{output}/networks/n{N}.npz` (weights, cx, cy, constraints)
- Log per-network: N, total_T, max_s, density, total_cost, b_layers
- Log overall: network count, total generation time

**Stage 2: `fit`**
- Load networks from Stage 1
- For each (N, family, constraint) combination:
  - Skip if infeasible (B with max_s > M*(N-1), etc.)
  - Run fitting with `--max-iterations` and `--tolerance`
  - Save fit result to `{output}/fits/n{N}_{family}_{constraint}.npz`
  - Log per-case: name, converged, iterations, seconds, max_residual
- For partial cases (if `--partial`):
  - Select `--known-fraction` of pairs (highest weight, non-diagonal)
  - Run partial fitting
  - Save results
- Log summary table after each N

**Stage 3: `sample`**
- Load fit results from Stage 2
- For each converged fit:
  - Sample one network from the fitted model
  - Save to `{output}/samples/n{N}_{family}_{constraint}.npz`
  - Log per-case: edges generated, total weight, time

**Stage 4: `check`**
- Load samples from Stage 3 and constraints from Stage 1
- For each sample:
  - Compute strength/degree/edge recovery error
  - Report max absolute error and relative error
  - Log per-case: name, max_err, relative_err, pass/fail
- Log summary: pass count, fail count, worst cases

**Stage 5: `filter`** (optional, only if `filter` stage explicitly requested)
- Load fit results and generate ensemble
- Apply significance filtering at alpha=0.01, 0.05, 0.10
- Report FPR and edge detection rates
- Log per-case: alpha, FPR, power

#### Logging and resilience

- Each stage saves a `{output}/{stage}_log.jsonl` with one JSON line per case
- If a case crashes, the error is logged and the next case proceeds
- Stages can resume: if `fits/n100_me_strength.npz` exists, skip it
- Final summary is also saved as `{output}/summary.json`
- Progress: one line per case on stderr (unless `--quiet`)
- Results: JSON on stdout if `--json`, else human table

#### File layout after a full run

```
benchmarks/results/
├── networks/
│   ├── n25.npz
│   ├── n50.npz
│   └── ...
├── fits/
│   ├── n25_me_strength.npz
│   ├── n25_b_strength.npz
│   └── ...
├── samples/
│   └── ...
├── generate_log.jsonl
├── fit_log.jsonl
├── sample_log.jsonl
├── check_log.jsonl
└── summary.json
```

#### Files to delete after implementing

Remove all of:
- `benchmarks/bench_e2e.py`
- `benchmarks/bench_e2e_full.py`
- `benchmarks/bench_fitting.py`
- `benchmarks/bench_generation.py`
- `benchmarks/bench_filter.py`
- `benchmarks/compare_legacy.py`
- `benchmarks/common.py`
- `benchmarks/__main__.py`
- `benchmarks/regression_baselines.json`

Replace with:
- `benchmarks/__init__.py` (empty)
- `benchmarks/cli.py` (Typer app, one command: `bench`)
- `benchmarks/generate.py` (Stage 1)
- `benchmarks/fit.py` (Stage 2)
- `benchmarks/sample.py` (Stage 3)
- `benchmarks/check.py` (Stage 4)
- `benchmarks/filter.py` (Stage 5)
- `benchmarks/types.py` (shared dataclasses, CaseSpec, etc.)

Register in `pyproject.toml` as `[project.scripts] odme-bench = "benchmarks.cli:app"`.

### 9. Further work (lower priority)

- Fix P1 (W convergence) via Anderson acceleration or L-BFGS
- Fix P2 (B cost-coord) via bisection over gamma
- Fix P3 (partial W) — depends on P1
- Archive/remove legacy thesis-era folders
- Final rename decision: ODME → MENoBiS
- Publish MkDocs site
- Write tutorials and notebooks with real examples

## Checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
mkdocs build --strict
```
