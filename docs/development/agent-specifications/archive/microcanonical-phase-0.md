
# MENoBiS Phase 0 Engineering Plan

## Foundation refactor before the microcanonical engine

**Status:** Final implementation specification  
**Audience:** Harness / Pi coding agents and MENoBiS maintainers  
**Repository:** `uladribia/menobis`  
**Target repository path:** `docs/development/agent-specifications/microcanonical-phase-0.md`

---

## 0. Why this refactor exists

MENoBiS is a package for **non-binary networks**. A non-binary network is defined by integer occupation numbers

\[
t_{ij}\in\mathbb N_0
\]

assigned to ordered node pairs \((i,j)\). A node pair is an edge precisely when

\[
t_{ij}>0.
\]

The present codebase already contains:

- grand-canonical fitting and sampling for the ME, W, and B families;
- a limited canonical fixed-\(T\) sampler;
- one direct microcanonical ME sampler for exact strengths with self-loops;
- partial constraints and fixed-pair masking;
- filtering, analysis, and model-routing infrastructure.

The next project stage is to add a **general microcanonical generation engine**. That engine will sample networks while preserving exact integer constraints such as strengths, degrees, total events, and total edges. Later phases will require dependent-pair sampling, exact or exact-stationary MCMC, Coolen-style backbone swaps, and family-specific degeneracy factors.

The current code was primarily organized around independent pair distributions. A general microcanonical engine cannot be added cleanly until the shared scientific concepts are extracted and unified. Phase 0 performs that foundation refactor.

Phase 0 is therefore not cosmetic. It is required because the future microcanonical engine must share the same:

- ME/W/B family semantics;
- occupation-number support;
- pair masking;
- fixed-pair accounting;
- residual constraints;
- validation;
- model routing;
- output types;
- analysis;
- filtering pair laws.

The refactor must leave one coherent framework instead of separate grand-canonical, canonical, microcanonical, and filtering implementations that each redefine the same mathematics.

---

## 0.1 Terminology is part of the architecture

This project must use the ontology of the thesis.

The package is about **NonBinaryNetworks**, not generic weighted graphs.

Mandatory terminology:

| Avoid | Use |
|---|---|
| weighted network | non-binary network |
| weight | occupation number |
| weighted edge | occupied pair |
| edge weight | pair occupation number |
| rate, when referring to \(t_{ij}\) or a fixed pair value | occupation number |
| sampled edges | sampled network or occupied pairs |
| weight distribution | occupation-number distribution |
| weighted clustering | occupation-based clustering |
| `WeightFamily` | `OccupationFamily` |
| `WeightedEdge` | `OccupiedPair` |
| `SampledEdges` | `SampledNetwork` |
| `weight` field | `occ_num` |
| `weights` field | `occ_nums` |
| `known_rate` | `known_occnum` |
| `validate_weight` | `validate_occnum` |

The word **rate** may remain only when it is mathematically a continuous grand-canonical intensity or external third-party terminology. It must not be used as a synonym for an observed integer occupation number.

The words **weight** and **weighted** must not remain in MENoBiS-owned public types, function names, fields, modules, docs, error messages, examples, tests, or CLI options after Phase 0. Third-party APIs and citations are exempt only where renaming is impossible.

No backward-compatibility shim is required. Prefer the clean final API.

Introduce a scalar alias in Rust:

```rust
pub type OccNum = u64;
```

and use it consistently for integer pair occupations.

---

## 0.2 Ensemble context

MENoBiS distinguishes three ensemble classes.

### Grand canonical

Pair occupations are sampled independently:

\[
P(\mathbf T)=\prod_{ij}P_{ij}(t_{ij}).
\]

Constraints are enforced in expectation through fitted multipliers.

### Canonical

At least one global integer quantity is exact, for example total events \(T\), while other observables may remain expected.

### Microcanonical

All declared hard observables are exact:

\[
\mathcal C(\mathbf T)=\mathcal C^\star.
\]

The target measure is not generally uniform over occupation matrices. It is proportional to the family-specific base measure:

\[
P_F(\mathbf T\mid\mathcal C)
\propto
D_F(\mathbf T)\,
\mathbf 1[\mathcal C(\mathbf T)=\mathcal C^\star].
\]

This is why the family-degeneracy abstraction must exist before the general microcanonical engine.

### Mixed strength-cost case

The `STRENGTH_COST` case is deliberately hybrid:

- strengths are exact;
- cost remains an expected observable.

Its target is

\[
P_{F,\gamma}(\mathbf T\mid \mathbf s)
\propto
D_F(\mathbf T)
e^{-\gamma C(\mathbf T)}
\mathbf 1[\mathbf s(\mathbf T)=\mathbf s^\star].
\]

The cost is not fixed microcanonically.

---

## 2. Scientific invariants

### 2.1 Family base measures

For integer occupations \(t_{ij}\):

\[
d_{ME}(t)=1/t!,
\]

\[
d_{W,M}(t)=\binom{M+t-1}{t},
\]

\[
d_{B,M}(t)=\binom{M}{t}, \quad 0\le t\le M.
\]

The global \(T!\) factor in ME is constant whenever total events are fixed and therefore cancels in local probability ratios.

### 2.2 Zero is a valid occupation number

A pair fixed to zero is not a sepaoccupation parameter state from a prohibited pair. It is a node pair whose known occupation number is

\[
t_{ij}=0.
\]

The code must not introduce a special structural-event ontology.

### 2.3 Preprocessing occurs once

The following operations happen once before ensemble dispatch:

1. input normalization;
2. dimension and type validation;
3. family and layer validation;
4. fixed-pair validation;
5. sparse mask construction;
6. fixed-pair contribution calculation;
7. residualization;
8. generic feasibility checks;
9. prepared-problem creation.

No grand-canonical, canonical, or microcanonical backend may repeat these operations.

### 2.4 Memory rule

No production path may materialize an auxiliary \(N\times N\) matrix.

The correct bound is:

\[
O(N)+O(E_{occupied})+O(K_{fixed})+O(output).
\]

An algorithm may require \(O(N^2)\) time while iterating over implicit all-pairs support, but it must not allocate \(O(N^2)\) working memory.

## 3. Existing components to reuse

Phase 0 must refactor, not replace:

- `distribution.rs`: `OccupationFamily`, `PairOccupationDistribution`;
- `pairs.rs`: providers, candidate support, costs, chunk seeds;
- `generation.rs`: independent sampling, multinomial sampling, stub matching, `SampledNetwork`;
- `fitting/mask.rs`: sparse `PairMask`;
- `graph.rs`: `OccupiedPair` and summaries;
- `stats.rs` and `clustering.rs`;
- Python `Ensemble`, `ModelFamily`, `Constraint`;
- `routing.py`;
- Python generation adapters and result types;
- the existing partial-fitting implementation;
- current ensemble-equivalence tests.

## 4. Required Rust organization

The generation layer must be organized by ensemble:

```text
coccupation parameters/menobis-core/src/
    family/
        mod.rs
        ontology.rs
        measure.rs
        support.rs
        generating_functions.rs

    constraints/
        mod.rs
        ontology.rs
        validation.rs
        fixed_pairs.rs
        residuals.rs
        prepared.rs
        capability.rs
        mask.rs

    generation/
        mod.rs
        output.rs
        config.rs

        grandcanonical/
            mod.rs
            independent_pairs.rs
            providers.rs
            dense_iteration.rs
            sparse_iteration.rs

        canonical/
            mod.rs
            multinomial.rs
            fixed_total.rs

        microcanonical/
            mod.rs
            stub_matching.rs
            placeholders.rs

    analysis/
        mod.rs
        graph_view.rs
        summary.rs
        node_statistics.rs
        distributions.rs
        clustering.rs
        ensemble.rs
```

You do not need to keep existing public paths under `menobis_core::generation::*` through re-exports. But make sure the corresponding python and API endpoints are adapted.

## 5. Constraint ontology

### 5.1 Public constraints

Add:

```python
Constraint.EDGES_EVENTS = "edges_events"
```

The complete set becomes:

- `STRENGTH`
- `STRENGTH_COST`
- `STRENGTH_EDGES`
- `STRENGTH_DEGREE`
- `DEGREE_EVENTS`
- `EDGES_EVENTS`

Make sure the corresponding code (it already exists for grand canonical) is well routed in the API python interface for the new constraint.

### 5.2 Elementary observables

Internally decompose named cases into:

```rust
pub enum ObservableKind {
    TotalEvents,
    TotalEdges,
    OutStrength,
    InStrength,
    OutDegree,
    InDegree,
    PairCost,
}
```

Mapping:

| Public constraint | Hard observables in the new constrained sampler | Expected observables |
|---|---|---|
| `STRENGTH` | out/in strengths | none |
| `STRENGTH_COST` | out/in strengths | pair cost |
| `STRENGTH_EDGES` | out/in strengths, total edges | none |
| `STRENGTH_DEGREE` | out/in strengths, out/in degrees | none |
| `DEGREE_EVENTS` | out/in degrees, total events | none |
| `EDGES_EVENTS` | total edges, total events | none |

This decomposition must support future constraints without requiring changes to every sampler.

### 5.3 Extensibility contract

Adding a new constraint requires:

1. observable composition;
2. required public inputs;
3. hard/expected treatment by ensemble. All hard constraints can naturally be converted to expected constraints, but not the other way around (unless they are strictly integer constraints like degrees or strengths or occupation numbers);
4. common validation;
5. capability declaration;
6. a compatible solver or sampling backend.

It must not require a new family implementation or a duplicate mask path.

## 6. Family abstraction

Keep `OccupationFamily` and add thesis semantics.

```rust
pub enum ModelFamilyKind {
    MultiEdge,
    Weighted,
    BinaryLayers,
}
```

Required shared methods:

```rust
impl OccupationFamily {
    pub fn model_family(self) -> ModelFamilyKind;
    pub fn layers(self) -> Option<u32>;
    pub fn occupation_support(self) -> OccupationSupport;
    pub fn validate_occnum(self, occ_num: OccNum) -> bool;
    pub fn log_local_degeneracy(self, occ_num: OccNum) -> f64;
    pub fn delta_log_local_degeneracy(
        self,
        old_occ_num: OccNum,
        new_occ_num: OccNum,
    ) -> f64;
}
```

Use stable log-gamma formulas. B must reject weights above \(M\).

Required tests:

- family mapping;
- layer extraction;
- bounded/unbounded support;
- small exact degeneracies;
- delta/full-difference equivalence;
- B capacity rejection;
- consistency with current grand-canonical PMFs;
- large-occupation numerical stability.

## 7. Shared mask and fixed-pair preprocessing

### 7.1 Reuse `PairMask`

Move the existing sparse `PairMask` out of fitting-only ownership into `constraints/mask.rs`, adapting the necessary paths (avoid re-export if the implementation is cleaner).

Do not create another domain or mask abstraction unless a concrete missing operation cannot be added to `PairMask`.

### 7.2 Fixed pairs

Keep fixed values sepaoccupation parameter from mask membership:

```rust
pub struct FixedPairs {
    pub sources: Vec<u64>,
    pub targets: Vec<u64>,
    pub occ_nums: Vec<OccNum>,
    pub mask: PairMask,
}
```

A fixed-zero pair is represented explicitly with occupation number zero.

### 7.3 Contribution calculation

Compute once:

```rust
pub struct FixedContributions {
    pub total_events: u64,
    pub total_edges: u64,
    pub strength_out: Vec<u64>,
    pub strength_in: Vec<u64>,
    pub degree_out: Vec<u64>,
    pub degree_in: Vec<u64>,
    pub total_cost: Option<f64>,
}
```

Complexity must be \(O(N+K)\).

### 7.4 Residual constraints

Create one residual structure:

```rust
pub struct ResidualConstraints {
    pub total_events: Option<u64>,
    pub total_edges: Option<u64>,
    pub strength_out: Option<Vec<u64>>,
    pub strength_in: Option<Vec<u64>>,
    pub degree_out: Option<Vec<u64>>,
    pub degree_in: Option<Vec<u64>>,
    pub expected_cost: Option<f64>,
}
```

Use borrowed slices or `Cow` where useful to avoid unnecessary copies.

### 7.5 Prepared problem

All ensemble backends consume one prepared representation:

```rust
pub struct PreparedProblem<'a> {
    pub node_count: usize,
    pub family: OccupationFamily,
    pub constraint: ConstraintKind,
    pub self_loops: bool,
    pub mask: &'a PairMask,
    pub fixed_pairs: FixedPairView<'a>,
    pub fixed_contributions: FixedContributions,
    pub residual: ResidualConstraints,
    pub cost_provider: Option<&'a dyn PairCostProvider>,
}
```

The concrete generic design may vary, but the semantic boundary is mandatory.

## 8. Common validation

Before dispatch, validate:

- array lengths;
- balanced strengths;
- balanced degrees;
- fixed-pair parallel-array lengths;
- duplicate coordinates;
- out-of-range indices;
- self-loop conflicts;
- finite coordinates and costs;
- \(M\ge1\);
- B fixed occupation numbers within capacity;
- nonnegative residuals;
- event and edge consistency;
- elementary feasibility bounds.

Examples:

\[
E^R\le T^R,
\]

\[
s_i^R\ge k_i^R
\]

for hard strength-degree residuals.

Keep only algorithm-specific checks in backends, such as graphical realization, kernel availability, or optimizer convergence.

## 9. Capability registry and routing

Replace scattered support checks with a central registry keyed by:

```python
(verb, ensemble, family, constraint)
```

Each entry defines:

```python
@dataclass(frozen=True)
class ModelCapability:
    supported: bool
    requires_fit: bool
    backend: str
    required_arguments: frozenset[str]
    optional_arguments: frozenset[str]
    supports_self_loops: bool
    supports_no_self_loops: bool
    result_kind: str
```

This registry dispatches to existing heterogeneous solvers. It does not replace their numerical methods.

`EDGES_EVENTS` should receive full public recognition. Prefer implementing its grand-canonical ME/W/B case during Phase 0 because it will serve as the reference for later \((E,T)\) microcanonical tests.

## 10. Generation refactor by ensemble

Convert `generation.rs` into a module tree while preserving numerical semantics.

### `grandcanonical`

Move:

- provider-backed independent sampling;
- all-pairs row iteration;
- sparse chunk iteration;
- Poisson/geometric/binomial/negative-binomial samplers;
- zero-inflated samplers.

### `canonical`

Move:

- custom multinomial;
- strength multinomial;
- fixed-total utilities.

### `microcanonical`

Move:

- exact ME strength stub matching.

### Shared output

Move `SampledNetwork` to `generation/output.rs`.

### API migration

`generation/mod.rs` re-exports all existing public functions. Update PyO3 imports and Python wrappers to the clean renamed API.

Add deterministic regression tests for numerical semantics, seeds where intentionally preserved, and output ordering.

## 11. ME fixed-strength stub matching

Retain it as the direct fast path for:

- ME;
- hard strengths;
- self-loops allowed.

Correct documentation:

> The method is uniform over compatible labelled stub matchings. Aggregated weighted matrices are sampled with probability proportional to \(T!/\prod t_{ij}!\).

For `self_loops=False`, do not add rejection-based stub matching. Until the general MCMC backend exists, raise a clear unsupported-case error.

Add `self_loops` as an optional keyword-only argument to public sampling if required, preserving the current default.

## 12. Public Python sampling contract

Define the clean final public sampling entry point. It may replace the current signature because backward compatibility is not required.

Provide:

```python
sample_model_detailed(...) -> SamplingResult
```

Suggested result:

```python
@dataclass(frozen=True)
class SamplingResult:
    edges: EdgeTable
    ensemble: Ensemble
    family: ModelFamily
    constraint: Constraint
    method: str
    exactness: SamplingExactness
    seed: int
    diagnostics: SamplingDiagnostics | None
```

`sample_model()` should delegate and return `.edges`.

Add exactness values such as:

- `EXACT_INDEPENDENT`
- `EXACT_DIRECT`
- `EXACT_STATIONARY_MCMC`
- `EXACT_PSEUDO_MARGINAL`
- `APPROXIMATE`
- `HEURISTIC`

Replace legacy partial-fitting inputs with the clean fixed-pair occupation API and update all callers.

## 13. Analysis consolidation

Public `menobis.analysis` is already a facade, but internals should be compacted.

Create a sparse graph view:

```rust
pub struct SparseGraphView<'a> {
    pub node_count: usize,
    pub edges: &'a [OccupiedPair],
}
```

Create lazy sparse adjacency only when adjacency-based metrics are requested.

Add a composable analysis request/result API, but provide clean occupation-based equivalents for:

- `directed_strengths`;
- `directed_degrees`;
- `compute_all_stats`;
- `occupation_distribution`;
- `clustering_coefficient`;
- `occupation_clustering_coefficient`.

Do not force clustering into the basic statistics pass. Clustering is algorithmically distinct and may need sparse adjacency. The unification should be at the facade and graph-view level.

Remove duplicate node-count helpers.

All analysis paths must use \(O(N+E)\) memory.

## 14. Ensemble-equivalence tests

Rename the current broad test as an ME observable-convergence smoke test.

Create sepaoccupation parameter suites for:

1. exact conditioning identities;
2. direct sampler correctness;
3. asymptotic observable convergence;
4. specific relative entropy where defined.

For small ME strength cases with self-loops, enumeoccupation parameter states and compare stub-matching frequencies against:

\[
P(\mathbf T\mid s)\propto T!/\prod t_{ij}!.
\]

Do not treat absence of significant differences as proof of equivalence.


## 14. Filtering integration

Filtering is a first-class consumer of the Phase 0 refactor.

The filtering module must remain responsible for:

- pair-level tail probabilities;
- observed-pair significance;
- absent-pair significance;
- multiple-testing correction;
- filtering-specific result tables.

Filtering must **not** own duplicate implementations of:

- ME/W/B family semantics;
- occupation-number support;
- pair occupation distributions;
- positive-occupation generating functions;
- pair masks;
- fixed-pair preprocessing;
- cost providers;
- occupation validation;
- common model constraints.

These concepts must move to shared core modules and be imported by fitting, generation, and filtering.

### 14.1 Shared pair-law source of truth

Grand-canonical generation and filtering both use pair occupation laws. They must obtain them from the same `PairOccupationDistribution` and provider abstractions.

For every supported pair law, the same object must provide or support:

- probability mass;
- cumulative or survival probability;
- probability of zero;
- probability of positive occupation;
- expected occupation;
- random sampling;
- family and layer metadata;
- support validation.

A distribution formula must not be reimplemented separately in filtering and generation.

### 14.2 Mandatory filtering rename

The package-wide terminology migration applies fully to filtering.

Required examples:

```text
filter_weighted_*            -> filter_nonbinary_* or family/constraint-specific name
weight                       -> occ_num
weights                      -> occ_nums
observed_weight              -> observed_occnum
rate                         -> occupation parameter, intensity, or occ_num depending on meaning
weighted tail                -> occupation tail
weighted network filter      -> non-binary network filter
```

Choose names based on actual mathematical meaning. Do not mechanically replace a true Poisson intensity with `occ_num`; distinguish continuous model parameters from integer observed occupations.

### 14.3 Filtering API adaptation

Because backward compatibility is not required, update all Python and Rust filtering APIs to the clean occupation-number terminology.

The final public filtering layer should accept and return:

- `OccupiedPair` / `EdgeTable`-equivalent occupation data;
- `occ_num` arrays;
- occupation-tail probabilities;
- family and constraint metadata using the shared enums.

Update:

- Python wrappers;
- PyO3 function names;
- Rust function names;
- dataclass fields;
- CLI options;
- docs;
- notebooks;
- tests.

### 14.4 No filtering dependency inversion

The dependency direction must be:

```text
shared family / constraint / pair-law modules
        -> fitting
        -> generation
        -> filtering
        -> analysis
```

Filtering must not call generation internally to obtain formulas. Generation and filtering consume shared mathematical kernels.

### 14.5 Filtering regression requirements

For every currently supported filtering combination:

1. old and refactored formulas must agree numerically before old code is removed;
2. observed-pair and absent-pair calculations must be tested;
3. ME, W, and B cases must be covered;
4. zero occupation must be tested explicitly;
5. B upper support \(t_{ij}\le M\) must be tested;
6. shared occupation PMFs and filtering tails must satisfy probability identities;
7. all renamed public endpoints must be exercised through Python.


## 15. Documentation location

Add:

```text
docs/development/agent-specifications/
    README.md
    microcanonical-phase-0.md
```

Add both to `mkdocs.yml` under `Development`.

The index must explain that these are implementation specifications for autonomous coding agents and are intentionally more detailed than end-user documentation.

## 16. Work packages

### P0.1 Mandatory package-wide terminology migration

- introduce `OccNum`;
- rename all MENoBiS-owned `weight`, `weighted`, and ambiguous `rate` identifiers;
- rename Rust structs, enums, fields, modules, and functions;
- rename Python functions, arguments, dataclasses, and exports;
- rename PyO3 endpoints;
- rename CLI options and help text;
- rename docs, tests, fixtures, and examples;
- update filtering, fitting, generation, and analysis together;
- run a repository-wide search proving no prohibited terminology remains except an explicit reviewed allowlist.

### P0.2 Public API inventory and clean API definition

- snapshot signatures and enums;
- public export tests;
- default behavior tests;
- seed reproducibility;
- baseline benchmarks.

### P0.3 Constraint ontology and `EDGES_EVENTS`

- Python/Rust enum;
- observable mapping;
- capabilities;
- validation;
- docs;
- preferably grand-canonical ME/W/B implementation.

### P0.4 Shared family measure

- thesis family kind;
- support;
- degeneracy methods;
- numerical tests.

### P0.5 Shared mask and preprocessing

- move/re-export `PairMask`;
- fixed-pair normalization;
- contribution calculation;
- residualization;
- common validation;
- migoccupation parameter existing partial fitting;
- delete duplicate residual code only after equivalence tests.

### P0.6 Generation split by ensemble

- `grandcanonical`;
- `canonical`;
- `microcanonical`;
- shared output;
- re-exports;
- deterministic regression.

### P0.7 Sampling result and routing cleanup

- capability-driven routing;
- detailed result;
- exactness metadata;
- unchanged `sample_model`;
- explicit stub-matching self-loop rules.

### P0.8 Analysis consolidation

- Rust analysis tree;
- graph view;
- optional sparse adjacency;
- composable facade;
- renamed clean wrappers;
- memory benchmarks.

### P0.9 Ensemble-equivalence test split

- rename smoke test;
- exact ME direct-sampler test;
- conditioning scaffold;
- asymptotic cleanup.

### P0.10 Filtering adaptation and final audit

- migrate filtering to shared occupation-family and pair-law abstractions;
- rename all filtering terminology and endpoints;
- remove duplicated filtering formulas;
- run filtering regression tests across ME/W/B and all supported constraints;
- support matrix;
- architecture diagrams;
- migration notes;
- strict docs build;
- API and performance audit.

## 17. Test commands

Rust:

```bash
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
```

Python and docs:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run mkdocs build --strict
```

Required test classes:

- unit tests;
- exact small-state enumeration;
- deterministic sampling regression;
- mask/residual equivalence against dense test-only calculations;
- public API contracts;
- routing matrix;
- large-\(N\), sparse-\(E\) memory benchmarks.



## 18. Mandatory benchmark baseline and three-regime preservation

Phase 0 is a structural and terminology refactor. It must not silently improve, degrade, hide, or reinterpret the current numerical behaviour of the existing grand-canonical solvers.

The repository benchmark CLI is:

```bash
uv run python -m benchmarks all ...
```

It exercises synthetic-network generation, fitting, sampling, filtering, convergence diagnostics, residual precision, runtime, and memory.

The three benchmark regimes are:

| Regime | Parameters | Purpose |
|---|---|---|
| `sparse` | `average_degree=3.0`, `events_per_edge=3.0` | nearly binary occupations; difficult for zero-inflated models |
| `dense` | `average_degree=N/5`, `events_per_edge=8.0` | recommended baseline |
| `saturated` | `average_degree=0.85*(N-1)`, `events_per_edge=8.0` | degree and occupation boundary stress |

### 18.1 Capture the baseline before editing production code

Before the first production-code change, run the benchmark on the current default branch:

```bash
uv run python -m benchmarks all \
  --nodes 100 \
  --families me,b,w \
  --constraints strength,strength-cost,strength-edges,strength-degree \
  --regime sparse \
  --known-pairs 0.0,0.02

uv run python -m benchmarks all \
  --nodes 100 \
  --families me,b,w \
  --constraints strength,strength-cost,strength-edges,strength-degree \
  --regime dense \
  --known-pairs 0.0,0.02

uv run python -m benchmarks all \
  --nodes 100 \
  --families me,b,w \
  --constraints strength,strength-cost,strength-edges,strength-degree \
  --regime saturated \
  --known-pairs 0.0,0.02
```

Run both self-loop policies when exposed by the CLI.

Also run a larger sentinel, preferably `N=1000`, for combinations that finish in reasonable time.

Store:

- exact command;
- git commit SHA;
- OS, CPU, Python, and Rust metadata;
- raw JSON rows;
- convergence summary;
- residuals;
- peak memory;
- wall time;
- failure messages.

Recommended location:

```text
benchmarks/results/phase0-baseline/
    metadata.json
    sparse.json
    dense.json
    saturated.json
    summary.md
```

### 18.2 Known current solver behaviour

The current documentation identifies the following expected behaviour. Phase 0 must preserve it unless a separate solver-improvement task is approved.

| Family / constraint | Dense | Sparse | Saturated |
|---|---|---|---|
| ME strength | reliable and fast | reliable | reliable |
| ME strength-cost | reliable and fast | normally reliable | inspect cost and boundary diagnostics |
| ME strength-edges | reliable baseline | can be ill-conditioned | boundary handling may be difficult |
| ME strength-degree | reliable baseline but slower | can be ill-conditioned when degree approaches strength | difficult near pair capacity |
| B strength | reliable when layer capacity is feasible | feasible if layer bounds permit | capacity checks essential |
| B strength-cost | generally reliable with feasible layers | inspect residuals | inspect boundary behaviour |
| B strength-edges | generally usable | needs care | needs care near saturation |
| B strength-degree | generally usable | needs care | needs care near saturation |
| W strength | usable; require valid `q<1` diagnostics | inspect domain margin | inspect domain margin |
| W strength-cost | slower and sensitive as \(N\) grows | sensitive | sensitive |
| W strength-edges | experimental | known difficult regime | known difficult regime |
| W strength-degree | experimental | known difficult regime | known difficult regime |

Preserving behaviour means preserving the observed result class:

- convergent remains convergent;
- maximum-iteration cases may remain maximum-iteration cases;
- known non-convergent experimental cases may remain non-convergent;
- infeasible B cases remain classified as infeasible;
- no case may crash, return NaN diagnostics, or emit invalid occupation numbers;
- no weakness may be hidden by changing fixtures, loosening tolerances, suppressing diagnostics, or removing combinations.

### 18.3 Before/after comparison

After every work package touching fitting, generation, filtering, family laws, masks, routing, or analysis, rerun the affected benchmark slice.

At the final gate, rerun the complete three-regime matrix and compare:

- completion versus exception;
- `converged`;
- status class;
- maximum-iteration flag;
- iteration count;
- strength residuals;
- degree residuals;
- edge-count residual;
- cost residual;
- sampled occupied-pair count;
- filtering false-positive rate;
- peak memory;
- runtime.

Rules:

1. Categorical outcomes must match unless an approved change is documented.
2. Residuals must remain in the same accuracy class.
3. A previously convergent case must not become non-convergent.
4. A known non-convergent case is not required to become convergent.
5. Known difficult cases must still terminate cleanly with finite diagnostics.
6. Memory must not acquire \(O(N^2)\) scaling.
7. Performance regressions above the agreed threshold require approval.
8. Benchmark fixtures and tolerances must not be weakened to obtain a pass.

### 18.4 Benchmark terminology migration

The benchmark code and JSON schema must also be renamed to occupation terminology.

Examples:

```text
weight                    -> occ_num
weights                   -> occ_nums
sampled_edges             -> sampled_occupied_pairs
known_rate                -> known_occnum
weight residual           -> occupation residual
```

A true continuous Poisson intensity or family parameter must receive a mathematically correct name such as `intensity` or `occupation_parameter`, not `occ_num`.

Use a one-time adapter to compare the renamed output against the frozen baseline. Do not retain old names in the final schema.

### 18.5 Extend benchmark coverage

The current benchmark covers:

- strength;
- strength-cost;
- strength-edges;
- strength-degree.

After the ontology refactor, include implemented routes for:

- degree-events;
- edges-events.

Unsupported combinations must appear explicitly in the capability matrix and benchmark report.

## 18. Mandatory full supported-combination integration run

Phase 0 is not complete after unit tests alone.

Create one machine-readable support matrix covering every supported combination of:

- ensemble;
- family;
- constraint;
- self-loop policy;
- masked versus unmasked input;
- relevant layer count for W/B;
- fitting;
- generation;
- filtering;
- analysis.

The matrix must be the source of truth used by routing tests and end-to-end tests.

For every entry marked supported, the integration suite must:

1. construct a valid small synthetic problem;
2. run preprocessing and validation;
3. fit the model when fitting is required;
4. generate at least one network;
5. verify every generated occupation number is an integer and lies in family support;
6. verify masked/fixed pairs are unchanged;
7. verify hard constraints exactly;
8. verify expected constraints through deterministic formula checks or statistically justified ensemble checks;
9. run all applicable filtering operations;
10. run the analysis facade;
11. verify Rust and Python endpoints;
12. verify no panic, NaN, invalid occupation, duplicate occupied pair, or inconsistent node count;
13. verify the output uses the renamed occupation-number API;
14. record the exactness category of the generator.

At minimum, the run must cover all supported combinations in the following conceptual grid:

| Family | Constraint |
|---|---|
| ME | strength |
| ME | strength-cost |
| ME | strength-edges |
| ME | strength-degree |
| ME | degree-events |
| ME | edges-events |
| W | all currently supported thesis constraints |
| B | all currently supported thesis constraints |

Each row must be exercised for every ensemble that the capability registry marks as supported.

A test must fail if:

- the capability registry says a combination is supported but no integration fixture exists;
- a public endpoint exists but is absent from the support matrix;
- the generated network cannot be analyzed;
- filtering uses a separate untested formula;
- prohibited terminology appears in returned public objects.

Produce a final CI artifact or textual report listing:

- combinations tested;
- combinations unsupported by design;
- combinations deferred to later microcanonical phases;
- failures;
- performance and memory summary.


## 20. Exit gate

Phase 0 is complete only when all conditions below hold.

1. The package consistently describes itself as a non-binary network package.
2. `OccNum` is the canonical Rust scalar for integer occupations.
3. No MENoBiS-owned identifier uses `weight`, `weighted`, or ambiguous `rate`, except a reviewed allowlist.
4. Docs, tests, examples, CLI, Python, PyO3, Rust, filtering, and benchmark fields use occupation terminology.
5. `EDGES_EVENTS` exists in Python and Rust and is correctly routed.
6. ME/W/B share one tested occupation-family and base-measure abstraction.
7. Pair occupation laws are shared by generation and filtering.
8. Filtering contains no duplicated PMF, support, or generating-function logic.
9. `PairMask` is shared outside fitting.
10. Fixed-pair accounting and residualization occur once before ensemble dispatch.
11. Generic validation occurs once before ensemble dispatch.
12. Partial fitting uses the shared preprocessing path.
13. Generation is split into `grandcanonical`, `canonical`, and `microcanonical`.
14. Python and Rust APIs use the clean final naming.
15. Detailed sampling results and exactness metadata exist.
16. Stub matching is documented with the correct ME degeneracy interpretation.
17. No-self-loop stub matching is not approximated by naive rejection.
18. No production path materializes \(N^2\) auxiliary memory.
19. Fixed-zero and fixed-positive occupations use the same mask semantics.
20. Analysis has a compact facade and reusable sparse graph view.
21. Clustering remains specialized and sparse.
22. Ensemble-equivalence tests are mathematically separated.
23. Exact small-state tests validate the current ME direct sampler.
24. Filtering regression tests pass for every supported combination.
25. A pre-refactor benchmark baseline exists for sparse, dense, and saturated regimes.
26. The final benchmark CLI has been run for all three regimes.
27. The final run includes ME, B, W, and every currently benchmarked constraint.
28. Implemented degree-events and edges-events routes are included.
29. Every benchmark row is compared for completion, convergence, status, residuals, iterations, memory, and runtime.
30. Every previously convergent baseline combination remains convergent.
31. Known solver limitations reproduce their prior diagnostic behaviour.
32. No difficult sparse, saturated, or W zero-inflated case is removed or relabelled successful.
33. No benchmark tolerance or synthetic regime is weakened to pass.
34. The full supported-combination integration matrix runs successfully.
35. Every supported combination can be fit, generated, filtered when applicable, and analyzed.
36. Generated occupation numbers satisfy family support and hard constraints.
37. Masked pairs remain exactly fixed.
38. Rust tests, Python tests, typing, lint, strict docs, and benchmarks pass.
39. The support matrix, benchmark comparison, known limitations, and integration report are committed.
40. The DeepSeek Flash V4 implementation handoff is complete and self-contained.

## 21. Agent execution protocol

Each coding agent must record:

- task ID;
- files inspected;
- public interfaces affected;
- scientific invariants;
- memory impact;
- tests added;
- compatibility risk.

Each PR must state:

- behavior changes;
- whether it is a pure refactor;
- tests and benchmarks run;
- API compatibility;
- remaining limitations;
- the relevant section of this specification.

Before completing any PR, agents must run repository-wide terminology checks for:

```text
weight
weighted
known_rate
edge_rate
weight_distribution
WeightedEdge
WeightFamily
SampledEdges
```

Every match must either be renamed or documented in a narrow allowlist with a reason.

Agents must not:

- create a second family ontology;
- create a second mask system;
- duplicate residualization inside an ensemble backend;
- implement naive no-self-loop stub rejection;
- allocate \(N^2\) production arrays;
- leave mixed old/new terminology in public return types;
- claim general ensemble equivalence from loose observable agreement;
- put sampling implementation into filtering;
- leave duplicated pair-law formulas in filtering and generation.


## 21.1 DeepSeek Flash V4 implementation handoff

The implementation agent is expected to be **DeepSeek Flash V4**. Each task must be self-contained and must not rely on this conversation.

Each task prompt must include:

- exact objective;
- mathematical context;
- files to inspect first;
- expected files to change;
- symbols to rename;
- invariants that must not change;
- benchmark slice to run before editing;
- tests to add;
- commands to run;
- expected outputs;
- prohibited shortcuts;
- completion checklist.

Do not ask the model to infer:

- ME/W/B semantics;
- occupation numbers versus continuous parameters;
- hard versus expected constraints;
- why filtering shares pair laws with generation;
- which benchmark failures are already known;
- whether a changed solver result is acceptable.

Required loop for every task:

1. Read the cited specification sections.
2. Inspect the named files.
3. Run the requested baseline test or benchmark slice.
4. Record current behaviour.
5. Make the smallest coherent change.
6. Run focused tests.
7. Rerun the benchmark slice.
8. Compare before and after.
9. Run formatting, lint, and typing.
10. Produce a concise implementation report.

Use mandatory review checkpoints after:

- terminology and type migration;
- family-law extraction;
- shared mask/preparation extraction;
- generation module split;
- filtering migration;
- analysis consolidation;
- benchmark and integration completion.

Do not assign the complete Phase 0 as one prompt.

## 22. Target runtime flow

```text
Python request
    -> capability lookup
    -> input normalization
    -> shared sparse mask and fixed-pair contributions
    -> shared residualization and validation
    -> PreparedProblem
    -> grandcanonical | canonical | microcanonical backend
    -> SampledNetwork
    -> EdgeTable
    -> shared analysis facade
    -> filtering consumes the same shared pair-law and occupation-family kernels
```

This is the required foundation for the later general microcanonical engine.
