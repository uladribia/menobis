# MENoBiS Documentation, Notebook, and Ensemble-Equivalence Update Plan

## 0. Purpose and scope

This is a **documentation and example update only**. Do not redesign MENoBiS, change public APIs, introduce new scientific algorithms, or significantly reorganize the documentation tree.

Work from the current `master` branch of `uladribia/menobis` and preserve the present documentation structure. The goals are:

1. Update `docs/examples/main-use-cases.ipynb` so it visibly demonstrates the current microcanonical functionality.
2. Replace the notebook's current narrow fixed-strength ME ensemble-equivalence check with a precise **fixed-`N`, large-occupation (`T -> infinity`) ME-versus-W comparison** whose practical purpose is to show when grand-canonical calculations can stand in for harder constrained ensembles.
3. Add a concise scientific documentation section/page explaining when the grand-canonical ensemble is asymptotically interchangeable for theoretical calculations in ME, why this fails for W, and why ME cases with binary/support constraints require separate care.
4. Review the existing documentation for clarity and concision for three audiences:
   - scientists using MENoBiS;
   - developers extending MENoBiS;
   - readers interested in the mathematics, statistical mechanics, and sampling methods.
5. Make only targeted edits. **Do not expand the documentation substantially.** Prefer replacing, shortening, cross-linking, and clarifying existing text over adding new pages or long explanations.

Known out-of-scope microcanonical cases remain out of scope:

- fixed `(s,E)`;
- fixed `(s,k)`.

These must be mentioned as unavailable where relevant, but must not be implemented.

---

## 1. Non-negotiable constraints

The implementing agent must follow these rules.

### 1.1 No structural refactor

Do not significantly alter `mkdocs.yml` navigation. Keep the existing major groups:

- Start here
- Tutorials
- Model selection and maths
- Reference
- Development

At most add **one** new scientific page under **Model selection and maths** if the ensemble-equivalence material cannot be made sufficiently clear inside an existing page. Preferred option: create `docs/concepts/ensemble-equivalence.md` and add one nav entry adjacent to `Equations` / `Microcanonical sampling`.

Do not create additional documentation categories.

### 1.2 No documentation bloat

Target lengths:

- new ensemble-equivalence page: **1,200–2,000 words maximum**;
- new notebook explanatory prose: **short markdown cells, normally 50–150 words each**;
- edits to existing pages: prefer net-neutral or net-negative word count;
- no new long derivations already available in the thesis or cited literature.

Do not duplicate the same explanation in the notebook, concepts page, tutorial, and API docs. Use cross-links.

### 1.3 Scientific precision

The ensemble-equivalence discussion is about exactly one asymptotic regime:

\[
N \text{ fixed}, \qquad T \to \infty.
\]

Do not introduce a large-`N` thermodynamic limit; it is not relevant to the requested argument.

The practical/theoretical question is:

> When may calculations performed in the grand-canonical ensemble be used in place of the harder constrained ensemble because soft magnitude constraints become relatively sharp as occupation grows?

Required distinctions:

- **ME with magnitude-only constraints:** relative fluctuations of the soft occupation constraints vanish as \(T\to\infty\); the saddle-point approximation becomes sharp, so GC calculations can become asymptotically interchangeable for the relevant network magnitude/topology statistics.
- **ME with binary/support constraints:** do not automatically transfer the preceding result. Constraints involving \(a_{ij}=\mathbf 1[t_{ij}>0]\), \(E\), or \(k\) do not scale with occupation in the same way.
- **W:** relative fluctuations remain non-negligible at large occupation, so the GC ensemble cannot generally be substituted for the harder constrained ensemble on the basis of large \(T\).

Never make any of these claims:

- “all ME ensembles are interchangeable at large occupation”;
- “all ensembles become identical”;
- “large `N` implies ensemble equivalence”;
- “conditioning proves the asymptotic result”;
- “W becomes equivalent if `T` is made large enough”;
- “binary constraints become irrelevant at high occupation”;
- “the full probability measures are identical.”

Conditional equality after fixing the soft constraints may be mentioned in at most one short orientation sentence. It must **not** be the organizing argument.

### 1.4 Respect exactness semantics

Use the current semantics consistently:

- fixed `(E,T)`: exact hard constraints;
- fixed `(k,T)`: exact hard constraints at stationarity of the implemented chain;
- fixed strengths: strengths exact at stationarity;
- fixed strengths + expected cost: strengths exact, cost constrained **in expectation**, not exactly.

Do not describe the whole microcanonical family as “all constraints exact” without qualification.

### 1.5 Do not duplicate historical implementation documents

`docs/development/agent-specifications/` is historical/internal material and remains excluded from public MkDocs navigation. Do not use historical plans as user-facing documentation. Current code and current public docs are authoritative.

---

## 2. Required repository inspection before editing

Before changing files, inspect the current versions of at least:

- `docs/examples/main-use-cases.ipynb`
- `docs/concepts/microcanonical.md`
- `docs/concepts/equations.md`
- `docs/concepts/choose-null-model.md`
- `docs/concepts/solvers-and-scaling.md`
- `docs/development/scalability.md`
- `docs/development/benchmarking.md`
- `docs/development/architecture.md`
- `docs/development/testing.md`
- `docs/development/extending-thesis-cases.md`
- `docs/tutorials/microcanonical-sampling.md`
- `docs/api/python.md`
- `docs/api/rust.md`
- `docs/index.md`
- `docs/getting-started.md`
- `mkdocs.yml`

Also inspect the current sampling APIs used by the notebook. Do not invent parameter names from memory.

Important current fact: `main-use-cases.ipynb` already contains a section titled approximately **“ME fixed-strength ensemble equivalence”** comparing grand-canonical, canonical, and microcanonical samples. This must be **reworked**, not duplicated.

Gate: do not begin edits until the agent can state which existing notebook cells will be retained, modified, replaced, or removed.

---

## 3. Deliverable A — scientific ensemble-equivalence section

### 3.1 Preferred location

Create:

`docs/concepts/ensemble-equivalence.md`

Add a single entry under **Model selection and maths** in `mkdocs.yml`, preferably after `Equations` and before or after `Microcanonical sampling`.

If the current docs make a new page clearly unnecessary, the material may instead be integrated into `docs/concepts/choose-null-model.md`, but do not spread the main exposition across multiple pages.

### 3.2 Required page structure

Use exactly these conceptual sections, with minor wording changes allowed:

1. `# Ensemble equivalence at large occupation`
2. `## Regime: fixed N, T -> infinity`
3. `## ME: why the grand-canonical ensemble becomes interchangeable`
4. `## Saddle point and vanishing relative fluctuations`
5. `## W: why the equivalence fails`
6. `## Binary constraints: the ME caveat`
7. `## Consequences for theoretical calculations`
8. `## Numerical illustration in MENoBiS`
9. `## References`

Do not add more top-level sections unless strictly necessary.

### 3.3 Required scientific content

The page must explain the following argument in this order.

#### A. Start from the practical theoretical-calculation question

Open with the reason the result matters:

> MENoBiS often permits the same scientific null hypothesis to be represented with soft or hard constraints. The useful question for analytical work is whether, at high occupation, one can calculate in the simpler grand-canonical ensemble and obtain the same relevant network statistics as in the corresponding harder constrained ensemble.

State immediately that the regime is

\[
N \text{ fixed},\qquad T\to\infty.
\]

Do not center the discussion on conditional equivalence. Conditional on the values of the soft constraints, equivalence is by construction; that fact does not answer whether the unconstrained GC calculation is asymptotically sufficient.

#### B. ME: concentration is the reason GC becomes usable

For ME magnitude constraints, explain the scaling of a generic extensive occupation constraint \(C\):

\[
\langle C\rangle = O(T),\qquad
\operatorname{Var}(C)=O(T),
\]

therefore

\[
\sigma_C=O(\sqrt T),\qquad
\frac{\sigma_C}{\langle C\rangle}=O(T^{-1/2})\to 0.
\]

The exact constraint(s) used in the derivation must be matched to the thesis/MENoBiS equations. Do not apply the formula mechanically to quantities for which it has not been established.

The critical wording is:

> Absolute fluctuations need not vanish; **relative fluctuations do**.

Then state the practical implication:

> For the ME magnitude-only cases covered by this scaling, the soft GC constraints become sharply concentrated around their target values. Consequently, for the relevant smooth network magnitude/topology observables, the GC calculation can be used asymptotically in place of the corresponding harder constrained calculation.

Do not claim exact finite-\(T\) equality of all observables.

#### C. Entropy and saddle-point mechanism

Connect the fluctuation result to the thesis entropy calculation.

Use the existing derivation/citations and summarize only the essential mechanism:

\[
Z \sim \int d\lambda\, e^{T\phi(\lambda)}
\]

or the repository's equivalent expression.

Required explanation:

- the leading entropy/action is proportional to \(T\);
- the saddle-point contribution therefore dominates increasingly strongly as \(T\) grows;
- deviations around the saddle are subleading;
- the relative oscillations of the soft magnitude constraints go to zero;
- this is why the GC description becomes asymptotically interchangeable for the stated ME calculations.

Do not add a long Laplace-method derivation if the thesis already contains it. Link/cite it.

#### D. The conditional identity is secondary

If useful, include no more than one compact note:

> Conditioning the soft ensemble on the exact constraint values recovers the corresponding hard constrained measure by construction.

Then immediately state that this is **not** the large-\(T\) result being discussed. The relevant result is that in ME the conditioning becomes asymptotically unnecessary for the stated observables because the soft constraints concentrate.

Do not create a separate section around Poisson-factor cancellation.

#### E. W: failure of concentration

W is the mandatory comparison family.

Verify the exact MENoBiS W mean/variance formulas and `layers` convention before writing the asymptotics.

The page must establish that W contains a fluctuation contribution quadratic in the occupation scale, schematically

\[
\operatorname{Var}(C)=O(T^2)
\]

for the relevant high-occupation regime, or the precise equivalent implied by the MENoBiS parameterization.

Thus

\[
\frac{\sigma_C}{\langle C\rangle}=O(1)
\]

rather than tending to zero.

State plainly:

> Increasing occupation does not make W soft constraints relatively sharp. The hard constraint remains macroscopically relevant, so the GC ensemble is not generally interchangeable with the harder W ensemble even as \(T\to\infty\).

The exact asymptotic coefficient must be checked against the current W formulas. Do not insert a generic negative-binomial coefficient from memory.

#### F. Required ME vs W comparison table

Include exactly one compact table, maximum 6 rows:

| Property | ME | W |
|---|---|---|
| mean magnitude scale | \(O(T)\) | \(O(T)\) |
| variance scale | \(O(T)\) | contains \(O(T^2)\) term / exact verified form |
| relative fluctuation | \(O(T^{-1/2})\to0\) | remains \(O(1)\) |
| saddle concentration | sharpens with \(T\) | does not sharpen in the required sense |
| GC usable as asymptotic theoretical proxy? | yes, for stated magnitude-only observables | generally no |
| hard conditioning at large \(T\) | asymptotically weak for stated observables | remains relevant |

Use the exact W wording/formula verified from the thesis/code.

#### G. Binary constraints are the ME caveat

This is the second failure mode and must be stated separately from W.

Define

\[
a_{ij}=\mathbf 1[t_{ij}>0],
\qquad
E=\sum_{ij}a_{ij},
\qquad
k_i^{out}=\sum_j a_{ij},
\qquad
k_j^{in}=\sum_i a_{ij}.
\]

Explain:

- these are support variables, not occupation magnitudes;
- at fixed \(N\), increasing \(T\) can make an already occupied pair arbitrarily large without changing \(a_{ij}\);
- therefore these constraints do not acquire the same \(T^{-1/2}\) relative-concentration mechanism;
- the pure-magnitude ME saddle-point/interchangeability argument cannot simply be transferred once \(E\), \(k\), or another binary support constraint is present.

Required sentence, or extremely close equivalent:

> In ME, grand-canonical calculations are asymptotically interchangeable in the large-occupation limit only for the magnitude-constraint setting to which the concentration argument applies; binary constraints must be checked separately.

Use current fixed `(E,T)` and fixed `(k,T)` as examples. Mention future `(s,E)` and `(s,k)` only if needed to clarify the taxonomy; do not discuss their algorithms.

#### H. Consequence for model choice and theoretical work

Give the reader a concise practical rule:

- **ME + magnitude-only constraints + large \(T\):** GC is the preferred analytical proxy when the observable lies within the equivalence regime; use the harder ensemble when exact realization-level constraints or finite-\(T\) corrections matter.
- **ME + binary/support constraints:** do not assume GC interchangeability merely because \(T\) is large; inspect the support-constraint fluctuations/derivation.
- **W:** do not use large occupation alone to justify replacing the constrained ensemble with GC; its relative fluctuations remain non-negligible.

This theoretical-calculation consequence is the main purpose of the page.

Do not suggest collapsing MENoBiS ensemble APIs. Computational interchangeability in an asymptotic calculation is not the same as identical model semantics.

#### I. Limits of the claim

Keep this short.

Do not claim equivalence for:

- arbitrary microscopic pair probabilities;
- tail/extreme statistics without derivation;
- observables discontinuous in the fluctuating constraints;
- binary support observables;
- W by analogy with ME.

Do not introduce a separate large-\(N\) discussion. Simply state once that \(N\) is held fixed throughout.

### 3.4 References

Use existing bibliography/citations already present in the repository where possible. Add only well-established ensemble-equivalence/statistical-mechanics references if needed.

Do not add a long literature review. Maximum 3–5 references specifically for this page.

Where the repository already cites Coolen or the thesis for the relevant theory, reuse those references consistently.

### 3.5 Cross-links

Add only minimal links:

- `choose-null-model.md` → ensemble-equivalence page: one short paragraph or note;
- `microcanonical.md` → ensemble-equivalence page: one sentence near motivation/model choice;
- notebook → ensemble-equivalence page: one short link in explanatory prose.

Do not replicate the full content elsewhere.

Gate A: the scientific page must make it impossible for a careful reader to confuse **large occupation** with **large N**, or **observable-level agreement** with equality of probability measures.

---

## 4. Deliverable B — update `main-use-cases.ipynb`

### 4.1 Preserve the notebook's role

The notebook is a **main-use-cases notebook**, not a benchmark suite, implementation tutorial, or research paper.

Preserve the current applied flow:

- generate an observed non-binary network;
- filtering workflow;
- sampling/network-magnitude workflow;
- partial constraints where currently useful.

Do not turn the notebook into an exhaustive tour of all constraints.

### 4.2 Add one concise “Microcanonical sampling” section

Add a clear section after the main sampling workflow and before the ensemble-equivalence experiment, or in the closest logically equivalent location.

The section must demonstrate the new microcanonical API with **three compact cases**:

1. fixed `(E,T)` / `Constraint.EDGES_EVENTS`;
2. fixed `(k,T)` / `Constraint.DEGREE_EVENTS`;
3. fixed strengths / `Constraint.STRENGTH`.

Do **not** add a full strength-cost microcanonical run to the main notebook if it materially increases runtime. Instead, mention it in one markdown sentence with a link to the dedicated microcanonical tutorial/concept page.

Do not demonstrate all three ME/B/W families. Use **ME for the primary notebook examples** unless there is a compelling existing lightweight B/W example that can be retained without increasing complexity. Mention that the APIs also support the implemented B/W cases.

### 4.3 Required invariant checks

Each microcanonical example must visibly verify its hard constraints using simple calculations/dataframes/assertions.

#### fixed `(E,T)`

Show:

- sampled occupied pair count equals target `E`;
- sampled total occupation equals target `T`.

#### fixed `(k,T)`

Show:

- sampled out-degree sequence equals target out-degree sequence;
- sampled in-degree sequence equals target in-degree sequence;
- sampled total occupation equals target `T`.

#### fixed strengths

Show:

- sampled out-strength sequence equals target out-strength sequence;
- sampled in-strength sequence equals target in-strength sequence.

Use concise output. Prefer one small dataframe per case or one combined summary table rather than printing full vectors unnecessarily.

### 4.4 Explicit unsupported-case note

Add a single short note:

> Fixed `(s,E)` and fixed `(s,k)` microcanonical constraints are not currently implemented.

Do not discuss their planned algorithms in this notebook.

### 4.5 Rework the existing ensemble-equivalence notebook section

Rework the existing “ME fixed-strength ensemble equivalence” cells into a **fixed-`N`, large-`T`, ME-versus-W demonstration of when the GC ensemble is or is not a valid theoretical proxy**.

Do not make conditional equality the experimental result.

The notebook must tell this compact story:

1. Fix a small node count \(N\) and a normalized magnitude/strength profile.
2. Increase total occupation over exactly three practical scales, for example \(T,4T,16T\).
3. In ME, show that relative fluctuations of the soft magnitude constraints decrease approximately as \(T^{-1/2}\).
4. Show that a selected nontrivial network statistic computed from the GC ensemble approaches the corresponding hard-constraint result as occupation grows.
5. Repeat the fluctuation comparison for W and show that relative fluctuations remain non-negligible; the same GC substitution is therefore not justified.
6. Use one binary-constrained ME example from the microcanonical examples to show why high occupation alone is insufficient once support information is constrained.

The notebook illustrates the analytical argument; it is not a proof.

#### Experiment design

Keep \(N\) fixed throughout.

Use the same normalized strength/magnitude profile across occupation scales. Increase the occupation scale without changing the scientific interpretation of the model.

Before coding:

- verify the ME mean/variance scaling from the current equations;
- verify the W mean/variance and `layers` parameterization;
- determine which hard ensemble provides the cleanest comparison for the selected statistic using the current public API.

For **ME**:

- sample the relevant GC model at each occupation scale;
- estimate the mean and standard deviation of the soft magnitude constraint(s);
- calculate the coefficient of variation;
- compare one selected network statistic against the corresponding hard-constraint ensemble;
- the expected result must show decreasing relative constraint fluctuations and convergence of that statistic.

For **W**:

- repeat the same occupation scaling with the closest scientifically matched W model;
- calculate the same relative fluctuation quantity;
- compare the same or directly analogous network statistic where the public API permits;
- the expected result is that relative fluctuations do not vanish in the ME fashion and the GC-vs-hard difference remains relevant.

Do not force exact symmetry between ME and W APIs if the mathematically corresponding hard ensemble is not exposed. The scientific comparison is fluctuation scaling and the validity of GC as a proxy, not API symmetry.

Do not use B in this experiment.

### 4.6 Required observables

Use **two quantities maximum**:

1. **Required:** coefficient of variation of a relevant soft magnitude constraint or magnitude aggregate.
2. **Required:** one nontrivial network topology/magnitude statistic whose GC and hard-ensemble predictions can meaningfully be compared.

The second statistic must not be identically fixed by the microcanonical constraint.

Prefer a statistic already used elsewhere in MENoBiS rather than inventing a new diagnostic.

The notebook must make the theoretical inference visually/readably possible:

- ME: constraint CV \(\to0\), and GC/hard statistic difference decreases.
- W: constraint CV remains non-negligible, and no corresponding GC interchangeability should be inferred.

### 4.7 Required visualization

Use at most **two figures total** for ensemble equivalence.

**Figure 1 — mandatory**

- x-axis: total occupation \(T\) or occupation scale;
- y-axis: coefficient of variation;
- show ME and W;
- optionally overlay/reference the \(T^{-1/2}\) ME scaling without turning the notebook into a fit exercise.

**Figure 2 — optional but preferred**

- x-axis: total occupation \(T\);
- y-axis: the selected network statistic or normalized difference between GC and hard-ensemble estimates;
- show the ME approach explicitly;
- include W only if the comparison is scientifically like-for-like.

Do not add decorative figures.

### 4.8 Required interpretation

End with a concise interpretation stating all of the following:

- throughout this experiment \(N\) is fixed and \(T\) increases;
- for ME magnitude constraints, the mean scales extensively while relative fluctuations vanish;
- the saddle-point approximation becomes sharp, so GC calculations become asymptotically interchangeable with the harder constrained ensemble for the demonstrated class of network statistics;
- this is the useful reason to use GC for theoretical calculations in that ME regime;
- W does not acquire the same relative concentration, so large occupation does not justify replacing the constrained W ensemble with GC;
- ME models with binary/support constraints (`E`, `k`, support indicators) also require separate care because those constraints do not scale with occupation like strengths/magnitudes;
- conditional equivalence is not the result being demonstrated.

Link once to the ensemble-equivalence concept page.

### 4.9 Notebook reproducibility/runtime gate

Use fixed seeds and only public MENoBiS APIs. Avoid `N=1000`. Regenerate outputs. Keep the added microcanonical + ME/W comparison under about two minutes if feasible; reduce network size/sample counts before altering the scientific comparison.

Gate B: the rendered notebook must show precisely when GC becomes a usable large-occupation theoretical proxy in ME, why that inference fails for W, and why ME binary constraints remain a separate caveat.

## 5. Deliverable C — three-audience documentation review

Do not rewrite everything. Review each page according to its intended audience and make targeted edits.

### 5.1 Audience 1 — scientists using MENoBiS

Primary pages:

- `docs/index.md`
- `docs/getting-started.md`
- `docs/concepts/choose-null-model.md`
- tutorials
- `docs/api/python.md`
- CLI pages
- `docs/examples/main-use-cases.ipynb`

Required review criteria:

- first paragraph says what the page helps the user do;
- terminology is occupation-number based and consistent;
- mathematical detail is enough to interpret the model but not implementation-heavy;
- examples use current high-level APIs;
- limitations are visible before users commit to expensive runs;
- model-choice guidance distinguishes expected vs hard constraints;
- no historical refactor terminology unless needed for migration/history.

Edits should be small. Delete redundant architecture or implementation detail from scientist-facing pages and link to Development/Maths instead.

### 5.2 Audience 2 — developers extending MENoBiS

Primary pages:

- `docs/development/architecture.md`
- `docs/development/responsibilities.md`
- `docs/development/testing.md`
- `docs/development/scalability.md`
- `docs/development/benchmarking.md`
- `docs/development/extending-thesis-cases.md`
- Rust API reference

Required review criteria:

- module responsibility boundaries match the current code;
- dependency direction is explicit;
- scalable production vs oracle-only algorithms are clearly separated;
- adding a new constraint/family points developers toward the correct abstraction (`PreparedProblem`, shared constraints, `SamplingPlan`, backend, routing/capability layer);
- tests are described by level: unit, Python integration/E2E, oracle, heavy/scaling;
- no old stub-matching/direct/exact-production claims remain;
- no instructions encourage `O(N^2)`/exact production fallbacks for marginal small-N cases.

Do not add long class-by-class documentation. Developers can use Rust/Python API reference for symbols.

### 5.3 Audience 3 — science/maths/sampling readers

Primary pages:

- `docs/concepts/equations.md`
- `docs/concepts/microcanonical.md`
- new ensemble-equivalence page
- `docs/concepts/solvers-and-scaling.md`
- `docs/thesis-context.md`

Required review criteria:

- constraints and ensemble semantics are mathematically unambiguous;
- ME/B/W terminology and occupation support are consistent;
- exact vs expectation constraints are correctly labeled;
- microcanonical target measures/degeneracies are stated without unnecessary implementation detail;
- MCMC claims use “at stationarity” where appropriate;
- feasibility/convergence claims do not overstate mathematical guarantees;
- empirical validation is distinguished from proof;
- large-occupation ensemble equivalence is framed at fixed `N` as a practical GC-interchangeability result for ME magnitude constraints, with W and binary-constraint failure modes stated explicitly.

### 5.4 Specific wording corrections required

Review and fix at least these known issues if still present on `master`:

1. In `docs/development/architecture.md`, avoid a blanket statement that microcanonical constraints are “all exact” because strength+cost has expected cost.
2. In `docs/concepts/microcanonical.md`, review wording equivalent to “constructor plus repairs always yield a feasible state.” Replace theorem-like absolute wording with a precise statement about supported feasible instances and targeted repair guarantees unless completeness is actually established and documented.
3. Remove any remaining wording suggesting stub matching is a production strength sampler.
4. Keep fixed `(s,E)` and `(s,k)` consistently marked as unavailable/deferred.

### 5.5 Concision pass

After correctness edits, perform a second pass whose only goal is shortening.

For each changed page:

- remove duplicated definitions already given elsewhere;
- replace repeated model tables with links where possible;
- remove release/refactor history from conceptual pages;
- remove obvious statements a code example already demonstrates;
- collapse repeated warnings into one authoritative warning;
- avoid consecutive “TL;DR”, note, warning, and summary blocks saying the same thing.

Target: except for the new ensemble-equivalence page and notebook additions, total prose across existing docs should not increase materially.

Gate C: each public page should have one obvious primary audience. Cross-audience information should be linked, not duplicated.

---

## 6. Documentation ownership map to preserve

Use this as the duplication-prevention rule.

| Topic | Authoritative location |
|---|---|
| First-use workflow | `getting-started.md` |
| End-to-end applied examples | `main-use-cases.ipynb` |
| Which null model/ensemble to choose | `choose-null-model.md` |
| Core mathematical equations | `equations.md` |
| Ensemble equivalence / large occupation | new `ensemble-equivalence.md` |
| Microcanonical target/samplers/constraints | `microcanonical.md` |
| Practical microcanonical API workflow | `tutorials/microcanonical-sampling.md` |
| Solver caveats | `solvers-and-scaling.md` |
| Scaling guarantees/results | `development/scalability.md` |
| Benchmark methodology | `development/benchmarking.md` |
| Code/module responsibilities | `development/architecture.md` + `responsibilities.md` |
| Extension workflow | `development/extending-thesis-cases.md` |
| Validation/test strategy | `development/testing.md` |
| Function signatures | API reference |
| Deferred work | `development/todos.md` |

If content belongs to another row, link to it instead of duplicating it.

---

## 7. Implementation order

Follow this exact order.

### Phase 1 — inventory and edit map

1. Pull current `master`.
2. Read the files listed in Section 2.
3. Produce a private edit map: file → exact sections/cells to edit → purpose.
4. Confirm the current notebook's existing fixed-strength ME ensemble-comparison cells.
5. Identify existing citations for ensemble equivalence/concentration.

No content changes yet.

### Phase 2 — scientific concept page

1. Write `ensemble-equivalence.md` within the required structure and length.
2. Add one nav entry.
3. Add minimal cross-links from `choose-null-model.md` and `microcanonical.md`.
4. Check mathematical language against current ensemble semantics.

Gate: page passes the scientific precision requirements in Section 3.

### Phase 3 — notebook microcanonical update

1. Add the three compact microcanonical cases.
2. Add invariant checks.
3. Replace/rework the old fixed-strength ME equivalence section into the large-occupation scaling experiment.
4. Add at most two figures.
5. Add the unsupported `(s,E)` / `(s,k)` note.
6. Re-run notebook from top to bottom.

Gate: all examples execute and all hard-constraint checks pass.

### Phase 4 — targeted three-audience docs audit

Review the pages in Section 5. Make only required correctness, audience, duplication, and concision edits.

Do not modify files merely to “improve style” if they are already clear and correct.

### Phase 5 — validation

Run the repository's normal documentation and fast-test gates. At minimum:

- strict MkDocs build;
- notebook execution/render validation;
- Python tests relevant to public sampling/routing APIs;
- microcanonical fast tests relevant to the examples;
- formatting/lint checks required by the repository.

Do not run or modify unrelated heavy benchmark matrices unless a changed scientific claim depends on them.

### Phase 6 — final audit report

Produce a short implementation report containing only:

- files changed;
- notebook examples added/reworked;
- scientific claims added/clarified;
- docs wording problems fixed;
- validation commands and results;
- any unresolved issue that blocked a required item.

Maximum approximately 500 words.

---

## 8. Acceptance checklist

All boxes must be satisfied before completion.

### Ensemble equivalence

- [ ] One concise authoritative explanation exists.
- [ ] The regime is explicitly fixed \(N\), \(T\to\infty\).
- [ ] Conditional equality is not the main argument.
- [ ] ME magnitude constraints have verified \(O(T)\) means and \(O(T)\) variance in the stated regime.
- [ ] ME relative fluctuations are shown to vanish as \(O(T^{-1/2})\).
- [ ] The saddle-point / leading-entropy explanation is stated correctly.
- [ ] The practical conclusion is explicit: GC can be used as an asymptotic theoretical proxy for the stated ME magnitude/topology observables.
- [ ] W is the explicit comparison family and its large-\(T\) relative fluctuations remain non-negligible according to the verified MENoBiS formula.
- [ ] Large \(T\) is explicitly *not* used to justify GC substitution for W.
- [ ] Binary/support constraints are explicitly identified as an ME failure/caution case.
- [ ] Claims do not exceed the observables and constraint classes covered by the derivation.

### Notebook

- [ ] Existing fixed-strength ensemble-equivalence section was reworked, not duplicated.
- [ ] Fixed `(E,T)` microcanonical example exists.
- [ ] Fixed `(k,T)` microcanonical example exists.
- [ ] Fixed-strength microcanonical example exists.
- [ ] Exact hard constraints are visibly checked.
- [ ] Fixed `(s,E)` and `(s,k)` are explicitly marked unavailable.
- [ ] Large-occupation experiment holds `N` fixed and scales `T`.
- [ ] ME constraint coefficient of variation visibly decreases with the expected large-`T` behavior.
- [ ] A nontrivial ME network statistic shows GC/hard-ensemble convergence.
- [ ] W relative fluctuations remain visibly non-negligible over the scaling experiment.
- [ ] The notebook does not infer GC interchangeability for W.
- [ ] One binary-constrained ME example is explicitly used as the caveat to the pure-magnitude result.
- [ ] At most 2 new equivalence figures.
- [ ] Notebook executes cleanly and deterministically.

### Three audiences

- [ ] User-facing pages prioritize tasks and model choice.
- [ ] Developer pages prioritize extension boundaries, scalability, and testing.
- [ ] Science pages prioritize mathematical semantics and sampling validity.
- [ ] Significant duplicate explanations were removed or replaced with links.
- [ ] Existing docs did not grow materially apart from the required new material.

### Known wording issues

- [ ] No blanket “microcanonical = all exact” wording remains where cost-in-expectation is included.
- [ ] No unjustified absolute feasibility claim remains.
- [ ] No stale production stub-matching description remains.
- [ ] Deferred fixed `(s,E)` and `(s,k)` status is consistent everywhere.

### Validation

- [ ] `mkdocs` strict build passes.
- [ ] Notebook executes/render checks pass.
- [ ] Relevant fast Python/microcanonical tests pass.
- [ ] No public API changes were introduced.
- [ ] No production algorithm changes were introduced.

---

## 9. Explicitly forbidden scope expansion

Do **not** do any of the following during this task:

- implement fixed `(s,E)` or fixed `(s,k)`;
- refactor Rust sampling architecture;
- create a universal MCMC abstraction;
- redesign capability/routing APIs;
- add new benchmark infrastructure;
- run a new N=1000 matrix merely for documentation;
- add several new conceptual pages;
- rewrite every documentation page for tone consistency;
- add long derivations copied from the thesis;
- add speculative mathematical claims about thermodynamic ensemble equivalence;
- rename established ME/B/W concepts;
- modify archived agent specifications;
- change documentation navigation beyond the single ensemble-equivalence entry and any required broken-link correction.

When uncertain, make the **smaller edit**.

---

## 10. Definition of done

The task is complete when a new scientist can use the main notebook to discover and verify MENoBiS microcanonical sampling, a scientifically interested reader can understand when large-occupation ME calculations may use GC interchangeably, why W does not permit that substitution, and why binary-constrained ME cases require separate care, and a developer can still navigate the existing architecture/testing documentation without additional clutter.

The final result should make MENoBiS documentation **more precise and easier to navigate, not larger for its own sake**.
