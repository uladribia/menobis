# MENoBiS Microcanonical Framework

**Version:** August 2026

## 1. Purpose

This document defines the scientific and architectural principles governing the implementation of canonical and microcanonical ensembles in MENoBiS.

MENoBiS is a Rust/Python library for maximum-entropy models of non-binary networks. The project already contains a complete grand-canonical framework, family abstractions for ME, W and B, fitting and filtering infrastructure, sparse graph representations, benchmarking support, and a layered Rust/Python architecture.

The purpose of the microcanonical project is not to add disconnected samplers. It is to extend MENoBiS into a unified framework supporting:

- Grand Canonical ensembles;
- Canonical ensembles;
- Microcanonical ensembles;

for all supported non-binary families and constraint classes.

The mathematical framework developed in the doctoral thesis is the scientific specification. The repository implements that framework; the implementation must not redefine it.

Primary scientific reference:

> O. J. Sagarra, *Non-binary Maximum Entropy Network Ensembles and their Application to Urban Mobility*, Universitat de Barcelona.

## 2. Scope

This document establishes the common rules for all implementation phases. Individual phase documents must provide the detailed mathematical derivation, algorithms, software architecture, tests, benchmarks, and completion criteria for a particular ensemble and constraint set.

The planned implementation order is:

1. ME with fixed \((E,T)\);
2. W and B with fixed \((E,T)\);
3. fixed \((k,T)\);
4. fixed strengths;
5. fixed strengths with expected cost;
6. fixed \((s,E)\);
7. fixed \((s,k)\);
8. optimized backbone samplers, pseudo-marginal methods, and advanced MCMC kernels.

## 3. Terminology

MENoBiS models integer occupation numbers.

For a pair \((i,j)\), the central variable is

\[
t_{ij}\in \mathbb{Z}_{\ge 0}.
\]

Preferred terms are:

- occupation number;
- `OccNum`;
- occupied pair;
- occupation distribution;
- occupation histogram.

An edge is simply a pair with

\[
t_{ij}>0.
\]

Zero is an admissible occupation number unless excluded by a local support rule. Observed occupation numbers must not be called weights in the code or public documentation.

The ME family may still be interpreted in terms of distinguishable events, because that interpretation determines its degeneracy, but individual events are not stored explicitly by the implementation.

## 4. Existing generation pipeline

Microcanonical generation must reuse the same preprocessing and reconstruction pipeline as the existing grand-canonical code.

The logical flow is:

```text
User parameters
      |
      v
Validation
      |
      v
Mask and fixed-occupation prefilter
      |
      v
Residual problem
      |
      v
Sampling backend
      |
      v
Residual graph
      |
      v
Reconstruction with fixed occupations
      |
      v
Final validation
      |
      v
Output graph
```

A sampler must never be given the raw user problem when preprocessing can reduce it to a simpler residual problem.

## 5. Masks and fixed occupations

MENoBiS supports restrictions that may make some pair occupations:

- forbidden;
- fixed to zero;
- fixed to a positive integer;
- free and admissible for sampling.

These restrictions must be resolved before the sampling backend runs.

The prefilter is responsible for:

1. validating mask dimensions and consistency;
2. removing forbidden pairs from the admissible domain;
3. extracting all fixed occupations;
4. computing their contribution to every hard observable;
5. subtracting those contributions from the original constraints;
6. constructing the residual admissible pair set;
7. rejecting infeasible residual problems before sampling.

The sampler receives only the residual problem. It must not branch on mask types, fixed occupations, self-loop policies, or forbidden pairs.

## 6. Residual formulation

Suppose the original problem fixes the number of occupied pairs and total occupation:

\[
E=\sum_{ij}\mathbf{1}(t_{ij}>0),
\qquad
T=\sum_{ij}t_{ij}.
\]

Let the fixed part contain \(E_{\mathrm{fix}}\) positive pairs and total occupation \(T_{\mathrm{fix}}\). The sampling backend solves

\[
E_{\mathrm{res}}=E-E_{\mathrm{fix}},
\qquad
T_{\mathrm{res}}=T-T_{\mathrm{fix}}.
\]

The residual admissible pair set contains only pairs whose occupation remains to be sampled.

After sampling, the reconstruction stage merges the residual sample with the fixed occupations. Final validation checks the original, not merely residual, constraints.

This residual formulation is mandatory for every microcanonical sampler.

## 7. Families

MENoBiS supports three non-binary families.

### 7.1 ME

ME represents distinguishable events. Its degeneracy is

\[
D_{\mathrm{ME}}(\mathbf{t})
=
\frac{T!}{\prod_{ij}t_{ij}!}.
\]

The local support is

\[
t_{ij}\in\mathbb{Z}_{\ge 0}.
\]

### 7.2 W

W represents indistinguishable events distributed among \(M\) layers. Its degeneracy is

\[
D_{\mathrm{W}}(\mathbf{t})
=
\prod_{ij}
\binom{M+t_{ij}-1}{t_{ij}}.
\]

The local support is unbounded above:

\[
t_{ij}\in\mathbb{Z}_{\ge 0}.
\]

### 7.3 B

B represents the aggregation of \(M\) binary layers. Its degeneracy is

\[
D_{\mathrm{B}}(\mathbf{t})
=
\prod_{ij}
\binom{M}{t_{ij}},
\]

with bounded local support

\[
0\le t_{ij}\le M.
\]

## 8. Family abstraction

A family is mathematically defined by:

1. its local support;
2. its degeneracy.

From these two ingredients follow its pair law, generating function, local log-weight, and acceptance ratios.

The family layer should provide, at minimum:

- local support checks;
- minimum and maximum admissible occupation;
- local log-degeneracy;
- local log-degeneracy differences;
- reusable pair-distribution and generating-function logic already present in MENoBiS.

A family must not implement:

- constraints;
- graph moves;
- direct sampling backends;
- MCMC control logic;
- mask handling;
- preprocessing.

Filtering and generation must share the same mathematical family layer.

## 9. Constraint abstraction

Constraints define observables, not families or algorithms.

A constraint specification should distinguish:

- hard observables, satisfied exactly;
- expected observables, enforced through fitted parameters;
- residual values after preprocessing;
- feasibility conditions;
- updates under local moves.

Target constraints include:

- fixed \((E,T)\);
- fixed \((k,T)\);
- fixed strengths;
- fixed strengths plus expected cost;
- fixed \((s,E)\);
- fixed \((s,k)\).

The strength-cost ensemble remains mixed: strengths are hard, while cost is expected, because an exactly fixed global cost is not considered practical for the intended applications.

## 10. Moves

Moves preserve hard constraints. They are separate from both family laws and sampling backends.

Each move family must document:

- the state components it modifies;
- the hard observables it preserves;
- preconditions for feasibility;
- whether the proposal is symmetric;
- proposal-probability corrections when asymmetric;
- computational complexity;
- connectivity or ergodicity assumptions.

Examples include:

- occupation transfers;
- degree-preserving edge swaps;
- strength-preserving cycle moves;
- switch-and-hold graph moves.

The move definition must not contain the Metropolis acceptance probability. Acceptance belongs to the backend and is computed from the family degeneracy and proposal ratio.

## 11. Sampling backends

Sampling backends implement strategies, not mathematical families.

The preferred order is:

1. exact direct sampler;
2. exact conditional sampler;
3. Coolen-style backbone sampler;
4. joint-state occupation MCMC;
5. biased backbone MCMC;
6. approximate methods.

MCMC is a fallback, not the default.

A backend should interact only with:

- the residual constraint specification;
- the family interface;
- move or direct-sampling components;
- the sparse output builder;
- an explicit RNG.

No generic backend should contain family-name branches such as `if family == ME`. Family-specific mathematics must be supplied through abstractions or a family-specific direct algorithm explicitly scoped to that family.

## 12. Exact conditioning identity

For a grand-canonical law of the form

\[
P_{\mathrm{GC}}(\mathbf{t})
\propto
D(\mathbf{t})
\exp[-\boldsymbol{\theta}\cdot\mathbf{C}(\mathbf{t})],
\]

conditioning on

\[
\mathbf{C}(\mathbf{t})=\mathbf{c}
\]

gives

\[
P_{\mathrm{GC}}(\mathbf{t}\mid \mathbf{C}=\mathbf{c})
=
P_{\mathrm{MC}}(\mathbf{t}\mid \mathbf{C}=\mathbf{c}).
\]

This is an exact finite-size identity. It is not an asymptotic ensemble-equivalence statement.

It is therefore one of the strongest validation tools available: a microcanonical sampler can be tested against conditioned samples from the corresponding grand-canonical implementation.

## 13. Ensemble equivalence

Exact conditioning must not be confused with thermodynamic equivalence.

Observable equivalence between grand-canonical and microcanonical ensembles can fail, especially for:

- sparse limits;
- binary constraints;
- W close to its convergence boundary;
- B close to saturation;
- extensive local constraints.

Differences between unconditional grand-canonical and microcanonical observables are not automatically implementation errors.

Correctness must first be established through finite-size identities, exact enumeration, and hard-constraint validation.

## 14. Sparse implementation requirements

MENoBiS must not allocate \(O(N^2)\) memory merely because the underlying graph has \(N\) nodes.

Microcanonical implementations should use:

- compact admissible-pair indexing;
- sparse vectors of occupied pairs;
- occupation arrays of length \(E\), not \(N^2\);
- streaming or indexed mask iterators;
- reusable scratch buffers;
- bounded auxiliary maps only where unavoidable.

Every phase specification must state its time and memory complexity in terms of the residual problem size.

## 15. Rejection sampling policy

Rejection sampling is permitted only when all of the following are provided:

- a precomputed or estimated acceptance probability;
- an activation threshold;
- a bounded retry count;
- a non-rejection fallback that preserves correctness.

A rejection-based algorithm must never retry indefinitely.

## 16. Validation hierarchy

Every sampler should eventually satisfy:

1. exact enumeration on very small systems;
2. the conditioned grand-canonical identity;
3. exact hard constraints;
4. mask and reconstruction correctness;
5. detailed balance for MCMC implementations;
6. connectivity or ergodicity of the move graph;
7. agreement between independent exact and MCMC implementations where both exist;
8. integration with the benchmark framework.

Only after these checks pass should thermodynamic-limit or ensemble-equivalence experiments be interpreted scientifically.

## 17. Repository organization

The intended top-level structure is:

```text
generation/
    grandcanonical/
    canonical/
    microcanonical/
```

Within `microcanonical/`, shared components should be separated from phase- or family-specific implementations. Exact module names should follow the existing repository conventions rather than introducing an unrelated parallel architecture.

Likely shared responsibilities include:

- residual constraint types;
- support sampling;
- occupation-allocation backends;
- move traits;
- backend selection;
- errors;
- validation helpers.

## 18. Implementation discipline

Before adding new code, contributors must inspect and reuse existing:

- family laws;
- generating functions;
- pair distributions;
- masks;
- preprocessing;
- sparse graph builders;
- validation logic;
- RNG conventions;
- benchmark conventions;
- Python/Rust API patterns.

