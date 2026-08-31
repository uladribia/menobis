---
description: The six structural constraints, their grand-canonical and microcanonical semantics, the zero-inflated layer, and feasibility.
---

# Constraints

## TL;DR

A **constraint** is a structural quantity you decide to control in the null
model. A grand-canonical model matches it **in expectation**; a
microcanonical model fixes it **exactly** (realization by realization),
except for the explicitly documented hybrid strength+cost route. Which
combinations are implemented is listed in the
[generated capability matrix](../guide/supported-models.md).

## The six constraints

### STRENGTH

Out and in strengths:

\[
s_i^{\mathrm{out}}=\sum_j t_{ij},
\qquad
s_j^{\mathrm{in}}=\sum_i t_{ij}.
\]

- **Grand canonical:** expected strengths match the input sequence,
  \(s_i^{\mathrm{out}}=\sum_j\mathbb E[t_{ij}]\).
- **Microcanonical:** row and column sums are exact.

### STRENGTH_COST

Total pair cost

\[
C(t)=\sum_{ij} t_{ij}d_{ij}.
\]

- **Grand canonical:** expected cost \(C^\star=\sum_{ij}\mathbb E[t_{ij}]d_{ij}\)
  is matched; expected strengths are also matched.
- **Microcanonical (hybrid):** strengths are exact; cost is matched **in
  expectation** through the cost multiplier \(\gamma\) (`f_{ij}=e^{-\gamma d_{ij}}`).
  This route is **hybrid**, never summarized as "everything exact".

### STRENGTH_EDGES

Occupied-pair count:

\[
E(t)=\sum_{ij}\mathbf 1[t_{ij}>0].
\]

- **Grand canonical:** expected strengths and expected \(E\) are matched.
- **Microcanonical:** strengths and \(E\) are exact.

### STRENGTH_DEGREE

Out/in degree sequences:

\[
k_i^{\mathrm{out}}=\sum_j a_{ij},
\qquad
k_j^{\mathrm{in}}=\sum_i a_{ij}.
\]

- **Grand canonical:** expected strengths and expected degree sequences are
  matched.
- **Microcanonical:** strengths and degree sequences are exact.

### DEGREE_EVENTS

Degree sequences plus total events:

\[
T=\sum_{ij} t_{ij}.
\]

- **Grand canonical:** expected degree sequences and expected \(T\) are matched.
- **Microcanonical:** degree sequences and \(T\) are exact.

### EDGES_EVENTS

Occupied-pair count plus total events;

\[
E=\sum_{ij} a_{ij},
\qquad
T=\sum_{ij} t_{ij}.
\]

- **Grand canonical:** expected \(E\) and expected \(T\) are matched; all
  pair parameters collapse to global multipliers (a global-parameter special
  case of the generic pair parameterization).
- **Microcanonical:** \(E\) and \(T\) are exact.

`EDGES_EVENTS` is one of the six official constraint types exposed by the
capability registry.

## The zero-inflated support layer

Constraints that involve \(E\) or the degree sequences condition on whether
a pair is occupied; they use a **zero-inflated** pair law. Define the
positive-support partition factor

\[
G_F(q)=\sum_{t\ge 1}d_F(t)q^t,
\]

which evaluates to

\[
G_{\mathrm{ME}}(q)=e^q-1,
\qquad
G_{\mathrm B}(q)=(1+q)^M-1,
\qquad
G_{\mathrm W}(q)=(1-q)^{-M}-1.
\]

With support fugacity \(\ell_{ij}\),

\[
Z_{ij}=1+\ell_{ij}G_F(q_{ij}),
\]

\[
P(t_{ij}>0)=\frac{\ell_{ij}G_F(q_{ij})}{1+\ell_{ij}G_F(q_{ij})},
\]

\[
\mathbb E[t_{ij}]
=
\frac{\ell_{ij}q_{ij}G_F'(q_{ij})}
     {1+\ell_{ij}G_F(q_{ij})}.
\]

Equivalently, the conditional mean on the positive support is

\[
\mathbb E[t_{ij}\mid t_{ij}>0]=\frac{q_{ij}G_F'(q_{ij})}{G_F(q_{ij})},
\]

which does **not** depend on the support fugacity \(\ell_{ij}\).

The support-aware constraints (`STRENGTH_EDGES`, `STRENGTH_DEGREE`,
`DEGREE_EVENTS`, `EDGES_EVENTS`) use this layer; `STRENGTH` and
`STRENGTH_COST` are non-zero-inflated.

## High-level constraint table

| Constraint | Symbolic quantities | GC semantics | Canonical semantics | MC semantics |
|---|---|---|---|---|
| Strength | \(s^{out},s^{in}\) | expected | registry-derived | exact |
| Strength + cost | \(s,C\) | expected | registry-derived | strengths exact; cost expected |
| Strength + edges | \(s,E\) | expected | registry-derived | exact |
| Strength + degree | \(s,k\) | expected | registry-derived | exact |
| Degree + events | \(k,T\) | expected | registry-derived | exact |
| Edges + events | \(E,T\) | expected | registry-derived | exact |

The "registry-derived" canonical cells reflect the current capability
registry; see the
[generated capability matrix](../guide/supported-models.md) for what is
actually implemented.

## Feasibility

Necessary relations for any feasible non-binary network:

\[
\sum_i s_i^{\mathrm{out}}=\sum_j s_j^{\mathrm{in}}=T,
\qquad
\sum_i k_i^{\mathrm{out}}=\sum_j k_j^{\mathrm{in}}=E.
\]

For positive-support edges (every occupied pair contributes at least one
event):

\[
s_i^{\mathrm{out}}\ge k_i^{\mathrm{out}},
\qquad
s_i^{\mathrm{in}}\ge k_i^{\mathrm{in}}.
\]

For B under fixed degree, each occupied pair holds at most \(M\) events:

\[
s_i^{\mathrm{out}}\le M k_i^{\mathrm{out}},
\qquad
s_i^{\mathrm{in}}\le M k_i^{\mathrm{in}}.
\]

For fixed \((E,T)\):

\[
E\le T,
\qquad
\text{and for B:}\quad T\le M E.
\]

Add domain bounds: no self loops requires \(k_i\le N-1\); with self loops
\(k_i\le N\); the strength budget must fit the admissible pair domain.

!!! warning "Necessary, not sufficient"
    These are **necessary** conditions. Some sparse-domain problems require
    additional feasibility conditions; simple scalar checks do not prove
    every sparse constrained instance feasible. In practice, derive
    constraints from a valid witness network as the documenting examples do.

## Fixed / known pairs

Fixed pairs are not a separate constraint enum. They transform the problem
to a residual domain after subtracting the fixed contributions (see
[Fixed / known pairs](../guide/fixed-pairs.md)).