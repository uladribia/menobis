---
description: The ME, B and W event families — their degeneracies, pair laws and interpretation.
---

# Event families: ME, B, W

## TL;DR

MENoBiS offers three occupation families. They share the same constraint
machinery but differ in what a unit of occupation means and in the resulting
pair statistics. Choose a family by the **nature of the events**, not by
speed (see [Choose a model](../guide/choose-model.md)).

## What are ME, B and W?

Each family is defined by a pair degeneracy \(d_F(t)\): the number of
microscopic arrangements of \(t\) events on pair \((i,j)\). The degeneracy
drives the conditional occupation law given a pair fugacity \(q_{ij}\).

| Family | Name | Degeneracy \(d_F(t)\) | Occupation range | Interpretation |
|---|---|---|---|---|
| ME | Multi-edge | \(1/t!\) | \(t\ge 0\) | distinguishable events |
| B | Aggregated binary layers | \(\binom Mt\) | \(0\le t\le M\) | aggregate of \(M\) binary layers/trials |
| W | Weighted | \(\binom{M+t-1}{t}\) | \(t\ge 0\) | indistinguishable events |

## ME: distinguishable events

\[
d_{\mathrm{ME}}(t)=\frac{1}{t!},\qquad t\ge 0.
\]

\[
P_{\mathrm{ME}}(t_{ij}=t)=e^{-q_{ij}}\frac{q_{ij}^{t}}{t!}.
\]

\[
\mathbb E[t_{ij}]=q_{ij}.
\]

Interpretation: occupation counts distinguishable events (for instance,
individual trips in an origin–destination table). The Poisson pair law is
the maximum-entropy law given an expected occupation \(q_{ij}\).

## B: aggregated binary layers

\[
d_{\mathrm B}(t)=\binom Mt,\qquad 0\le t\le M.
\]

\[
P_{\mathrm B}(t_{ij}=t)=\binom Mt\frac{q_{ij}^{t}}{(1+q_{ij})^M}.
\]

\[
\mathbb E[t_{ij}]=M\frac{q_{ij}}{1+q_{ij}}.
\]

Interpretation: the occupation is the aggregate of \(M\) binary
layers/trials, each open with odds \(q_{ij}\). Occupations are bounded by
\(M\): \(t_{ij}\le M\). The special case \(M=1\) is a **Bernoulli**
(binary) pair.

## W: indistinguishable events

\[
d_{\mathrm W}(t)=\binom{M+t-1}{t},\qquad t\ge 0.
\]

\[
P_{\mathrm W}(t_{ij}=t)=\binom{M+t-1}{t}(1-q_{ij})^M q_{ij}^{t},
\qquad 0<q_{ij}<1.
\]

\[
\mathbb E[t_{ij}]=M\frac{q_{ij}}{1-q_{ij}}.
\]

Interpretation: occupation counts indistinguishable events (e.g.
multi-occupancy of a shared resource). The pair law is negative binomial;
the special case \(M=1\) is the **geometric** distribution.

!!! warning "Parameter domain"
    The W family requires \(q_{ij}\in(0,1)\). Solvers keep every fitted pair
    parameter inside this domain; infeasible or degenerate inputs surface as
    solver status messages rather than silently out-of-domain parameters.

## The B \(M=1\) invariant

For B with \(M=1\), \(t_{ij}\in\{0,1\}\), so support and occupation coincide:

\[
s_i^{\mathrm{out}}=k_i^{\mathrm{out}},
\qquad
s_i^{\mathrm{in}}=k_i^{\mathrm{in}}.
\]

This is a mathematical consequence of the Bernoulli pair law, not an
implementation quirk. It matters for feasibility: a B \(M=1\) fixed-degree
problem is only feasible when strengths equal degrees.

## Pair parameterization

The pair fugacity is written generically as

\[
q_{ij}=x_i y_j f_{ij}.
\]

For exponential spatial cost \(f_{ij}=e^{-\gamma d_{ij}}\) (see
[Spatial costs](../guide/spatial-costs.md)).

This factorization is a convenient parameterization, not a claim that every
route needs distinct node-specific multipliers. Global-constraint models
(notably `EDGES_EVENTS`) are special cases in which the multipliers collapse
to constants or global parameters.

## Comparison

| Question | ME | B | W |
|---|---|---|---|
| Occupation bound | unbounded | \(\le M\) | unbounded |
| Fitted pair param range | \(q>0\) | \(q>0\) | \(0<q<1\) |
| \(M\) layer parameter | not used | required | used |
| \(M=1\) special case | — | Bernoulli | geometric |
| Interpretation | distinguishable events | M binary layers | indistinguishable events |

!!! warning "Identical constraints do not make families interchangeable"
    The same strength sequence fitted under ME, B and W produces different
    pair statistics and different sampled networks. A comparison across
    families is a comparison of event interpretations, not a numerically
    interchangeable choice.

Family availability across ensembles and constraints is listed in the
[generated capability matrix](../guide/supported-models.md), included into the
supported-models guide and authoritative for what is supported today.