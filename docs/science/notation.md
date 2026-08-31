---
description: Canonical notation used throughout the MENoBiS scientific documentation.
---

# Notation

## TL;DR

MENoBiS models **directed non-binary networks**: to every ordered node pair
\((i,j)\) we assign an integer **occupation number** \(t_{ij}\) counting
events. Everything else — families, ensembles, constraints, filtering — is
defined on top of this single object.

## Occupations and support

Let \(t_{ij}\in\{0,1,2,\ldots\}\) be the occupation number of the ordered
pair \((i,j)\). The network is **directed**: \((i,j)\) and \((j,i)\) are
different coordinates with their own occupation numbers.

The **binary support** of the network is the occupation indicator

\[
a_{ij}=\Theta(t_{ij})=\mathbf 1[t_{ij}>0].
\]

A pair with \(a_{ij}=1\) is an **occupied pair**.

The canonical MENoBiS sparse representation lists only occupied pairs as
three columns `source target occ_num`; unoccupied pairs are implicit.

## Symbols

| Symbol | Meaning |
|---|---|
| \(N\) | number of nodes |
| \(t_{ij}\) | occupation number of ordered pair \((i,j)\) |
| \(a_{ij}\) | binary support indicator, \(\mathbf 1[t_{ij}>0]\) |
| \(T\) | total occupation \(\sum_{ij} t_{ij}\) (total number of events) |
| \(E\) | occupied-pair count \(\sum_{ij} a_{ij}\) |
| \(s^{\mathrm{out}}_i,\ s^{\mathrm{in}}_i\) | out/in strength sequences |
| \(k^{\mathrm{out}}_i,\ k^{\mathrm{in}}_i\) | out/in binary degree sequences |
| \(d_{ij}\) | pair cost (e.g. Euclidean distance) |
| \(C(t)\) | total cost \(\sum_{ij} t_{ij}d_{ij}\) |
| \(M\) | B/W layer/degeneracy parameter |
| \(q_{ij}\) | pair fugacity/intensity |
| \(\ell_{ij}\) | support fugacity (zero-inflated models) |

## Derived quantities

Out and in strengths:

\[
s_i^{\mathrm{out}}=\sum_j t_{ij},
\qquad
s_j^{\mathrm{in}}=\sum_i t_{ij}.
\]

Out and in binary degrees:

\[
k_i^{\mathrm{out}}=\sum_j a_{ij},
\qquad
k_j^{\mathrm{in}}=\sum_i a_{ij}.
\]

Total occupation and occupied-pair count:

\[
T=\sum_{ij} t_{ij},
\qquad
E=\sum_{ij} a_{ij}.
\]

## Admissible pairs and self loops

The basic coordinates are **ordered pairs**. The self-loop policy changes
which pairs are admissible:

\[
\text{no self loops:}\quad L=N(N-1),
\qquad
\text{self loops allowed:}\quad L=N^2,
\]

where \(L\) is the number of candidate pairs.

## Terminology

Preferred MENoBiS vocabulary:

- **non-binary network** — a network with integer occupation numbers;
- **occupation number** — the integer event count \(t_{ij}\);
- **occupied pair** — a pair with \(a_{ij}=1\);
- **event** — one unit of occupation;
- **binary support** — the occupation indicator \(a_{ij}\);
- **strength** — out/in event sums;
- **degree** — out/in support sums.

The phrase **weighted network** appears in the historical literature
(thesis and related papers) as a synonym for non-binary networks. Within
MENoBiS documentation it is treated as historical literature terminology;
the primary terms are the ones above.