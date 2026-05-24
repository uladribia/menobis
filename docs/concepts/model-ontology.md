---
description: Source-of-truth model ontology for ODME families, ensembles, and constraints.
---

# Model ontology

## TL;DR

ODME has three independent choices: **ensemble**, **weight family**, and
**constraint layer**. `AGENTS.md` is the coding source of truth; this page is the
public, non-duplicated map used by the rest of the docs.

## Axes

| Axis | Values | Rule |
|---|---|---|
| Ensemble | grand canonical, canonical, microcanonical | B and W are grand-canonical only. ME also has canonical multinomial and microcanonical fixed-strength stub matching. |
| Family | ME, B, W | ME = Poisson; B = Binomial(M), with M=1 Bernoulli; W = Negative Binomial(M), with M=1 geometric. |
| Constraint | strength, strength-cost, strength-edges, strength-degree, degree-events, partial | Linear constraints use expected weights. Binary constraints use zero inflation. Partial fits subtract frozen pairs then call the matching full fit. |

## Constraint levels

| Level | Examples | Saturation rule |
|---|---|---|
| Global | total events, total binary edges, total cost | Saturated constraints are deducted from the free problem. |
| Node | strength and degree sequences | Nodes at capacity must be peeled or handled by a mixed solver. |
| Pair | frozen known pairs | Frozen pairs are removed from support and their contribution is subtracted. |

## Grand-canonical pair notation

Let

$$
q_{ij}=x_i y_j f_{ij},
$$

where $f_{ij}=1$ unless a cost constraint supplies
$f_{ij}=\exp(-\gamma d_{ij})$. Zero-inflated constraints add a raw binary
multiplier $\ell_{ij}$: node-factorized for degree constraints, or scalar for a
global edge-count constraint.

## Non-zero-inflated means

| Family | Expected weight | Domain |
|---|---|---|
| ME / Poisson | $\mathbb{E}[t_{ij}] = q_{ij}$ | $q_{ij}\in(0,\infty)$ |
| B / Binomial(M) | $\mathbb{E}[t_{ij}] = Mq_{ij}/(1+q_{ij})$ | $q_{ij}\in(0,\infty)$ |
| W / NegBin(M) | $\mathbb{E}[t_{ij}] = Mq_{ij}/(1-q_{ij})$ | $q_{ij}\in(0,1)$ |

## Zero-inflated partition factors

| Family | Positive-support factor $G_F(q)$ |
|---|---|
| ME / Poisson | $e^q - 1$ |
| B / Binomial(M) | $(1+q)^M - 1$ |
| W / NegBin(M) | $(1-q)^{-M} - 1$ |

For raw $\ell_{ij}$:

$$
\mathbb{E}[\Theta(t_{ij}>0)] =
\frac{\ell_{ij}G_F(q_{ij})}{1+\ell_{ij}G_F(q_{ij})},
$$

$$
\mathbb{E}[t_{ij}] =
\frac{\ell_{ij}q_{ij}G'_F(q_{ij})}{1+\ell_{ij}G_F(q_{ij})}.
$$

Equivalently,
$\mathbb{E}[t_{ij}\mid t_{ij}>0]=q_{ij}G'_F(q_{ij})/G_F(q_{ij})$.

## Zero-truncated means

| Family | $\mathbb{E}[t_{ij}\mid t_{ij}>0]$ |
|---|---|
| ME / Poisson | $q_{ij}e^{q_{ij}}/(e^{q_{ij}}-1)=q_{ij}/(1-e^{-q_{ij}})$ |
| B / Binomial(M) | $M q_{ij}(1+q_{ij})^{M-1}/((1+q_{ij})^M-1)$ |
| W / NegBin(M) | $M q_{ij}(1-q_{ij})^{-M-1}/((1-q_{ij})^{-M}-1)$ |

## Reuse rules

- Every family has its own solver; B and W must not call the ME solver and
  relabel its output.
- Constraint variants are composition: family kernel plus constraint layer.
- Rust owns numerical kernels, masks, excess computation, cost providers,
  gamma searches, IPF loops, and rate-table assembly.
- Python validates inputs, calls Rust, and wraps typed numpy/dataclass results.
- Cost providers are family-orthogonal: no cost, sparse triples, and Euclidean
  coordinates must work for ME, B, and W.

## See also

- [Thesis cases](thesis-cases.md) maps public names to this ontology.
- [Ontology conformance audit](../development/ontology-conformance-audit.md)
  tracks current code gaps against this source of truth.
