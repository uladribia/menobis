---
description: The statistics behind filtering — conditional p-values on positive support, absent-edge tests, and multiple-testing corrections.
---

# Filtering statistics

## TL;DR

Filtering compares each observed occupation with the fitted null
distribution and reports how many null networks would reach the observed
value. Because filtering only applies to occupied pairs, MENoBiS uses
p-values **conditioned on positive support**.

## Upper and lower p-values

For an occupied pair with observed occupation \(t_{ij}^{\mathrm{obs}}\),
the raw null tail probabilities are

\[
p^{\mathrm{raw,upper}}_{ij}=P(T_{ij}\ge t_{ij}^{\mathrm{obs}}\mid \text{null}),
\qquad
p^{\mathrm{raw,lower}}_{ij}=P(T_{ij}\le t_{ij}^{\mathrm{obs}}\mid \text{null}).
\]

MENoBiS filters **observed positive pairs**, so the reported p-values are
conditioned on the pair being occupied. With
\(\omega_{ij}=P(T_{ij}>0)\) the occupation probability under the null,

\[
p^{\mathrm{upper}}_{ij}
=
\frac{P(T_{ij}\ge t_{ij}^{\mathrm{obs}})}{\omega_{ij}},
\qquad
p^{\mathrm{lower}}_{ij}
=
\frac{P(T_{ij}\le t_{ij}^{\mathrm{obs}})-(1-\omega_{ij})}{\omega_{ij}},
\]

clamped to \([0,1]\). This is the positive-support conditioning actually
used by MENoBiS: an observed pair can only be "surprising" relative to
pairs that are occupied under the null.

## Absent-edge tests

Absent pairs (observed \(t_{ij}=0\)) are tested separately: an absent pair
is flagged when its expected occupation under the null is large enough that
observing zero is statistically unusual in the lower tail, bounded by
`min_occupation`, `min_expected`, and `max_absent`.

## Multiple testing

Each observed pair yields one hypothesis; a network with \(E\) occupied
pairs yields \(E\) tests.

- **No correction** — per-test \(\alpha\).
- **Bonferroni** — effective threshold \(\alpha/m\) (controls family-wise
  error; conservative).
- **FDR (Benjamini–Hochberg)** — controls the false discovery rate; less
  conservative, appropriate when screening many pairs.

The choice of \(m\) for Bonferroni and the FDR rank set is the number of
observed pairs under consideration. Document the correction you use.

## Selection of observed positive pairs

Filtering operates on the observed occupied pairs only (plus the separate
absent-pair scan). Cells with observed zero occupation are outside the
observed-pair filter domain unless absent-edge detection is enabled.

CLI mechanics for these options live in the
[Filter CLI](../cli/filter.md).