---
description: When grand-canonical calculations can substitute for harder constrained ensembles at large total occupation.
---

# Ensemble equivalence at large occupation

**TL;DR:** At fixed node count \(N\) and large total occupation \(T\), the
grand-canonical (GC) ensemble becomes an asymptotically interchangeable
theoretical proxy for the harder constrained ensemble in ME magnitude-only
cases. This does **not** hold for W, nor for ME with binary/support
constraints.

## Regime: fixed N, T → ∞

MENoBiS often permits the same scientific null hypothesis to be represented
with either soft (expectation) or hard (exact) constraints. The useful
question for analytical work is whether, at high occupation, one can
calculate in the simpler grand-canonical ensemble and obtain the same
relevant network statistics as in the corresponding harder constrained
ensemble.

The regime studied here is

\[
N \text{ fixed},\qquad T \to \infty,
\]

where \(T = \sum_{ij} t_{ij}\) is the total occupation (total number of
events). The node count \(N\) is held constant throughout. This is **not**
a thermodynamic limit; the concentration mechanism arises from the scaling
of occupation, not from system size.

!!! note "Conditional equality is not the answer"
    Conditioning the soft GC ensemble on the exact constraint values recovers
    the corresponding hard constrained measure by construction. That is a
    definitional identity, not the large-occupation result discussed here.
    The relevant result is that in ME the conditioning becomes asymptotically
    unnecessary for the stated observables because the soft constraints
    concentrate.

## ME: why the grand-canonical ensemble becomes interchangeable

For ME magnitude constraints (e.g., strength sequences), the occupation
variables are Poisson-distributed in the GC ensemble (see
[Equations](equations.md)):

\[
\mathbb{E}[t_{ij}] = q_{ij},\qquad \operatorname{Var}(t_{ij}) = q_{ij}.
\]

Consider a generic extensive magnitude constraint \(C = \sum_{ij} t_{ij}\)
(aggregate strength, or a subset thereof). In the GC ensemble:

\[
\langle C\rangle = O(T),\qquad \operatorname{Var}(C) = O(T).
\]

Hence the standard deviation scales as \(\sigma_C = O(\sqrt{T})\) and the
coefficient of variation satisfies

\[
\frac{\sigma_C}{\langle C\rangle} = O(T^{-1/2}) \to 0.
\]

**Absolute fluctuations need not vanish; relative fluctuations do.**

The practical implication: for the ME magnitude-only cases covered by this
scaling, the soft GC constraints become sharply concentrated around their
target values. Consequently, for smooth network magnitude and topology
observables that depend continuously on the occupation numbers, the GC
calculation can be used asymptotically in place of the corresponding harder
constrained calculation. This is **not** a claim of exact finite-\(T\)
equality of all observables.

## Saddle point and vanishing relative fluctuations

The concentration mechanism reflects a deeper saddle-point structure
familiar from the maximum-entropy formalism (thesis §3.2–3.3). The
grand-canonical partition function can be written schematically as

\[
Z \sim \int d\lambda\; e^{T\phi(\lambda)},
\]

where \(\lambda\) represents the Lagrange multipliers that enforce the
constraints and \(\phi(\lambda)\) is the intensive entropic action per unit
occupation. The leading term in the exponent is proportional to \(T\). As
\(T\) grows:

1. The saddle-point contribution dominates increasingly strongly.
2. Deviations around the saddle are subleading in \(T\).
3. The relative oscillations of the soft magnitude constraints around their
   target values vanish as \(T^{-1/2}\).

This is why the GC description becomes asymptotically interchangeable for
the stated ME calculations: the ensemble is sharply peaked around the
constraint values that define the corresponding hard ensemble, and smooth
observables coincide at leading order. See thesis §3.2–3.3 for the complete
Laplace-method derivation; see Coolen, Annibale & Roberts (2017), Chapter 2,
for the binary-network analogue.

## W: why the equivalence fails

W (negative-binomial, \(M\) layers) is the mandatory comparison family
because it shares the same multiplier structure \(q_{ij} = x_i y_j f_{ij}\)
but has fundamentally different fluctuation properties.

The MENoBiS W grand-canonical formulas (verified from the codebase) are:

\[
\mathbb{E}[t_{ij}] = \frac{M q_{ij}}{1 - q_{ij}},\qquad q_{ij}\in(0,1),
\]

\[
\operatorname{Var}(t_{ij}) = \mathbb{E}[t_{ij}] +
\frac{\mathbb{E}[t_{ij}]^2}{M}.
\]

For an aggregate magnitude \(C = \sum_{ij} t_{ij}\) over independent pairs:

\[
\operatorname{Var}(C) = \underbrace{\sum_{ij} \mathbb{E}[t_{ij}]}_{=O(T)}
\;+\; \frac{1}{M}\sum_{ij} \mathbb{E}[t_{ij}]^2.
\]

At fixed \(N\), the second sum contains terms of order \(O(T^2)\): for any
non-zero strength profile, scaling all pair expectations by a common factor
\(c\) grows \(\sum_{ij}\mathbb{E}[t_{ij}]^2\) as \(c^2\), i.e. as \(T^2\). Consequently

\[
\frac{\sigma_C}{\langle C\rangle} = O(1)
\]

and **does not tend to zero** as \(T\to\infty\).

The plain statement: increasing occupation does **not** make W soft
constraints relatively sharp. The hard constraint remains macroscopically
relevant, so the GC ensemble is not generally interchangeable with the
harder W ensemble even as \(T\to\infty\).

### ME vs W comparison

| Property | ME | W |
|---|---|---|
| Mean magnitude scale | \(O(T)\) | \(O(T)\) |
| Variance scale | \(O(T)\) | Contains \(O(T^2)\) term: \(\operatorname{Var} = \mathbb{E} + \mathbb{E}^2/M\) |
| Relative fluctuation | \(O(T^{-1/2}) \to 0\) | Remains \(O(1)\) |
| Saddle concentration | Sharpens with \(T\) | Does not sharpen in the required sense |
| GC usable as asymptotic theoretical proxy? | Yes, for stated magnitude-only observables | Generally no |
| Hard conditioning at large \(T\) | Vanishes for stated observables | Remains relevant |

## Binary constraints: the ME caveat

Even within ME, the magnitude-concentration argument does **not**
automatically extend to binary (support) constraints. Define

\[
a_{ij} = \mathbf{1}[t_{ij} > 0],\qquad
E = \sum_{ij} a_{ij},\qquad
k_i^{\text{out}} = \sum_j a_{ij},\qquad
k_j^{\text{in}} = \sum_i a_{ij}.
\]

These are support variables, not occupation magnitudes. At fixed \(N\),
increasing \(T\) can make an already-occupied pair arbitrarily large without
changing \(a_{ij}\). The binary constraints therefore **do not** acquire the
\(T^{-1/2}\) relative-concentration mechanism: a strongly occupied pair
contributes \(O(T)\) to a magnitude aggregate but contributes exactly 1 to
\(E\) regardless of its occupation.

The practical caveat:

> In ME, grand-canonical calculations are asymptotically interchangeable in
> the large-occupation limit only for the magnitude-constraint setting to
> which the concentration argument applies; binary constraints must be
> checked separately.

Current implemented examples include fixed \((E,T)\) (edges and events) and
fixed \((k,T)\) (degree sequence and events), both of which mix binary
support constraints with total occupation. Deferred cases (fixed \((s,E)\)
and fixed \((s,k)\)) would also involve mixed magnitude-and-support
constraints; apply the same caution.

## Consequences for theoretical calculations

The practical rule for model choice and analytical work:

- **ME + magnitude-only constraints + large \(T\):** The GC ensemble is the
  preferred analytical proxy. Use the harder constrained ensemble when exact
  realization-level constraints or finite-\(T\) corrections are required.
- **ME + binary/support constraints:** Do **not** assume GC
  interchangeability merely because \(T\) is large. Inspect the
  support-constraint fluctuations separately.
- **W:** Do **not** use large occupation alone to justify replacing the
  constrained ensemble with GC. The relative fluctuations remain
  non-negligible at all occupation scales.

This is a theoretical-calculation consequence. It does **not** suggest
collapsing MENoBiS ensemble APIs: computational interchangeability in an
asymptotic calculation is not the same as identical model semantics.

### Limits of the claim

The equivalence described above is bounded. Do not extend it to:

- Arbitrary microscopic pair probabilities (the result holds for the
  aggregate constraint structure, not for every individual pair
  probability).
- Tail or extreme-value statistics without a separate derivation.
- Observables that are discontinuous in the fluctuating magnitude
  constraints.
- Binary or support observables (as detailed above).
- W models, by analogy with ME.

Throughout, \(N\) is held fixed. The regime is **not** the large-\(N\)
thermodynamic limit, and no separate large-\(N\) equivalence is claimed.

## Numerical illustration in MENoBiS

The notebook [`examples/main-use-cases.ipynb`](../examples/main-use-cases.ipynb)
contains a fixed-\(N\), increasing-\(T\) experiment that demonstrates the
results above:

- **For ME:** the coefficient of variation of a soft magnitude constraint
  decreases as \(T^{-1/2}\), and a selected network statistic converges to
  its hard-ensemble value.
- **For W:** the same coefficient of variation remains of order unity
  across occupation scales.
- **For ME with fixed \((E,T)\):** the binary constraint does not become
  interchangeable at large \(T\).

See the notebook section **"Ensemble equivalence at large occupation"** for
the complete experiment, figures, and inference.

## References

1. O. Sagarra, *Non-binary maximum entropy network ensembles and their
   application to the study of urban mobility*, PhD thesis, 2015.
   [HDL 10803/400560](https://hdl.handle.net/10803/400560). — Primary
   reference for the Poisson/negative-binomial occupation families,
   saddle-point derivation, and ensemble-equivalence analysis (§3.2–3.3).

2. A. C. C. Coolen, A. Annibale, E. Roberts, *Generating Random Networks
   and Bipartite Graphs*, Oxford University Press, 2017.
   ISBN 978-0-19-879016-9. — Comprehensive treatment of microcanonical
   ensemble theory and the binary-network equivalence framework.

3. A. Annibale, A. C. C. Coolen, L. P. Fernandes, F. Fraternali,
   J. Kleinjung, *Tailored graph ensembles as proxies for biological
   network data*, J. Phys. A: Math. Theor. 42, 485001 (2009).
   DOI: [10.1088/1751-8113/42/48/485001](https://doi.org/10.1088/1751-8113/42/48/485001).
   — Foundational microcanonical ensemble-equivalence results for binary
   networks.