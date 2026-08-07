# Phase 1 Design — Exact ME Microcanonical Ensemble with Fixed \((E,T)\)

**Version:** August 2026

## 1. Purpose

Phase 1 implements the first microcanonical generator in MENoBiS:

- family: ME;
- hard constraints: fixed occupied-pair count \(E\) and fixed total occupation \(T\);
- sampling method: exact direct sampling;
- graph representation: existing sparse MENoBiS representation;
- preprocessing: existing mask and fixed-occupation prefilter;
- MCMC: not used.

This phase is restricted to fixed \((E,T)\). It must not introduce degree, strength, cost, W, B, or generic MCMC logic beyond interfaces that are directly justified by this implementation.

The implementation should become:

- the first correctness oracle for the microcanonical subsystem;
- the reference for residual-problem handling;
- the reference for conditioned grand-canonical validation;
- a reusable basis for later fixed-\((E,T)\) families.

## 2. Target state space

Let the residual admissible pair set be

\[
\mathcal{A}_{\mathrm{res}}
=
\{a_1,\dots,a_L\},
\]

where \(L\) is the number of pair locations not fixed or forbidden by preprocessing.

A residual configuration is an occupation vector

\[
\mathbf{t}_{\mathrm{res}}
=
(t_1,\dots,t_L),
\qquad
t_\ell\in\mathbb{Z}_{\ge 0}.
\]

The hard constraints are

\[
\sum_{\ell=1}^{L}\mathbf{1}(t_\ell>0)=E_{\mathrm{res}},
\]

and

\[
\sum_{\ell=1}^{L}t_\ell=T_{\mathrm{res}}.
\]

For readability, the remainder of this document writes \(E\) and \(T\) for the residual values unless the distinction matters.

The residual microcanonical state space is

\[
\Omega_{L,E,T}
=
\left\{
\mathbf{t}\in\mathbb{Z}_{\ge0}^{L}:
\sum_{\ell}\mathbf{1}(t_\ell>0)=E,
\sum_{\ell}t_\ell=T
\right\}.
\]

## 3. Target probability

For the ME family, the degeneracy is

\[
D_{\mathrm{ME}}(\mathbf{t})
=
\frac{T!}{\prod_{\ell=1}^{L}t_\ell!}.
\]

The fixed-\((E,T)\) microcanonical distribution is

\[
P_{\mathrm{ME}}(\mathbf{t}\mid E,T)
=
\frac{1}{Z_{L,E,T}}
\frac{T!}{\prod_{\ell=1}^{L}t_\ell!}
\mathbf{1}[\mathbf{t}\in\Omega_{L,E,T}].
\]

Because \(T\) is fixed, \(T!\) is constant on the constrained state space. Therefore

\[
P_{\mathrm{ME}}(\mathbf{t}\mid E,T)
\propto
\frac{1}{\prod_{\ell:t_\ell>0}t_\ell!}.
\]

## 4. Feasibility conditions

After preprocessing, the residual problem is feasible if and only if:

1. \(0\le E\le L\);
2. \(T\ge0\);
3. \(E=0\) if and only if \(T=0\);
4. if \(E>0\), then \(T\ge E\).

Equivalently:

- \((E,T)=(0,0)\) is feasible;
- for \(E>0\), feasibility requires \(E\le\min(L,T)\).

No sampler should run before these conditions are checked.

Special cases should be resolved without entering the general algorithm:

- \(E=0,T=0\): return an empty residual graph;
- \(E=1\): choose one admissible pair uniformly and assign occupation \(T\);
- \(E=T\): choose \(E\) admissible pairs uniformly and assign occupation one to every selected pair;
- \(E=L\): support is deterministic; only occupations are sampled.

## 5. Prefilter and residual problem contract

The Phase 1 sampler must consume the residual problem produced by the existing preprocessing layer.

The preprocessing layer is responsible for:

- excluding forbidden pair locations;
- resolving self-loop policy;
- separating fixed-zero and fixed-positive entries;
- validating fixed occupations;
- computing
  \[
  E_{\mathrm{fix}}
  =
  \sum_{a\in\mathcal{F}}\mathbf{1}(t_a^{\mathrm{fix}}>0),
  \]
  and
  \[
  T_{\mathrm{fix}}
  =
  \sum_{a\in\mathcal{F}}t_a^{\mathrm{fix}};
  \]
- computing residual constraints
  \[
  E_{\mathrm{res}}=E_{\mathrm{user}}-E_{\mathrm{fix}},
  \qquad
  T_{\mathrm{res}}=T_{\mathrm{user}}-T_{\mathrm{fix}};
  \]
- creating a compact indexable representation of residual admissible pairs;
- rejecting inconsistent or impossible problems.

The sampler must not inspect or modify the original mask. It receives:

```text
ResidualFixedETProblem {
    admissible_pairs,
    residual_edge_count,
    residual_total_occupation,
}
```

The exact Rust type names should follow repository conventions.

After residual sampling, reconstruction merges the sampled residual pairs with the fixed-positive occupations. Final validation checks the original requested \((E,T)\).

## 6. Factorization theorem

Let \(S\subseteq\mathcal{A}_{\mathrm{res}}\) be the support of a residual configuration, with \(|S|=E\). Let

\[
\mathbf{n}=(n_1,\dots,n_E)
\]

be the strictly positive occupations assigned to the selected support, satisfying

\[
n_i\ge1,
\qquad
\sum_{i=1}^{E}n_i=T.
\]

For any fixed support \(S\), the conditional weight of \(\mathbf{n}\) is

\[
w(\mathbf{n})=rac{1}{\prod_{i=1}^{E}n_i!}.
\]

This weight does not depend on the identities of the pair locations in \(S\). Therefore every support has the same conditional partition function

\[
Z_{E,T}^{+}
=
\sum_{\substack{n_i\ge1\\\sum_i n_i=T}}
\frac{1}{\prod_i n_i!}.
\]

It follows that support and positive occupation allocation factorize:

\[
P(S,\mathbf{n}\mid L,E,T)
=
P(S\mid L,E)P(\mathbf{n}\mid E,T),
\]

with

\[
P(S\mid L,E)=\frac{1}{\binom{L}{E}},
\]

and

\[
P(\mathbf{n}\mid E,T)
=
\frac{1}{Z_{E,T}^{+}}
\frac{1}{\prod_i n_i!}.
\]

Thus exact sampling separates into:

1. uniform support selection without replacement;
2. exact positive occupation allocation.

## 7. Surjection representation of the occupation law

Consider \(T\) labelled distinguishable events and \(E\) labelled selected pairs. Assign each event independently and uniformly to one of the \(E\) pairs.

For a particular positive occupation vector \(\mathbf{n}\), the number of assignments yielding those counts is the multinomial coefficient

\[
\frac{T!}{\prod_i n_i!}.
\]

Conditioning on every selected pair receiving at least one event gives

\[
P(\mathbf{n}\mid n_i>0\ \forall i)
\propto
\frac{T!}{\prod_i n_i!}
\propto
\frac{1}{\prod_i n_i!}.
\]

Therefore the required positive occupation law is exactly the occupancy distribution of a uniform random surjection from \(T\) labelled events to \(E\) labelled pairs.

This representation yields two exact sampling methods:

1. independent multinomial assignment with rejection when any pair is empty;
2. direct recursive sampling of a uniform surjection using Stirling numbers of the second kind.

Phase 1 uses a hybrid of these methods.

## 8. Chosen hybrid algorithm

The preferred path is multinomial rejection because it is simple and fast when acceptance is high.

The fallback is Stirling-recursion sampling because it does not reject and remains efficient when \(T\) is close to \(E\).

The backend-selection threshold is fixed for Phase 1 at

\[
r_{\max}=0.8,
\]

where \(r\) denotes the predicted rejection probability.

The policy is:

```text
if estimated_rejection_probability <= 0.8:
    try multinomial rejection with a bounded number of attempts
    if the bound is reached:
        fall back to Stirling recursion
else:
    use Stirling recursion directly
```

Both paths are exact conditional samplers. Backend selection affects only runtime, not the target distribution.

## 9. Why the rejection path is first

One rejected attempt costs \(O(T+E)\): draw \(T\) labels, build \(E\) counts, and test whether every count is positive.

If the acceptance probability is \(p_{\mathrm{acc}}\), the expected number of attempts is

\[
\mathbb{E}[A]=\frac{1}{p_{\mathrm{acc}}}.
\]

The threshold \(r_{\max}=0.8\) corresponds to

\[
p_{\mathrm{acc}}\ge0.2,
\]

and hence at most five attempts in expectation.

This is intentionally permissive. Phase 1 prioritizes a simple fast path while retaining a guaranteed fallback. Benchmarking may later motivate a cost-based threshold, but the public statistical behavior must remain unchanged.

## 10. Fast rejection-probability estimate

The exact acceptance probability is

\[
p_{\mathrm{acc}}(T,E)
=
\frac{E!\,S(T,E)}{E^T},
\]

where \(S(T,E)\) is a Stirling number of the second kind.

Computing the exact value by dynamic programming before every rejection sample can erase the advantage of the fast path. Phase 1 should therefore use a cheap estimate for backend selection and reserve exact Stirling dynamic programming for the fallback.

The default estimate is the independent-empty-box approximation:

\[
\widehat{p}_{\mathrm{acc}}
=
\left(1-e^{-T/E}\right)^E,
\]

with

\[
\widehat{r}=1-\widehat{p}_{\mathrm{acc}}.
\]

For numerical stability, compute

\[
\log\widehat{p}_{\mathrm{acc}}
=
E\log\left(1-e^{-T/E}\right),
\]

using stable `log1p`/`expm1` formulations.

This estimate is used only to choose a backend. It does not enter the generated probability law and cannot bias accepted samples.

A conservative implementation may also use a small inclusion-exclusion approximation or exact precomputed lookup values for common \((T,E)\), but Phase 1 must keep backend selection cheap and deterministic.

## 11. Bounded rejection and exact fallback

Even when \(\widehat{r}\le0.8\), the true acceptance probability may differ from the approximation. Rejection must therefore be bounded.

Phase 1 should use a fixed default retry limit of

```text
MAX_REJECTION_ATTEMPTS = 20
```

This gives a very small probability of exhausting the limit when the true acceptance probability is at least \(0.2\):

\[
P(\text{20 failures})
=(1-p_{\mathrm{acc}})^{20}
\le0.8^{20}.
\]

If the retry limit is reached, the sampler must not fail and must not return an approximate result. It must construct the Stirling fallback state and draw an exact sample.

## 12. Uniform support sampling

The support consists of \(E\) distinct indices chosen uniformly from \(L\) admissible residual pairs.

The support sampler must:

- sample without replacement;
- produce every size-\(E\) subset with probability \(1/\binom{L}{E}\);
- use \(O(E)\) auxiliary memory when possible;
- avoid materializing an \(O(N^2)\) pair matrix;
- operate on compact residual pair indices.

Appropriate algorithms include:

- partial Fisher-Yates when the admissible pair index vector is already materialized and mutable;
- Floyd's algorithm for sampling \(E\) distinct integers from \([0,L)\) with \(O(E)\) memory;
- reservoir sampling when admissible pairs are available only as a stream.

The choice should follow the existing residual-mask representation. If compact random-access indexing already exists, Floyd's algorithm or a partial shuffle is preferred.

The output is a vector of selected residual pair indices. The order may be arbitrary, provided occupation labels are assigned exchangeably.

## 13. Exchangeability and assignment

The occupation allocator returns counts for \(E\) labelled boxes. The selected support vector also defines \(E\) labelled positions. Assigning count \(n_i\) to support position \(i\) is already exchangeable if both the multinomial and Stirling algorithms treat the box labels symmetrically.

An additional shuffle of occupation counts is therefore not mathematically required. It may still be used defensively if an implementation detail gives the fallback vector a construction order. Any such shuffle must be uniform and should be documented as an exchangeability safeguard rather than part of the target law.

## 14. End-to-end exactness

Let \(S\) be the sampled support and \(\mathbf{n}\) the positive occupation vector.

The support sampler gives

\[
P(S)=\frac{1}{\binom{L}{E}}.
\]

Either occupation backend gives

\[
P(\mathbf{n})
=
\frac{1}{Z_{E,T}^{+}}
\frac{1}{\prod_i n_i!}.
\]

Therefore

\[
P(S,\mathbf{n})
=
\frac{1}{\binom{L}{E}Z_{E,T}^{+}}
\frac{1}{\prod_i n_i!},
\]

which is proportional to the ME degeneracy under fixed \((E,T)\). The sampler is exact up to ordinary finite-precision effects in random-number generation and floating-point branch probabilities in the Stirling backend.

## 15. Non-goals

Phase 1 must not:

- introduce MCMC;
- handle W or B degeneracies;
- impose degree or strength constraints;
- implement dense masks;
- duplicate family laws already present elsewhere;
- expose backend selection as a user-facing scientific parameter unless existing API conventions require it;
- treat backend disagreement as acceptable.

The two occupation backends must generate the same target law.
