---
description: Compact MENoBiS equations mapped to thesis terminology.
---

# Equations

## TL;DR

Grand-canonical MENoBiS models are independent over node pairs. Families share
multipliers but differ in the expected occupation equation.

## Pair parameter

For every allowed ordered pair:

$$q_{ij}=x_i y_j f_{ij}$$

where `x` and `y` are strength multipliers. Without cost, $f_{ij}=1$. With cost,
$f_{ij}=\exp(-\gamma d_{ij})$.

## Non-zero-inflated families

| Family | MENoBiS | Expected occupation | Domain |
|---|---|---|---|
| ME | Poisson | $\mathbb{E}[t_{ij}]=q_{ij}$ | $q>0$ |
| B | Binomial(M) | $\mathbb{E}[t_{ij}]=Mq_{ij}/(1+q_{ij})$ | $q>0$ |
| W | NegBin(M) | $\mathbb{E}[t_{ij}]=Mq_{ij}/(1-q_{ij})$ | $0<q<1$ |

For W, `M=1` is the geometric case.

!!! note "Event nature matters"
    The same strength or degree constraints generate different statistics when
    events are distinguishable, aggregated binary layers, or indistinguishable.

## Zero-inflated constraints

Strength-edges and strength-degree constraints also control binary occupation.
They use a raw binary multiplier $\ell_{ij}$ and a positive-support factor
$G_F(q)$:

| Family | $G_F(q)$ |
|---|---|
| ME | $e^q-1$ |
| B | $(1+q)^M-1$ |
| W | $(1-q)^{-M}-1$ |

The occupation probability is:

$$
\mathbb{E}[\Theta(t_{ij}>0)] =
\frac{\ell_{ij}G_F(q_{ij})}{1+\ell_{ij}G_F(q_{ij})}.
$$

The expected occupation is:

$$
\mathbb{E}[t_{ij}] =
\frac{\ell_{ij}q_{ij}G'_F(q_{ij})}{1+\ell_{ij}G_F(q_{ij})}.
$$

## Constraint map

| MENoBiS constraint | Matched expectation |
|---|---|
| `STRENGTH` | outgoing and incoming strengths |
| `STRENGTH_COST` | strengths plus total cost $\sum_{ij}\mathbb{E}[t_{ij}]d_{ij}$ |
| `STRENGTH_EDGES` | strengths plus total binary edges |
| `STRENGTH_DEGREE` | strengths plus in/out degrees |
| `DEGREE_EVENTS` | in/out degrees plus total events |
| partial variants | parent constraints after subtracting frozen pairs |

## Microcanonical (exact-constraint) ensembles

Microcanonical ensembles fix constraints exactly. The target is proportional
to the **family degeneracy** conditioned on the constraints. Per-pair (local)
degeneracies (thesis Table 3.1):

| Family | Per-pair degeneracy $d_\mathrm{family}(t_{ij})$ | Support |
|---|---|---|
| ME (Poisson) | $d_\mathrm{ME}(t_{ij}) = 1/t_{ij}!$ | $t_{ij} \ge 0$ |
| B (Binomial, $M$ layers) | $d_\mathrm{B}(t_{ij}) = \binom{M}{t_{ij}}$ | $0 \le t_{ij} \le M$ |
| W (NegBin, $M$ layers) | $d_\mathrm{W}(t_{ij}) = \binom{M+t_{ij}-1}{t_{ij}}$ | $t_{ij} \ge 0$ |

$M=1$: $d_\mathrm{W}(t)=1$, $d_\mathrm{B}(t)=1$ for $t\in\{0,1\}$.

### Fixed $(E,T)$ — EDGES_EVENTS

Degeneracy for $E$ occupied pairs and $T$ total events:

$$
\Omega_\mathrm{family}(E, T) = \sum_{\{t_{ij}\}} \Bigl[\prod_{ij} d_\mathrm{family}(t_{ij})\Bigr] \,\delta\!\Bigl(\sum t_{ij} - T\Bigr) \,\delta\!\Bigl(\sum \Theta(t_{ij}) - E\Bigr).
$$

Conditional: $P(\{t_{ij}\} \mid E, T) \propto \prod_{ij} d_\mathrm{family}(t_{ij}) \cdot \delta(\Sigma t_{ij} = T) \cdot \delta(\#\{ij \mid t_{ij}>0\} = E)$. No closed form for $\Omega$; sampling uses the pair-Gibbs chain (ME: multinomial split; B/W: hypergeometric / negative-hypergeometric conditional).

### Fixed $(k,T)$ — DEGREE_EVENTS

As $(E,T)$ but with per-node degrees $k_i^\text{out}, k_i^\text{in}$ instead of global $E$. Pairs interact through shared nodes — no closed form. Sampling: binary support MCMC (edge switches) + pair-Gibbs occupation allocation.

### Fixed strengths $(s^\text{out}, s^\text{in})$ — STRENGTH

Degeneracy for occupation matrices with exact row/column sums:

$$
\Omega_\mathrm{family}(\mathbf{s}^\text{out}, \mathbf{s}^\text{in}) = \sum_{\{t_{ij}\}} \Bigl[\prod_{ij} d_\mathrm{family}(t_{ij})\Bigr] \,\prod_i \delta\!\Bigl(\sum_j t_{ij} - s_i^\text{out}\Bigr) \,\prod_j \delta\!\Bigl(\sum_i t_{ij} - s_j^\text{in}\Bigr).
$$

No closed form. Sampling uses an **occupied-cell Metropolis chain** on the compressed residual table. The elementary **4-cycle (rectangle)** $(i,j),(i,j'),(i',j),(i',j')$ changes occupation by $\pm\delta$, preserving all row/column sums; Hastings correction accounts for non-uniform proposal density. For STRENGTH_COST, the chain is augmented with $\exp(-\gamma d_{ij})$ and $\gamma$ fitted by stochastic bisection.

### References

- Per-pair degeneracy formulas: thesis Table 3.1 (O. Sagarra, 2015).
- 4-cycle move and MCMC: thesis §5.1, Coolen et al. (2009), Coolen, Annibale, Roberts (2017).
- Implementation: `OccupationFamily::log_local_degeneracy`, `generation::microcanonical`.

## Important rule

ME, B, and W are not interchangeable. B and W must use their own equations and
solvers; they must not call the ME solution and relabel the result.
