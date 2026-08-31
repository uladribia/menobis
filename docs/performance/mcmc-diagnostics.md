---
description: MCMC diagnostics — exact targets, initialization, burn-in, mixing, autocorrelation, and effective sample size.
---

# MCMC diagnostics

## TL;DR

MENoBiS microcanonical MCMC routes use kernels with **exactly the target
distribution as their stationary law**. Finite runs still require burn-in
and mixing assessment: the stationary law guarantees the limit, not the
finite-run quality. Inspect acceptance, autocorrelation, and effective
sample size (ESS) per statistic.

## Exact target

A kernel is **exact stationary MCMC** when its stationary distribution is
exactly the target constrained measure. Sample exactness is a property of
the kernel, not of how the initial state was found. See
[Validation](validation.md) for how exactness is verified.

## Initialization

The constructor provides one feasible state on the constraint fiber.
Initialization:

- must satisfy the hard constraints;
- need **not** satisfy detailed balance;
- contributes initial-state bias that burn-in removes.

## Burn-in

Burn-in discards early draws while the chain forgets the initial state.
Its required length depends on the route and instance; there is no universal
"more sweeps is enough" rule — measure the diagnostics below.

## Mixing

Mixing describes how fast the chain explores the target state space.
Fast-mixing chains visit diverse configurations quickly; slow mixing shows
up as high autocorrelation and small ESS.

## Autocorrelation

For a statistic \(g\) evaluated on chain states \(t_1,\ldots,t_R\), the lag-
\(\tau\) autocorrelation is

\[
\rho_g(\tau)
=
\operatorname{Corr}(g(t_r),g(t_{r+\tau})).
\]

## Effective sample size

\[
ESS_g
\approx
\frac{R}
{1+2\sum_{\tau\ge 1}\rho_g(\tau)}.
\]

ESS is **statistic-specific**: a statistic that is insensitive to the slow
directions of the chain can have a larger ESS than one that probes them.
Estimate tails of \(g\) only with enough effective samples
(see [Ensemble statistics](../guide/ensemble-statistics.md)).

## Diagnostics to monitor

- acceptance rate (too low → wasteful proposals; too high → small moves);
- effective movement rate;
- support change rate (how often the occupied-pair set changes);
- trace returns for trace-based kernels;
- cost ESS for cost-influenced routes;
- repeated-chain agreement (independent chains, same seed offset);
- autocorrelation of the high-level metrics you will report.

## Wording

Avoid "MCMC may not converge". Prefer:

> The kernel has the correct stationary target, but finite-run mixing can be
> slow in tight or heterogeneous fibers.

Distinguish throughout: deterministic/iterative fit convergence, stochastic
gamma-fit convergence, stationarity, burn-in, mixing, and Monte Carlo
error — they are different things (see
[Validation](validation.md)).