---
description: Compare observed network statistics with an ensemble — metrics, helper functions, and sampling-count guidance.
---

# Ensemble statistics

## TL;DR

Fit a null, sample many networks, compute a statistic \(g\) on the observed
network \(t^{\mathrm{obs}}\) and on each sampled network, then compare:

\[
g_{\mathrm{obs}} \quad\text{vs}\quad
\{g(t^{(1)}),\ldots,g(t^{(R)})\}.
\]

If \(g_{\mathrm{obs}}\) lies in the tails of the sampled distribution, the
observed value is unusual under the null.

## The workflow

```python
from menobis.analysis import compute_all_stats, ensemble_average
from menobis.models import Constraint, Ensemble, ModelFamily, fit_model
from menobis.routing import sample_model

fit = fit_model(family=ModelFamily.ME, constraint=Constraint.STRENGTH,
                strength_out=strength_out, strength_in=strength_in)
if not fit.converged:
    raise RuntimeError(fit.status)

observed = compute_all_stats(observed_edges)

samples = [sample_model(ensemble=Ensemble.GRAND_CANONICAL,
                        family=ModelFamily.ME, constraint=Constraint.STRENGTH,
                        fit=fit, seed=r)
           for r in range(100)]

stats_per_sample = [compute_all_stats(s) for s in samples]
y2_obs = observed.y2_out.mean()
y2_ensemble = [s.y2_out.mean() for s in stats_per_sample]
print("observed mean Y2:", y2_obs, "ensemble mean:", sum(y2_ensemble) / len(y2_ensemble))
```

For the ensemble aggregation use the keyword helpers
`ensemble_average(generate=..., analyze=..., repetitions=...)` (per-node
arrays) or `ensemble_scalar_average(generate=..., compute=...,
repetitions=...)` (scalars):

```python
from menobis.analysis import compute_all_stats, ensemble_scalar_average

m, s = ensemble_scalar_average(
    generate=lambda seed: sample_model(
        ensemble=Ensemble.GRAND_CANONICAL, family=ModelFamily.ME,
        constraint=Constraint.STRENGTH, fit=fit, seed=seed,
    ),
    compute=lambda edges: float(compute_all_stats(edges).y2_out.mean()),
    repetitions=100,
)
```

## Metrics

### Degree

\[
k_i^{\mathrm{out}}=\sum_j\mathbf 1[t_{ij}>0],
\qquad
k_j^{\mathrm{in}}=\sum_i\mathbf 1[t_{ij}>0].
\]

Use `directed_degrees(edges)` (`DirectedSequences` with `out` and `incoming`).

### \(Y_2\) disparity

Strength concentration of node \(i\):

\[
Y_{2,i}^{\mathrm{out}}
=
\sum_j
\left(
\frac{t_{ij}}{s_i^{\mathrm{out}}}
\right)^2
=
\frac{\sum_j t_{ij}^2}{(s_i^{\mathrm{out}})^2}.
\]

\(Y_2=1\) means all events go to a single neighbour; small \(Y_2\) means
even spread. For nodes with zero strength the implemented value is `0.0`.
Available as `y2_out` / `y2_in` from `compute_all_stats`.

### Nearest-neighbour strength

The implemented nearest-neighbour strength is **occupation-weighted** and
**directed**. For the out direction:

\[
s^{\mathrm{nn,out}}_i
=
\frac{\sum_j t_{ij}\,s_j^{\mathrm{in}}}{s_i^{\mathrm{out}}},
\qquad s_i^{\mathrm{out}}>0,
\]

the occupation-weighted mean of the *in-strengths* of the destinations
reached by \(i\). The in direction mirrors it:

\[
s^{\mathrm{nn,in}}_j
=
\frac{\sum_i t_{ij}\,s_i^{\mathrm{out}}}{s_j^{\mathrm{in}}},
\qquad s_j^{\mathrm{in}}>0.
\]

Available as `s_nn_out` / `s_nn_in` (`compute_all_stats`). A separate
topological (unweighted) nearest-neighbour degree is `k_nn_out` /
`k_nn_in`.

### Clustering

MENoBiS exposes two per-node clustering helpers:

- `clustering_coefficient(edges)` — **binary-support** clustering;
- `occupation_clustering_coefficient(edges)` — **occupation-based**
  clustering.

Report which convention you use; the occupation-based function is *not*
exported under a "weighted" name.

### Occupation distribution

`occupation_distribution(edges)` returns the histogram
\(P(t_{ij}=t)\) over occupied pairs (`OccupationDistribution` with `occ_num`
and `count`).

## How many samples?

Avoid fixed universal claims like "100 quick / 1000 for reporting". The
required number of samples follows from Monte Carlo precision. For an
estimated probability

\[
\operatorname{SE}(\hat p)
\approx
\sqrt{\frac{p(1-p)}{R_{\mathrm{eff}}}},
\]

where

- for independent direct samples: \(R_{\mathrm{eff}}=R\);
- for correlated MCMC samples: \(R_{\mathrm{eff}}<R\) (account for
  autocorrelation — see
  [MCMC diagnostics](../performance/mcmc-diagnostics.md)).

Estimating tails (\(p\) close to 0 or 1) needs many more effective samples
than estimating means. Decide \(R\) from the precision you need, not from a
round number.