---
description: Ensemble equivalence validation for fixed-strength ME models.
---

# Ensemble equivalence

## TL;DR

The three statistical ensembles for fixed-strength multi-edge networks —
microcanonical, canonical, and grand-canonical — produce identical marginal
statistics in the limit of large total events `T`. ODME validates this
numerically.

## Three ensembles

| Ensemble | Sampler | What is fixed exactly | What fluctuates |
|----------|---------|----------------------|-----------------|
| Microcanonical | `sample_microcanonical` | `s_out`, `s_in`, `T` | nothing |
| Canonical | `sample_multinomial` | `T` | `s_out`, `s_in` |
| Grand-canonical | `sample_poisson` | nothing | `s_out`, `s_in`, `T` |

All three share the same analytical expectation:

\[
E[t_{ij}] = \frac{s_i^{out} s_j^{in}}{T} = x_i y_j.
\]

## Microcanonical sampler

The microcanonical ensemble is implemented as a **stub-matching** algorithm:

1. Create `s_out[i]` outgoing stubs for each node `i`.
2. Create `s_in[j]` incoming stubs for each node `j`.
3. Shuffle the incoming stubs uniformly at random.
4. Pair outgoing stub `k` with incoming stub `k`.
5. Count edge weights from the pairings.

This produces an **unbiased uniform sample** from the space of all directed
integer-weight graphs with exactly the given strength sequence.

**Important**: uniform sampling is only unbiased when self-loops are allowed.
Without self-loops, the rejection of diagonal pairings introduces bias that
requires MCMC or other correction methods. ODME therefore only implements the
self-loop-allowed version.

## Validation protocol

1. Fix a heterogeneous relative strength profile `p_out`, `p_in` (Pareto-like).
2. For increasing `T` values (100, 500, 2000, 10000):
   - Compute integer strengths: `s = round(T × p)`, balanced.
   - Generate 200 ensemble samples from each of the three samplers.
   - Compute **all** higher-order graph statistics per sample:
     strengths, degrees, Y2 disparity, k_nn, s_nn, clustering.
3. Assert:
   - Ensemble means converge across all three ensembles as `T` grows.
   - Ensemble variances (per unit T) decrease with `T`.

## Results

### Mean convergence

At large `T`, the maximum normalized difference between any pair of ensemble
means decreases as ~1/T:

![Ensemble equivalence convergence](../figures/ensemble_equivalence.png)

### Per-statistic comparison at T=10000

At `T = 10000`, all three ensembles produce statistically indistinguishable
means for every computed graph statistic:

![Per-statistic ensemble means](../figures/ensemble_per_statistic.png)

## Running the validation

```bash
uv run pytest tests/test_odme_ensemble_equivalence.py -q
```

The test generates the figures in `docs/figures/` automatically.
