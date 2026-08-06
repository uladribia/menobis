---
description: Complexity, memory model, and practical scaling guidance.
---

# Scalability

## TL;DR

MENoBiS is designed to keep public workflows sparse or O(N) in memory. Large N is
possible when you can afford the all-pairs time required by some constraints.

![Example scalability benchmark](../figures/scalability_example.png)

!!! note "Benchmark context"
    The plot uses stored repository benchmark results for ME/B strength routes.
    Re-run the benchmark CLI on your machine for local wall-clock numbers.

## Operation costs

| Operation | Typical complexity | Notes |
|---|---:|---|
| edge-list I/O and statistics | O(E) | single pass over observed pairs |
| ME strength fit | O(N) | very large N is practical |
| B/W strength fit | O(N² I) | time grows with all-pairs sweeps |
| degree-events fit | O(N² I) | usually low iteration count |
| strength-edges fit | O(N² I) | zero-inflated; inspect convergence |
| strength-degree fit | O(N² I) | high constant, often slowest |
| strength-cost fit | O(N² I) | costs computed on the fly or bounded caches |
| generation | O(P + E_s) | streams candidate pairs |
| filtering | O(E) or O(P) | absent-edge filtering scans candidates |

## Memory principles

| Data | Preferred representation |
|---|---|
| observed network | sparse `EdgeTable` |
| sampled network | sparse `EdgeTable` |
| custom probabilities | sparse triples |
| multipliers | O(N) arrays |
| costs | on-the-fly provider or sparse state |
| frozen pairs | sparse mask |

!!! tip "Time, not dense storage"
    An O(N²) solver can still have O(N) public state. Expect large runs to be
    CPU-bound rather than blocked by an `N x N` rate matrix.

## Practical guidance

| Goal | Good first choice |
|---|---|
| huge N baseline | ME strength |
| medium N with spatial effect | ME strength-cost |
| support-aware null | ME/B strength-edges before strength-degree |
| W family science | start at small N and inspect diagnostics |
| absent-edge filtering | cap `max_absent` during exploration |

## Microcanonical sampler operating ranges

The microcanonical samplers (exact-constraint generation) have different
scaling characteristics depending on the constraint type, regime, and family.
The table below shows the **maximum N** tested at each regime × constraint
combination (dense regime, T = E × events_per_edge = 8E).

| Constraint | Regime | ME | B | W | Bottleneck |
|---|---|---|---|---|---|
| fixed (E,T) | sparse | ≥500 | ≥500 | ≤100 | W rejection acceptance at λ=3; ME/B work via exact DP (small E×T) |
| fixed (E,T) | dense | ≤300 | ≤400 | ≤100 | DP table E×T > 2×10⁸; rejection P(acc) ∝ exp(−E·e^{−λ}) ≈ 10⁻⁷ at N=500 |
| fixed (E,T) | saturated | ≤300 | ≤400 | ≤100 | same as dense, T/E=8 fixed; rejection estimate underflows |
| fixed (k,T) | sparse | ≥500 | ≥500 | ≤100 | inherits fixed-(E,T) occupation kernel; same E,T bottleneck for W |
| fixed (k,T) | dense | ≤300 | ≤300 | ≤100 | same as fixed-(E,T) dense |
| fixed (k,T) | saturated | ≤300 | ≤300 | ≤100 | same as fixed-(E,T) saturated |
| fixed strengths | any | **≥500** | **≥500** | **≥500** | 4-cycle MCMC; scales linearly with occupied pairs |
| strengths + cost | any | ≈100 | ≈100 | ≈100 | warm-start variance estimate collapses at larger N; needs per-case tuning |

**Key limitations:**

1. **Fixed-(E,T) and fixed-(k,T)** use an exact DP table (limit 2×10⁸ cells) or
   brute-force rejection. At λ = T/E = events_per_edge = 8 (dense/saturated),
   the acceptance probability is (1−e^{−8})^{E} ≈ exp(−0.000335·E), which
   decays exponentially with E. At N=500 (E≈50000) this is ≈5×10⁻⁸, and the
   fallback attempt budget is too small. The sparse regime (λ=3) works at
   N=500 for ME/B because E is smaller and acceptance is higher.

2. **W is the weakest family** for these exact methods: its Negative-Binomial
   occupation allocation has the lowest rejection acceptance at any given E,T.

3. **Fixed-strength (Phase 4)** is the only microcanonical constraint that
   genuinely scales to N=500+ in all regimes. The 4-cycle MCMC preserves
   strengths exactly and has no DP/rejection bottleneck.

4. **Strength + cost (Phase 5)** inherits the Phase 4 chain but adds a gamma
   fitter whose warm-start variance estimate becomes unreliable at large N.
   The expected cost variance shrinks relative to its mean as N grows, making
   the initial γ₀ = (µ₀−C_obs)/Var₀ unstable. A fix would use more sweeps
   or an autocorrelation-aware estimator.

### Fixing the scaling gap

The fixed-(E,T) and fixed-(k,T) samplers could be made to scale via a simple
MCMC alternative: sample the E-edge support uniformly, allocate T events
multinomially across it (some edges will get 0), then run a short corrective
MCMC that transfers 1 event from an occupied edge to an empty edge until all
E edges are filled. This replaces the exponential-in-E rejection wall with
O(E·exp(−λ)) corrective steps — about 17 steps for N=500 dense. The approach
is family-agnostic (ME/B/W) and preserves the correct stationary distribution.

## Regenerate local numbers

```bash
uv run python -m benchmarks all --nodes 100,1000 --families me,b \
  --constraints strength --regime dense --known-pairs 0.0
```

Use larger `--nodes` values once the small run validates your environment.
Long-running benchmark improvements are tracked in [TODOs](todos.md).
