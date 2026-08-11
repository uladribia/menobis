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
| microcanonical MCMC | O(E_occ) per sweep | scales linearly with occupied pairs; no N² memory |
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

Validated N per constraint × regime from the §35 benchmark matrix
(N=100/500/1000; dense regime is `T = E × events_per_edge = 8E`):

| Constraint | Regime | Max N tested | Bottleneck |
|---|---|---|---|
| fixed (E,T) | sparse | ≥1000 | pair-Gibbs chain: O(E) |
| fixed (E,T) | dense | ≥1000 | pair-Gibbs chain: O(E) |
| fixed (k,T) | sparse | ≥1000 | support MCMC + pair-Gibbs |
| fixed (k,T) | dense | ≥1000 | support MCMC + pair-Gibbs |
| fixed strengths | sparse | ≥1000 | compressed constructor + MCMC: O(E_occ) |
| fixed strengths | dense | ≥1000 | B capacity repair at high occupancy |
| strengths + cost | sparse | ≥1000 | gamma fitting: 10–30 s per cell |
| strengths + cost | dense | ≥1000 | gamma fitting: 200–1200 s per cell |

The **pair-Gibbs chain** replaces the earlier DP/rejection approach — see
[docs/concepts/microcanonical.md](../concepts/microcanonical.md). Production
backends have no DP tables, no rejection acceptance walls, and no
family-specific weakness: W runs through the same Gibbs chain with no
rejection. Fixed-(E,T) and fixed-(k,T) are O(E) memory and scale with the
number of occupied pairs.

## Regenerate local numbers

```bash
uv run python -m benchmarks matrix   # §35 microcanonical fixed-strength matrix

uv run python -m benchmarks all --nodes 100,1000 --families me,b \
  --constraints strength --regime dense --known-pairs 0.0
```

Use larger `--nodes` values once the small run validates your environment.
Long-running benchmark improvements are tracked in [TODOs](todos.md).
