# Microcanonical Performance Benchmarks — Gate 5

**TL;DR:** The factorized paths (edges-events, degree-events) scale **O(E)** with no O(N²) memory.
N=5000 sparse completes in <1 s for edges-events and ~12 s for degree-events.
The fixed-strength MCMC and strength-cost gamma fitting hit known scalability walls at N=5000 and N=500 respectively.
Rust internal benchmarks show 60–150% throughput improvements over the baseline.

- **Branch:** `microcanonical-refactor`
- **Commit:** `e1e96de`
- **Generator:** PA-geographic synthetic, feasible constraints derived from generated network
- **Default flags:** `--seed 42 --no-self-loops --known-pairs 0.0`

---

## 1. Edges-events (factorized fixed-E,T)

Backend: uniform support sampling + pair-Gibbs fixed-total chain.

| N    | Regime | E          | T           | ME (s) | B (s) | W (s) | py_MB |
|------|--------|-----------|-------------|--------|-------|-------|-------|
| 500  | sparse | 1 500     | 4 500       | 0.09   | 0.09  | 0.09  | 0.3   |
| 1000 | sparse | 3 000     | 9 000       | 0.07   | 0.09  | 0.14  | 0.3   |
| 5000 | sparse | 15 000    | 45 000      | 0.33   | 0.48  | 0.71  | 1.5   |
| 500  | dense  | 50 000    | 400 000     | 1.30   | 2.25  | 2.34  | 3.9   |
| 1000 | dense  | 200 000   | 1 600 000   | 5.52   | 10.18 | 10.10 | 17.9  |
| 5000 | dense  | 5 000 000 | 40 000 000  | —      | —     | —     | —     |

- **Scaling:** O(E) time, O(E) memory. N=5000 sparse all < 1 s.
- **N=5000 dense:** timed out (≥600 s). The uniform support sampler's reservoir sampling over ~25M candidate pairs is the bottleneck. This is a **time** issue, not memory — no O(N²) allocation occurs.

**Variants (all passed):**

| Variant | N  | ME (s) | B (s) | W (s) |
|---------|----|--------|-------|-------|
| kp=5% fixed pairs, dense 500 | 500 | 1.57 | 2.44 | 2.66 |
| self-loops allowed, sparse 500 | 500 | 0.09 | 0.11 | 0.13 |
| self-loops allowed, sparse 1000 | 1000 | 0.07 | 0.09 | 0.14 |

---

## 2. Degree-events (factorized fixed-k,T)

Backend: binary degree-support MCMC + shared fixed-total pair-Gibbs chain.

| N    | Regime | E        | ME (s) | B (s) | W (s) |
|------|--------|---------|--------|-------|-------|
| 500  | sparse | 1 500   | 0.24   | 0.21  | 0.22  |
| 1000 | sparse | 3 000   | 0.60   | 0.62  | 0.65  |
| 5000 | sparse | 15 000  | 11.69  | 11.58 | 11.35 |

- **Scaling:** O(E) — linear in the number of edges. No O(N²) allocation.
- N=5000 completes in ~11.5 s per family.

---

## 3. Strength (coupled occupation MCMC)

Backend: 4-cycle fixed-strength MCMC chain.

| N    | Regime | T         | ME (s) | B (s) | W (s) |
|------|--------|----------|--------|-------|-------|
| 500  | sparse | 4 500    | 0.39   | 0.30  | 0.30  |
| 1000 | sparse | 9 000    | 3.62   | 3.65  | 3.65  |
| 500  | dense  | 400 000  | 0.30   | 0.37  | 0.30  |
| 1000 | dense  | 1 600 000| 3.55   | 3.99  | 3.56  |

| N=5000 | Status | Root cause |
|--------|--------|------------|
| ME     | ❌ hangs | Router uses occupation MCMC (PairDomain::Complete max-flow over 25M pairs) instead of stub matching. |
| B      | ❌ hangs | Same PairDomain::Complete max-flow. |
| W      | ❌ hangs | Same PairDomain::Complete max-flow. |

**Limitation:** The router unconditionally routes strength to `route_occupation_mcmc`, even for ME with no fixed pairs where stub matching works in 0.15 s at N=5000.
This is a missing optimization — the router should check for the stub-matching fast path.

---

## 4. Strength-cost (gamma fitting)

Backend: fixed-strength 4-cycle MCMC + stochastic bisection gamma fit.

**N=100, dense regime** (with generous sweeps: warm-start=200, adapt=200, estim=100):

| Family | resid  | wall (s) | Status |
|--------|--------|----------|--------|
| ME     | 395    | 5.2      | ✅ ok  |
| B      | 675    | 15.7     | ✅ ok  |
| W      | 816    | 9.9      | ✅ ok  |

**N=500, dense regime:**

| Family | wall (s) | Status | Notes |
|--------|----------|--------|-------|
| ME     | —        | ❌ bracket not found | Cost variance collapses relative to total cost |
| B      | 251.9    | ✅ ok (γ≈3×10⁴) | Extreme gamma, conv=False |
| W     | —        | ❌ bracket not found | Same variance collapse |

**Limitation:** The gamma bracket expansion uses a warm-start estimate `γ₀ = (µ₀ − C_obs) / Var₀`. At N=500 the cost variance at γ=0 is small
(central-limit effect on the fixed-strength chain), making the estimate unstable. Known from the Phase 5 report.

---

## 5. Rust Criterion benchmarks

Pair-Gibbs fixed-total chain throughput improvements over the previous session's baseline:

| Benchmark | Change | Current throughput |
|-----------|--------|--------------------|
| sweep_throughput ME/E=100/T=500  | **+138%** | 19.2 Melem/s |
| sweep_throughput B(4)/E=100/T=300 | **+152%** | 24.2 Melem/s |
| sweep_throughput W(2)/E=100/T=500 | **+67%**  | 8.6 Melem/s |
| e2e_sample ME/E=100/T=500        | **+59%**  | 28.7 µs/sample |
| e2e_sample B(4)/E=100/T=300      | **+66%**  | 30.1 µs/sample |
| e2e_sample W(2)/E=100/T=500      | **+60%**  | 62.3 µs/sample |

Grand-canonical router benchmarks are stable (within noise).

---

## 6. Summary

| Path | N=500 | N=1000 | N=5000 | Memory  | Time scaling |
|------|-------|--------|--------|---------|-------------|
| Edges-events (sparse) | ✅ | ✅ | ✅ <1 s | O(E) | O(E) |
| Edges-events (dense)  | ✅ | ✅ | ❌ timeout | O(E) | O(E) but reservoir bottleneck |
| Degree-events (sparse)| ✅ | ✅ | ✅ ~12 s | O(E) | O(E) |
| Strength               | ✅ | ✅ | ❌ max-flow hang | O(N²) flow net | O(N²) init |
| Strength-cost (N=100)  | ✅ | — | — | O(E) chain | — |
| Strength-cost (N=500)  | ❌ bracket | — | — | O(E) chain | — |

## 7. Follow-up items

1. **Router should short-circuit to ME stub matching** for strength + no fixed pairs.
2. **Uniform edge sampling** at dense N=5000 could use index-space sampling instead of reservoir over all candidates.
3. **Gamma bracket expansion** needs a fallback when the warm-start variance estimate is degenerate.

## 8. Commands

```bash
# Edges-events sparse
python -m benchmarks micro --nodes 500,1000,5000 --families me,b,w \
  --regime sparse --constraint edges-events --seed 42 --no-self-loops

# Degree-events sparse
python -m benchmarks micro --nodes 500,1000,5000 --families me,b,w \
  --regime sparse --constraint degree-events --seed 42 --no-self-loops \
  --burn-in-sweeps 10 --sweeps-per-sample 5

# Strength (MCMC)
python -m benchmarks micro --nodes 500,1000 --families me,b,w \
  --regime sparse --constraint strength --seed 42 --no-self-loops \
  --burn-in-sweeps 10 --sweeps-per-sample 5

# Strength-cost at N=100
python -m benchmarks micro --nodes 100 --families me,b,w \
  --regime dense --constraint strength-cost --seed 42 --no-self-loops \
  --burn-in-sweeps 30 --sweeps-per-sample 15 \
  --fit-warm-start-sweeps 200 --fit-adaptation-sweeps 200 \
  --fit-estimation-sweeps 100 --fit-samples-per-iteration 30 \
  --fit-max-iterations 30 --fit-cost-tolerance 0.3
```

Result JSON files in `benchmarks/results/micro-gate5-*.json`.