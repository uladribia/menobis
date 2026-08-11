# Migration Decision — Exact Fixed-(E,T) → Pair-Gibbs

**Status:** implemented (Phase 4/5)
**Policy:** legacy exact backends live in `menobis-test-oracles`, not on an
archive branch.

## 1. Decision

The exact rejection/DP occupation backends (ME/B/W) were replaced in
production by the shared pair-Gibbs chain (`fixed_total/`).  The legacy
backends are **not** archived on a branch; they were moved into the
oracle crate as `menobis-test-oracles/src/legacy_fixed_et/`.

Rationale: an archive branch is a fossil — it never gets compiled,
never gets tested, and rots.  Keeping the legacy code in the oracle
crate means it is compiled and exercised on every CI run and remains a
usable reference oracle.

## 2. What lives in `legacy_fixed_et/`

- `FixedETOccupancy` trait and `MeFamily` / `BFamily` / `WFamily` impls;
- bounded-rejection fast paths (multinomial, binary-cell, weak-composition);
- exact DP fallbacks (Stirling numbers of the second kind for ME, bounded
  and unbounded composition tables for B and W);
- scaled-attempts rejection fallback for table-too-large cases;
- the legacy orchestrator `sample_fixed_et_core` and the occupation
  backend `sample_positive_occupations`.

This module is **test/reference infrastructure only** and must never
become a production dependency of `menobis-core`.

## 3. Why the legacy backends remain valuable

- **Scientific validity**: they are exact samplers of the fixed-(E,T)
  fiber (up to the DP table limit `E×T ≈ 2e8`).
- **Medium-scale oracle**: state enumeration
  (`enumeration::enumerate_fixed_total`) explodes combinatorially beyond
  `E ≈ 10`, while the DP backends reach `E ≈ 14000`.  They validate the
  Gibbs chain where enumeration cannot (Level 2 validation).
- **Reference for observable comparison** across ME/B/W.

## 4. Validation performed

### Level 1 — Exact enumeration (small fibers, TV distance)

`tests/fixed_total_stationarity.rs` (oracle crate): Gibbs empirical state
probabilities vs exact enumeration:

- ME `E=3,T=6`: TV < 0.03
- B(4) `E=3,T=8`: TV < 0.03
- B(3) `E=3,T=9` (saturated, single state): TV = 0
- W(2) `E=3,T=6`: TV < 0.03
- W(1) `E=3,T=6` (uniform law): TV < 0.03

### Level 2 — Legacy backends (medium fibers, observables)

`tests/fixed_total_legacy_comparison.rs` (oracle crate): Gibbs vs legacy
DP on `E=10` fibers, comparing mean max occupation, mean Σt², and mean
occupation within relative tolerance (4000 samples per backend):

- ME `E=10,T=30`: agree
- B(5) `E=10,T=30`: agree
- W(2) `E=10,T=25`: agree

### Conditional exactness (by construction)

Each pair-Gibbs conditional draws exactly from the closed-form family
PMF (Binomial / Hypergeometric / Beta-Binomial), so the chain targets
the same law as the exact backends.

### Invariants

Total `T`, positivity, B capacity preserved; deterministic
reproducibility for fixed seeds.

## 5. Remaining recommended validation

- Level 2 extension to larger `E` (e.g., `E=50, 100`) with formal ESS /
  autocorrelation-aware comparison;
- a full legacy-vs-Gibbs performance benchmark matrix (the design doc
  §15–§16 `--occupation-backend legacy|gibbs|compare` CLI was not built).

These are follow-ups; the switch to Gibbs as production default is
already complete and validated at Levels 1–2 above.

## 6. Reproducibility

- Legacy backends: `menobis-test-oracles::legacy_fixed_et`
  (compiled and tested on every CI run).
- Exact enumeration: `menobis-test-oracles::enumeration`.
- No archive branch or tag is required; git history preserves the
  pre-migration state (`microcanonical-refactor` before the Phase 5
  merge).
