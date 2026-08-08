//! Exact two-cell conditionals for the fixed-total pair-Gibbs kernel.
//!
//! Given two cells with sum `q = t_a + t_b`, redraw the split
//! `(k, q−k)` from the exact family conditional
//!
//! \[
//! P_F(k \mid q) \propto d_F(k)\, d_F(q-k),
//! \]
//!
//! subject to `1 ≤ k ≤ q−1` and family bounds.
//!
//! - ME: `k ~ Binomial(q, 1/2)` conditioned on `1 ≤ k ≤ q−1`.
//! - B:  `k ~ Hypergeometric(2M, M, q)` conditioned on the same range.
//! - W:  `k ~ BetaBinomial(q, M, M)` conditioned on the same range
//!   (generated as `p ~ Beta(M,M)`, `k ~ Binomial(q,p)`); `M=1`
//!   degenerates to uniform on `{1,…,q−1}`.
//!
//! All families special-case `q = 2` → `(1, 1)`, the only valid split.
//! Bounded rejection is used with an exact log-space enumeration
//! fallback, so the procedure is exact and always terminates.

use rand::Rng;
use rand_distr::{Beta, Binomial, Distribution, Hypergeometric};

use crate::model::family::OccupationFamily;
use crate::OccNum;

/// Maximum rejection attempts before the exact fallback is used.
const MAX_REJECT_ATTEMPTS: usize = 64;

/// Draw a split of `q` into two positive parts under the family law.
///
/// # Panics
///
/// Panics if `q < 2` (a valid split requires at least 2) or if no
/// feasible split exists for the family (infeasible input).
pub fn sample_split(family: OccupationFamily, q: OccNum, rng: &mut impl Rng) -> (OccNum, OccNum) {
    assert!(q >= 2, "split requires q ≥ 2, got {q}");
    if q == 2 {
        return (1, 1);
    }
    match family {
        OccupationFamily::ME => sample_me_split(q, rng),
        OccupationFamily::B { layers } => sample_b_split(layers as u64, q, rng),
        OccupationFamily::W { layers } => sample_w_split(layers as u64, q, rng),
    }
}

/// ME: `k ~ Binomial(q, 1/2) | 1 ≤ k ≤ q−1`.
fn sample_me_split(q: OccNum, rng: &mut impl Rng) -> (OccNum, OccNum) {
    let dist = match Binomial::new(q, 0.5) {
        Ok(d) => d,
        Err(_) => return sample_split_exact_fallback(OccupationFamily::ME, q, rng),
    };
    for _ in 0..MAX_REJECT_ATTEMPTS {
        let k = dist.sample(rng);
        if k >= 1 && k < q {
            return (k, q - k);
        }
    }
    sample_split_exact_fallback(OccupationFamily::ME, q, rng)
}

/// B: `k ~ Hypergeometric(2M, M, q) | 1 ≤ k ≤ q−1`.
///
/// The native hypergeometric support is `[max(0, q−M), min(M, q)]`; the
/// positivity conditioning is applied by rejection.
fn sample_b_split(m: u64, q: OccNum, rng: &mut impl Rng) -> (OccNum, OccNum) {
    debug_assert!(q <= 2 * m, "B split q={q} exceeds 2M={}", 2 * m);
    let dist = match Hypergeometric::new(2 * m, m, q) {
        Ok(d) => d,
        Err(_) => {
            return sample_split_exact_fallback(OccupationFamily::B { layers: m as u32 }, q, rng)
        }
    };
    for _ in 0..MAX_REJECT_ATTEMPTS {
        let k = dist.sample(rng);
        if k >= 1 && k < q {
            return (k, q - k);
        }
    }
    sample_split_exact_fallback(OccupationFamily::B { layers: m as u32 }, q, rng)
}

/// W: `k ~ BetaBinomial(q, M, M) | 1 ≤ k ≤ q−1`.
///
/// For `M = 1` the conditional is uniform on `{1,…,q−1}`.
fn sample_w_split(m: u64, q: OccNum, rng: &mut impl Rng) -> (OccNum, OccNum) {
    if m == 1 {
        let k = rng.random_range(1..q);
        return (k, q - k);
    }
    let beta = match Beta::new(m as f64, m as f64) {
        Ok(b) => b,
        Err(_) => {
            return sample_split_exact_fallback(OccupationFamily::W { layers: m as u32 }, q, rng)
        }
    };
    for _ in 0..MAX_REJECT_ATTEMPTS {
        let p = beta.sample(rng);
        if let Ok(dist) = Binomial::new(q, p.clamp(0.0, 1.0)) {
            let k = dist.sample(rng);
            if k >= 1 && k < q {
                return (k, q - k);
            }
        }
    }
    sample_split_exact_fallback(OccupationFamily::W { layers: m as u32 }, q, rng)
}

/// Exact fallback: sample the split from the normalized log weights
/// `w(k) = ln d_F(k) + ln d_F(q−k)` over `k ∈ [1, q−1]`.
///
/// Used only if bounded rejection is exhausted (practically never; the
/// fallback guarantees termination and exactness).  Its cost is `O(q)`
/// and it is triggered only when `q` is small.
fn sample_split_exact_fallback(
    family: OccupationFamily,
    q: OccNum,
    rng: &mut impl Rng,
) -> (OccNum, OccNum) {
    let mut log_w = Vec::with_capacity((q - 1) as usize);
    for k in 1..q {
        log_w.push(family.log_base_measure(k) + family.log_base_measure(q - k));
    }
    let max = log_w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = log_w.iter().map(|&w| (w - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    assert!(
        total.is_finite() && total > 0.0,
        "no feasible split for q={q} under {family:?}"
    );
    let mut u = rng.random::<f64>() * total;
    for (idx, w) in weights.iter().enumerate() {
        u -= w;
        if u <= 0.0 {
            let k = idx as OccNum + 1;
            return (k, q - k);
        }
    }
    (q - 1, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    fn mean_of_splits(family: OccupationFamily, q: OccNum, n: usize, seed: u64) -> f64 {
        let mut r = rng(seed);
        let mut sum = 0.0;
        for _ in 0..n {
            let (k, _) = sample_split(family, q, &mut r);
            sum += k as f64;
        }
        sum / n as f64
    }

    #[test]
    fn q_two_always_one_one() {
        for family in [
            OccupationFamily::ME,
            OccupationFamily::B { layers: 3 },
            OccupationFamily::W { layers: 2 },
            OccupationFamily::W { layers: 1 },
        ] {
            assert_eq!(sample_split(family, 2, &mut rng(1)), (1, 1));
        }
    }

    #[test]
    fn me_split_preserves_sum_and_bounds() {
        let mut r = rng(7);
        let q = 10u64;
        for _ in 0..1000 {
            let (k, rest) = sample_split(OccupationFamily::ME, q, &mut r);
            assert!(k >= 1 && k < q);
            assert_eq!(k + rest, q);
        }
        // Symmetric conditional → mean ≈ q/2
        let mean = mean_of_splits(OccupationFamily::ME, 10, 10_000, 3);
        assert!((mean - 5.0).abs() < 0.1, "mean {mean}");
    }

    #[test]
    fn b_split_respects_cap() {
        // B(3): each part ≤ 3, so q=4 allows k ∈ [1,3]
        let mut r = rng(7);
        for _ in 0..2000 {
            let (k, rest) = sample_split(OccupationFamily::B { layers: 3 }, 4, &mut r);
            assert!((1..=3).contains(&k));
            assert!((1..=3).contains(&rest));
            assert_eq!(k + rest, 4);
        }
    }

    #[test]
    fn w_m1_uniform() {
        // W(1): uniform on {1..q-1} → mean ≈ q/2
        let mean = mean_of_splits(OccupationFamily::W { layers: 1 }, 8, 10_000, 5);
        assert!((mean - 4.0).abs() < 0.1, "mean {mean}");
    }

    #[test]
    fn w_split_preserves_sum() {
        let mut r = rng(11);
        for _ in 0..2000 {
            let (k, rest) = sample_split(OccupationFamily::W { layers: 2 }, 6, &mut r);
            assert!((1..=5).contains(&k));
            assert_eq!(k + rest, 6);
        }
    }

    #[test]
    fn deterministic_same_seed() {
        let a = sample_split(OccupationFamily::ME, 9, &mut rng(42));
        let b = sample_split(OccupationFamily::ME, 9, &mut rng(42));
        assert_eq!(a, b);
    }
}
