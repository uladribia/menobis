//! W (Weighted / NegativeBinomial) family implementation for fixed-(E,T) sampling.
//!
//! # Target law
//!
//! Positive occupations ∏ C(M+tᵢ−1, tᵢ) for tᵢ ≥ 1, sum = T.
//!
//! # Backends
//!
//! * Rejection: uniform weak composition (stars and bars) of T into ME
//!   microscopic boxes; accept if no empty macro-group.
//!   Uses the smaller of separator‑mode and star‑mode decoding.
//! * Fallback: unbounded composition DP storing Z_W(k, s) in log space.

use rand::rngs::StdRng;
use rand::Rng;

use super::core::{FixedETOccupancy, MAX_DP_CELLS};
use super::errors::FixedETError;
use crate::generation::microcanonical::support::uniform_edges::{
    sample_uniform_support, shuffle_slice,
};
use crate::OccNum;

/// W (Weighted / NegativeBinomial) family with M layers.
pub struct WFamily {
    pub layers: OccNum,
}

impl FixedETOccupancy for WFamily {
    fn family_name(&self) -> &'static str {
        "W/DP"
    }

    fn max_occnum(&self) -> Option<OccNum> {
        None // W is unbounded
    }

    fn validate_residual(&self, _e: usize, _t: OccNum) -> Result<(), FixedETError> {
        Ok(()) // only basic T ≥ E is needed
    }

    fn estimate_rejection(&self, t: OccNum, e: usize) -> f64 {
        estimate_w_rejection(self.layers, t, e)
    }

    fn try_rejection(
        &self,
        t: OccNum,
        e: usize,
        max_attempts: usize,
        rng: &mut StdRng,
    ) -> Result<Vec<OccNum>, ()> {
        try_w_weak_composition_rejection(self.layers, t, e, max_attempts, rng)
    }

    fn sample_exact(
        &self,
        t: OccNum,
        e: usize,
        rng: &mut StdRng,
    ) -> Result<Vec<OccNum>, FixedETError> {
        sample_w_dp(self.layers, t, e, rng)
    }

    fn rejection_cost_per_attempt(&self, t: OccNum, e: usize) -> u64 {
        self.layers * e as OccNum + t
    }
}

// ---------------------------------------------------------------------------
// Rejection-probability estimate
// ---------------------------------------------------------------------------

/// q₀ = C(M(E-1)+T-1, T) / C(ME+T-1, T),  p̂_acc = (1−q₀)^E.
fn estimate_w_rejection(m: OccNum, t: OccNum, e: usize) -> f64 {
    let m_u = m;
    let e_u = e as u64;
    let t_u = t;
    let numer = m_u * (e_u - 1) + t_u;
    let denom = m_u * e_u + t_u;

    // log_q0 = log_binom(numer, T) - log_binom(denom, T)
    let log_q0 = log_binomial(numer, t_u) - log_binomial(denom, t_u);
    let q0 = log_q0.exp();
    let log_p_acc = (e as f64) * (-q0).ln_1p();
    (-log_p_acc.exp_m1()).clamp(0.0, 1.0)
}

fn log_binomial(n: u64, k: u64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    lg(n + 1) - lg(k + 1) - lg(n - k + 1)
}

fn lg(x: u64) -> f64 {
    libm::lgamma(x as f64)
}

// ---------------------------------------------------------------------------
// Weak-composition rejection (stars and bars)
// ---------------------------------------------------------------------------

/// Sample a uniform weak composition of T into ME boxes, accept if no
/// empty group.  Uses separator mode when K-1 ≤ T, star mode otherwise.
fn try_w_weak_composition_rejection(
    m: OccNum,
    t: OccNum,
    e: usize,
    max_attempts: usize,
    rng: &mut StdRng,
) -> Result<Vec<OccNum>, ()> {
    let m_u = m as usize;
    let t_u = t as usize;
    let k = e * m_u; // microscopic boxes
    let total_len = t_u + k - 1; // stars-and-bars string length
    let n_sep = k - 1; // number of separators

    for _ in 0..max_attempts {
        let mut occupations = vec![0_u64; e];

        if n_sep <= t_u {
            // ---- separator mode ----
            let mut separators = sample_uniform_support(total_len, n_sep, rng);
            separators.sort_unstable();

            // Decode: x_1 = b_0, x_j = b_j - b_{j-1} - 1, x_K = total_len - 1 - b_{K-2}
            let mut prev: i64 = -1;
            for (box_idx, &b) in separators.iter().enumerate() {
                let x = (b as i64 - prev - 1) as u64;
                occupations[box_idx / m_u] += x;
                prev = b as i64;
            }
            // Last segment
            let x_last = (total_len as i64 - 1 - prev) as u64;
            occupations[(k - 1) / m_u] += x_last;
        } else {
            // ---- star mode ----
            let mut stars = sample_uniform_support(total_len, t_u, rng);
            stars.sort_unstable();

            // For sorted star s_i: segment = s_i - i, group = segment / M
            for (i, &s) in stars.iter().enumerate() {
                let seg = s - i;
                occupations[seg / m_u] += 1;
            }
        }

        if occupations.iter().all(|&c| c > 0) {
            return Ok(occupations);
        }
    }
    Err(())
}

// ---------------------------------------------------------------------------
// Unbounded-composition DP
// ---------------------------------------------------------------------------

/// Precompute log C(M+t-1, t) for t = 0..=T.
fn precompute_w_degeneracy(m: OccNum, t_max: usize) -> Vec<f64> {
    let m_u = m;
    (0..=t_max)
        .map(|t| log_binomial(m_u + t as u64 - 1, t as u64))
        .collect()
}

/// Build Z_W(k, s) in log space and sample sequentially.
#[allow(clippy::needless_range_loop)]
fn sample_w_dp(
    m: OccNum,
    t: OccNum,
    e: usize,
    rng: &mut StdRng,
) -> Result<Vec<OccNum>, FixedETError> {
    let t_u = t as usize;

    // Check memory: rows[k] has length T - k + 1, total ≈ (E+1)(T+1)/2
    let max_cells = (e + 1).saturating_mul(t_u + 1);
    if max_cells > MAX_DP_CELLS {
        return Err(FixedETError::TableTooLarge {
            t: t_u,
            e,
            required_cells: max_cells,
            max_cells: MAX_DP_CELLS,
            family: "W/DP",
        });
    }

    // Check time: the naive recurrence is O(E·T²).  Cap it so large problems
    // fail fast with a clear error instead of hanging for minutes.
    let work = (e as u64)
        .saturating_mul(t_u as u64)
        .saturating_mul(t_u as u64);
    const MAX_W_DP_WORK: u64 = 1_000_000_000; // ~1s of log-domain ops
    if work > MAX_W_DP_WORK {
        return Err(FixedETError::TableTooLarge {
            t: t_u,
            e,
            required_cells: max_cells,
            max_cells: MAX_W_DP_WORK as usize,
            family: "W/DP",
        });
    }

    let log_degen = precompute_w_degeneracy(m, t_u);

    // Build table: rows[k] = log Z_W(k, s) for s in k..=T
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(e + 1);
    rows.push(vec![0.0]); // k=0: Z_W(0,0)=1

    for k in 1..=e {
        let s_max = t_u;
        let s_min = k;
        let len = s_max + 1 - s_min;
        let mut row = vec![f64::NEG_INFINITY; len];
        let prev = &rows[k - 1];
        for s in s_min..=s_max {
            // t in [1, s - (k-1)]
            let t_max = s - (k - 1);
            let mut best = f64::NEG_INFINITY;
            for t_val in 1..=t_max {
                let prev_idx = (s - t_val) - (k - 1);
                if prev_idx < prev.len() {
                    let w = log_degen[t_val] + prev[prev_idx];
                    best = logaddexp(best, w);
                }
            }
            row[s - s_min] = best;
        }
        rows.push(row);
    }

    // Sequential sampling
    let mut occupations = Vec::with_capacity(e);
    let mut remaining = t_u;
    let mut k = e;
    while k > 0 {
        let t_max = remaining - (k - 1);
        let mut branches: Vec<(OccNum, f64)> = Vec::new();
        let mut log_sum = f64::NEG_INFINITY;
        let prev = &rows[k - 1];
        for t_val in 1..=t_max {
            let prev_idx = (remaining - t_val) - (k - 1);
            if prev_idx < prev.len() {
                let w = log_degen[t_val] + prev[prev_idx];
                log_sum = logaddexp(log_sum, w);
                branches.push((t_val as OccNum, w));
            }
        }
        if branches.is_empty() {
            return Err(FixedETError::NoBackendAvailable);
        }
        let u = rng.random::<f64>();
        let mut cum = 0.0f64;
        let mut selected = branches.last().unwrap().0;
        for (t_val, w) in &branches {
            cum += (w - log_sum).exp();
            if u < cum {
                selected = *t_val;
                break;
            }
        }
        occupations.push(selected);
        remaining -= selected as usize;
        k -= 1;
    }
    debug_assert_eq!(remaining, 0);
    shuffle_slice(&mut occupations, rng);
    Ok(occupations)
}

fn logaddexp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn estimate_range() {
        for (m, t, e) in [(2, 10, 3), (3, 20, 5), (5, 50, 10)] {
            let r = estimate_w_rejection(m, t, e);
            assert!((0.0..=1.0).contains(&r));
        }
    }

    #[test]
    fn rejection_works() {
        let mut rng = StdRng::seed_from_u64(7);
        let r = try_w_weak_composition_rejection(2, 10, 3, 100, &mut rng).expect("accept");
        assert_eq!(r.len(), 3);
        assert_eq!(r.iter().copied().sum::<OccNum>(), 10);
        assert!(r.iter().all(|&c| c > 0));
    }

    #[test]
    fn star_mode() {
        // T small, K = M*E large → star mode
        let mut rng = StdRng::seed_from_u64(7);
        // E=2, M=5, T=6: K=10 boxes, K-1=9 > T=6 → star mode. T/E=3, feasible.
        let r = try_w_weak_composition_rejection(5, 6, 2, 200, &mut rng).expect("accept");
        assert_eq!(r.len(), 2);
        assert_eq!(r.iter().copied().sum::<OccNum>(), 6);
        assert!(r.iter().all(|&c| c > 0));
    }

    #[test]
    fn dp_produces_valid_vectors() {
        let mut rng = StdRng::seed_from_u64(42);
        for (m, t, e) in [(2, 10, 3), (3, 15, 5), (5, 20, 5)] {
            let sizes = sample_w_dp(m, t, e, &mut rng).unwrap();
            assert_eq!(sizes.len(), e);
            assert_eq!(sizes.iter().copied().sum::<OccNum>(), t);
            assert!(sizes.iter().all(|&s| s > 0));
        }
    }

    #[test]
    fn dp_table_too_large() {
        let err = sample_w_dp(10_000, 10_000, 10_000, &mut StdRng::seed_from_u64(0)).unwrap_err();
        assert!(matches!(err, FixedETError::TableTooLarge { .. }));
    }
}
