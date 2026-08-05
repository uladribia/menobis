//! B (BinaryLayers) family implementation for fixed-(E,T) sampling.
//!
//! # Target law
//!
//! Positive occupations ∏ C(M, tᵢ) for 1 ≤ tᵢ ≤ M, sum = T.
//!
//! # Backends
//!
//! * Rejection: uniform binary-cell subset (T cells from EM), accept if no
//!   empty row.  Complement mode when T > EM/2.
//! * Fallback: bounded-composition DP storing Z_B(k, s) in log space.

use rand::rngs::StdRng;
use rand::Rng;

use super::core::{FixedETOccupancy, MAX_DP_CELLS};
use super::errors::FixedETError;
use crate::generation::microcanonical::support::uniform_edges::{
    sample_uniform_support, shuffle_slice,
};
use crate::OccNum;

/// B (BinaryLayers) family with M layers.
pub struct BFamily {
    pub layers: OccNum,
}

impl FixedETOccupancy for BFamily {
    fn family_name(&self) -> &'static str {
        "B/DP"
    }

    fn max_occnum(&self) -> Option<OccNum> {
        Some(self.layers)
    }

    fn validate_residual(&self, e: usize, t: OccNum) -> Result<(), FixedETError> {
        let m = self.layers as OccNum;
        if t > m * e as OccNum {
            return Err(FixedETError::InvalidResidual(format!(
                "B residual total {t} exceeds M·E = {m}×{e} = {}",
                m * e as OccNum
            )));
        }
        Ok(())
    }

    fn estimate_rejection(&self, t: OccNum, e: usize) -> f64 {
        estimate_b_rejection(self.layers, t, e)
    }

    fn try_rejection(
        &self,
        t: OccNum,
        e: usize,
        max_attempts: usize,
        rng: &mut StdRng,
    ) -> Result<Vec<OccNum>, ()> {
        try_b_subset_rejection(self.layers, t, e, max_attempts, rng)
    }

    fn sample_exact(
        &self,
        t: OccNum,
        e: usize,
        rng: &mut StdRng,
    ) -> Result<Vec<OccNum>, FixedETError> {
        sample_b_dp(self.layers, t, e, rng)
    }

    fn rejection_cost_per_attempt(&self, t: OccNum, e: usize) -> u64 {
        let total = self.layers * e as OccNum;
        total.saturating_sub(t).min(t).max(1) // min(T, M*E - T)
    }
}

// ---------------------------------------------------------------------------
// Rejection-probability estimate
// ---------------------------------------------------------------------------

/// Hypergeometric estimate: q₀ = C(M(E-1), T) / C(ME, T), p̂_acc = (1−q₀)^E.
fn estimate_b_rejection(m: OccNum, t: OccNum, e: usize) -> f64 {
    let m_u = m;
    let e_u = e as u64;
    let t_u = t;
    let total_cells = m_u * e_u;
    if t_u > m_u * (e_u - 1) {
        return 0.0; // q₀ = 0 → p_acc = 1 → no rejection
    }

    let log_q0 = log_binomial(m_u * (e_u - 1), t_u) - log_binomial(total_cells, t_u);
    let q0 = log_q0.exp();
    let log_p_acc = (e as f64) * (-q0).ln_1p();
    (-log_p_acc.exp_m1()).clamp(0.0, 1.0)
}

/// log C(n, k) via log-gamma.
fn log_binomial(n: u64, k: u64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    lg(n + 1) - lg(k + 1) - lg(n - k + 1)
}

fn lg(x: u64) -> f64 {
    libm::lgamma(x as f64)
}

/// Precompute log C(M, t) for t = 0..=M.
fn precompute_b_degeneracy(m: OccNum) -> Vec<f64> {
    let m_u = m as usize;
    (0..=m_u)
        .map(|t| log_binomial(m_u as u64, t as u64))
        .collect()
}

// ---------------------------------------------------------------------------
// Cell-subset rejection  (with complement mode)
// ---------------------------------------------------------------------------

/// Sample a uniform subset of T cells from EM cells, accept if every row > 0.
fn try_b_subset_rejection(
    m: OccNum,
    t: OccNum,
    e: usize,
    max_attempts: usize,
    rng: &mut StdRng,
) -> Result<Vec<OccNum>, ()> {
    let m_u = m as usize;
    let total_cells = e * m_u;
    let t_u = t as usize;
    let use_complement = t_u > total_cells / 2;

    for _ in 0..max_attempts {
        let mut counts: Vec<OccNum> = if use_complement {
            vec![m; e] // start at max, decrement for holes
        } else {
            vec![0; e]
        };

        let sample_size = if use_complement {
            total_cells - t_u
        } else {
            t_u
        };
        let cells = sample_uniform_support(total_cells, sample_size, rng);

        if use_complement {
            for &cell in &cells {
                let row = cell / m_u;
                counts[row] -= 1;
            }
            if counts.iter().all(|&c| c > 0) {
                return Ok(counts);
            }
        } else {
            let mut occupied = 0usize;
            for &cell in &cells {
                let row = cell / m_u;
                if counts[row] == 0 {
                    occupied += 1;
                }
                counts[row] += 1;
            }
            if occupied == e {
                return Ok(counts);
            }
        }
    }
    Err(())
}

// ---------------------------------------------------------------------------
// Bounded-composition DP
// ---------------------------------------------------------------------------

/// Build Z_B(k, s) in log space and sample sequentially.
#[allow(clippy::needless_range_loop)]
fn sample_b_dp(
    m: OccNum,
    t: OccNum,
    e: usize,
    rng: &mut StdRng,
) -> Result<Vec<OccNum>, FixedETError> {
    let m_u = m as usize;
    let t_u = t as usize;

    // Precompute log C(M, t)
    let log_binom = precompute_b_degeneracy(m);

    // Check memory
    // Table: row k (0..=e) stores s from k to min(T, M*k)
    // Max cells ≈ sum_{k=0}^{e} (min(T, M*k) - k + 1) ≤ (e+1)*(T+1)
    let max_cells = (e + 1).saturating_mul(t_u + 1);
    if max_cells > MAX_DP_CELLS {
        return Err(FixedETError::TableTooLarge {
            t: t_u,
            e,
            required_cells: max_cells,
            max_cells: MAX_DP_CELLS,
            family: "B/DP",
        });
    }

    // Build table: rows[k] = Vec<f64>, length = min(T, M*k) - k + 1
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(e + 1);
    rows.push(vec![0.0]); // k=0: Z_B(0,0) = 1

    for k in 1..=e {
        let s_max = t_u.min(m_u * k);
        let s_min = k;
        let len = s_max + 1 - s_min;
        let mut row = vec![f64::NEG_INFINITY; len];
        for s in s_min..=s_max {
            let t_min = 1.max(s as i64 - (m_u * (k - 1)) as i64) as usize;
            let t_max = m_u.min(s - (k - 1));
            let mut best = f64::NEG_INFINITY;
            for t_val in t_min..=t_max {
                let prev_s = s - t_val;
                // prev_s >= k-1 by construction; index into rows[k-1]
                let prev_idx = prev_s - (k - 1);
                if prev_idx < rows[k - 1].len() {
                    let w = log_binom[t_val] + rows[k - 1][prev_idx];
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
        let t_min = 1.max(remaining as i64 - (m_u * (k - 1)) as i64) as usize;
        let t_max = m_u.min(remaining - (k - 1));
        let mut branches: Vec<(OccNum, f64)> = Vec::new();
        let mut log_sum = f64::NEG_INFINITY;
        for t_val in t_min..=t_max {
            let prev_s = remaining - t_val;
            let prev_idx = prev_s - (k - 1);
            if prev_idx < rows[k - 1].len() {
                let w = log_binom[t_val] + rows[k - 1][prev_idx];
                log_sum = logaddexp(log_sum, w);
                branches.push((t_val as OccNum, w));
            }
        }
        if branches.is_empty() {
            return Err(FixedETError::NoBackendAvailable);
        }
        // Sample from categorical
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
    // Shuffle for exchangeability
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
            let r = estimate_b_rejection(m, t, e);
            assert!((0.0..=1.0).contains(&r));
        }
    }

    #[test]
    fn rejection_works() {
        let mut rng = StdRng::seed_from_u64(7);
        let r = try_b_subset_rejection(3, 15, 5, 100, &mut rng).expect("accept");
        assert_eq!(r.len(), 5);
        assert_eq!(r.iter().copied().sum::<OccNum>(), 15);
        assert!(r.iter().all(|&c| c > 0 && c <= 3));
    }

    #[test]
    fn complement_mode() {
        // T close to M*E → complement mode
        let mut rng = StdRng::seed_from_u64(7);
        let r = try_b_subset_rejection(5, 23, 5, 100, &mut rng).expect("accept");
        assert_eq!(r.len(), 5);
        assert_eq!(r.iter().copied().sum::<OccNum>(), 23);
        assert!(r.iter().all(|&c| c > 0 && c <= 5));
    }

    #[test]
    fn dp_produces_valid_vectors() {
        let mut rng = StdRng::seed_from_u64(42);
        for (m, t, e) in [(2, 6, 3), (3, 15, 5), (4, 16, 5)] {
            let sizes = sample_b_dp(m, t, e, &mut rng).unwrap();
            assert_eq!(sizes.len(), e);
            assert_eq!(sizes.iter().copied().sum::<OccNum>(), t);
            assert!(sizes.iter().all(|&s| s > 0 && s <= m));
        }
    }

    #[test]
    fn all_ones_when_t_equals_e() {
        // Not a unit test of the DP per se, but B special case: when T=E, occupations are all 1.
        let mut rng = StdRng::seed_from_u64(42);
        let sizes = sample_b_dp(5, 5, 5, &mut rng).unwrap();
        assert_eq!(sizes, vec![1; 5]);
    }

    #[test]
    fn dp_table_too_large() {
        let err = sample_b_dp(10_000, 10_000, 10_000, &mut StdRng::seed_from_u64(0)).unwrap_err();
        assert!(matches!(err, FixedETError::TableTooLarge { .. }));
    }
}
