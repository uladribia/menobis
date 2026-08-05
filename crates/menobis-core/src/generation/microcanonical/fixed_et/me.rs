//! ME (MultiEdge) family implementation for fixed-(E,T) sampling.
//!
//! # Target law
//!
//! Positive occupations ∝ 1 / ∏ tᵢ!
//!
//! # Backends
//!
//! * Rejection: multinomial draw (labels with replacement), accept if no
//!   empty box.
//! * Fallback: Stirling-numbers-of-the-second-kind surjection (flat table
//!   with checked capacity).

use rand::rngs::StdRng;
use rand::Rng;

use super::core::{FixedETOccupancy, MAX_DP_CELLS};
use super::errors::FixedETError;
use crate::OccNum;

/// ME (MultiEdge) family: no extra parameters.
pub(crate) struct MeFamily;

impl FixedETOccupancy for MeFamily {
    fn family_name(&self) -> &'static str {
        "ME/Stirling"
    }

    fn max_occnum(&self) -> Option<OccNum> {
        None
    }

    fn validate_residual(&self, _e: usize, _t: OccNum) -> Result<(), FixedETError> {
        Ok(()) // only basic T ≥ E is needed
    }

    fn estimate_rejection(&self, t: OccNum, e: usize) -> f64 {
        estimate_me_rejection(t, e)
    }

    fn try_rejection(
        &self,
        t: OccNum,
        e: usize,
        max_attempts: usize,
        rng: &mut StdRng,
    ) -> Result<Vec<OccNum>, ()> {
        try_multinomial_rejection(t, e, max_attempts, rng)
    }

    fn sample_exact(
        &self,
        t: OccNum,
        e: usize,
        rng: &mut StdRng,
    ) -> Result<Vec<OccNum>, FixedETError> {
        sample_stirling_sizes(t, e, rng)
    }

    fn rejection_cost_per_attempt(&self, t: OccNum, e: usize) -> u64 {
        t + e as u64
    }
}

// ---------------------------------------------------------------------------
// Rejection-probability estimate
// ---------------------------------------------------------------------------

/// p_acc ≈ (1 − exp(−T/E))^E, computed in log space.
fn estimate_me_rejection(t: OccNum, e: usize) -> f64 {
    let lambda = t as f64 / e as f64;
    let one_minus_exp = -(-lambda).exp_m1();
    if one_minus_exp <= 0.0 {
        return 1.0;
    }
    let log_p_acc = (e as f64) * one_minus_exp.ln();
    (-log_p_acc.exp_m1()).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Multinomial rejection backend
// ---------------------------------------------------------------------------

/// Draw T events uniformly into E boxes, accept if all boxes are occupied.
fn try_multinomial_rejection(
    t: OccNum,
    e: usize,
    max_attempts: usize,
    rng: &mut StdRng,
) -> Result<Vec<OccNum>, ()> {
    let t_u = t as usize;
    let mut counts = vec![0_usize; e];
    for _ in 0..max_attempts {
        counts.fill(0);
        let mut occupied = 0usize;
        for _ in 0..t_u {
            let j = rng.random_range(0..e);
            if counts[j] == 0 {
                occupied += 1;
            }
            counts[j] += 1;
        }
        if occupied == e {
            return Ok(counts.iter().map(|&c| c as OccNum).collect());
        }
    }
    Err(())
}

// ---------------------------------------------------------------------------
// Stirling-numbers fallback (flat table)
// ---------------------------------------------------------------------------

/// Stable logaddexp.
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

/// Build the flat log-Stirling table and sample backward.
fn sample_stirling_sizes(
    t: OccNum,
    e: usize,
    rng: &mut StdRng,
) -> Result<Vec<OccNum>, FixedETError> {
    let t_u = t as usize;
    let stride = e + 1;
    let total_cells = (t_u + 1).saturating_mul(stride);
    if total_cells > MAX_DP_CELLS {
        return Err(FixedETError::TableTooLarge {
            t: t_u,
            e,
            required_cells: total_cells,
            max_cells: MAX_DP_CELLS,
            family: "ME/Stirling",
        });
    }

    let mut table = vec![f64::NEG_INFINITY; total_cells];
    let idx = |n: usize, k: usize| -> usize { n * stride + k };

    table[idx(0, 0)] = 0.0;

    for n in 1..=t_u {
        let max_k = n.min(e);
        for k in 1..=max_k {
            let term1 = if k < n {
                let v = table[idx(n - 1, k)];
                if v.is_finite() {
                    (k as f64).ln() + v
                } else {
                    f64::NEG_INFINITY
                }
            } else {
                f64::NEG_INFINITY
            };
            let term2 = table[idx(n - 1, k - 1)];
            table[idx(n, k)] = logaddexp(term1, term2);
        }
    }

    // Backward walk
    let mut n = t_u;
    let mut k = e;
    let mut decisions: Vec<bool> = Vec::with_capacity(t_u);

    while k > 1 && n > k {
        let log_new = table[idx(n - 1, k - 1)];
        let log_join = if table[idx(n - 1, k)].is_finite() {
            (k as f64).ln() + table[idx(n - 1, k)]
        } else {
            f64::NEG_INFINITY
        };
        let log_den = logaddexp(log_new, log_join);
        let p_new = ((log_new - log_den).exp()).clamp(0.0, 1.0);

        if rng.random::<f64>() < p_new {
            decisions.push(true);
            n -= 1;
            k -= 1;
        } else {
            decisions.push(false);
            n -= 1;
        }
    }

    let mut sizes: Vec<OccNum> = if k == 1 {
        vec![n as OccNum]
    } else {
        debug_assert_eq!(n, k);
        vec![1; k]
    };

    for &decision in decisions.iter().rev() {
        if decision {
            sizes.push(1);
        } else {
            let j = rng.random_range(0..sizes.len());
            sizes[j] += 1;
        }
    }

    // Shuffle for exchangeability
    super::support::shuffle_slice(&mut sizes, rng);
    Ok(sizes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn estimate_range() {
        for (t, e) in [(10, 3), (20, 5), (50, 10)] {
            let r = estimate_me_rejection(t, e);
            assert!((0.0..=1.0).contains(&r));
        }
    }

    #[test]
    fn rejection_works() {
        let mut rng = StdRng::seed_from_u64(7);
        let r = try_multinomial_rejection(100, 20, 100, &mut rng).expect("accept");
        assert_eq!(r.len(), 20);
        assert_eq!(r.iter().copied().sum::<OccNum>(), 100);
        assert!(r.iter().all(|&c| c > 0));
    }

    #[test]
    fn stirling_produces_valid_vectors() {
        let mut rng = StdRng::seed_from_u64(42);
        for (t, e) in [(10, 3), (15, 5)] {
            let sizes = sample_stirling_sizes(t, e, &mut rng).unwrap();
            assert_eq!(sizes.len(), e);
            assert_eq!(sizes.iter().copied().sum::<OccNum>(), t);
            assert!(sizes.iter().all(|&s| s > 0));
        }
    }

    #[test]
    fn stirling_table_too_large() {
        let err = sample_stirling_sizes(10_000, 10_000, &mut StdRng::seed_from_u64(0)).unwrap_err();
        assert!(matches!(err, FixedETError::TableTooLarge { .. }));
    }
}
