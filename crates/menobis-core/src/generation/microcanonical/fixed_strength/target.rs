//! Fixed-strength target: family degeneracy + optional cost potential.
//!
//! The [`StrengthTarget`] encapsulates the local log-weight calculation
//! used by the Metropolis acceptance step of the fixed-strength MCMC
//! chain.  For Phase 4 (no cost), `gamma = 0.0` and the cost provider is
//! `None`.  For Phase 5, a fitted `gamma` and a [`PairCostProvider`] are
//! injected.

use crate::distribution::OccupationFamily;
use crate::pairs::PairCostProvider;
use crate::OccNum;

/// Local log-weight delta for a pair occupation change.
///
/// For a proposal changing cell `(src, tgt)` from `old` to `new`,
/// and letting \(d_F(t)\) be the local degeneracy for family \(F\):
///
/// \[
/// \Delta\log\pi = \log d_F(t_{\text{new}}) - \log d_F(t_{\text{old}})
///     - \gamma \, c_{ij}\, (t_{\text{new}} - t_{\text{old}}).
/// \]
///
/// For Phase 4, `gamma = 0.0` and only the degeneracy ratio contributes.
pub struct StrengthTarget<'a> {
    pub family: OccupationFamily,
    /// Cost multiplier \(\gamma\).  Set to `0.0` for Phase 4.
    pub gamma: f64,
    /// Optional cost provider (Phase 5).  `None` disables the cost term.
    pub costs: Option<&'a dyn PairCostProvider>,
}

impl<'a> StrengthTarget<'a> {
    pub fn new(family: OccupationFamily, gamma: f64) -> Self {
        Self {
            family,
            gamma,
            costs: None,
        }
    }

    /// Construct a target with a cost provider (Phase 5 readiness).
    pub fn with_costs(
        family: OccupationFamily,
        gamma: f64,
        costs: &'a dyn PairCostProvider,
    ) -> Self {
        Self {
            family,
            gamma,
            costs: Some(costs),
        }
    }

    /// Compute \(\Delta\log\pi\) for changing one pair from `old` to `new`.
    ///
    /// Returns `None` if the new occupation would violate the family's
    /// occupation support (e.g., B occupation exceeding layers).
    ///
    /// The cost contribution is `-gamma * cost * (new - old)`.  When
    /// `gamma == 0.0` (Phase 4) this term vanishes regardless of cost.
    #[inline]
    pub fn delta_log_weight(&self, _src: u64, _tgt: u64, old: OccNum, new: OccNum) -> Option<f64> {
        // Validate new occupation against family support.
        if !self.family.validate_occnum(new) {
            return None;
        }

        // Degeneracy contribution.
        let log_old = self.family.log_local_degeneracy(old);
        let log_new = self.family.log_local_degeneracy(new);
        let mut delta = log_new - log_old;

        // Cost contribution (Phase 5: gamma > 0 with cost provider).
        if let Some(costs) = self.costs {
            if let Some(c) = costs.cost(_src as usize, _tgt as usize) {
                delta -= self.gamma * c * ((new as i64) - (old as i64)) as f64;
            } else {
                // Pair excluded from cost-constrained domain: reject.
                return None;
            }
        }

        Some(delta)
    }

    /// Compute \(\Delta\log\pi\) for a batch of pair changes.
    ///
    /// Each element is `(src, tgt, old_occ, new_occ)`.  Returns the sum
    /// of individual deltas, or `None` if any individual change is invalid.
    pub fn delta_log_weight_batch(&self, changes: &[(u64, u64, OccNum, OccNum)]) -> Option<f64> {
        let mut total = 0.0;
        for &(src, tgt, old, new) in changes {
            match self.delta_log_weight(src, tgt, old, new) {
                Some(d) => total += d,
                None => return None,
            }
        }
        Some(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::OccupationFamily;

    /// Reference log-local-degeneracy formulas for verification.
    fn me_log_degen(t: OccNum) -> f64 {
        if t == 0 {
            return 0.0; // 0! = 1, log(1) = 0
        }
        // log(1/t!) = -lgamma(t+1)
        -libm::lgamma((t as f64) + 1.0)
    }

    fn b_log_degen(t: OccNum, m: u32) -> f64 {
        if t > m as OccNum {
            return f64::NEG_INFINITY;
        }
        let mf = m as f64;
        let tf = t as f64;
        libm::lgamma(mf + 1.0) - libm::lgamma(tf + 1.0) - libm::lgamma(mf - tf + 1.0)
    }

    fn w_log_degen(t: OccNum, m: u32) -> f64 {
        let mf = m as f64;
        let tf = t as f64;
        libm::lgamma(mf + tf) - libm::lgamma(mf) - libm::lgamma(tf + 1.0)
    }

    #[test]
    fn me_delta_increment() {
        let target = StrengthTarget::new(OccupationFamily::Poisson, 0.0);
        // d_ME(t+1)/d_ME(t) = 1/(t+1) → log = -log(t+1)
        // So delta_log = -log(6) - (-log(5)) = log(5/6)
        let delta = target.delta_log_weight(0, 0, 5, 6).unwrap();
        let expected = me_log_degen(6) - me_log_degen(5);
        assert!((delta - expected).abs() < 1e-12);
    }

    #[test]
    fn me_delta_decrement() {
        let target = StrengthTarget::new(OccupationFamily::Poisson, 0.0);
        // d_ME(t-1)/d_ME(t) = t → log = log(t)
        let delta = target.delta_log_weight(0, 0, 5, 4).unwrap();
        let expected = me_log_degen(4) - me_log_degen(5);
        assert!((delta - expected).abs() < 1e-12);
    }

    #[test]
    fn me_zero_to_one() {
        let target = StrengthTarget::new(OccupationFamily::Poisson, 0.0);
        // log(1/1!) - log(1/0!) = 0 - 0 = 0
        let delta = target.delta_log_weight(0, 0, 0, 1).unwrap();
        assert!((delta - 0.0).abs() < 1e-12);
    }

    #[test]
    fn b_delta_increment() {
        // B with M=5: increment from 2 to 3
        // d(3)/d(2) = (M-2)/(2+1) = 3/3 = 1 → log = 0
        let target = StrengthTarget::new(OccupationFamily::Binomial(5), 0.0);
        let delta = target.delta_log_weight(0, 0, 2, 3).unwrap();
        let expected = b_log_degen(3, 5) - b_log_degen(2, 5);
        assert!((delta - expected).abs() < 1e-12);
    }

    #[test]
    fn b_delta_at_capacity() {
        // B with M=3: increment from 3 to 4 should be invalid
        let target = StrengthTarget::new(OccupationFamily::Binomial(3), 0.0);
        assert!(target.delta_log_weight(0, 0, 3, 4).is_none());
    }

    #[test]
    fn w_delta_increment() {
        // W with M=2: increment from 1 to 2
        // d(2)/d(1) = (M+1)/(1+1) = (2+1)/2 = 3/2 → log = ln(1.5)
        let target = StrengthTarget::new(OccupationFamily::NegativeBinomial(2), 0.0);
        let delta = target.delta_log_weight(0, 0, 1, 2).unwrap();
        let expected = w_log_degen(2, 2) - w_log_degen(1, 2);
        assert!((delta - expected).abs() < 1e-12);
    }

    #[test]
    fn batch_sum() {
        let target = StrengthTarget::new(OccupationFamily::Poisson, 0.0);
        let changes = vec![
            (0, 1, 2, 3), // ME: log(1/3) - log(1/2) = -log(3) + log(2) = log(2/3)
            (1, 0, 5, 4), // ME: log(1/4!) - log(1/5!) = log(5)
        ];
        let expected = (me_log_degen(3) - me_log_degen(2)) + (me_log_degen(4) - me_log_degen(5));
        let delta = target.delta_log_weight_batch(&changes).unwrap();
        assert!((delta - expected).abs() < 1e-12);
    }

    /// A test-only cost provider: `cost(i, j) = |i − j|`.
    struct LinearCost;
    impl crate::pairs::PairCostProvider for LinearCost {
        fn cost(&self, source: usize, target: usize) -> Option<f64> {
            Some((source as i64 - target as i64).unsigned_abs() as f64)
        }
    }

    #[test]
    fn cost_potential_adds_gamma_term() {
        // Phase 5 readiness: gamma>0 with a cost provider adds the cost
        // term to the local log-weight delta.
        let costs = LinearCost;
        let target = StrengthTarget::with_costs(OccupationFamily::Poisson, 2.0, &costs);

        // Increment (0,1) from 2 to 3:
        // Δ = log(1/3!) − log(1/2!) − γ·c·(3−2) = log(2/3) − 2·1·1
        let delta = target.delta_log_weight(0, 1, 2, 3).unwrap();
        let expected = me_log_degen(3) - me_log_degen(2) - 2.0 * 1.0 * 1.0;
        assert!((delta - expected).abs() < 1e-12);
    }

    #[test]
    fn cost_potential_requires_admissible_pair() {
        // A provider returning None for a pair excludes it (Phase 5
        // semantics match grand-canonical admissibility).
        struct ExcludingCost;
        impl crate::pairs::PairCostProvider for ExcludingCost {
            fn cost(&self, _source: usize, _target: usize) -> Option<f64> {
                None
            }
        }
        let costs = ExcludingCost;
        let target = StrengthTarget::with_costs(OccupationFamily::Poisson, 1.0, &costs);
        assert!(target.delta_log_weight(0, 1, 2, 3).is_none());
    }
}
