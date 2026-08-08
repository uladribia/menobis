//! Fixed-strength target: family degeneracy + optional cost potential.
//!
//! The [`StrengthTarget`] encapsulates the local log-weight calculation
//! used by the Metropolis acceptance step of the fixed-strength MCMC
//! chain.  Gamma is always fitted from observed cost data — it is never
//! user-supplied.

use crate::model::family::OccupationFamily;
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
/// Gamma is set by the fitting procedure.  Missing or non-finite cost
/// values are fatal configuration errors.
pub struct StrengthTarget<'a> {
    pub family: OccupationFamily,
    /// Cost multiplier \(\gamma\).  Always fitted from observed cost.
    /// Not directly settable by users.
    gamma: f64,
    /// Optional cost provider.  `None` disables the cost term.
    pub costs: Option<&'a dyn PairCostProvider>,
}

impl<'a> StrengthTarget<'a> {
    /// Create a target without cost (Phase 4 mode).  Gamma is 0.
    pub fn new(family: OccupationFamily) -> Self {
        Self {
            family,
            gamma: 0.0,
            costs: None,
        }
    }

    /// Create a target with a cost provider.  Gamma starts at 0.0
    /// and must be set via [`set_gamma`](Self::set_gamma) after fitting.
    pub fn with_costs(family: OccupationFamily, costs: &'a dyn PairCostProvider) -> Self {
        Self {
            family,
            gamma: 0.0,
            costs: Some(costs),
        }
    }

    /// Set the cost multiplier \(\gamma\).  Called by the fitting loop.
    pub fn set_gamma(&mut self, gamma: f64) {
        self.gamma = gamma;
    }

    /// Current gamma value.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Whether the target has a non-trivial cost term (provider present
    /// and gamma != 0).
    pub fn has_nontrivial_cost(&self) -> bool {
        self.costs.is_some() && self.gamma != 0.0
    }

    /// Compute \(\Delta\log\pi\) for changing one pair from `old` to `new`.
    ///
    /// Returns `None` if the new occupation would violate the family's
    /// occupation support (e.g., B occupation exceeding layers).
    ///
    /// # Panics
    ///
    /// Panics if a cost provider is present, gamma is non-zero, and the
    /// provider returns `None` or a non-finite value for the pair.  This
    /// is a configuration error: all admissible pairs must have valid costs.
    #[inline]
    pub fn delta_log_weight(&self, src: u64, tgt: u64, old: OccNum, new: OccNum) -> Option<f64> {
        // Validate new occupation against family support.
        if !self.family.validate_occnum(new) {
            return None;
        }

        // Degeneracy contribution.
        let delta = self.family.delta_log_base_measure(old, new);

        // Cost contribution (only when gamma != 0 and provider present).
        if let Some(costs) = self.costs {
            if self.gamma != 0.0 {
                let c = costs.cost(src as usize, tgt as usize).unwrap_or_else(|| {
                    panic!("cost provider returned None for admissible pair ({src}, {tgt})");
                });
                assert!(
                    c.is_finite(),
                    "cost provider returned non-finite cost {c} for pair ({src}, {tgt})"
                );
                let occ_delta = (new as i64) - (old as i64);
                Some(delta - self.gamma * c * (occ_delta as f64))
            } else {
                Some(delta)
            }
        } else {
            Some(delta)
        }
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
    use crate::model::family::OccupationFamily;

    /// Reference log-local-degeneracy formulas for verification.
    fn me_log_degen(t: OccNum) -> f64 {
        if t == 0 {
            return 0.0;
        }
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
        let target = StrengthTarget::new(OccupationFamily::ME);
        let delta = target.delta_log_weight(0, 0, 5, 6).unwrap();
        let expected = me_log_degen(6) - me_log_degen(5);
        assert!((delta - expected).abs() < 1e-12);
    }

    #[test]
    fn me_delta_decrement() {
        let target = StrengthTarget::new(OccupationFamily::ME);
        let delta = target.delta_log_weight(0, 0, 5, 4).unwrap();
        let expected = me_log_degen(4) - me_log_degen(5);
        assert!((delta - expected).abs() < 1e-12);
    }

    #[test]
    fn me_zero_to_one() {
        let target = StrengthTarget::new(OccupationFamily::ME);
        let delta = target.delta_log_weight(0, 0, 0, 1).unwrap();
        assert!((delta - 0.0).abs() < 1e-12);
    }

    #[test]
    fn b_delta_increment() {
        let target = StrengthTarget::new(OccupationFamily::B { layers: 5 });
        let delta = target.delta_log_weight(0, 0, 2, 3).unwrap();
        let expected = b_log_degen(3, 5) - b_log_degen(2, 5);
        assert!((delta - expected).abs() < 1e-12);
    }

    #[test]
    fn b_delta_at_capacity() {
        let target = StrengthTarget::new(OccupationFamily::B { layers: 3 });
        assert!(target.delta_log_weight(0, 0, 3, 4).is_none());
    }

    #[test]
    fn w_delta_increment() {
        let target = StrengthTarget::new(OccupationFamily::W { layers: 2 });
        let delta = target.delta_log_weight(0, 0, 1, 2).unwrap();
        let expected = w_log_degen(2, 2) - w_log_degen(1, 2);
        assert!((delta - expected).abs() < 1e-12);
    }

    #[test]
    fn batch_sum() {
        let target = StrengthTarget::new(OccupationFamily::ME);
        let changes = vec![(0, 1, 2, 3), (1, 0, 5, 4)];
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
        let costs = LinearCost;
        let mut target = StrengthTarget::with_costs(OccupationFamily::ME, &costs);
        target.set_gamma(2.0);

        // Increment (0,1) from 2 to 3:
        // Δ = log(1/3!) − log(1/2!) − γ·c·(3−2) = log(2/3) − 2·1·1
        let delta = target.delta_log_weight(0, 1, 2, 3).unwrap();
        let expected = me_log_degen(3) - me_log_degen(2) - 2.0 * 1.0 * 1.0;
        assert!((delta - expected).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "cost provider returned None")]
    fn missing_cost_panics_when_gamma_nonzero() {
        struct ExcludingCost;
        impl crate::pairs::PairCostProvider for ExcludingCost {
            fn cost(&self, _source: usize, _target: usize) -> Option<f64> {
                None
            }
        }
        let costs = ExcludingCost;
        let mut target = StrengthTarget::with_costs(OccupationFamily::ME, &costs);
        target.set_gamma(1.0);
        // This should panic, not return None.
        target.delta_log_weight(0, 1, 2, 3);
    }

    #[test]
    fn gamma_defaults_to_zero() {
        let target = StrengthTarget::new(OccupationFamily::ME);
        assert!((target.gamma() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn with_costs_gamma_starts_zero() {
        let costs = LinearCost;
        let target = StrengthTarget::with_costs(OccupationFamily::ME, &costs);
        assert!((target.gamma() - 0.0).abs() < 1e-12);
        assert!(!target.has_nontrivial_cost());
    }

    #[test]
    fn set_gamma_works() {
        let costs = LinearCost;
        let mut target = StrengthTarget::with_costs(OccupationFamily::ME, &costs);
        target.set_gamma(1.5);
        assert!((target.gamma() - 1.5).abs() < 1e-12);
        assert!(target.has_nontrivial_cost());
    }

    #[test]
    fn zero_gamma_cost_term_vanishes() {
        let costs = LinearCost;
        let mut target = StrengthTarget::with_costs(OccupationFamily::ME, &costs);
        // gamma is 0.0 → cost term disabled → same as no-cost target.
        let without = StrengthTarget::new(OccupationFamily::ME);
        let d_with = target.delta_log_weight(0, 1, 2, 3).unwrap();
        let d_without = without.delta_log_weight(0, 1, 2, 3).unwrap();
        assert!((d_with - d_without).abs() < 1e-12);

        // After setting gamma=0 explicitly, still no cost term.
        target.set_gamma(0.0);
        let d_zero = target.delta_log_weight(0, 1, 2, 3).unwrap();
        assert!((d_zero - d_without).abs() < 1e-12);
    }
}
