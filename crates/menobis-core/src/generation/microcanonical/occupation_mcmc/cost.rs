//! Cost measurement and validation for fixed-strength cost-constrained sampling.
//!
//! Provides:
//! - [`state_cost`] — total cost of a [`StrengthState`] under a [`PairCostProvider`].
//! - [`validate_cost_value`] — checks a single cost value for NaN/Inf.
//! - [`fixed_pairs_cost`] — total cost contributed by fixed pairs.
//! - [`residual_cost_target`] — observed cost minus fixed-pair cost.

use super::errors::FixedStrengthCostError;
use super::state::StrengthState;
use crate::pairs::PairCostProvider;

/// Validate a single cost value from a [`PairCostProvider`].
///
/// # Errors
///
/// Returns [`NonFiniteCost`](FixedStrengthCostError::NonFiniteCost) if
/// the value is `NaN`, `+inf`, or `-inf`.
#[inline]
pub fn validate_cost_value(
    source: u64,
    target: u64,
    value: f64,
) -> Result<(), FixedStrengthCostError> {
    if !value.is_finite() {
        return Err(FixedStrengthCostError::NonFiniteCost {
            source,
            target,
            value,
        });
    }
    Ok(())
}

/// Compute the total cost of a [`StrengthState`] under a [`PairCostProvider`].
///
/// Iterates only over occupied pairs.  Each occupied pair contributes
/// `c_ij * t_ij` to the total.
///
/// # Errors
///
/// - [`MissingCost`](FixedStrengthCostError::MissingCost) if the
///   provider returns `None` for an occupied pair.
/// - [`NonFiniteCost`](FixedStrengthCostError::NonFiniteCost) if the
///   provider returns a non-finite value.
pub fn state_cost(
    state: &StrengthState,
    costs: &dyn PairCostProvider,
) -> Result<f64, FixedStrengthCostError> {
    let mut total = 0.0;
    for ((src, tgt), occ) in state.iter_occupied() {
        let c =
            costs
                .cost(src as usize, tgt as usize)
                .ok_or(FixedStrengthCostError::MissingCost {
                    source: src,
                    target: tgt,
                })?;
        validate_cost_value(src, tgt, c)?;
        total += c * (occ as f64);
    }
    Ok(total)
}

/// Compute the total cost contributed by a set of fixed (pre-determined) pairs.
///
/// # Errors
///
/// Same as [`state_cost`] — any pair with a missing or non-finite cost
/// returns an error.
pub fn fixed_pairs_cost(
    fixed_pairs: &[(u64, u64, crate::OccNum)],
    costs: &dyn PairCostProvider,
) -> Result<f64, FixedStrengthCostError> {
    let mut total = 0.0;
    for &(src, tgt, occ) in fixed_pairs {
        let c =
            costs
                .cost(src as usize, tgt as usize)
                .ok_or(FixedStrengthCostError::MissingCost {
                    source: src,
                    target: tgt,
                })?;
        validate_cost_value(src, tgt, c)?;
        total += c * (occ as f64);
    }
    Ok(total)
}

/// Compute the residual cost target for the MCMC problem.
///
/// `residual = observed_total - fixed_cost`.  Validates that the result
/// is finite.
///
/// # Errors
///
/// - [`NonFiniteObservedCost`](FixedStrengthCostError::NonFiniteObservedCost)
///   if the observed total is non-finite.
/// - [`ResidualCostInconsistent`](FixedStrengthCostError::ResidualCostInconsistent)
///   if the residual would be negative (indicating fixed-pair cost exceeds total).
pub fn residual_cost_target(
    observed_total: f64,
    fixed_cost: f64,
) -> Result<f64, FixedStrengthCostError> {
    if !observed_total.is_finite() {
        return Err(FixedStrengthCostError::NonFiniteObservedCost {
            value: observed_total,
        });
    }
    let residual = observed_total - fixed_cost;
    if residual < 0.0 {
        return Err(FixedStrengthCostError::ResidualCostInconsistent {
            total: observed_total,
            fixed: fixed_cost,
            residual,
        });
    }
    Ok(residual)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::microcanonical::occupation_mcmc::domain::PairDomain;
    use crate::generation::microcanonical::occupation_mcmc::initializer::initialize_table;
    use crate::generation::microcanonical::occupation_mcmc::state::StrengthState;
    use crate::model::family::OccupationFamily;

    struct LinearCost;
    impl PairCostProvider for LinearCost {
        fn cost(&self, source: usize, target: usize) -> Option<f64> {
            Some((source as i64 - target as i64).unsigned_abs() as f64)
        }
    }

    struct NoOpCost;
    impl PairCostProvider for NoOpCost {
        fn cost(&self, _source: usize, _target: usize) -> Option<f64> {
            Some(0.0)
        }
    }

    struct PartialCost {
        /// Pairs (src, tgt) that return None.
        excluded: std::collections::HashSet<(usize, usize)>,
    }
    impl PairCostProvider for PartialCost {
        fn cost(&self, source: usize, target: usize) -> Option<f64> {
            if self.excluded.contains(&(source, target)) {
                None
            } else {
                Some(1.0)
            }
        }
    }

    fn make_state(n: usize, so: &[u64], si: &[u64]) -> StrengthState {
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };
        let table = initialize_table(so, si, OccupationFamily::ME, &domain).unwrap();
        StrengthState::new(n, table)
    }

    #[test]
    fn empty_state_zero_cost() {
        let state = StrengthState::new(3, vec![]);
        let costs = NoOpCost;
        let total = state_cost(&state, &costs).unwrap();
        assert!((total - 0.0).abs() < 1e-12);
    }

    #[test]
    fn state_cost_single_pair() {
        let state = StrengthState::new(2, vec![((0, 1), 5)]);
        // |0-1| = 1, so cost = 5 * 1 = 5.0
        let costs = LinearCost;
        let total = state_cost(&state, &costs).unwrap();
        assert!((total - 5.0).abs() < 1e-12);
    }

    #[test]
    fn state_cost_heterogeneous() {
        // N=3, strengths balanced
        let so = vec![5u64, 3, 2];
        let si = vec![2u64, 4, 4];
        let state = make_state(3, &so, &si);
        let costs = LinearCost;
        let total = state_cost(&state, &costs).unwrap();
        // Compute expected: sum over occupied pairs of |src-tgt| * occ
        let mut expected = 0.0;
        for ((src, tgt), occ) in state.iter_occupied() {
            expected += (src as i64 - tgt as i64).unsigned_abs() as f64 * (occ as f64);
        }
        assert!((total - expected).abs() < 1e-12);
    }

    #[test]
    fn missing_cost_error() {
        let state = StrengthState::new(2, vec![((0, 1), 3)]);
        let mut excluded = std::collections::HashSet::new();
        excluded.insert((0, 1));
        let costs = PartialCost { excluded };
        let err = state_cost(&state, &costs).unwrap_err();
        assert!(
            matches!(
                err,
                FixedStrengthCostError::MissingCost {
                    source: 0,
                    target: 1
                }
            ),
            "expected MissingCost, got {err:?}"
        );
    }

    #[test]
    fn non_finite_cost_error() {
        struct InfCost;
        impl PairCostProvider for InfCost {
            fn cost(&self, _source: usize, _target: usize) -> Option<f64> {
                Some(f64::INFINITY)
            }
        }
        let state = StrengthState::new(2, vec![((0, 1), 3)]);
        let err = state_cost(&state, &InfCost).unwrap_err();
        assert!(
            matches!(err, FixedStrengthCostError::NonFiniteCost { .. }),
            "expected NonFiniteCost, got {err:?}"
        );
    }

    #[test]
    fn fixed_pairs_cost_empty() {
        let costs = NoOpCost;
        let total = fixed_pairs_cost(&[], &costs).unwrap();
        assert!((total - 0.0).abs() < 1e-12);
    }

    #[test]
    fn fixed_pairs_cost_single() {
        let costs = LinearCost;
        let pairs = vec![(0u64, 1u64, 5u64)];
        let total = fixed_pairs_cost(&pairs, &costs).unwrap();
        // |0-1| = 1, cost = 5 * 1 = 5.0
        assert!((total - 5.0).abs() < 1e-12);
    }

    #[test]
    fn residual_target_positive() {
        let obs = 100.0;
        let fixed = 30.0;
        let residual = residual_cost_target(obs, fixed).unwrap();
        assert!((residual - 70.0).abs() < 1e-12);
    }

    #[test]
    fn residual_target_negative_errors() {
        let obs = 30.0;
        let fixed = 100.0;
        let err = residual_cost_target(obs, fixed).unwrap_err();
        assert!(
            matches!(err, FixedStrengthCostError::ResidualCostInconsistent { .. }),
            "expected ResidualCostInconsistent, got {err:?}"
        );
    }

    #[test]
    fn non_finite_observed_cost() {
        let err = residual_cost_target(f64::NAN, 0.0).unwrap_err();
        assert!(
            matches!(err, FixedStrengthCostError::NonFiniteObservedCost { .. }),
            "expected NonFiniteObservedCost, got {err:?}"
        );
    }

    #[test]
    fn validate_cost_finite_ok() {
        assert!(validate_cost_value(0, 0, 1.5).is_ok());
        assert!(validate_cost_value(0, 0, 0.0).is_ok());
        assert!(validate_cost_value(0, 0, -3.0).is_ok());
    }

    #[test]
    fn validate_cost_non_finite() {
        assert!(validate_cost_value(0, 0, f64::NAN).is_err());
        assert!(validate_cost_value(0, 0, f64::INFINITY).is_err());
        assert!(validate_cost_value(0, 0, f64::NEG_INFINITY).is_err());
    }
}
