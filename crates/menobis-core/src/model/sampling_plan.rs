//! Sampling-plan routing — encodes how a constraint set decomposes.
//!
//! The [`SamplingPlan`] enum classifies an ensemble by its probabilistic
//! structure, not its constraint name.  There are exactly three
//! structural categories (matching §48 of the refactor plan):
//!
//! | Plan | Structure |
//! |---|---|
//! | `GrandCanonical` | direct factorized pair sampling |
//! | `FactorizedMicrocanonical` | binary support stage + occupation allocation |
//! | `OccupationMcmc` | coupled occupation-number MCMC |
//!
//! The specific support strategy (uniform fixed-\(E\) vs degree-constrained
//! vs ...) is an implementation detail of the factorized branch, not a
//! separate plan variant.

use super::problem::PreparedProblem;

/// High-level sampling plan, determined from constraint structure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SamplingPlan {
    /// Grand-canonical: independent pair distributions.
    GrandCanonical,
    /// Factorized microcanonical: sample binary support, then occupations.
    FactorizedMicrocanonical,
    /// Coupled occupation-number MCMC (fixed strengths, strength+E, …).
    OccupationMcmc,
}

impl SamplingPlan {
    /// Classify a prepared problem into a sampling plan.
    ///
    /// Rules (checked in order — §54 routing priority):
    ///
    /// 1. Residual **strength** constraint → occupation MCMC **first**.
    ///    Coupled strengths (`s`, `s+E`, `s+k`) always require the
    ///    occupation-MCMC branch; a strength constraint must never fall
    ///    through to the factorized branch, which would silently ignore
    ///    the strengths (§80 release blocker).
    /// 2. Residual degree or edge constraint → factorized MC
    ///    (binary support + fixed-total occupation).
    /// 3. Otherwise (no hard constraints) → grand canonical.
    pub fn classify(problem: &PreparedProblem) -> Self {
        // Strength constraints win routing priority (§54): within the
        // branch the backend dispatches s+k → s+E → s in that order.
        if problem.residual_out_strengths.is_some() {
            return Self::OccupationMcmc;
        }

        // Degree or edge constraint factorizes: support then occupations.
        if problem.residual_out_degrees.is_some() || problem.residual_edges.is_some() {
            return Self::FactorizedMicrocanonical;
        }

        // No hard constraints → expectation constraints only → GC.
        Self::GrandCanonical
    }
}

#[cfg(test)]
mod tests {
    use super::super::family::OccupationFamily;
    use super::super::problem::PreparedProblem;
    use super::*;

    #[test]
    fn grand_canonical_when_no_total() {
        let p = PreparedProblem::new(
            OccupationFamily::ME,
            10,
            false,
            90,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert_eq!(SamplingPlan::classify(&p), SamplingPlan::GrandCanonical);
    }

    #[test]
    fn factorized_edges() {
        let p = PreparedProblem::new(
            OccupationFamily::B { layers: 3 },
            10,
            false,
            90,
            Some(15),
            Some(30),
            None,
            None,
            None,
            None,
        );
        assert_eq!(
            SamplingPlan::classify(&p),
            SamplingPlan::FactorizedMicrocanonical
        );
    }

    #[test]
    fn factorized_degrees() {
        let p = PreparedProblem::new(
            OccupationFamily::ME,
            5,
            false,
            20,
            None,
            Some(10),
            Some(vec![1, 2, 1, 0, 0]),
            Some(vec![0, 1, 2, 1, 0]),
            None,
            None,
        );
        assert_eq!(
            SamplingPlan::classify(&p),
            SamplingPlan::FactorizedMicrocanonical
        );
    }

    #[test]
    fn occupation_mcmc() {
        let p = PreparedProblem::new(
            OccupationFamily::ME,
            5,
            true,
            25,
            None,
            Some(20),
            None,
            None,
            Some(vec![5; 5]),
            Some(vec![5; 5]),
        );
        assert_eq!(SamplingPlan::classify(&p), SamplingPlan::OccupationMcmc);
    }

    #[test]
    fn occupation_mcmc_wins_over_degrees() {
        // §54/§80 release blocker: a problem with BOTH residual
        // strengths and residual degrees must route to occupation MCMC
        // (fixed-(s,k)), never to the factorized fixed-(k,T) branch,
        // which would silently ignore the strengths.
        let p = PreparedProblem::new(
            OccupationFamily::ME,
            5,
            false,
            20,
            None,
            None,
            Some(vec![2, 2, 1, 0, 0]),
            Some(vec![0, 1, 2, 1, 0]),
            Some(vec![3, 4, 1, 0, 0]),
            Some(vec![0, 1, 4, 3, 0]),
        );
        assert_eq!(SamplingPlan::classify(&p), SamplingPlan::OccupationMcmc);
        // The companion s+E case also belongs to the occupation branch.
        let q = PreparedProblem::new(
            OccupationFamily::ME,
            5,
            false,
            20,
            Some(6),
            Some(20),
            None,
            None,
            Some(vec![3, 4, 1, 0, 0]),
            Some(vec![0, 1, 4, 3, 0]),
        );
        assert_eq!(SamplingPlan::classify(&q), SamplingPlan::OccupationMcmc);
    }
}
