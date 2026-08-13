//! Unified microcanonical routing: constraint structure → sampling plan → backend.
//!
//! [`sample_microcanonical`] is the single entry point for microcanonical
//! sampling.  It classifies a prepared problem via
//! [`SamplingPlan::classify`] and dispatches to the matching backend:
//!
//! | Plan | Backend |
//! |---|---|
//! | `GrandCanonical` | not routed here (use the `grandcanonical` module) |
//! | `FactorizedMicrocanonical` (degrees) | `sample_fixed_kt_core` |
//! | `FactorizedMicrocanonical` (edges) | `sample_{me,b,w}_fixed_et` |
//! | `OccupationMcmc` (strengths) | `sample_fixed_strength` |
//!
//! The router encodes the factorization structure, not constraint names:
//! degree- and edge-constrained problems share the same factorized plan
//! and differ only in the binary-support stage.

use super::super::output::SampledNetwork;
use super::binary::core::{sample_fixed_kt_core, FixedKTConfig};
use super::binary::sampler::FixedDegreeMcmcConfig;
use super::conditional::fixed_et::{sample_b_fixed_et, sample_me_fixed_et, sample_w_fixed_et};
use super::mcmc::McmcConfig;
use super::occupation_mcmc::domain::PairDomain;
use super::occupation_mcmc::problem::FixedStrengthProblem;
use super::occupation_mcmc::sample_fixed_strength;
use crate::model::family::OccupationFamily;
use crate::model::problem::PreparedProblem;
use crate::model::sampling_plan::SamplingPlan;
use crate::OccNum;

/// Configuration shared by all microcanonical sampling plans.
#[derive(Clone, Debug)]
pub struct MicrocanonicalConfig {
    /// RNG seed (deterministic reproducibility).
    pub seed: u64,
    /// MCMC burn-in sweeps (occupation chain and degree-support chain).
    pub burn_in_sweeps: usize,
    /// Thinning sweeps between samples.
    pub sweeps_per_sample: usize,
    /// Whether self-loops are admissible.
    pub self_loops: bool,
}

impl Default for MicrocanonicalConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            burn_in_sweeps: 50,
            sweeps_per_sample: 10,
            self_loops: false,
        }
    }
}

/// Errors from the microcanonical router.
#[derive(Clone, Debug)]
pub enum MicrocanonicalError {
    /// The plan is grand-canonical; the `grandcanonical` module handles it.
    GrandCanonicalNotRouted,
    /// The prepared problem is missing a residual constraint required by
    /// its sampling plan.
    MissingConstraint(&'static str),
    /// A backend failed.
    Backend(String),
}

impl std::fmt::Display for MicrocanonicalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GrandCanonicalNotRouted => {
                write!(
                    f,
                    "grand-canonical plans are handled by the grandcanonical module"
                )
            }
            Self::MissingConstraint(name) => {
                write!(
                    f,
                    "prepared problem is missing residual constraint `{name}`"
                )
            }
            Self::Backend(msg) => write!(f, "microcanonical backend failed: {msg}"),
        }
    }
}

impl std::error::Error for MicrocanonicalError {}

/// Sample a microcanonical network from a prepared problem.
///
/// Dispatches on [`SamplingPlan::classify`].  Fails with a structured
/// error for grand-canonical problems and for missing residual
/// constraints.
pub fn sample_microcanonical(
    problem: &PreparedProblem,
    config: &MicrocanonicalConfig,
) -> Result<SampledNetwork, MicrocanonicalError> {
    match SamplingPlan::classify(problem) {
        SamplingPlan::GrandCanonical => Err(MicrocanonicalError::GrandCanonicalNotRouted),
        SamplingPlan::FactorizedMicrocanonical => route_factorized(problem, config),
        SamplingPlan::OccupationMcmc => route_occupation_mcmc(problem, config),
    }
}

/// Factorized branch: binary support stage + fixed-total occupation stage.
fn route_factorized(
    problem: &PreparedProblem,
    config: &MicrocanonicalConfig,
) -> Result<SampledNetwork, MicrocanonicalError> {
    // Degree constraints take priority over the edge count (mirrors classify).
    if let (Some(out), Some(in_)) = (&problem.residual_out_degrees, &problem.residual_in_degrees) {
        let total = problem
            .residual_total
            .ok_or(MicrocanonicalError::MissingConstraint("residual_total"))?;
        let kt_config = FixedKTConfig {
            mcmc: FixedDegreeMcmcConfig {
                burn_in_sweeps: config.burn_in_sweeps,
                sweeps_per_sample: config.sweeps_per_sample,
                proposals_per_sweep: None,
                seed: config.seed,
                self_loops: config.self_loops,
            },
            self_loops: config.self_loops,
            admissible_pairs: None,
        };
        return sample_fixed_kt_core(problem.family, out, in_, total, &kt_config)
            .map_err(|e| MicrocanonicalError::Backend(e.to_string()));
    }

    let e = problem
        .residual_edges
        .ok_or(MicrocanonicalError::MissingConstraint("residual_edges"))?;
    let t = problem
        .residual_total
        .ok_or(MicrocanonicalError::MissingConstraint("residual_total"))?;
    let n = problem.node_count;
    let sl = config.self_loops;
    match problem.family {
        OccupationFamily::ME => sample_me_fixed_et(n, sl, e, t, config.seed),
        OccupationFamily::B { layers } => {
            sample_b_fixed_et(n, sl, layers as OccNum, e, t, config.seed)
        }
        OccupationFamily::W { layers } => {
            sample_w_fixed_et(n, sl, layers as OccNum, e, t, config.seed)
        }
    }
    .map_err(|e| MicrocanonicalError::Backend(e.to_string()))
}

/// Coupled branch: fixed-strength occupation MCMC, with ME stub-matching fast path.
fn route_occupation_mcmc(
    problem: &PreparedProblem,
    config: &MicrocanonicalConfig,
) -> Result<SampledNetwork, MicrocanonicalError> {
    let out =
        problem
            .residual_out_strengths
            .clone()
            .ok_or(MicrocanonicalError::MissingConstraint(
                "residual_out_strengths",
            ))?;
    let in_ =
        problem
            .residual_in_strengths
            .clone()
            .ok_or(MicrocanonicalError::MissingConstraint(
                "residual_in_strengths",
            ))?;
    let family = problem.family;

    // Build the complete pair domain.
    let domain = PairDomain::Complete {
        node_count: problem.node_count,
        self_loops: config.self_loops,
    };
    let full = FixedStrengthProblem::new(family, out, in_, domain, vec![])
        .map_err(|e| MicrocanonicalError::Backend(e.to_string()))?;
    let residual = full
        .into_residual()
        .map_err(|e| MicrocanonicalError::Backend(e.to_string()))?;
    let mcmc = McmcConfig {
        burn_in_sweeps: config.burn_in_sweeps,
        sweeps_per_sample: config.sweeps_per_sample,
        proposals_per_sweep: None,
        seed: config.seed,
    };
    let net = sample_fixed_strength(residual, mcmc)
        .map_err(|e| MicrocanonicalError::Backend(e.to_string()))?;
    Ok(net)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(seed: u64) -> MicrocanonicalConfig {
        MicrocanonicalConfig {
            seed,
            burn_in_sweeps: 10,
            sweeps_per_sample: 5,
            self_loops: false,
        }
    }

    #[test]
    fn routes_fixed_et_me() {
        let p = PreparedProblem::new(
            OccupationFamily::ME,
            8,
            false,
            56,
            Some(6),
            Some(12),
            None,
            None,
            None,
            None,
        );
        let net = sample_microcanonical(&p, &cfg(1)).unwrap();
        assert_eq!(net.sources.len(), 6);
        assert_eq!(net.occ_nums.iter().sum::<OccNum>(), 12);
    }

    #[test]
    fn routes_fixed_kt() {
        let out = vec![1u32, 1, 1, 1];
        let inp = vec![1u32, 1, 1, 1];
        let p = PreparedProblem::new(
            OccupationFamily::ME,
            4,
            false,
            12,
            None,
            Some(8),
            Some(out),
            Some(inp),
            None,
            None,
        );
        let net = sample_microcanonical(&p, &cfg(1)).unwrap();
        assert_eq!(net.sources.len(), 4);
        assert_eq!(net.occ_nums.iter().sum::<OccNum>(), 8);
    }

    #[test]
    fn routes_occupation_mcmc_strengths() {
        let p = PreparedProblem::new(
            OccupationFamily::ME,
            3,
            false,
            6,
            None,
            Some(6),
            None,
            None,
            Some(vec![2, 2, 2]),
            Some(vec![2, 2, 2]),
        );
        let net = sample_microcanonical(&p, &cfg(1)).unwrap();
        let mut out = vec![0u64; 3];
        let mut inp = vec![0u64; 3];
        for ((&s, &t), &o) in net
            .sources
            .iter()
            .zip(net.targets.iter())
            .zip(net.occ_nums.iter())
        {
            out[s as usize] += o;
            inp[t as usize] += o;
        }
        assert_eq!(out, vec![2, 2, 2]);
        assert_eq!(inp, vec![2, 2, 2]);
    }

    #[test]
    fn rejects_grand_canonical() {
        let p = PreparedProblem::new(
            OccupationFamily::ME,
            8,
            false,
            56,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(matches!(
            sample_microcanonical(&p, &cfg(1)),
            Err(MicrocanonicalError::GrandCanonicalNotRouted)
        ));
    }

    #[test]
    fn missing_constraint_error() {
        // Degree path declared but in-degrees missing → falls through to the
        // edge branch, which is also missing → structured error.
        let p = PreparedProblem::new(
            OccupationFamily::ME,
            4,
            false,
            12,
            None,
            Some(8),
            Some(vec![1, 1, 1, 1]),
            None,
            None,
            None,
        );
        assert!(matches!(
            sample_microcanonical(&p, &cfg(1)),
            Err(MicrocanonicalError::MissingConstraint(_))
        ));
    }

    #[test]
    fn reproducible() {
        let p = PreparedProblem::new(
            OccupationFamily::ME,
            8,
            false,
            56,
            Some(6),
            Some(12),
            None,
            None,
            None,
            None,
        );
        let a = sample_microcanonical(&p, &cfg(42)).unwrap();
        let b = sample_microcanonical(&p, &cfg(42)).unwrap();
        assert_eq!(a.sources, b.sources);
        assert_eq!(a.targets, b.targets);
        assert_eq!(a.occ_nums, b.occ_nums);
    }
}
