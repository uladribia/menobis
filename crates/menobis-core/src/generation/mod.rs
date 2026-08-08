//! Seeded network generation, organized by ensemble.
//!
//! - [`output`]: shared `SampledNetwork` result type.
//! - [`grandcanonical`]: independent pair sampling for fitted multipliers.
//! - [`canonical`]: fixed-total multinomial sampling.
//! - [`microcanonical`]: exact-constraint direct samplers (stub matching).
//!
//! This module re-exports the full public generation API so existing callers
//! keep working; the ensemble submodules own the implementations.

pub mod canonical;
pub mod grandcanonical;
pub mod microcanonical;
pub mod output;

pub use canonical::{sample_custom_multinomial, sample_strength_multinomial};
pub use grandcanonical::{
    sample_custom_poisson, sample_degree_events_binomial, sample_degree_events_geometric,
    sample_degree_events_negative_binomial, sample_degree_events_poisson, sample_edges_events,
    sample_strength_binomial, sample_strength_cost_binomial_coordinates,
    sample_strength_cost_geometric_coordinates, sample_strength_cost_negative_binomial_coordinates,
    sample_strength_cost_poisson_coordinates, sample_strength_degree_binomial,
    sample_strength_degree_geometric, sample_strength_degree_negative_binomial,
    sample_strength_degree_poisson, sample_strength_edges_binomial,
    sample_strength_edges_geometric, sample_strength_edges_negative_binomial,
    sample_strength_edges_poisson, sample_strength_geometric, sample_strength_negative_binomial,
    sample_strength_poisson,
};
pub use microcanonical::binary::core::{sample_fixed_kt_core, FixedKTConfig};
pub use microcanonical::binary::diagnostics::FixedDegreeDiagnostics;
pub use microcanonical::binary::sampler::FixedDegreeMcmcConfig;
pub use microcanonical::occupation_mcmc::cost_fit::{
    fit_gamma, FixedStrengthCostFitConfig, FixedStrengthCostFitResult,
};
pub use microcanonical::occupation_mcmc::sample_fixed_strength;
pub use microcanonical::occupation_mcmc::sample_fixed_strength_with_cost;
pub use microcanonical::route::{sample_microcanonical, MicrocanonicalConfig, MicrocanonicalError};
pub use microcanonical::{
    sample_b_fixed_et, sample_b_fixed_et_explicit, sample_me_fixed_et, sample_me_fixed_et_explicit,
    sample_strength_stub_matching, sample_w_fixed_et, sample_w_fixed_et_explicit,
};
pub use output::SampledNetwork;

#[cfg(test)]
mod tests {
    use super::{
        sample_strength_degree_poisson, sample_strength_multinomial, sample_strength_poisson,
        sample_strength_stub_matching,
    };

    #[test]
    fn poisson_is_reproducible() {
        let x = vec![3.0, 5.0];
        let y = vec![4.0, 6.0];
        let a = sample_strength_poisson(&x, &y, true, 42);
        let b = sample_strength_poisson(&x, &y, true, 42);
        assert_eq!(a.sources, b.sources);
        assert_eq!(a.targets, b.targets);
        assert_eq!(a.occ_nums, b.occ_nums);
    }

    #[test]
    fn multinomial_preserves_total() {
        let x = vec![3.0, 5.0, 2.0];
        let y = vec![4.0, 6.0, 1.0];
        let total = 1000;
        let edges = sample_strength_multinomial(&x, &y, total, true, 42);
        let sum: u64 = edges.occ_nums.iter().sum();
        assert_eq!(sum, total);
    }

    #[test]
    fn zip_is_reproducible() {
        let dx = vec![1.0, 2.0];
        let dy = vec![1.5, 0.5];
        let ex = vec![10.0, 20.0];
        let ey = vec![30.0, 40.0];
        let a = sample_strength_degree_poisson(&dx, &dy, &ex, &ey, true, 42);
        let b = sample_strength_degree_poisson(&dx, &dy, &ex, &ey, true, 42);
        assert_eq!(a.sources, b.sources);
        assert_eq!(a.targets, b.targets);
        assert_eq!(a.occ_nums, b.occ_nums);
        assert!(a.occ_nums.iter().all(|&w| w > 0));
    }

    #[test]
    fn stub_matching_preserves_exact_strengths() {
        let s_out = vec![10, 20, 30];
        let s_in = vec![15, 25, 20];
        let result = sample_strength_stub_matching(&s_out, &s_in, 42);
        assert!(result.is_ok(), "stub matching failed: {:?}", result);
        let edges = result.unwrap();
        let total: u64 = edges.occ_nums.iter().sum();
        assert_eq!(total, 60);
        let mut actual_out = vec![0u64; 3];
        let mut actual_in = vec![0u64; 3];
        for ((&src, &tgt), &w) in edges
            .sources
            .iter()
            .zip(edges.targets.iter())
            .zip(edges.occ_nums.iter())
        {
            actual_out[src as usize] += w;
            actual_in[tgt as usize] += w;
        }
        assert_eq!(actual_out, s_out);
        assert_eq!(actual_in, s_in);
    }

    #[test]
    fn no_self_loops() {
        let x = vec![10.0, 10.0, 10.0];
        let y = vec![10.0, 10.0, 10.0];
        let edges = sample_strength_poisson(&x, &y, false, 42);
        for (s, t) in edges.sources.iter().zip(edges.targets.iter()) {
            assert_ne!(s, t, "self-loop found: {s} -> {t}");
        }
    }

    #[test]
    fn degree_events_with_unit_positive_weight_mean_does_not_hang() {
        let x = vec![0.5; 100];
        let y = vec![0.5; 100];
        let edges = super::sample_degree_events_poisson(&x, &y, 0.0, true, 42);
        assert!(edges.occ_nums.iter().all(|&w| w >= 1));
    }
}
