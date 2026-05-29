//! Lagrange multiplier fitting for MENoBiS null models.

pub mod b;
pub mod b_lbfgs;
pub mod mask;
pub mod me;
pub mod me_lbfgs;
pub mod partial;
pub mod strength_cost;
pub(crate) mod support;
pub mod types;
pub mod w;
pub mod w_lbfgs;

#[cfg(test)]
use b::binary_probability;
pub use b::{
    balance_degree_bernoulli, balance_sparse_masked_degree_bernoulli,
    balance_sparse_masked_strength_binomial, balance_strength_binomial,
    fit_strength_degree_binomial, fit_strength_edges_binomial,
};
pub use b_lbfgs::{fit_strength_degree_binomial_lbfgs, fit_strength_edges_binomial_lbfgs};
pub use me::{
    balance_sparse_masked_strength_degree_poisson, balance_sparse_masked_strength_poisson,
    balance_strength_degree_poisson, balance_strength_edges_poisson, balance_strength_poisson,
    balance_weighted_factors, fit_strength_cost_poisson_coordinates, fit_strength_poisson,
    CostFitOptions,
};
pub use me_lbfgs::{fit_strength_degree_poisson_lbfgs, fit_strength_edges_poisson_lbfgs};
pub use partial::{
    fit_partial_degree, fit_partial_strength, fit_partial_strength_cost_binomial_coordinates,
    fit_partial_strength_cost_coordinates, fit_partial_strength_cost_w_coordinates,
    fit_partial_strength_degree, fit_partial_strength_edges,
};
pub use strength_cost::fit_strength_cost_binomial_coordinates;
pub use types::{
    FitResult, PartialFitResult, StrengthCostFitResult, StrengthDegreeFitResult,
    StrengthEdgesFitResult, WConicFitOptions, WFitStatus, WProblemMetrics, WStrengthCostFitResult,
    WStrengthDegreeFitResult, WStrengthEdgesFitResult, WStrengthFitResult, WStrengthResiduals,
};
pub use w::{
    fit_degree_events_geometric, fit_degree_events_negative_binomial,
    fit_strength_degree_geometric, fit_strength_degree_negative_binomial,
    fit_strength_edges_geometric, fit_strength_edges_negative_binomial, fit_strength_geometric,
    fit_strength_negative_binomial, independent_strength_residuals, strength_cost_residuals,
    strength_edges_residuals, DegreeEventsFitResult,
};
pub use w_lbfgs::fit_strength_cost_w_lbfgs;

#[cfg(test)]
mod tests {
    use super::{
        balance_degree_bernoulli, balance_strength_poisson, binary_probability,
        fit_degree_events_geometric, fit_degree_events_negative_binomial,
    };

    #[test]
    fn recovers_binary_degrees() {
        let k_out = vec![0.8, 1.2, 1.0];
        let k_in = vec![1.1, 0.9, 1.0];
        let result = balance_degree_bernoulli(&k_out, &k_in, true, 1e-12, 50000);

        assert!(result.converged);
        for (i, &expected) in k_out.iter().enumerate() {
            let row_sum: f64 = (0..3)
                .map(|j| binary_probability(result.x[i], result.y[j]))
                .sum();
            assert!((row_sum - expected).abs() < 1e-6);
        }
        for (j, &expected) in k_in.iter().enumerate() {
            let col_sum: f64 = (0..3)
                .map(|i| binary_probability(result.x[i], result.y[j]))
                .sum();
            assert!((col_sum - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn recovers_symmetric_strengths() {
        let s_out = vec![10.0, 20.0, 30.0];
        let s_in = vec![15.0, 25.0, 20.0];

        let result = balance_strength_poisson(&s_out, &s_in, 1e-10, 50000);

        assert!(result.converged);
        // Check: sum_{j!=i} x_i * y_j ≈ s_out_i
        for (i, &s_out_i) in s_out.iter().enumerate() {
            let row_sum: f64 = (0..3)
                .filter(|&j| j != i)
                .map(|j| result.x[i] * result.y[j])
                .sum();
            assert!(
                (row_sum - s_out_i).abs() < 1e-6,
                "s_out[{i}]: expected {s_out_i}, got {row_sum}",
            );
        }
        for (j, &s_in_j) in s_in.iter().enumerate() {
            let col_sum: f64 = (0..3)
                .filter(|&i| i != j)
                .map(|i| result.x[i] * result.y[j])
                .sum();
            assert!(
                (col_sum - s_in_j).abs() < 1e-6,
                "s_in[{j}]: expected {s_in_j}, got {col_sum}",
            );
        }
    }

    #[test]
    fn degree_events_geometric_recovers_q() {
        let k_out = vec![2.0, 1.0, 1.0];
        let k_in = vec![1.0, 2.0, 1.0];
        let result = fit_degree_events_geometric(&k_out, &k_in, 10, true, 1e-10, 50000);
        assert!(result.converged);
        // avg_weight = 10/4 = 2.5, q = 1 - 1/2.5 = 0.6
        assert!((result.q - 0.6).abs() < 1e-12);
        assert!((result.positive_mean - 2.5).abs() < 1e-12);
    }

    #[test]
    fn degree_events_negative_binomial_q_in_range() {
        let k_out = vec![2.0, 1.0, 1.0];
        let k_in = vec![1.0, 2.0, 1.0];
        let result = fit_degree_events_negative_binomial(&k_out, &k_in, 12, 3, true, 1e-10, 50000);
        assert!(result.converged);
        assert!(result.q > 0.0);
        assert!(result.q < 1.0);
        // Verify positive negative binomial mean matches
        let q = result.q;
        let m = 3.0;
        let p0 = (1.0 - q).powi(3);
        let ztnb_mean = m * q / ((1.0 - q) * (1.0 - p0));
        assert!((ztnb_mean - result.positive_mean).abs() < 1e-8);
    }

    #[test]
    fn degree_saturated_node_converges_bernoulli() {
        // Node 0 has degree = N-1 = 2 (saturated without self-loops)
        let k_out = vec![2.0, 0.8, 1.2];
        let k_in = vec![1.5, 1.3, 1.2];
        let result = balance_degree_bernoulli(&k_out, &k_in, false, 1e-8, 50000);
        assert!(result.converged, "degree-saturated Bernoulli must converge");
        // Saturated node should have large multiplier
        assert!(result.x[0] > 100.0);
    }

    #[test]
    fn b_strength_saturated_node_converges() {
        use super::b::balance_strength_binomial;
        // Node 0 has strength = M*(N-1) = 3*2 = 6 (saturated, M=3, no self-loops)
        let s_out = vec![6.0, 2.0, 1.0];
        let s_in = vec![4.0, 3.0, 2.0];
        let result = balance_strength_binomial(&s_out, &s_in, 3, false, 1e-6, 50000);
        assert!(result.converged, "B-strength-saturated must converge");
        assert!(result.x[0] > 100.0);
    }
}
