//! Gamma fitting for fixed-strength cost-constrained sampling.
//!
//! Fits the scalar cost multiplier \(\gamma\) so that the expected cost
//! under the constrained chain matches the observed cost:
//!
//! \[
//! \mu_C(\gamma) = C_{\mathrm{obs}}.
//! \]
//!
//! # Strategy
//!
//! 1. **Warm start**: run a short chain at \(\gamma = 0\), estimate
//!    \(\mu_C(0)\) and \(\operatorname{Var}_C(0)\), set
//!    \(\gamma_0 = (\mu_C(0) - C_{\mathrm{obs}}) / \operatorname{Var}_C(0)\).
//! 2. **Bracket expansion**: geometrically expand around \(\gamma_0\) until
//!    \(\mu_C(\gamma_{\mathrm{low}}) \ge C_{\mathrm{obs}} \ge
//!    \mu_C(\gamma_{\mathrm{high}})\).
//! 3. **Stochastic bisection**: at each midpoint, run the chain, estimate
//!    \(\mu_C\) via batch means, update the bracket, check convergence.

use rand::Rng;

use super::chain::FixedStrengthChain;
use super::cost::{residual_cost_target, state_cost};
use super::errors::FixedStrengthCostError;
use super::target::StrengthTarget;
use crate::distribution::OccupationFamily;
use crate::pairs::PairCostProvider;

// ---------------------------------------------------------------------------
// Configuration & result
// ---------------------------------------------------------------------------

/// Configuration for gamma fitting via stochastic bisection.
#[derive(Clone, Debug)]
pub struct FixedStrengthCostFitConfig {
    /// Number of sweeps to estimate µ_C(0) and Var_C(0) for the warm start.
    pub warm_start_sweeps: usize,
    /// Number of adaptation sweeps after each gamma change, before
    /// collecting estimation samples.
    pub adaptation_sweeps: usize,
    /// Number of sweeps to collect in each estimation batch.
    pub estimation_sweeps: usize,
    /// Number of cost measurements (one per sweep) to collect per iteration
    /// after adaptation.
    pub samples_per_iteration: usize,
    /// Maximum number of bisection iterations.
    pub max_iterations: usize,
    /// Absolute cost tolerance for convergence.
    pub absolute_cost_tolerance: f64,
    /// Relative cost tolerance for convergence.
    pub relative_cost_tolerance: f64,
    /// Multiplier for the standard error in the convergence check
    /// (e.g., 2.09 ≈ t_{0.975, 19} for 20 batches).
    pub confidence_multiplier: f64,
    /// Factor by which to expand the bracket geometrically.
    pub bracket_expansion_factor: f64,
    /// Maximum number of bracket expansions.
    pub max_bracket_expansions: usize,
    /// Number of batches for batch-means standard error estimation.
    pub batch_count: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for FixedStrengthCostFitConfig {
    fn default() -> Self {
        Self {
            warm_start_sweeps: 50,
            adaptation_sweeps: 100,
            estimation_sweeps: 80,
            samples_per_iteration: 20,
            max_iterations: 40,
            absolute_cost_tolerance: 1e-1,
            relative_cost_tolerance: 1e-1,
            confidence_multiplier: 2.09, // t_{0.975, 19}
            bracket_expansion_factor: 2.0,
            max_bracket_expansions: 20,
            batch_count: 10,
            seed: 42,
        }
    }
}

/// Result of the gamma fitting procedure.
#[derive(Clone, Debug)]
pub struct FixedStrengthCostFitResult {
    /// Fitted gamma.
    pub gamma: f64,
    /// Estimated expected cost at the fitted gamma.
    pub expected_cost_estimate: f64,
    /// Monte Carlo standard error of the expected-cost estimate.
    pub expected_cost_standard_error: f64,
    /// Observed (target) cost used for fitting.
    pub observed_cost: f64,
    /// Fixed-pair cost that was subtracted.
    pub fixed_cost: f64,
    /// Residual cost target for the MCMC.
    pub residual_cost_target: f64,
    /// Residual after convergence: |µ_C − C_obs|.
    pub residual: f64,
    /// Number of bisection iterations used.
    pub iterations: usize,
    /// Whether the fit converged.
    pub converged: bool,
    /// Lower bracket bound at termination.
    pub bracket_lower: f64,
    /// Upper bracket bound at termination.
    pub bracket_upper: f64,
    /// Total MCMC proposals made.
    pub mcmc_proposals: u64,
    /// Total MCMC acceptances.
    pub mcmc_accepted: u64,
    /// Family used.
    pub family: OccupationFamily,
    /// Number of cost samples collected in the final measurement.
    pub sample_count: usize,
    /// Seed used.
    pub seed: u64,
}

// ---------------------------------------------------------------------------
// Batch-means standard error
// ---------------------------------------------------------------------------

/// Estimate the standard error of a sequence of correlated MCMC samples
/// using the batch-means method.
///
/// Splits `samples` into `n_batches` contiguous batches, computes the mean
/// of each batch, and returns `std(batch_means) / sqrt(n_batches)`.
///
/// Falls back to IID standard error if `samples.len() < 2 * n_batches`.
pub fn batch_means_se(samples: &[f64], n_batches: usize) -> f64 {
    let n = samples.len();
    if n < 2 {
        return 0.0;
    }

    let effective_batches = n_batches.min(n / 2);
    if effective_batches < 2 {
        // Fall back to IID standard error.
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        return (var / n as f64).sqrt();
    }

    let batch_size = n / effective_batches;
    let mut batch_means = Vec::with_capacity(effective_batches);
    for b in 0..effective_batches {
        let start = b * batch_size;
        let end = if b == effective_batches - 1 {
            n
        } else {
            start + batch_size
        };
        let sum: f64 = samples[start..end].iter().sum();
        batch_means.push(sum / (end - start) as f64);
    }

    let grand_mean = batch_means.iter().sum::<f64>() / effective_batches as f64;
    let variance = batch_means
        .iter()
        .map(|&m| (m - grand_mean).powi(2))
        .sum::<f64>()
        / (effective_batches - 1) as f64;

    (variance / effective_batches as f64).sqrt()
}

// ---------------------------------------------------------------------------
// Warm-start gamma via moment matching
// ---------------------------------------------------------------------------

/// Compute a warm-start \(\gamma\) by moment matching at \(\gamma = 0\).
///
/// 1. Run `sweeps` of the chain at \(\gamma = 0\), collecting cost per sweep.
/// 2. Compute sample mean \(\mu_0\) and variance \(\sigma_0^2\).
/// 3. Return \(\gamma_0 = (\mu_0 - C_{\mathrm{obs}}) / \sigma_0^2\).
///
/// If \(\sigma_0^2\) is effectively zero, returns
/// [`CostNotIdentifiable`](FixedStrengthCostError::CostNotIdentifiable).
pub fn warm_start_gamma<'a>(
    chain: &mut FixedStrengthChain<'a>,
    rng: &mut impl Rng,
    costs: &'a dyn PairCostProvider,
    observed_cost: f64,
    sweeps: usize,
) -> Result<f64, FixedStrengthCostError> {
    // Ensure gamma is 0 for warm-start measurement.
    {
        let mut target = StrengthTarget::new(chain.target.family);
        target.costs = chain.target.costs;
        chain.set_target(target);
    }

    // Collect cost samples at gamma = 0.
    // First run a short burn-in at gamma=0 (adaptation_sweeps/2).
    for _ in 0..sweeps / 2 {
        chain.sweep(rng);
    }

    let mut cost_samples = Vec::with_capacity(sweeps);
    for _ in 0..sweeps {
        chain.sweep(rng);
        let total = state_cost(&chain.state, costs)?;
        cost_samples.push(total);
    }

    let n = cost_samples.len() as f64;
    let mu_0 = cost_samples.iter().sum::<f64>() / n;
    let var_0 = if n > 1.0 {
        cost_samples
            .iter()
            .map(|&c| (c - mu_0).powi(2))
            .sum::<f64>()
            / (n - 1.0)
    } else {
        0.0
    };

    // Relative threshold: if std / |mu_0| < 1e-10 (or mu_0 is zero), cost is constant.
    let std_0 = var_0.sqrt();
    let relative_spread = if mu_0.abs() > 1e-12 {
        std_0 / mu_0.abs()
    } else {
        std_0
    };

    if relative_spread < 1e-10 {
        return Err(FixedStrengthCostError::CostNotIdentifiable);
    }

    let gamma_0 = (mu_0 - observed_cost) / var_0;

    // Clamp to finite range.
    Ok(if gamma_0.is_finite() { gamma_0 } else { 0.0 })
}

// ---------------------------------------------------------------------------
// Bracket expansion
// ---------------------------------------------------------------------------

/// Expand a bracket geometrically around `gamma_start` until the expected
/// cost at the bracket endpoints brackets `observed_cost`.
///
/// Returns `(gamma_low, gamma_high)`.
pub fn expand_bracket<'a>(
    chain: &mut FixedStrengthChain<'a>,
    rng: &mut impl Rng,
    costs: &'a dyn PairCostProvider,
    observed_cost: f64,
    gamma_start: f64,
    config: &FixedStrengthCostFitConfig,
) -> Result<(f64, f64), FixedStrengthCostError> {
    let factor = config.bracket_expansion_factor;
    let max_expansions = config.max_bracket_expansions;

    // Measure expected cost at gamma_start.
    let mu_start = estimate_mu_at_gamma(chain, rng, costs, gamma_start, config)?;

    // Determine search direction.
    // Since expected cost decreases with gamma:
    //   mu > observed → gamma too small → need larger gamma
    //   mu < observed → gamma too large → need smaller gamma
    let (mut low, mut high) = if mu_start > observed_cost {
        // Need to go up (positive direction from gamma_start).
        (gamma_start, gamma_start * (1.0 + factor))
    } else {
        // Need to go down (negative direction from gamma_start).
        (gamma_start * (1.0 - factor), gamma_start)
    };

    // Ensure low < high.
    if low > high {
        std::mem::swap(&mut low, &mut high);
    }

    for _ in 0..max_expansions {
        let mu_low = estimate_mu_at_gamma(chain, rng, costs, low, config)?;
        if mu_low >= observed_cost - config.absolute_cost_tolerance {
            // Low side is above observed cost (within tolerance), good.
            let mu_high = estimate_mu_at_gamma(chain, rng, costs, high, config)?;
            if mu_high <= observed_cost + config.absolute_cost_tolerance {
                // Both sides bracket.
                return Ok((low, high));
            }
            // High side is too high, expand upward.
            let expansion = if high >= 0.0 {
                high * (1.0 + factor)
            } else {
                high * (1.0 - factor)
            };
            high = expansion;
        } else {
            // Low side is too low, expand downward.
            let expansion = if low >= 0.0 {
                low * (1.0 - factor)
            } else {
                low * (1.0 + factor)
            };
            low = expansion.min(high);
        }
    }

    Err(FixedStrengthCostError::BracketNotFound)
}

/// Estimate expected cost at a specific gamma value.
///
/// Sets gamma on the chain, runs adaptation, collects samples, returns
/// batch-means estimate of expected cost.
fn estimate_mu_at_gamma<'a>(
    chain: &mut FixedStrengthChain<'a>,
    rng: &mut impl Rng,
    costs: &'a dyn PairCostProvider,
    gamma: f64,
    config: &FixedStrengthCostFitConfig,
) -> Result<f64, FixedStrengthCostError> {
    // Set gamma.
    let mut target = StrengthTarget::with_costs(chain.target.family, costs);
    target.set_gamma(gamma);
    chain.set_target(target);

    // Adaptation.
    for _ in 0..config.adaptation_sweeps.max(1) {
        chain.sweep(rng);
    }

    // Collect samples (use estimation_sweeps for more stable estimates
    // during bracket expansion and bisection).
    let n_samples = config.estimation_sweeps.max(config.samples_per_iteration);
    let mut samples = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        chain.sweep(rng);
        let total = state_cost(&chain.state, costs)?;
        samples.push(total);
    }

    // Batch-means estimate.
    let mu = samples.iter().sum::<f64>() / samples.len() as f64;
    Ok(mu)
}

// ---------------------------------------------------------------------------
// Main fitting routine
// ---------------------------------------------------------------------------

/// Fit \(\gamma\) for a fixed-strength cost-constrained problem.
///
/// # Arguments
///
/// * `chain` — A pre-built and initialized [`FixedStrengthChain`] with a
///   cost-aware target (gamma=0).  The chain's state is modified in place.
/// * `costs` — The [`PairCostProvider`] used by the target.
/// * `observed_total_cost` — The total observed cost \(C_{\mathrm{obs}}\).
/// * `fixed_cost` — Cost contributed by fixed pairs (0 if none).
/// * `config` — Fitting configuration.
///
/// # Returns
///
/// A [`FixedStrengthCostFitResult`] with the fitted gamma and diagnostics.
///
/// # Errors
///
/// Returns [`FixedStrengthCostError`] if cost is not identifiable, a
/// bracket cannot be found, or the fit does not converge.
pub fn fit_gamma<'a>(
    chain: &mut FixedStrengthChain<'a>,
    rng: &mut impl Rng,
    costs: &'a dyn PairCostProvider,
    observed_total_cost: f64,
    fixed_cost: f64,
    config: &FixedStrengthCostFitConfig,
) -> Result<FixedStrengthCostFitResult, FixedStrengthCostError> {
    // Compute residual cost target.
    let residual_obs = residual_cost_target(observed_total_cost, fixed_cost)?;

    // --- Warm start ---
    let mut gamma_0 = warm_start_gamma(chain, rng, costs, residual_obs, config.warm_start_sweeps)?;

    // Clamp to avoid near-zero gamma (which would make bracket expansion
    // too slow).  If gamma_0 is tiny, pick a sensible default direction.
    if gamma_0.abs() < 1e-6 {
        gamma_0 = if gamma_0 >= 0.0 { 0.1 } else { -0.1 };
    }

    // --- Bracket expansion ---
    let (gamma_low, gamma_high) = expand_bracket(chain, rng, costs, residual_obs, gamma_0, config)?;

    // --- Stochastic bisection ---
    let mut low = gamma_low;
    let mut high = gamma_high;
    let mut best_gamma = gamma_0;
    let mut best_residual = f64::MAX;
    let mut best_mu = 0.0;
    let mut best_se = 0.0;
    let mut best_samples = Vec::new();
    let mut converged = false;

    let mut iteration_count = 0;
    for _iteration in 0..config.max_iterations {
        iteration_count += 1;
        let gamma_mid = (low + high) / 2.0;

        // Set gamma and adapt.
        let mut target = StrengthTarget::with_costs(chain.target.family, costs);
        target.set_gamma(gamma_mid);
        chain.set_target(target);

        for _ in 0..config.adaptation_sweeps.max(1) {
            chain.sweep(rng);
        }

        // Collect samples (use estimation_sweeps for stability).
        let n_samples = config.estimation_sweeps.max(config.samples_per_iteration);
        let mut samples = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            chain.sweep(rng);
            let total = state_cost(&chain.state, costs)?;
            samples.push(total);
        }

        let n = samples.len() as f64;
        let mu = samples.iter().sum::<f64>() / n;
        let se = batch_means_se(&samples, config.batch_count);
        let residual = (mu - residual_obs).abs();

        // Track best.
        if residual < best_residual {
            best_gamma = gamma_mid;
            best_residual = residual;
            best_mu = mu;
            best_se = se;
            best_samples = samples;
        }

        // Check convergence: residual within tolerance AND SE sufficiently small.
        let tol = (config.absolute_cost_tolerance)
            .max(config.relative_cost_tolerance * residual_obs.abs());
        if residual <= tol && config.confidence_multiplier * se <= tol {
            converged = true;
            break;
        }

        // Update bracket: expected cost decreases with gamma.
        // If mu > residual_obs, gamma_mid is too low → move low up.
        // If mu < residual_obs, gamma_mid is too high → move high down.
        if mu > residual_obs {
            low = gamma_mid;
        } else {
            high = gamma_mid;
        }

        // If bracket collapsed, use the best found so far.
        if (high - low).abs() < 1e-15 {
            converged = true;
            break;
        }
    }

    let result = FixedStrengthCostFitResult {
        gamma: best_gamma,
        expected_cost_estimate: best_mu,
        expected_cost_standard_error: best_se,
        observed_cost: observed_total_cost,
        fixed_cost,
        residual_cost_target: residual_obs,
        residual: best_residual,
        iterations: iteration_count,
        converged,
        bracket_lower: gamma_low,
        bracket_upper: gamma_high,
        mcmc_proposals: chain.counters.proposals,
        mcmc_accepted: chain.counters.accepted,
        family: chain.target.family,
        sample_count: best_samples.len(),
        seed: config.seed,
    };

    if !converged {
        // Return the best result even if not converged, but with an error.
        // The caller can decide whether to accept it.
        return Err(FixedStrengthCostError::FitDidNotConverge {
            iterations: config.max_iterations,
            residual: best_residual,
        });
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::OccupationFamily;
    use crate::generation::microcanonical::fixed_strength::domain::PairDomain;
    use crate::generation::microcanonical::fixed_strength::initializer::initialize_table;
    use crate::generation::microcanonical::fixed_strength::problem::FixedStrengthProblem;
    use crate::generation::microcanonical::fixed_strength::state::StrengthState;
    use crate::generation::microcanonical::mcmc::McmcConfig;
    use crate::pairs::PairCostProvider;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    // -----------------------------------------------------------------------
    // Batch-means unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn batch_means_iid_fallback() {
        let samples = vec![1.0, 2.0, 3.0];
        let se = batch_means_se(&samples, 20);
        assert!(se.is_finite());
        assert!(se >= 0.0);
    }

    #[test]
    fn batch_means_constant_samples() {
        let samples = vec![5.0; 100];
        let se = batch_means_se(&samples, 20);
        assert!((se - 0.0).abs() < 1e-12);
    }

    #[test]
    fn batch_means_increasing() {
        let samples: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let se = batch_means_se(&samples, 20);
        assert!(se.is_finite());
        assert!(se > 0.0);
    }

    #[test]
    fn cost_fit_config_defaults() {
        let cfg = FixedStrengthCostFitConfig::default();
        assert!(cfg.warm_start_sweeps > 0);
        assert!(cfg.batch_count == 10);
        assert!((cfg.confidence_multiplier - 2.09).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Cost provider helpers
    // -----------------------------------------------------------------------

    /// Cost = |i - j|.
    struct LinearCost;
    impl PairCostProvider for LinearCost {
        fn cost(&self, source: usize, target: usize) -> Option<f64> {
            Some((source as i64 - target as i64).unsigned_abs() as f64)
        }
    }

    /// Cost = 1 for every pair (constant non-zero).
    struct ConstantCost;
    impl PairCostProvider for ConstantCost {
        fn cost(&self, _source: usize, _target: usize) -> Option<f64> {
            Some(1.0)
        }
    }

    // -----------------------------------------------------------------------
    // Exact enumeration helpers (ME, N=2)
    // -----------------------------------------------------------------------

    type OccupiedState = Vec<((u64, u64), crate::OccNum)>;

    /// Enumerate all ME occupation states for given strengths.
    /// Returns (state, log_degeneracy).
    fn enumerate_me_states(
        s_out: &[crate::OccNum],
        s_in: &[crate::OccNum],
        self_loops: bool,
    ) -> Vec<(OccupiedState, f64)> {
        let n = s_out.len();
        let mut results = Vec::new();

        let cells: Vec<(u64, u64)> = (0..n as u64)
            .flat_map(|i| (0..n as u64).map(move |j| (i, j)))
            .filter(|&(i, j)| self_loops || i != j)
            .collect();

        fn recurse(
            idx: usize,
            cells: &[(u64, u64)],
            remaining_out: &mut [crate::OccNum],
            remaining_in: &mut [crate::OccNum],
            current: &mut OccupiedState,
            results: &mut Vec<(OccupiedState, f64)>,
        ) {
            if idx == cells.len() {
                if remaining_out.iter().all(|&s| s == 0) && remaining_in.iter().all(|&s| s == 0) {
                    let log_degen: f64 = current
                        .iter()
                        .map(|&(_, occ)| -libm::lgamma((occ as f64) + 1.0))
                        .sum();
                    results.push((current.clone(), log_degen));
                }
                return;
            }

            let (src, tgt) = cells[idx];
            let max_possible = remaining_out[src as usize].min(remaining_in[tgt as usize]);

            for occ in 0..=max_possible {
                remaining_out[src as usize] -= occ;
                remaining_in[tgt as usize] -= occ;
                if occ > 0 {
                    current.push(((src, tgt), occ));
                }
                recurse(
                    idx + 1,
                    cells,
                    remaining_out,
                    remaining_in,
                    current,
                    results,
                );
                if occ > 0 {
                    current.pop();
                }
                remaining_out[src as usize] += occ;
                remaining_in[tgt as usize] += occ;
            }
        }

        let mut remaining_out = s_out.to_vec();
        let mut remaining_in = s_in.to_vec();
        let mut current: OccupiedState = Vec::new();
        recurse(
            0,
            &cells,
            &mut remaining_out,
            &mut remaining_in,
            &mut current,
            &mut results,
        );
        results
    }

    /// Compute total cost of an enumerated state.
    fn state_cost_from_pairs(pairs: &OccupiedState, costs: &dyn PairCostProvider) -> f64 {
        let mut total = 0.0;
        for &((s, t), occ) in pairs {
            let c = costs.cost(s as usize, t as usize).unwrap_or(0.0);
            total += c * (occ as f64);
        }
        total
    }

    // -----------------------------------------------------------------------
    // Test 1: Constant cost leaves probabilities unchanged
    // -----------------------------------------------------------------------

    #[test]
    fn constant_cost_does_not_affect_distribution() {
        // When all pair costs are equal, ΔC = 0 for every 4-cycle,
        // so the cost term vanishes regardless of gamma.
        // The MCMC distribution should match Phase 4.

        let s_out = vec![2u64, 2];
        let s_in = vec![2u64, 2];
        let n = 2;

        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };

        let config = McmcConfig::new(10, 5, 42);

        // Phase 4: no cost.
        let target_no_cost = StrengthTarget::new(OccupationFamily::Poisson);
        let mut chain_no_cost = super::super::chain::FixedStrengthChain::new(
            StrengthState::new(
                n,
                initialize_table(&s_out, &s_in, OccupationFamily::Poisson, &domain).unwrap(),
            ),
            target_no_cost,
            domain.clone(),
            config.clone(),
        );

        // Phase 5: constant cost with gamma = 10.0 (large).
        let costs = ConstantCost;
        let mut target_const = StrengthTarget::with_costs(OccupationFamily::Poisson, &costs);
        target_const.set_gamma(10.0);
        let mut chain_const = super::super::chain::FixedStrengthChain::new(
            StrengthState::new(
                n,
                initialize_table(&s_out, &s_in, OccupationFamily::Poisson, &domain).unwrap(),
            ),
            target_const,
            domain,
            config,
        );

        // Quick check: a single 4-cycle step should give same acceptance.
        // For constant cost, delta_C = 1+1-1-1 = 0, so cost term always 0.
        // The degeneracy deltas are identical.
        let mut rng_no = StdRng::seed_from_u64(99);
        let mut rng_const = StdRng::seed_from_u64(99);

        for _ in 0..50 {
            let outcome_no = chain_no_cost.step(&mut rng_no);
            let outcome_const = chain_const.step(&mut rng_const);
            // Both chains should make identical decisions (same RNG state).
            assert_eq!(
                outcome_no, outcome_const,
                "constant cost chain diverges from no-cost chain"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: Zero-gamma cost sampling = Phase 4
    // -----------------------------------------------------------------------

    #[test]
    fn zero_gamma_cost_equals_no_cost() {
        // When gamma = 0, the cost term is disabled even with a provider.
        let s_out = vec![2u64, 2];
        let s_in = vec![2u64, 2];
        let n = 2;

        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };

        let config = McmcConfig::new(10, 5, 42);

        // With cost provider but gamma = 0.
        let costs = LinearCost;
        let target = StrengthTarget::with_costs(OccupationFamily::Poisson, &costs);
        let chain = super::super::chain::FixedStrengthChain::new(
            StrengthState::new(
                n,
                initialize_table(&s_out, &s_in, OccupationFamily::Poisson, &domain).unwrap(),
            ),
            target,
            domain,
            config,
        );

        // The target's delta_log_weight at gamma=0 should equal the
        // no-cost target's delta_log_weight.
        let no_cost = StrengthTarget::new(OccupationFamily::Poisson);
        let d1 = no_cost.delta_log_weight(0, 1, 2, 3).unwrap();
        let d2 = chain.target.delta_log_weight(0, 1, 2, 3).unwrap();
        assert!(
            (d1 - d2).abs() < 1e-12,
            "gamma=0 delta ({d2}) != no-cost delta ({d1})"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: Exact enumeration with cost — MCMC vs exact Boltzmann
    // -----------------------------------------------------------------------

    #[test]
    fn me_cost_enumeration_agreement_n2() {
        // For ME N=2, s_out=[2,2], s_in=[2,2], self-loops allowed,
        // costs = |i-j|, gamma = 0.5, enumerate all 3 states and check
        // that MCMC frequencies match Boltzmann weights.

        let s_out = vec![2u64, 2];
        let s_in = vec![2u64, 2];

        // Enumerate all states with their (log_degen, cost).
        let states = enumerate_me_states(&s_out, &s_in, true);
        let costs = LinearCost;
        let gamma = 0.5;

        // Compute exact probabilities.
        let log_weights: Vec<f64> = states
            .iter()
            .map(|(pairs, log_d)| {
                let c = state_cost_from_pairs(pairs, &costs);
                log_d - gamma * c
            })
            .collect();

        // Stabilise with log-sum-exp.
        let max_log = log_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let total_weight: f64 = log_weights.iter().map(|w| (w - max_log).exp()).sum();
        let exact_probs: Vec<f64> = log_weights
            .iter()
            .map(|w| (w - max_log).exp() / total_weight)
            .collect();

        // Run MCMC many times and count state frequencies.
        let trials = 3000;
        let mut counts = std::collections::HashMap::<OccupiedState, u64>::new();

        for seed in 0..trials {
            let problem = FixedStrengthProblem::new(
                OccupationFamily::Poisson,
                s_out.clone(),
                s_in.clone(),
                PairDomain::Complete {
                    node_count: 2,
                    self_loops: true,
                },
                vec![],
            )
            .unwrap()
            .into_residual()
            .unwrap();

            let table = initialize_table(
                &problem.strength_out,
                &problem.strength_in,
                problem.family,
                &problem.domain,
            )
            .unwrap();
            let state = StrengthState::new(2, table);

            let mut tgt = StrengthTarget::with_costs(OccupationFamily::Poisson, &costs);
            tgt.set_gamma(gamma);

            let mcmc_config = McmcConfig {
                burn_in_sweeps: 10,
                sweeps_per_sample: 5,
                proposals_per_sweep: None,
                seed,
            };

            let mut chain = super::super::chain::FixedStrengthChain::new(
                state,
                tgt,
                PairDomain::Complete {
                    node_count: 2,
                    self_loops: true,
                },
                mcmc_config,
            );

            let mut rng = StdRng::seed_from_u64(seed);
            chain.burn_in(&mut rng);
            let net = chain.sample(&mut rng);

            let mut pairs: OccupiedState = net
                .sources
                .iter()
                .zip(net.targets.iter())
                .zip(net.occ_nums.iter())
                .map(|((&s, &t), &o)| ((s, t), o))
                .collect();
            pairs.sort_unstable();
            *counts.entry(pairs).or_default() += 1;
        }

        // Check each state appears with roughly the right frequency.
        for ((pairs, _), prob) in states.iter().zip(exact_probs.iter()) {
            let expected_count = prob * trials as f64;
            let observed = counts.get(pairs).copied().unwrap_or(0) as f64;
            let ratio = if expected_count > 0.0 {
                observed / expected_count
            } else {
                1.0
            };
            assert!(
                ratio > 0.3 && ratio < 2.5,
                "state {:?}: expected {expected_count:.1}, observed {observed}, ratio {ratio:.2}",
                pairs,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 4: Warm-start sign detection
    // -----------------------------------------------------------------------

    #[test]
    fn warm_start_positive_gamma_when_cost_above_uniform() {
        // If observed cost is lower than the uniform (gamma=0) expected cost,
        // warm start should produce gamma > 0.
        let s_out = vec![3u64, 3];
        let s_in = vec![3u64, 3];
        let n = 2;
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };

        let config = McmcConfig::new(5, 2, 42);
        let costs = LinearCost;

        let table = initialize_table(&s_out, &s_in, OccupationFamily::Poisson, &domain).unwrap();
        let mut chain = super::super::chain::FixedStrengthChain::new(
            StrengthState::new(n, table),
            StrengthTarget::with_costs(OccupationFamily::Poisson, &costs),
            domain,
            config,
        );

        let mut rng = StdRng::seed_from_u64(42);

        // Use a very low observed cost (lower than uniform expected cost),
        // so gamma should be positive.
        let low_observed = 0.1;
        let gamma_0 = warm_start_gamma(&mut chain, &mut rng, &costs, low_observed, 10).unwrap();
        assert!(
            gamma_0 > 0.0,
            "expected positive gamma for low observed cost, got {gamma_0}"
        );
    }

    #[test]
    fn warm_start_negative_gamma_when_cost_below_uniform() {
        // If observed cost is higher than the uniform expected cost,
        // warm start should produce gamma < 0.
        let s_out = vec![3u64, 3];
        let s_in = vec![3u64, 3];
        let n = 2;
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: true,
        };

        let config = McmcConfig::new(5, 2, 42);
        let costs = LinearCost;

        let table = initialize_table(&s_out, &s_in, OccupationFamily::Poisson, &domain).unwrap();
        let mut chain = super::super::chain::FixedStrengthChain::new(
            StrengthState::new(n, table),
            StrengthTarget::with_costs(OccupationFamily::Poisson, &costs),
            domain,
            config,
        );

        let mut rng = StdRng::seed_from_u64(42);

        // Use a very high observed cost, so gamma should be negative.
        let high_observed = 100.0;
        let gamma_0 = warm_start_gamma(&mut chain, &mut rng, &costs, high_observed, 10).unwrap();
        assert!(
            gamma_0 < 0.0,
            "expected negative gamma for high observed cost, got {gamma_0}"
        );
    }
}
