//! Generic orchestrator for all fixed-(E,T) microcanonical families.
//!
//! Defines [`FixedETOccupancy`] — the trait that ME, B, and W implement —
//! and the generic [`sample_fixed_et_core`] function that drives the
//! shared pipeline: validation → support → positive occupations → output.

use rand::rngs::StdRng;
use rand::Rng;

use super::super::super::output::SampledNetwork;
use super::errors::FixedETError;
use crate::generation::microcanonical::support::uniform_edges::sample_uniform_support;
use crate::OccNum;

// ---------------------------------------------------------------------------
// Constants (shared across all families)
// ---------------------------------------------------------------------------

/// Maximum rejection attempts before falling back to the exact DP backend.
pub const MAX_REJECTION_ATTEMPTS: usize = 200;

/// Estimated-rejection threshold: below this the fast rejection path is tried
/// first; above it the exact DP sampler is used directly.
pub const REJECTION_THRESHOLD: f64 = 0.95;

/// Maximum number of cells in a flat fallback DP table (~1.6 GB at 8 bytes).
/// On machines with >8 GB RAM this safely handles most practical networks.
pub const MAX_DP_CELLS: usize = 200_000_000;

/// Minimum estimated acceptance probability for the scaled rejection fallback.
/// Below this threshold the problem is effectively infeasible for rejection
/// (e.g. W when T ≈ M·E), so we fail immediately instead of burning attempts.
pub const MIN_ESTIMATED_P_ACC: f64 = 1e-12;

// ---------------------------------------------------------------------------
// The trait
// ---------------------------------------------------------------------------

/// Family-specific occupancy logic for fixed-(E,T) microcanonical sampling.
///
/// Each family (ME, B, W) implements the five methods below.  The shared
/// orchestrator handles validation dispatch, support sampling, special
/// cases, and output construction.
pub trait FixedETOccupancy {
    /// Human-readable family name for error messages.
    fn family_name(&self) -> &'static str;

    /// Upper bound per admissible pair (None = unbounded, as in ME and W).
    fn max_occnum(&self) -> Option<OccNum>;

    /// Family-specific residual-problem validation.
    ///
    /// Called after the basic `0 ≤ E ≤ L` and `(E=0) == (T=0)` checks pass
    /// and E > 0.  Should check bounds like `T ≥ E` (always) and
    /// `T ≤ M·E` (B only).
    fn validate_residual(&self, e: usize, t: OccNum) -> Result<(), FixedETError>;

    /// Fast-path rejection-probability estimate (0.0–1.0).
    fn estimate_rejection(&self, t: OccNum, e: usize) -> f64;

    /// Bounded-rejection fast proposal.
    ///
    /// Returns `Ok(counts)` on success or `Err(())` if retries exhausted.
    #[allow(clippy::result_unit_err)]
    fn try_rejection(
        &self,
        t: OccNum,
        e: usize,
        max_attempts: usize,
        rng: &mut StdRng,
    ) -> Result<Vec<OccNum>, ()>;

    /// Exact fallback — sequential DP sampling.
    fn sample_exact(
        &self,
        t: OccNum,
        e: usize,
        rng: &mut StdRng,
    ) -> Result<Vec<OccNum>, FixedETError>;

    /// Approximate work per rejection attempt (used to bound the
    /// scaled-attempts fallback when the DP table is unavailable).
    fn rejection_cost_per_attempt(&self, t: OccNum, e: usize) -> u64;
}

// ---------------------------------------------------------------------------
// Positive-occupation selector (shared special cases + backend routing)
// ---------------------------------------------------------------------------

/// Draw `E` positive integers summing to `T` with the family-specific law.
pub(crate) fn sample_positive_occupations<F: FixedETOccupancy>(
    family: &F,
    t: OccNum,
    e: usize,
    rng: &mut StdRng,
) -> Result<Vec<OccNum>, FixedETError> {
    // E == 1 handled at higher level
    debug_assert!(e >= 2);
    debug_assert!(t > e as OccNum);

    // B special case: T == M * E → all pairs at capacity
    if let Some(max) = family.max_occnum() {
        if t == max * e as OccNum {
            return Ok(vec![max; e]);
        }
    }

    // Always try the cheap exact rejection path first (≤20 attempts).
    // Rejection is exact on acceptance, and each attempt costs far less than
    // building the DP table.  The estimate-based routing is therefore only a
    // performance hint for which path is *likely* to win; it never changes
    // correctness.
    if let Ok(counts) = family.try_rejection(t, e, MAX_REJECTION_ATTEMPTS, rng) {
        return Ok(counts);
    }
    match family.sample_exact(t, e, rng) {
        Ok(counts) => Ok(counts),
        Err(FixedETError::TableTooLarge { .. }) => {
            // The exact DP table would be too large.  Fall back to the
            // rejection proposal with an attempt budget scaled by the
            // estimated acceptance probability, bounded by total work.
            let rej = family.estimate_rejection(t, e);
            scaled_rejection_fallback(family, t, e, rej, rng)
        }
        Err(e) => Err(e),
    }
}

/// Total work budget for the scaled-attempts rejection fallback.
const FALLBACK_WORK_BUDGET: u64 = 1_000_000_000;

/// Retry rejection with an attempt budget proportional to 1/p_acc, capped so
/// the total work stays bounded.  If acceptance is effectively impossible,
/// returns a clear error immediately instead of hanging.
fn scaled_rejection_fallback<F: FixedETOccupancy>(
    family: &F,
    t: OccNum,
    e: usize,
    rej: f64,
    rng: &mut StdRng,
) -> Result<Vec<OccNum>, FixedETError> {
    let p_acc = 1.0 - rej;
    if p_acc < MIN_ESTIMATED_P_ACC {
        return Err(FixedETError::NoBackendAvailable);
    }
    let p_acc = p_acc.clamp(MIN_ESTIMATED_P_ACC, 1.0);
    let cost = family.rejection_cost_per_attempt(t, e).max(1);
    let max_by_work = (FALLBACK_WORK_BUDGET / cost).max(MAX_REJECTION_ATTEMPTS as u64);
    let scaled = (5.0 / p_acc).ceil() as u64;
    let attempts = scaled
        .clamp(MAX_REJECTION_ATTEMPTS as u64, max_by_work)
        .min(1_000_000) as usize;
    family
        .try_rejection(t, e, attempts, rng)
        .map_err(|_| FixedETError::NoBackendAvailable)
}

// ---------------------------------------------------------------------------
// Core orchestrator
// ---------------------------------------------------------------------------

/// Run the full fixed-(E,T) sampling pipeline with a given family.
///
/// `get_pair(idx)` maps a linear index to a `(source, target)` pair.
pub fn sample_fixed_et_core<F, G>(
    family: &F,
    l: usize,
    e: usize,
    t: OccNum,
    rng: &mut StdRng,
    get_pair: G,
) -> Result<SampledNetwork, FixedETError>
where
    F: FixedETOccupancy,
    G: Fn(usize) -> (u64, u64),
{
    // ---- basic validation (shared across families) ----
    if e > l {
        return Err(FixedETError::InvalidResidual(format!(
            "residual_edges ({e}) exceeds admissible pair count ({l})"
        )));
    }
    if (e == 0) != (t == 0) {
        return Err(FixedETError::InvalidResidual(format!(
            "inconsistent (E,T) = ({e},{t}): both must be zero or both positive"
        )));
    }
    if e > 0 && t < e as OccNum {
        return Err(FixedETError::InvalidResidual(format!(
            "residual total {t} < residual edges {e} (each edge needs ≥1 event)"
        )));
    }
    if e > 0 {
        family.validate_residual(e, t)?;
    }

    // ---- special cases ----
    if e == 0 {
        return Ok(SampledNetwork::default());
    }
    if e == 1 {
        let idx = rng.random_range(0..l);
        let (i, j) = get_pair(idx);
        return Ok(SampledNetwork {
            sources: vec![i],
            targets: vec![j],
            occ_nums: vec![t],
        });
    }
    if t == e as OccNum {
        let indices = sample_uniform_support(l, e, rng);
        let mut sources = Vec::with_capacity(e);
        let mut targets = Vec::with_capacity(e);
        for &idx in &indices {
            let (i, j) = get_pair(idx);
            sources.push(i);
            targets.push(j);
        }
        return Ok(SampledNetwork {
            sources,
            targets,
            occ_nums: vec![1; e],
        });
    }

    // ---- general case ----
    let support = sample_uniform_support(l, e, rng);
    let occupations = sample_positive_occupations(family, t, e, rng)?;

    let mut sources = Vec::with_capacity(e);
    let mut targets = Vec::with_capacity(e);
    let mut occ_nums = Vec::with_capacity(e);
    for (&idx, &occ) in support.iter().zip(occupations.iter()) {
        debug_assert!(occ > 0, "occupation allocator returned a zero");
        let (i, j) = get_pair(idx);
        sources.push(i);
        targets.push(j);
        occ_nums.push(occ);
    }
    let result = SampledNetwork {
        sources,
        targets,
        occ_nums,
    };

    debug_assert_eq!(result.sources.len(), e);
    debug_assert_eq!(result.occ_nums.iter().copied().sum::<OccNum>(), t);

    Ok(result)
}
