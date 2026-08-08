//! Legacy uniform-coordinate 4-cycle MCMC kernel (preserved for oracle
//! validation).
//!
//! # Purpose
//!
//! This is the **old** fixed-strength MCMC kernel before the Phase B
//! occupied-cell refactor.  It selects four-cycle cells uniformly from
//! the full coordinate domain \( \{0,\dots,N-1\}^4 \) with \( a \ne c \)
//! and \( b \ne d \), chooses a sign \( s \in \{+1,-1\} \) uniformly,
//! and relies on the proposal being symmetric (no Hastings correction).
//!
//! # Preserved for (§28)
//!
//! This copy is kept in the heavy Rust oracle suite for validation of
//! the new occupied-cell kernel.  It must **never** be used in production
//! code paths after Phase B.
//!
//! # Spec reference
//!
//! - Original proposal: uniform-coordinate per §22 (old behaviour).
//! - No Hastings because coordinate proposal is symmetric.
//! - Oracle migration rule: §28, §5.

use rand::Rng;

use menobis_core::generation::microcanonical::mcmc::McmcOutcome;
use menobis_core::generation::microcanonical::occupation_mcmc::domain::PairDomain;
use menobis_core::generation::microcanonical::occupation_mcmc::state::StrengthState;
use menobis_core::generation::microcanonical::occupation_mcmc::target::StrengthTarget;
use menobis_core::OccNum;

/// Four (cell, delta) pairs of one cycle proposal.
type Deltas = [(u64, u64, i64); 4];

/// Build a 4-cycle delta set from two sources, two targets, and a sign.
fn build_cycle4(a: u64, c: u64, b: u64, d: u64, sign: i64) -> Deltas {
    [(a, b, sign), (c, d, sign), (a, d, -sign), (c, b, -sign)]
}

/// Perform one 4-cycle MCMC step using the **legacy** uniform-coordinate
/// proposal.
///
/// This is identical to the pre-Phase-B `cycle4_step`.  It selects
/// \(a,c,b,d\) uniformly from node indices (with \(a \ne c, b \ne d\)),
/// chooses a sign \(\pm1\) uniformly, and accepts with probability
/// \(\min(1, e^{\Delta\log\pi})\) because the proposal is symmetric.
///
/// # Oracle use only
///
/// This function is **not** part of the production generation path.
/// Use it exclusively for validation against the new occupied-cell kernel.
pub fn legacy_cycle4_step(
    state: &mut StrengthState,
    target: &StrengthTarget,
    domain: &PairDomain,
    rng: &mut impl Rng,
) -> McmcOutcome {
    let n = state.node_count;
    if n < 2 {
        return McmcOutcome::Held;
    }

    // Choose two distinct source nodes.
    let a = rng.random_range(0..n) as u64;
    let mut c = rng.random_range(0..n - 1) as u64;
    if c >= a {
        c += 1;
    }

    // Choose two distinct target nodes.
    let b = rng.random_range(0..n) as u64;
    let mut d = rng.random_range(0..n - 1) as u64;
    if d >= b {
        d += 1;
    }

    // Choose sign uniformly.
    let sign = if rng.random_bool(0.5) { 1i64 } else { -1i64 };

    let deltas = build_cycle4(a, c, b, d, sign);
    let cap = domain.capacity(target.family);

    // ---- Validation + target ratio in one pass (no allocation) ----
    let mut delta_log_pi = 0.0f64;
    let mut applied = false;
    for &(src, tgt, d) in &deltas {
        let old = state.get(src, tgt);
        let new = (old as i64 + d) as OccNum;
        if old as i64 + d < 0 {
            return McmcOutcome::Held;
        }
        if !domain.is_admissible(src, tgt) {
            return McmcOutcome::Held;
        }
        if new > cap {
            return McmcOutcome::Held;
        }
        match target.delta_log_weight(src, tgt, old, new) {
            Some(w) => delta_log_pi += w,
            None => return McmcOutcome::Held,
        }
        if d != 0 {
            applied = true;
        }
    }
    if !applied {
        return McmcOutcome::Held;
    }

    // ---- Metropolis acceptance (symmetric proposal, no Hastings) ----
    if delta_log_pi < 0.0 {
        let log_u = (rng.random::<f64>() + f64::MIN_POSITIVE).ln();
        if log_u >= delta_log_pi {
            return McmcOutcome::Rejected;
        }
    }

    // ---- Apply directly (cells are distinct, no merging needed) ----
    for &(src, tgt, d) in &deltas {
        let old = state.get(src, tgt);
        state.set(src, tgt, (old as i64 + d) as OccNum);
    }

    McmcOutcome::Accepted
}