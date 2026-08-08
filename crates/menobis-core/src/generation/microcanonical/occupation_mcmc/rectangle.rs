//! Shared rectangle transaction helper for fixed-strength MCMC.
//!
//! Provides the four-cell (2×2) delta set used by the occupied-cell
//! proposal (§22) and by future loop-repair steps (§15, §17, §37).
//!
//! # Formula (spec §22, fixed direction)
//!
//! Given distinct source nodes \(a \ne c\) and distinct target nodes
//! \(b \ne d\), the four-cell transaction is:
//!
//! \[
//! t_{ab} \to t_{ab} - 1,\quad
//! t_{cd} \to t_{cd} - 1,\quad
//! t_{ad} \to t_{ad} + 1,\quad
//! t_{cb} \to t_{cb} + 1.
//! \]
//!
//! Every source and target strength is preserved exactly because each
//! row and column receives one \(-1\) and one \(+1\).
//!
//! # Minimal scope (§20, §37, §39)
//!
//! This is a **shared helper only**, not a generic trait framework.
//! It is intentionally minimal to avoid speculative generalization.

use super::domain::PairDomain;
use super::state::StrengthState;
use super::target::StrengthTarget;
use crate::OccNum;

/// Four (cell, delta) pairs of one rectangular cycle proposal.
type Deltas = [(u64, u64, i64); 4];

/// Build a 4-cycle delta set from two sources and two targets.
///
/// The direction is **fixed per spec §22**: decrement the diagonal
/// pairs `(a,b)` and `(c,d)`; increment the cross pairs `(a,d)` and
/// `(c,b)`.  No varied sign — the reverse move is always reachable
/// because the cross cells are occupied in the proposed state.
///
/// # Panics
///
/// In debug mode, panics if `a == c` or `b == d` (cells would not be
/// distinct).
#[inline]
pub fn build_four_cell(a: u64, c: u64, b: u64, d: u64) -> Deltas {
    debug_assert_ne!(a, c, "source nodes must be distinct");
    debug_assert_ne!(b, d, "target nodes must be distinct");
    [(a, b, -1i64), (c, d, -1i64), (a, d, 1i64), (c, b, 1i64)]
}

/// Validate that all four cells of a rectangle update are admissible.
///
/// Checks, for each `(src, tgt, delta)`:
/// - The new occupation is non-negative (`old + delta ≥ 0`).
/// - The pair is admitted by the domain.
/// - The new occupation does not exceed the domain capacity.
/// - The target accepts the transition (returns `Some`).
///
/// Returns `true` if all four cells pass; returns `false` (with no panic)
/// if any check fails.
pub fn validate_four_cell(
    state: &StrengthState,
    target: &StrengthTarget,
    domain: &PairDomain,
    deltas: &Deltas,
) -> bool {
    let cap = domain.capacity(target.family);
    for &(src, tgt, d) in deltas {
        let old = state.get(src, tgt);
        // Check state-independent bounds.
        let new_i = old as i64 + d;
        if new_i < 0 {
            return false;
        }
        let new = new_i as OccNum;
        if !domain.is_admissible(src, tgt) {
            return false;
        }
        if new > cap {
            return false;
        }
        // Check target weight (family support + cost).
        if target.delta_log_weight(src, tgt, old, new).is_none() {
            return false;
        }
    }
    true
}
