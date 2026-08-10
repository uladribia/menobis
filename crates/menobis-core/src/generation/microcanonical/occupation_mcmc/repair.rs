//! Targeted repair for fixed-strength microcanonical samplers.
//!
//! Provides:
//! - **Phase D**: Guaranteed loop repair for complete loopless ME/W (§14–17).
//! - **Phase E**: Capacity repair for B family (§18) and forbidden-pair
//!   repair for arbitrary masks (§19), with bounded randomized restarts
//!   (§21).
//!
//! # Algorithm (§14–21)
//!
//! 1. [Loop repair (§14–17)]: Guaranteed deterministic self-loop
//!    elimination via rectangle repair.
//! 2. [Capacity repair (§18)]: Rectangle repair with per-cell capacity
//!    checks for B family (t_ij ≤ M).
//! 3. [Forbidden-pair repair (§19)]: Rectangle repair that moves mass
//!    from inadmissible pairs to admissible ones.
//! 4. [Bounded restarts (§21)]: If any repair step gets stuck, discard
//!    and retry with a fresh random construction.
//!
//! # Design (§20)
//!
//! Repairs are **targeted functions**, not a generic framework.  No
//! `ViolationTrait`, `RepairConstraintSet`, or `GenericAnnealer`.

use rand::Rng;
use std::collections::HashSet;

use super::compressed::compressed_aggregated_matching;
use super::domain::PairDomain;
use super::errors::FixedStrengthError;
use super::state::StrengthState;
use crate::model::family::OccupationFamily;
use crate::OccNum;

/// Repair configuration (spec §21).
///
/// Controls bounded termination of structural repair.  Kept small —
/// no generic configuration framework.
#[derive(Clone, Debug)]
pub struct RepairConfig {
    /// Maximum rectangle-repair steps per attempt.
    pub max_steps: u64,
    /// Maximum reconstruction restarts when repair gets stuck.
    pub max_restarts: u32,
    /// Maximum random donor attempts before linear-scan fallback.
    pub max_random_donor_attempts: usize,
}

impl Default for RepairConfig {
    fn default() -> Self {
        Self {
            max_steps: 10_000_000,
            max_restarts: 5,
            max_random_donor_attempts: 20,
        }
    }
}

impl RepairConfig {
    /// Create a new `RepairConfig` with the given parameters.
    pub fn new(max_steps: u64, max_restarts: u32, max_random_donor_attempts: usize) -> Self {
        Self {
            max_steps,
            max_restarts,
            max_random_donor_attempts,
        }
    }
}

// ════════════════════════════════════════════════════════════════
// Phase D: Loop repair (spec §14–17)
// ════════════════════════════════════════════════════════════════

/// O(N) feasibility check for complete loopless ME/W (spec 14).
///
/// A loopless realization exists iff:
///
/// ```text
/// s_i^out + s_i^in ≤ T   ∀i,
/// ```
///
/// where T = Σ s_i^out = Σ s_i^in.
///
/// Returns `true` if feasible.
pub fn loopless_feasibility_check(strength_out: &[OccNum], strength_in: &[OccNum]) -> bool {
    let total: OccNum = strength_out.iter().sum();
    debug_assert_eq!(
        total,
        strength_in.iter().sum::<OccNum>(),
        "out and in totals must be equal"
    );
    strength_out
        .iter()
        .zip(strength_in.iter())
        .all(|(&s_out, &s_in)| s_out + s_in <= total)
}

/// Find all self-loops in the state.
///
/// Returns the list of `(node, occupation)` pairs where `src == tgt`.
fn find_self_loops(state: &StrengthState) -> Vec<(u64, OccNum)> {
    state
        .occupied_pairs()
        .iter()
        .filter_map(|&(src, tgt)| {
            if src == tgt {
                Some((src, state.get(src, tgt)))
            } else {
                None
            }
        })
        .collect()
}

/// Find a donor cell for self-loop repair (spec 17).
///
/// Attempts `config.max_random_donor_attempts` random draws from
/// occupied pairs, rejecting candidates where `src == i || tgt == i`.
/// If random sampling fails, falls back to a bounded linear scan over
/// all occupied pairs (guaranteed to succeed when feasibility holds,
/// spec 16).
///
/// Returns `None` only if no valid donor exists (should not happen
/// when the feasibility condition holds and `t_ii > 0`).
pub fn find_donor(
    state: &StrengthState,
    i: u64,
    config: &RepairConfig,
    rng: &mut impl Rng,
) -> Option<(u64, u64)> {
    // Phase 1: random sampling from occupied pairs.
    for _ in 0..config.max_random_donor_attempts {
        if let Some(((src, tgt), _)) = state.choose_random_occupied(rng) {
            if src != i && tgt != i {
                return Some((src, tgt));
            }
        }
    }

    // Phase 2: bounded linear scan fallback.
    for &(src, tgt) in state.occupied_pairs() {
        if src != i && tgt != i {
            return Some((src, tgt));
        }
    }

    None
}

/// Guaranteed loop repair for complete loopless ME/W (spec 15–17).
///
/// Eliminates all self-loops from the state while preserving every
/// source and target strength exactly.  The repair is guaranteed to
/// progress because each step strictly decreases total loop mass
/// L(t) (spec 16), and donor existence is guaranteed by the loopless
/// feasibility condition (spec 14).
///
/// # Algorithm
///
/// For each self-loop `(i, i)` with occupation `t_ii`:
/// 1. Find a donor cell `(c, d)` with `c ≠ i` and `d ≠ i` (spec 17).
/// 2. Compute `δ = min(t_ii, t_cd)`.
/// 3. Apply rectangle (spec 15).
///
/// # Errors
///
/// Returns [`FixedStrengthError::RepairDidNotConverge`] if the repair
/// exceeds `config.max_steps` steps or self-loops remain afterward.
pub fn repair_self_loops(
    state: &mut StrengthState,
    domain: &PairDomain,
    config: &RepairConfig,
    rng: &mut impl Rng,
) -> Result<(), FixedStrengthError> {
    debug_assert!(
        !domain.self_loops_allowed(),
        "repair_self_loops should only be called when self-loops are forbidden"
    );

    // Discover initial self-loops.
    let mut self_loops: Vec<(u64, OccNum)> = find_self_loops(state);
    let mut steps: u64 = 0;

    while !self_loops.is_empty() && steps < config.max_steps {
        // Pop one self-loop from the list (O(1) removal from back).
        let (i, _) = self_loops.swap_remove(self_loops.len() - 1);

        // Refresh t_ii from state — may have changed via cross-cell
        // increments from previous steps involving this node.
        let t_ii = state.get(i, i);

        if t_ii == 0 {
            continue;
        }

        // Find a donor cell (c, d) with c != i and d != i.
        let (c, d) = find_donor(state, i, config, rng).ok_or_else(|| {
            FixedStrengthError::RepairDidNotConverge {
                remaining_loops: self_loops.len() + 1,
                remaining_capacity_violations: 0,
                remaining_forbidden_occupations: 0,
                restart_count: 0,
                steps,
            }
        })?;

        let t_cd = state.get(c, d);
        debug_assert!(
            t_cd > 0,
            "donor cell ({c}, {d}) must be occupied (find_donor guarantees this)"
        );

        // Compute delta = min(t_ii, t_cd).
        let delta = t_ii.min(t_cd);
        debug_assert!(delta > 0, "delta must be positive");

        // Apply rectangle delta set via state.set().
        // Cells: (i,i,-δ), (c,d,-δ), (i,d,+δ), (c,i,+δ)
        let t_ii_new = t_ii - delta;
        let t_cd_new = t_cd - delta;
        let t_id_new = state.get(i, d) + delta;
        let t_ci_new = state.get(c, i) + delta;

        state.set(i, i, t_ii_new);
        state.set(c, d, t_cd_new);
        state.set(i, d, t_id_new);
        state.set(c, i, t_ci_new);

        steps += 1;

        // Re-add self-loop to list if it still has mass after this step.
        if t_ii_new > 0 {
            self_loops.push((i, t_ii_new));
        }
    }

    if !self_loops.is_empty() {
        return Err(FixedStrengthError::RepairDidNotConverge {
            remaining_loops: self_loops.len(),
            remaining_capacity_violations: 0,
            remaining_forbidden_occupations: 0,
            restart_count: 0,
            steps,
        });
    }

    // Verify no self-loops remain (debug-only assertion).
    debug_assert_eq!(
        find_self_loops(state).len(),
        0,
        "repair claimed success but self-loops remain"
    );

    Ok(())
}

// ════════════════════════════════════════════════════════════════
// Phase E: Capacity repair for B family (spec §18)
// ════════════════════════════════════════════════════════════════

/// Find all capacity violations in the state.
///
/// Returns the list of `(i, j, occupation)` where occupation exceeds
/// the per-pair capacity for the given family (§18).  For ME and W
/// capacity is `OccNum::MAX`, so no violations can occur — the list
/// will be empty.  For B, the capacity is the layer count `M`.
pub fn find_capacity_violations(
    state: &StrengthState,
    family: OccupationFamily,
    domain: &PairDomain,
) -> Vec<(u64, u64, OccNum)> {
    let cap = domain.capacity(family);
    if cap == OccNum::MAX {
        // Unbounded: no violations possible.
        return Vec::new();
    }
    state
        .iter_occupied()
        .filter_map(|((src, tgt), occ)| {
            if occ > cap {
                Some((src, tgt, occ))
            } else {
                None
            }
        })
        .collect()
}

/// Find a donor cell for capacity violation repair (spec §18).
///
/// We need a cell `(c, d)` with:
/// - `t_cd > 0` (occupied donor)
/// - `c ≠ i` and `d ≠ j` (distinct row/col from the violation)
/// - `t_id + δ_max ≤ cap` (spare capacity at increment cell `(i,d)`)
/// - `t_cj + δ_max ≤ cap` (spare capacity at increment cell `(c,j)`)
///
/// The largest feasible δ is:
/// ```text
/// δ_max = min(t_cd, cap - t_id, cap - t_cj)
/// ```
///
/// We accept any donor where `δ_max > 0`, i.e., the donor cell and
/// both increment cells have at least 1 unit of slack.
///
/// Returns `None` if no valid donor is found after random sampling
/// and linear-scan fallback.
pub fn find_donor_with_capacity(
    state: &StrengthState,
    i: u64,
    j: u64,
    cap: OccNum,
    config: &RepairConfig,
    rng: &mut impl Rng,
) -> Option<(u64, u64)> {
    // Phase 1: random sampling from occupied pairs.
    for _ in 0..config.max_random_donor_attempts {
        if let Some(((c, d), t_cd)) = state.choose_random_occupied(rng) {
            if c == i || d == j {
                continue;
            }
            let id_slack = cap.saturating_sub(state.get(i, d));
            let cj_slack = cap.saturating_sub(state.get(c, j));
            // δ_max = min(t_cd, cap - t_id, cap - t_cj)
            let delta_max = t_cd.min(id_slack).min(cj_slack);
            if delta_max > 0 {
                return Some((c, d));
            }
        }
    }

    // Phase 2: bounded linear scan fallback.
    for &(c, d) in state.occupied_pairs() {
        if c == i || d == j {
            continue;
        }
        let t_cd = state.get(c, d);
        let id_slack = cap.saturating_sub(state.get(i, d));
        let cj_slack = cap.saturating_sub(state.get(c, j));
        let delta_max = t_cd.min(id_slack).min(cj_slack);
        if delta_max > 0 {
            return Some((c, d));
        }
    }

    None
}

/// Repair B capacity violations using rectangle repair (spec §18).
///
/// For each capacity violation `(i, j)` where `t_ij > M`:
/// 1. Find a donor cell `(c, d)` with spare capacity at increment
///    cells `(i, d)` and `(c, j)`.
/// 2. Compute δ = min(t_ij - M, t_cd, M - t_id, M - t_cj).
/// 3. Apply rectangle:
///    ```text
///    t_ij' = t_ij - δ
///    t_cd' = t_cd - δ
///    t_id' = t_id + δ
///    t_cj' = t_cj + δ
///    ```
///
/// Each step preserves strengths exactly (each row/column gets one
/// `-δ` and one `+δ`).  Increment cells are checked to never exceed
/// the per-cell capacity M.
///
/// # Errors
///
/// Returns [`FixedStrengthError::RepairDidNotConverge`] if no donor
/// can be found for a remaining violation after `config.max_steps`
/// steps.
pub fn repair_capacity(
    state: &mut StrengthState,
    family: OccupationFamily,
    domain: &PairDomain,
    config: &RepairConfig,
    rng: &mut impl Rng,
) -> Result<(), FixedStrengthError> {
    let cap = domain.capacity(family);
    if cap == OccNum::MAX {
        // Unbounded: no repair needed.
        return Ok(());
    }

    let mut violations: Vec<(u64, u64, OccNum)> = find_capacity_violations(state, family, domain);
    let mut steps: u64 = 0;
    while !violations.is_empty() && steps < config.max_steps {
        let (i, j, _occ) = violations.swap_remove(violations.len() - 1);

        // Refresh current occupation.
        let t_ij = state.get(i, j);
        if t_ij <= cap {
            continue;
        }
        let over = t_ij - cap;

        // Find a donor cell with spare capacity at increment cells.
        let (c, d) = find_donor_with_capacity(state, i, j, cap, config, rng).ok_or_else(|| {
            FixedStrengthError::RepairDidNotConverge {
                remaining_loops: 0,
                remaining_capacity_violations: violations.len() + 1,
                remaining_forbidden_occupations: 0,
                restart_count: 0,
                steps,
            }
        })?;

        let t_cd = state.get(c, d);
        debug_assert!(t_cd > 0, "donor must be occupied");

        let id_slack = cap - state.get(i, d);
        let cj_slack = cap - state.get(c, j);
        let delta = over.min(t_cd).min(id_slack).min(cj_slack);
        debug_assert!(delta > 0, "delta must be positive for capacity repair");

        // Apply rectangle:
        // Cells: (i,j,-δ), (c,d,-δ), (i,d,+δ), (c,j,+δ)
        state.set(i, j, t_ij - delta);
        state.set(c, d, t_cd - delta);
        state.set(i, d, state.get(i, d) + delta);
        state.set(c, j, state.get(c, j) + delta);

        steps += 1;

        // Re-add to violations if still over capacity.
        let new_t_ij = t_ij - delta;
        if new_t_ij > cap {
            violations.push((i, j, new_t_ij));
        }
    }

    if !violations.is_empty() {
        return Err(FixedStrengthError::RepairDidNotConverge {
            remaining_loops: 0,
            remaining_capacity_violations: violations.len(),
            remaining_forbidden_occupations: 0,
            restart_count: 0,
            steps,
        });
    }

    Ok(())
}

// ════════════════════════════════════════════════════════════════
// Phase E: Forbidden-pair repair (spec §19)
// ════════════════════════════════════════════════════════════════

/// Find all forbidden-pair occupations.
pub fn find_forbidden_occupations(
    state: &StrengthState,
    forbidden_pairs: &HashSet<(u64, u64)>,
) -> Vec<(u64, u64, OccNum)> {
    state
        .iter_occupied()
        .filter_map(|((src, tgt), occ)| {
            if forbidden_pairs.contains(&(src, tgt)) {
                Some((src, tgt, occ))
            } else {
                None
            }
        })
        .collect()
}

/// Repair forbidden-pair occupations using rectangle repair (spec §19).
///
/// For each forbidden occupied pair `(i, j)`, move mass to admissible
/// cells via rectangle transaction.  The same capacity constraints may
/// apply when the family is B.
pub fn repair_forbidden_pairs(
    state: &mut StrengthState,
    family: OccupationFamily,
    domain: &PairDomain,
    forbidden_pairs: &HashSet<(u64, u64)>,
    config: &RepairConfig,
    rng: &mut impl Rng,
) -> Result<(), FixedStrengthError> {
    let cap = domain.capacity(family);
    let mut violations: Vec<(u64, u64, OccNum)> =
        find_forbidden_occupations(state, forbidden_pairs);
    let mut steps: u64 = 0;

    while !violations.is_empty() && steps < config.max_steps {
        let (i, j, _) = violations.swap_remove(violations.len() - 1);
        let t_ij = state.get(i, j);
        if t_ij == 0 {
            continue;
        }

        // Find a donor outside forbidden set with capacity slack.
        let (c, d) = find_donor_with_forbidden(state, i, j, cap, forbidden_pairs, config, rng)
            .ok_or_else(|| FixedStrengthError::RepairDidNotConverge {
                remaining_loops: 0,
                remaining_capacity_violations: 0,
                remaining_forbidden_occupations: violations.len() + 1,
                restart_count: 0,
                steps,
            })?;

        let t_cd = state.get(c, d);
        let delta = if cap == OccNum::MAX {
            t_ij.min(t_cd)
        } else {
            let id_slack = cap.saturating_sub(state.get(i, d));
            let cj_slack = cap.saturating_sub(state.get(c, j));
            t_ij.min(t_cd).min(id_slack).min(cj_slack)
        };
        debug_assert!(delta > 0, "delta must be positive");

        state.set(i, j, state.get(i, j) - delta);
        state.set(c, d, state.get(c, d) - delta);
        state.set(i, d, state.get(i, d) + delta);
        state.set(c, j, state.get(c, j) + delta);

        steps += 1;

        if state.get(i, j) > 0 && forbidden_pairs.contains(&(i, j)) {
            violations.push((i, j, state.get(i, j)));
        }
    }

    if !violations.is_empty() {
        return Err(FixedStrengthError::RepairDidNotConverge {
            remaining_loops: 0,
            remaining_capacity_violations: 0,
            remaining_forbidden_occupations: violations.len(),
            restart_count: 0,
            steps,
        });
    }

    Ok(())
}

/// Find a donor cell for forbidden-pair repair, avoiding the forbidden set.
fn find_donor_with_forbidden(
    state: &StrengthState,
    i: u64,
    j: u64,
    cap: OccNum,
    forbidden_pairs: &HashSet<(u64, u64)>,
    config: &RepairConfig,
    rng: &mut impl Rng,
) -> Option<(u64, u64)> {
    for _ in 0..config.max_random_donor_attempts {
        if let Some(((c, d), t_cd)) = state.choose_random_occupied(rng) {
            if c == i || d == j {
                continue;
            }
            if forbidden_pairs.contains(&(c, d)) {
                continue;
            }
            // Increment cells must also be outside forbidden set (semantic review M1).
            if forbidden_pairs.contains(&(i, d)) || forbidden_pairs.contains(&(c, j)) {
                continue;
            }
            if cap != OccNum::MAX {
                let id_slack = cap.saturating_sub(state.get(i, d));
                let cj_slack = cap.saturating_sub(state.get(c, j));
                if t_cd.min(id_slack).min(cj_slack) == 0 {
                    continue;
                }
            }
            return Some((c, d));
        }
    }

    for &(c, d) in state.occupied_pairs() {
        if c == i || d == j {
            continue;
        }
        if forbidden_pairs.contains(&(c, d)) {
            continue;
        }
        // Increment cells must also be outside forbidden set (semantic review M1).
        if forbidden_pairs.contains(&(i, d)) || forbidden_pairs.contains(&(c, j)) {
            continue;
        }
        if cap != OccNum::MAX {
            let t_cd = state.get(c, d);
            let id_slack = cap.saturating_sub(state.get(i, d));
            let cj_slack = cap.saturating_sub(state.get(c, j));
            if t_cd.min(id_slack).min(cj_slack) == 0 {
                continue;
            }
        }
        return Some((c, d));
    }

    None
}

// ════════════════════════════════════════════════════════════════
// Admissibility repair for sparse domains (spec §19)
// ════════════════════════════════════════════════════════════════

/// Find a donor cell for admissibility repair of sparse domains.
///
/// Uses `domain.is_admissible()` checks instead of a forbidden-pair
/// HashSet.  This avoids materialising the enormous forbidden set for
/// sparse domains.
///
/// We need a cell `(c, d)` with:
/// - `c ≠ i` and `d ≠ j` (distinct row/col from the violation)
/// - `domain.is_admissible(c, d)` (donor cell is admissible)
/// - `domain.is_admissible(i, d)` (increment cell is admissible)
/// - `domain.is_admissible(c, j)` (other increment cell is admissible)
/// - If `cap != MAX`: `t_cd.min(cap - t_id).min(cap - t_cj) > 0` (capacity slack)
///
/// Returns `None` if no valid donor is found after random sampling
/// and linear-scan fallback.
pub fn find_donor_by_admissible(
    state: &StrengthState,
    i: u64,
    j: u64,
    cap: OccNum,
    domain: &PairDomain,
    config: &RepairConfig,
    rng: &mut impl Rng,
) -> Option<(u64, u64)> {
    // Phase 1: random sampling from occupied pairs.
    for _ in 0..config.max_random_donor_attempts {
        if let Some(((c, d), t_cd)) = state.choose_random_occupied(rng) {
            if c == i || d == j {
                continue;
            }
            if !domain.is_admissible(c, d) {
                continue;
            }
            if !domain.is_admissible(i, d) {
                continue;
            }
            if !domain.is_admissible(c, j) {
                continue;
            }
            if cap != OccNum::MAX {
                let id_slack = cap.saturating_sub(state.get(i, d));
                let cj_slack = cap.saturating_sub(state.get(c, j));
                if t_cd.min(id_slack).min(cj_slack) == 0 {
                    continue;
                }
            }
            return Some((c, d));
        }
    }

    // Phase 2: bounded linear scan fallback.
    for &(c, d) in state.occupied_pairs() {
        if c == i || d == j {
            continue;
        }
        if !domain.is_admissible(c, d) {
            continue;
        }
        if !domain.is_admissible(i, d) {
            continue;
        }
        if !domain.is_admissible(c, j) {
            continue;
        }
        if cap != OccNum::MAX {
            let t_cd = state.get(c, d);
            let id_slack = cap.saturating_sub(state.get(i, d));
            let cj_slack = cap.saturating_sub(state.get(c, j));
            if t_cd.min(id_slack).min(cj_slack) == 0 {
                continue;
            }
        }
        return Some((c, d));
    }

    None
}

/// Repair inadmissible-occupation pairs using admissibility checks.
///
/// For each occupied pair `(i, j)` where `!domain.is_admissible(i, j)`,
/// move mass to admissible cells via rectangle transaction.  Uses
/// `find_donor_by_admissible` which checks admissibility dynamically
/// rather than looking up a forbidden set, avoiding O(N^2) memory
/// for sparse domains (spec §19).
///
/// # Errors
///
/// Returns [`FixedStrengthError::RepairDidNotConverge`] if no donor
/// can be found for a remaining violation after `config.max_steps`
/// steps.
pub fn repair_inadmissible_pairs(
    state: &mut StrengthState,
    family: OccupationFamily,
    domain: &PairDomain,
    config: &RepairConfig,
    rng: &mut impl Rng,
) -> Result<(), FixedStrengthError> {
    let cap = domain.capacity(family);

    // Collect all occupied pairs that are inadmissible.
    let mut violations: Vec<(u64, u64)> = state
        .iter_occupied()
        .filter_map(|((src, tgt), _)| {
            if !domain.is_admissible(src, tgt) {
                Some((src, tgt))
            } else {
                None
            }
        })
        .collect();

    let mut steps: u64 = 0;

    while !violations.is_empty() && steps < config.max_steps {
        let (i, j) = violations.swap_remove(violations.len() - 1);
        let t_ij = state.get(i, j);
        if t_ij == 0 {
            continue;
        }

        // Find a donor via admissibility check (no HashSet needed).
        let (c, d) =
            find_donor_by_admissible(state, i, j, cap, domain, config, rng).ok_or_else(|| {
                FixedStrengthError::RepairDidNotConverge {
                    remaining_loops: 0,
                    remaining_capacity_violations: 0,
                    remaining_forbidden_occupations: violations.len() + 1,
                    restart_count: 0,
                    steps,
                }
            })?;

        let t_cd = state.get(c, d);
        let delta = if cap == OccNum::MAX {
            t_ij.min(t_cd)
        } else {
            let id_slack = cap.saturating_sub(state.get(i, d));
            let cj_slack = cap.saturating_sub(state.get(c, j));
            t_ij.min(t_cd).min(id_slack).min(cj_slack)
        };
        debug_assert!(delta > 0, "delta must be positive");

        // Apply rectangle:
        // Cells: (i,j,-δ), (c,d,-δ), (i,d,+δ), (c,j,+δ)
        state.set(i, j, state.get(i, j) - delta);
        state.set(c, d, state.get(c, d) - delta);
        state.set(i, d, state.get(i, d) + delta);
        state.set(c, j, state.get(c, j) + delta);

        steps += 1;

        if state.get(i, j) > 0 && !domain.is_admissible(i, j) {
            violations.push((i, j));
        }
    }

    if !violations.is_empty() {
        return Err(FixedStrengthError::RepairDidNotConverge {
            remaining_loops: 0,
            remaining_capacity_violations: 0,
            remaining_forbidden_occupations: violations.len(),
            restart_count: 0,
            steps,
        });
    }

    Ok(())
}

// ════════════════════════════════════════════════════════════════
// Orchestrator: Bounded restarts (spec §21)
// ════════════════════════════════════════════════════════════════

/// Orchestrate capacity + forbidden-pair repair with bounded restarts.
///
/// 1. Try repair_capacity + repair_forbidden_pairs (if forbidden set provided).
/// 2. If stuck: discard state, rebuild via `compressed_aggregated_matching`, retry.
/// 3. After max_restarts exhausted: return RepairDidNotConverge.
#[allow(clippy::too_many_arguments)]
pub fn repair_all_violations(
    state: &mut StrengthState,
    family: OccupationFamily,
    domain: &PairDomain,
    forbidden_pairs: Option<&HashSet<(u64, u64)>>,
    config: &RepairConfig,
    rng: &mut impl Rng,
    strength_out: &[OccNum],
    strength_in: &[OccNum],
) -> Result<(), FixedStrengthError> {
    for restart in 0..config.max_restarts {
        if restart > 0 {
            let table =
                compressed_aggregated_matching(strength_out, strength_in, family, domain, rng)?;
            *state = StrengthState::new(domain.node_count(), table);
        }

        // Capacity repair (spec §18).
        if repair_capacity(state, family, domain, config, rng).is_err() {
            continue;
        }

        // Forbidden-pair repair (spec §19).
        if let Some(forbidden) = forbidden_pairs {
            if repair_forbidden_pairs(state, family, domain, forbidden, config, rng).is_err() {
                continue;
            }
        }

        return Ok(());
    }

    // All restarts exhausted: report remaining violations.
    let remaining_cap = find_capacity_violations(state, family, domain);
    let remaining_forbidden = forbidden_pairs
        .map(|f| find_forbidden_occupations(state, f))
        .unwrap_or_default();
    Err(FixedStrengthError::RepairDidNotConverge {
        remaining_loops: 0,
        remaining_capacity_violations: remaining_cap.len(),
        remaining_forbidden_occupations: remaining_forbidden.len(),
        restart_count: config.max_restarts,
        steps: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    // ════════════════════════════════════════════════════════════
    // Phase D: Loop repair tests
    // ════════════════════════════════════════════════════════════

    #[test]
    fn loopless_feasibility_check_passes() {
        let out = vec![5, 3, 2];
        let inp = vec![4, 4, 2];
        assert!(loopless_feasibility_check(&out, &inp));
    }

    #[test]
    fn loopless_feasibility_check_fails() {
        let out = vec![10, 0];
        let inp = vec![1, 9];
        assert!(!loopless_feasibility_check(&out, &inp));
    }

    #[test]
    fn loopless_feasibility_check_boundary() {
        let out = vec![5, 5];
        let inp = vec![5, 5];
        assert!(loopless_feasibility_check(&out, &inp));
    }

    #[test]
    fn repair_self_loops_eliminates_all() {
        let mut rng = StdRng::seed_from_u64(42);
        let domain = PairDomain::Complete {
            node_count: 3,
            self_loops: false,
        };
        let config = RepairConfig::default();
        let pairs: Vec<((u64, u64), OccNum)> = vec![
            ((0, 0), 3),
            ((0, 1), 2),
            ((1, 0), 1),
            ((1, 1), 1),
            ((1, 2), 1),
            ((2, 1), 1),
            ((2, 2), 1),
        ];
        let mut state = StrengthState::new(3, pairs);
        assert!(!find_self_loops(&state).is_empty());
        let out_before = state.out_strengths.clone();
        let in_before = state.in_strengths.clone();

        repair_self_loops(&mut state, &domain, &config, &mut rng).unwrap();

        assert_eq!(find_self_loops(&state).len(), 0);
        assert_eq!(state.out_strengths, out_before);
        assert_eq!(state.in_strengths, in_before);
    }

    #[test]
    fn repair_self_loops_no_op_when_none() {
        let mut rng = StdRng::seed_from_u64(42);
        let config = RepairConfig::default();
        let domain = PairDomain::Complete {
            node_count: 3,
            self_loops: false,
        };
        let pairs: Vec<((u64, u64), OccNum)> = vec![
            ((0, 1), 3),
            ((0, 2), 2),
            ((1, 0), 3),
            ((1, 2), 1),
            ((2, 0), 2),
            ((2, 1), 2),
        ];
        let mut state = StrengthState::new(3, pairs);
        assert_eq!(find_self_loops(&state).len(), 0);
        let out_before = state.out_strengths.clone();

        repair_self_loops(&mut state, &domain, &config, &mut rng).unwrap();

        assert_eq!(find_self_loops(&state).len(), 0);
        assert_eq!(state.out_strengths, out_before);
    }

    #[test]
    fn find_donor_none_for_one_node() {
        let mut rng = StdRng::seed_from_u64(42);
        let config = RepairConfig::default();
        let pairs: Vec<((u64, u64), OccNum)> = vec![((0, 0), 5)];
        let state = StrengthState::new(1, pairs);
        assert!(find_donor(&state, 0, &config, &mut rng).is_none());
    }

    #[test]
    fn repair_self_loops_deterministic() {
        let config = RepairConfig::default();
        let domain = PairDomain::Complete {
            node_count: 3,
            self_loops: false,
        };
        let pairs: Vec<((u64, u64), OccNum)> = vec![
            ((0, 0), 3),
            ((0, 1), 2),
            ((1, 0), 1),
            ((1, 1), 1),
            ((1, 2), 1),
            ((2, 1), 1),
            ((2, 2), 1),
        ];

        let mut state_a = StrengthState::new(3, pairs.clone());
        let mut state_b = StrengthState::new(3, pairs);
        let mut rng_a = StdRng::seed_from_u64(12345);
        let mut rng_b = StdRng::seed_from_u64(12345);

        repair_self_loops(&mut state_a, &domain, &config, &mut rng_a).unwrap();
        repair_self_loops(&mut state_b, &domain, &config, &mut rng_b).unwrap();

        let mut pairs_a: Vec<_> = state_a.iter_occupied().collect();
        let mut pairs_b: Vec<_> = state_b.iter_occupied().collect();
        pairs_a.sort_unstable();
        pairs_b.sort_unstable();
        assert_eq!(pairs_a, pairs_b);
    }

    // ════════════════════════════════════════════════════════════
    // Phase E: Capacity repair tests
    // ════════════════════════════════════════════════════════════

    #[test]
    fn find_capacity_violations_detects_excess() {
        let domain = PairDomain::Complete {
            node_count: 2,
            self_loops: true,
        };
        let family = OccupationFamily::B { layers: 3 };
        let pairs: Vec<((u64, u64), OccNum)> = vec![((0, 1), 5)];
        let state = StrengthState::new(2, pairs);
        let viols = find_capacity_violations(&state, family, &domain);
        assert_eq!(viols.len(), 1);
        assert_eq!(viols[0], (0, 1, 5));
    }

    #[test]
    fn find_capacity_violations_none_when_ok() {
        let domain = PairDomain::Complete {
            node_count: 2,
            self_loops: true,
        };
        let family = OccupationFamily::B { layers: 5 };
        let pairs: Vec<((u64, u64), OccNum)> = vec![((0, 1), 3), ((1, 0), 4)];
        let state = StrengthState::new(2, pairs);
        let viols = find_capacity_violations(&state, family, &domain);
        assert!(viols.is_empty());
    }

    #[test]
    fn find_capacity_violations_me_always_empty() {
        let domain = PairDomain::Complete {
            node_count: 2,
            self_loops: true,
        };
        let pairs: Vec<((u64, u64), OccNum)> = vec![((0, 1), OccNum::MAX)];
        let state = StrengthState::new(2, pairs);
        let viols = find_capacity_violations(&state, OccupationFamily::ME, &domain);
        assert!(viols.is_empty());
    }

    #[test]
    fn find_forbidden_occupations_detects_forbidden() {
        let pairs: Vec<((u64, u64), OccNum)> = vec![((0, 0), 2), ((0, 1), 3), ((1, 0), 1)];
        let state = StrengthState::new(2, pairs);
        let mut forbidden = HashSet::new();
        forbidden.insert((0, 0));
        let viols = find_forbidden_occupations(&state, &forbidden);
        assert_eq!(viols.len(), 1);
        assert_eq!(viols[0], (0, 0, 2));
    }

    #[test]
    fn repair_capacity_eliminates_violations() {
        let mut rng = StdRng::seed_from_u64(42);
        let config = RepairConfig::default();
        let domain = PairDomain::Complete {
            node_count: 2,
            self_loops: true,
        };
        let family = OccupationFamily::B { layers: 3 };
        // (0,1)=5 exceeds M=3
        let pairs: Vec<((u64, u64), OccNum)> = vec![((0, 1), 5), ((1, 0), 3)];
        let mut state = StrengthState::new(2, pairs);
        let out_before = state.out_strengths.clone();
        let in_before = state.in_strengths.clone();

        repair_capacity(&mut state, family, &domain, &config, &mut rng).unwrap();

        // All occupations should be ≤ M
        for ((_, _), occ) in state.iter_occupied() {
            assert!(occ <= 3, "occupation {occ} exceeds capacity 3");
        }
        // Strengths preserved
        assert_eq!(state.out_strengths, out_before);
        assert_eq!(state.in_strengths, in_before);
    }

    #[test]
    fn repair_capacity_preserves_strengths() {
        let mut rng = StdRng::seed_from_u64(99);
        let config = RepairConfig::default();
        let domain = PairDomain::Complete {
            node_count: 3,
            self_loops: true,
        };
        let family = OccupationFamily::B { layers: 2 };
        // 3 nodes, strengths out=[5,3,2], in=[4,4,2], total=10
        // Some occupations exceed M=2
        let pairs: Vec<((u64, u64), OccNum)> = vec![
            ((0, 1), 3),
            ((0, 2), 2),
            ((1, 0), 1),
            ((1, 1), 2),
            ((2, 0), 1),
            ((2, 1), 1),
        ];
        let mut state = StrengthState::new(3, pairs);
        let out_before = state.out_strengths.clone();
        let in_before = state.in_strengths.clone();

        repair_capacity(&mut state, family, &domain, &config, &mut rng).unwrap();

        assert_eq!(state.out_strengths, out_before);
        assert_eq!(state.in_strengths, in_before);
        let total: OccNum = state.iter_occupied().map(|(_, o)| o).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn repair_capacity_no_op_when_ok() {
        let mut rng = StdRng::seed_from_u64(42);
        let config = RepairConfig::default();
        let domain = PairDomain::Complete {
            node_count: 2,
            self_loops: true,
        };
        let family = OccupationFamily::B { layers: 5 };
        let pairs: Vec<((u64, u64), OccNum)> = vec![((0, 1), 3), ((1, 0), 2)];
        let mut state = StrengthState::new(2, pairs);
        let occupied_before: Vec<_> = state.occupied_pairs().to_vec();

        repair_capacity(&mut state, family, &domain, &config, &mut rng).unwrap();

        // State unchanged
        assert_eq!(state.occupied_pairs(), &occupied_before);
    }

    #[test]
    fn repair_config_defaults_sensible() {
        let config = RepairConfig::default();
        assert_eq!(config.max_steps, 10_000_000);
        assert_eq!(config.max_restarts, 5);
        assert_eq!(config.max_random_donor_attempts, 20);
    }

    #[test]
    fn repair_all_violations_no_restarts_when_repair_succeeds() {
        // B capacity with M=3, state already within capacity
        // repair_all_violations should succeed on first try (no reconstruct needed)
        let mut rng = StdRng::seed_from_u64(42);
        let config = RepairConfig::default();
        let domain = PairDomain::Complete {
            node_count: 2,
            self_loops: true,
        };
        let family = OccupationFamily::B { layers: 3 };
        let pairs: Vec<((u64, u64), OccNum)> = vec![((0, 1), 3), ((1, 0), 3)];
        let mut state = StrengthState::new(2, pairs);
        let out_before = state.out_strengths.clone();

        repair_all_violations(
            &mut state,
            family,
            &domain,
            None,
            &config,
            &mut rng,
            &[3, 3],
            &[3, 3],
        )
        .unwrap();

        assert_eq!(state.out_strengths, out_before);
    }
}
