//! Compressed aggregated matching for fixed-strength initial states.
//!
//! Implements the compressed aggregated constructor (spec §11–13):
//! a simple randomised greedy that builds an exact-strength sparse
//! occupation table while avoiding \(O(T)\) stub expansion,
//! \(O(N^2)\) pair enumeration, and max flow.
//!
//! The constructor is only a **starting-state generator** (§12).
//! It intentionally leaves self-loops and B capacity violations in
//! the output — these are repaired by Phase D (loop repair) and
//! Phase E (B repair) respectively.
//!
//! # Algorithm (§11)
//!
//! Maintain active source and target index lists.  Repeatedly:
//!
//! 1. Pick a random active source \(i\).
//! 2. Pick a random active target \(j\).
//! 3. Allocate a block \(x = \min(r_i, c_j)\).
//! 4. Record \(t_{ij} \mathrel{+}= x\).
//! 5. Decrease residuals.
//! 6. Remove exhausted entries from active lists.
//!
//! The block allocation avoids per-unit stub expansion
//! (no \(O(T)\) memory/work).  Active-index management avoids
//! scanning all \(N^2\) pairs (no \(O(N^2)\)).

use rand::Rng;

use super::domain::PairDomain;
use super::errors::FixedStrengthError;
use crate::model::family::OccupationFamily;
use crate::OccNum;

/// Compressed aggregated matching for fixed-strength initial states.
///
/// Returns a `FlowTable` (`Vec<((u64, u64), OccNum)>`) that satisfies
/// the strength sequences exactly.
///
/// # Properties
///
/// - **Exact strengths**: every source and target strength is satisfied.
/// - **Sparse output**: the table has at most one entry per active
///   source–target pair encountered (packed).
/// - **Self-loops permitted**: no `i ≠ j` check (Phase D repair).
/// - **B capacity NOT enforced**: occupations may exceed `M` (Phase E
///   repair).
/// - **No family-weight formulas**: purely combinatorial (§12).
/// - **Randomised**: source and target order is shuffled by the
///   provided RNG for diverse starting states.
/// - **Reproducible**: same RNG seed → same output.
///
/// # Errors
///
/// Returns [`FixedStrengthError::InitializationFailed`] if the
/// strength sequences are unbalanced (total out != total in).
pub fn compressed_aggregated_matching(
    strength_out: &[OccNum],
    strength_in: &[OccNum],
    _family: OccupationFamily,
    _domain: &PairDomain,
    rng: &mut impl Rng,
) -> Result<super::feasibility::FlowTable, FixedStrengthError> {
    let n = strength_out.len();
    let n_in = strength_in.len();

    // Basic consistency.
    if n != n_in {
        return Err(FixedStrengthError::InitializationFailed(
            "out and in strength arrays have different lengths".into(),
        ));
    }
    let total_out: OccNum = strength_out.iter().sum();
    let total_in: OccNum = strength_in.iter().sum();
    if total_out != total_in {
        return Err(FixedStrengthError::InitializationFailed(format!(
            "unbalanced strengths: out={total_out} ≠ in={total_in}"
        )));
    }

    let mut remaining_out = strength_out.to_vec();
    let mut remaining_in = strength_in.to_vec();
    let mut active_sources: Vec<usize> = (0..n).filter(|&i| remaining_out[i] > 0).collect();
    let mut active_targets: Vec<usize> = (0..n).filter(|&j| remaining_in[j] > 0).collect();

    let mut table = Vec::new();

    while !active_sources.is_empty() && !active_targets.is_empty() {
        // Pick random active source.
        let src_idx = rng.random_range(0..active_sources.len());
        let i = active_sources[src_idx];

        // Pick random active target.
        let tgt_idx = rng.random_range(0..active_targets.len());
        let j = active_targets[tgt_idx];

        let r_i = remaining_out[i];
        let r_j = remaining_in[j];
        debug_assert!(r_i > 0 && r_j > 0);

        // Block allocation: min of residuals.  Do NOT enforce domain
        // admissibility, self-loop policy, or B capacity — those are
        // deferred to Phase D/E.
        let x = r_i.min(r_j);
        debug_assert!(x > 0);

        // Record the block.
        table.push(((i as u64, j as u64), x));

        // Update residuals.
        remaining_out[i] -= x;
        remaining_in[j] -= x;

        // Remove exhausted sources.
        if remaining_out[i] == 0 {
            active_sources.swap_remove(src_idx);
        }

        // Remove exhausted targets.
        if remaining_in[j] == 0 {
            active_targets.swap_remove(tgt_idx);
        }
    }

    debug_assert!(
        remaining_out.iter().all(|&r| r == 0),
        "unallocated out-strength remains"
    );
    debug_assert!(
        remaining_in.iter().all(|&r| r == 0),
        "unallocated in-strength remains"
    );

    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_domain(n: usize, self_loops: bool) -> PairDomain {
        PairDomain::Complete {
            node_count: n,
            self_loops,
        }
    }

    fn verify_strengths(table: &[((u64, u64), OccNum)], s_out: &[OccNum], s_in: &[OccNum]) {
        let n = s_out.len();
        let mut check_out = vec![0u64; n];
        let mut check_in = vec![0u64; n];
        for &((s, t), o) in table {
            check_out[s as usize] += o;
            check_in[t as usize] += o;
        }
        assert_eq!(check_out, s_out, "out-strengths mismatch");
        assert_eq!(check_in, s_in, "in-strengths mismatch");
    }

    #[test]
    fn compressed_exact_strengths_me() {
        let s_out = vec![5u64, 3, 7, 2];
        let s_in = vec![4u64, 6, 3, 4];
        let domain = make_domain(4, true);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let table =
            compressed_aggregated_matching(&s_out, &s_in, OccupationFamily::ME, &domain, &mut rng)
                .unwrap();
        verify_strengths(&table, &s_out, &s_in);
    }

    #[test]
    fn compressed_exact_strengths_b() {
        let s_out = vec![4u64, 3, 2];
        let s_in = vec![2u64, 4, 3];
        let domain = make_domain(3, true);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let table = compressed_aggregated_matching(
            &s_out,
            &s_in,
            OccupationFamily::B { layers: 3 },
            &domain,
            &mut rng,
        )
        .unwrap();
        verify_strengths(&table, &s_out, &s_in);
        // B capacity may be exceeded — that's Phase E's job.
    }

    #[test]
    fn compressed_single_node() {
        let s_out = vec![10u64];
        let s_in = vec![10u64];
        let domain = make_domain(1, true);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let table =
            compressed_aggregated_matching(&s_out, &s_in, OccupationFamily::ME, &domain, &mut rng)
                .unwrap();
        verify_strengths(&table, &s_out, &s_in);
        assert_eq!(table.len(), 1, "single node should produce 1 pair");
        assert_eq!(table[0].1, 10, "single pair should have full occupation");
    }

    #[test]
    fn compressed_zero_strengths() {
        let s_out = vec![0u64; 3];
        let s_in = vec![0u64; 3];
        let domain = make_domain(3, true);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let table =
            compressed_aggregated_matching(&s_out, &s_in, OccupationFamily::ME, &domain, &mut rng)
                .unwrap();
        assert!(
            table.is_empty(),
            "zero strengths should produce empty table"
        );
    }

    #[test]
    fn compressed_reproducible() {
        let s_out = vec![5u64, 4, 3];
        let s_in = vec![3u64, 5, 4];
        let domain = make_domain(3, true);

        let run = |seed: u64| -> Vec<((u64, u64), OccNum)> {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            compressed_aggregated_matching(&s_out, &s_in, OccupationFamily::ME, &domain, &mut rng)
                .unwrap()
        };

        assert_eq!(run(42), run(42));
    }

    #[test]
    fn compressed_self_loops_not_checked() {
        // Compressed may produce self-loops; we do NOT assert their absence.
        // Phase D handles loop repair.
        let s_out = vec![5u64; 3];
        let s_in = vec![5u64; 3];
        let domain = make_domain(3, false); // loopless domain
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let table =
            compressed_aggregated_matching(&s_out, &s_in, OccupationFamily::ME, &domain, &mut rng)
                .unwrap();
        verify_strengths(&table, &s_out, &s_in);
        // We DO NOT assert no self-loops — that's Phase D.
        let _has_self_loop = table.iter().any(|&((s, t), _)| s == t);
        let total: OccNum = table.iter().map(|(_, o)| o).sum();
        assert_eq!(total, 15);
        // Note: it's valid for self-loops to be present or absent — the
        // constructor is just a starting-state generator.
    }

    #[test]
    fn compressed_occupied_count_compact() {
        // Compressed should produce at most (N_active_src + N_active_tgt) pairs,
        // which is much smaller than O(N^2).
        let s_out = vec![3u64; 10];
        let s_in = vec![3u64; 10];
        let domain = make_domain(10, true);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let table =
            compressed_aggregated_matching(&s_out, &s_in, OccupationFamily::ME, &domain, &mut rng)
                .unwrap();
        let n_active_src = s_out.iter().filter(|&&s| s > 0).count();
        let n_active_tgt = s_in.iter().filter(|&&s| s > 0).count();
        assert!(
            table.len() <= n_active_src + n_active_tgt,
            "occupied pairs {} > active src + tgt {} (too many pairs)",
            table.len(),
            n_active_src + n_active_tgt
        );
        verify_strengths(&table, &s_out, &s_in);
    }
}
