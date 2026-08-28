use super::compressed::compressed_aggregated_matching;
use super::compressed::FlowTable;
use super::domain::PairDomain;
use super::errors::FixedStrengthError;
use crate::model::family::OccupationFamily;
use crate::OccNum;
use rand::Rng;

/// Construct an initial occupation table for a residual strength problem.
///
/// - **Complete domain**: compressed aggregated matching (spec §11–13).
///   Self-loops and B capacity violations are deferred to Phase D/E.
/// - **Sparse domain**: also compressed aggregated matching (no max-flow
///   needed — admissibility violations are repaired in Phase E).
/// - **CompleteMinus domain**: the constructor ignores domain
///   restrictions (including the excluded set); inadmissible mass on
///   excluded coordinates is repaired by the structural repair phase.
///
/// The caller's RNG is used directly (never reseeded internally), so
/// different top-level seeds produce different initial states and every
/// reconstruction restart continues the same RNG stream (§13.4).
pub fn initialize_table(
    strength_out: &[OccNum],
    strength_in: &[OccNum],
    family: OccupationFamily,
    domain: &PairDomain,
    rng: &mut impl Rng,
) -> Result<FlowTable, FixedStrengthError> {
    let total: OccNum = strength_out.iter().sum();
    if total == 0 {
        return Ok(Vec::new());
    }
    match domain {
        PairDomain::Complete { .. }
        | PairDomain::CompleteMinus { .. }
        | PairDomain::Sparse { .. } => {
            compressed_aggregated_matching(strength_out, strength_in, family, domain, rng)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::collections::HashSet;

    #[test]
    fn compressed_preserves_strengths() {
        let domain = PairDomain::Complete {
            node_count: 3,
            self_loops: false,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let table = compressed_aggregated_matching(
            &[5, 5, 5],
            &[5, 5, 5],
            OccupationFamily::ME,
            &domain,
            &mut rng,
        )
        .unwrap();
        assert_eq!(table.iter().map(|(_, o)| o).sum::<OccNum>(), 15);
        let (mut co, mut ci) = (vec![0u64; 3], vec![0u64; 3]);
        for &((s, t), o) in &table {
            co[s as usize] += o;
            ci[t as usize] += o;
        }
        assert_eq!(co, [5, 5, 5]);
        assert_eq!(ci, [5, 5, 5]);
    }

    #[test]
    fn sparse_domain_compressed_matching() {
        // Compressed matching does not enforce domain admissibility
        // (admissibility violations are repaired in Phase E).
        // Verify that strengths are preserved.
        let mut allowed = HashSet::new();
        allowed.insert((0, 1));
        allowed.insert((1, 0));
        allowed.insert((2, 2));
        let domain = PairDomain::Sparse {
            node_count: 3,
            allowed,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let table = initialize_table(
            &[4, 3, 2],
            &[3, 4, 2],
            OccupationFamily::ME,
            &domain,
            &mut rng,
        )
        .unwrap();
        assert_eq!(table.iter().map(|(_, o)| o).sum::<OccNum>(), 9);
        // Verify strengths are preserved.
        let (mut co, mut ci) = (vec![0u64; 3], vec![0u64; 3]);
        for &((s, t), o) in &table {
            co[s as usize] += o;
            ci[t as usize] += o;
        }
        assert_eq!(co, [4, 3, 2]);
        assert_eq!(ci, [3, 4, 2]);
    }

    #[test]
    fn complete_minus_domain_compressed_matching() {
        // The constructor ignores the excluded set (structural repair
        // handles excluded-coordinate mass); strengths must be exact.
        let mut excluded = HashSet::new();
        excluded.insert((0, 1));
        excluded.insert((2, 2));
        let domain = PairDomain::CompleteMinus {
            node_count: 3,
            self_loops: true,
            excluded,
        };
        let mut rng = StdRng::seed_from_u64(7);
        let table = initialize_table(
            &[4, 3, 2],
            &[3, 4, 2],
            OccupationFamily::ME,
            &domain,
            &mut rng,
        )
        .unwrap();
        assert_eq!(table.iter().map(|(_, o)| o).sum::<OccNum>(), 9);
        let (mut co, mut ci) = (vec![0u64; 3], vec![0u64; 3]);
        for &((s, t), o) in &table {
            co[s as usize] += o;
            ci[t as usize] += o;
        }
        assert_eq!(co, [4, 3, 2]);
        assert_eq!(ci, [3, 4, 2]);
    }

    #[test]
    fn zero_total() {
        let domain = PairDomain::Complete {
            node_count: 3,
            self_loops: true,
        };
        let mut rng = StdRng::seed_from_u64(42);
        assert!(
            initialize_table(&[0; 3], &[0; 3], OccupationFamily::ME, &domain, &mut rng)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn same_seed_reproducible() {
        let domain = PairDomain::Complete {
            node_count: 5,
            self_loops: true,
        };
        let s_out = [7u64, 5, 3, 9, 2];
        let s_in = [4u64, 8, 6, 3, 5];
        let run = |seed: u64| -> Vec<((u64, u64), OccNum)> {
            let mut rng = StdRng::seed_from_u64(seed);
            initialize_table(&s_out, &s_in, OccupationFamily::ME, &domain, &mut rng).unwrap()
        };
        assert_eq!(run(42), run(42), "same seed must reproduce the table");
    }

    #[test]
    fn different_seeds_construct_differently() {
        let domain = PairDomain::Complete {
            node_count: 5,
            self_loops: true,
        };
        let s_out = [7u64, 5, 3, 9, 2];
        let s_in = [4u64, 8, 6, 3, 5];
        let tables: Vec<Vec<((u64, u64), OccNum)>> = (1..=8u64)
            .map(|seed| {
                let mut rng = StdRng::seed_from_u64(seed);
                initialize_table(&s_out, &s_in, OccupationFamily::ME, &domain, &mut rng).unwrap()
            })
            .collect();
        let distinct = tables
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(
            distinct > 1,
            "different seeds should produce at least two distinct tables, got {distinct}"
        );
    }
}
