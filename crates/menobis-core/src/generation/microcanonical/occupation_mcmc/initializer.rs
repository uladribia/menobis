use super::compressed::compressed_aggregated_matching;
use super::domain::PairDomain;
use super::errors::FixedStrengthError;
use crate::model::family::OccupationFamily;
use crate::OccNum;
use rand::SeedableRng;

/// Construct an initial occupation table for a residual strength problem.
///
/// - **Complete domain**: compressed aggregated matching (spec §11–13).
///   Self-loops and B capacity violations are deferred to Phase D/E.
/// - **Sparse domain**: also compressed aggregated matching (no max-flow
///   needed — admissibility violations are repaired in Phase E).
pub fn initialize_table(
    strength_out: &[OccNum],
    strength_in: &[OccNum],
    family: OccupationFamily,
    domain: &PairDomain,
) -> Result<super::feasibility::FlowTable, FixedStrengthError> {
    let total: OccNum = strength_out.iter().sum();
    if total == 0 {
        return Ok(Vec::new());
    }
    match domain {
        PairDomain::Complete { .. } | PairDomain::Sparse { .. } => {
            let mut rng = rand::rngs::StdRng::seed_from_u64(0);
            compressed_aggregated_matching(strength_out, strength_in, family, domain, &mut rng)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let table =
            initialize_table(&[4, 3, 2], &[3, 4, 2], OccupationFamily::ME, &domain).unwrap();
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
    fn zero_total() {
        let domain = PairDomain::Complete {
            node_count: 3,
            self_loops: true,
        };
        assert!(
            initialize_table(&[0; 3], &[0; 3], OccupationFamily::ME, &domain)
                .unwrap()
                .is_empty()
        );
    }
}
