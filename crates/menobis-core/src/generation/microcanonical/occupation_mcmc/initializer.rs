use super::compressed::compressed_aggregated_matching;
use super::domain::PairDomain;
use super::errors::FixedStrengthError;
use super::feasibility::feasibility_max_flow;
use crate::model::family::OccupationFamily;
use crate::OccNum;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[allow(clippy::needless_range_loop)]
pub fn greedy_complete(
    strength_out: &[OccNum],
    strength_in: &[OccNum],
    family: OccupationFamily,
    domain: &PairDomain,
) -> Result<super::feasibility::FlowTable, FixedStrengthError> {
    let n = strength_out.len();
    let mut out = strength_out.to_vec();
    let mut inp = strength_in.to_vec();
    let cap = domain.capacity(family);
    let self_loops = domain.self_loops_allowed();
    let mut table = Vec::new();
    for i in 0..n {
        if out[i] == 0 {
            continue;
        }
        for j in 0..n {
            if inp[j] == 0 {
                continue;
            }
            if !self_loops && i == j {
                continue;
            }
            let assign = cap.min(out[i]).min(inp[j]);
            if assign > 0 {
                table.push(((i as u64, j as u64), assign));
                out[i] -= assign;
                inp[j] -= assign;
            }
            if out[i] == 0 {
                break;
            }
        }
    }
    if out.iter().any(|&s| s > 0) || inp.iter().any(|&s| s > 0) {
        return Err(FixedStrengthError::InitializationFailed(
            "greedy fill failed: residual strengths remain".into(),
        ));
    }
    Ok(table)
}

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
        PairDomain::Complete { .. } => {
            let mut rng = StdRng::from_os_rng();
            compressed_aggregated_matching(strength_out, strength_in, family, domain, &mut rng)
        }
        PairDomain::Sparse { .. } => {
            let cap = domain.capacity(family);
            let admissible: Vec<_> = domain.iter_admissible().collect();
            match feasibility_max_flow(strength_out, strength_in, family, &admissible, cap) {
                Ok(Some(table)) => Ok(table),
                Ok(None) => Ok(Vec::new()),
                Err(msg) => Err(FixedStrengthError::InitializationFailed(msg)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    #[test]
    fn greedy_simple_2x2() {
        let domain = PairDomain::Complete {
            node_count: 2,
            self_loops: true,
        };
        let table = greedy_complete(&[5, 5], &[5, 5], OccupationFamily::ME, &domain).unwrap();
        assert_eq!(table.iter().map(|(_, o)| o).sum::<OccNum>(), 10);
        assert_eq!(table.len(), 2);
    }
    #[test]
    fn greedy_heterogeneous() {
        let domain = PairDomain::Complete {
            node_count: 3,
            self_loops: true,
        };
        let table =
            greedy_complete(&[10, 0, 5], &[3, 7, 5], OccupationFamily::ME, &domain).unwrap();
        assert_eq!(table.iter().map(|(_, o)| o).sum::<OccNum>(), 15);
        let (mut co, mut ci) = (vec![0u64; 3], vec![0u64; 3]);
        for &((s, t), o) in &table {
            co[s as usize] += o;
            ci[t as usize] += o;
        }
        assert_eq!(co, [10, 0, 5]);
        assert_eq!(ci, [3, 7, 5]);
    }
    #[test]
    fn compressed_preserves_strengths() {
        let domain = PairDomain::Complete {
            node_count: 3,
            self_loops: false,
        };
        let mut rng = StdRng::seed_from_u64(42);
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
    fn sparse_domain_via_max_flow() {
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
        for &((s, t), _) in &table {
            assert!(domain.is_admissible(s, t));
        }
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
    #[test]
    fn greedy_n5000() {
        let n = 5000;
        let t: OccNum = 45000;
        let mut out = vec![t / n as OccNum; n];
        let mut inp = out.clone();
        let rem = t - out.iter().sum::<OccNum>();
        out[n - 1] += rem;
        inp[n - 1] += rem;
        let domain = PairDomain::Complete {
            node_count: n,
            self_loops: false,
        };
        let table = greedy_complete(&out, &inp, OccupationFamily::ME, &domain).unwrap();
        assert_eq!(table.iter().map(|(_, o)| o).sum::<OccNum>(), t);
    }
    #[test]
    fn b_capacity_greedy() {
        let domain = PairDomain::Complete {
            node_count: 3,
            self_loops: true,
        };
        let table = greedy_complete(
            &[5, 0, 0],
            &[2, 2, 1],
            OccupationFamily::B { layers: 2 },
            &domain,
        )
        .unwrap();
        for &(_, occ) in &table {
            assert!(occ <= 2, "B occ {occ} > 2");
        }
        assert_eq!(table.iter().map(|(_, o)| o).sum::<OccNum>(), 5);
    }
}
