//! Legacy fixed-strength constructors (test oracles).
//!
//! Kept from the previous production code for regression testing.
//! Production uses `compressed_aggregated_matching` instead.

use menobis_core::generation::microcanonical::occupation_mcmc::domain::PairDomain;
use menobis_core::model::family::OccupationFamily;
use menobis_core::OccNum;

/// Table of occupied node pairs.
type OccTable = Vec<((u64, u64), OccNum)>;

/// Greedy O(N²) construction for complete pair domains (legacy oracle).
///
/// Scans row-by-row, column-by-column.  Always succeeds for ME/W with
/// self-loops.  Enforces B per-cell capacity.
#[allow(
    clippy::needless_range_loop,
    reason = "Index-based loops preserve row-by-row, column-by-column greedy fill with early break on out[i]==0; iterator conversion would harm clarity"
)]
pub fn greedy_complete(
    strength_out: &[OccNum],
    strength_in: &[OccNum],
    family: OccupationFamily,
    domain: &PairDomain,
) -> Result<OccTable, String> {
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
        return Err("greedy fill failed: residual strengths remain".into());
    }

    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

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
            assert!(occ <= 2);
        }
        assert_eq!(table.iter().map(|(_, o)| o).sum::<OccNum>(), 5);
    }
}
