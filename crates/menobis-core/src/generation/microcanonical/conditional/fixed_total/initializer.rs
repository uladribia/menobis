//! Bounded feasible initializer for the fixed-total Gibbs chain.
//!
//! Constructs a valid positive occupation vector directly (never by
//! generating zeros and repairing them).  The initializer is allowed to
//! be biased because it is never emitted before burn-in.

use rand::rngs::StdRng;

use super::errors::FixedTotalError;
use crate::generation::microcanonical::support::uniform_edges::shuffle_slice;
use crate::model::family::OccupationFamily;
use crate::OccNum;

/// Build a balanced feasible occupation vector: `E` positive values
/// summing to `T`, respecting the family's maximum occupation (B cap).
///
/// Balanced fill: all cells start at `1`, the residual `R = T − E` is
/// distributed as `q = R / E` to every cell plus one extra to `R mod E`
/// randomly selected cells, then the vector is shuffled.
pub fn initialize_balanced(
    family: OccupationFamily,
    e: usize,
    t: OccNum,
    rng: &mut StdRng,
) -> Result<Vec<OccNum>, FixedTotalError> {
    if e == 0 {
        return if t == 0 {
            Ok(Vec::new())
        } else {
            Err(FixedTotalError::InvalidResidual(
                "zero edges but total occupation > 0".into(),
            ))
        };
    }
    if t < e as OccNum {
        return Err(FixedTotalError::InvalidResidual(format!(
            "total occupation {t} < edge count {e} (each edge needs ≥1 event)"
        )));
    }
    if let Some(max) = family.max_occupation() {
        let capacity = max.saturating_mul(e as OccNum);
        if t > capacity {
            return Err(FixedTotalError::InvalidResidual(format!(
                "total occupation {t} exceeds B capacity {max} × {e} = {capacity}"
            )));
        }
    }

    let residual = t - e as OccNum;
    let q = residual / e as OccNum;
    let rem = residual % e as OccNum;

    let mut occ = vec![1 + q; e];
    for v in occ.iter_mut().take(rem as usize) {
        *v += 1;
    }
    shuffle_slice(&mut occ, rng);
    Ok(occ)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    #[test]
    fn balanced_total() {
        let v = initialize_balanced(OccupationFamily::ME, 4, 10, &mut rng(1)).unwrap();
        assert_eq!(v.len(), 4);
        assert_eq!(v.iter().sum::<OccNum>(), 10);
        assert!(v.iter().all(|&t| t >= 1));
    }

    #[test]
    fn b_cap_respected() {
        // B(3), E=4, T=10 → R=6, q=1, rem=2 → values {2,2,3,3}
        let v = initialize_balanced(OccupationFamily::B { layers: 3 }, 4, 10, &mut rng(1)).unwrap();
        assert_eq!(v.iter().sum::<OccNum>(), 10);
        assert!(v.iter().all(|&t| (1..=3).contains(&t)));
    }

    #[test]
    fn t_equals_e_all_ones() {
        let v = initialize_balanced(OccupationFamily::ME, 5, 5, &mut rng(1)).unwrap();
        assert_eq!(v, vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn e_one() {
        let v = initialize_balanced(OccupationFamily::ME, 1, 7, &mut rng(1)).unwrap();
        assert_eq!(v, vec![7]);
    }

    #[test]
    fn e_zero() {
        assert_eq!(
            initialize_balanced(OccupationFamily::ME, 0, 0, &mut rng(1)).unwrap(),
            Vec::<OccNum>::new()
        );
        assert!(initialize_balanced(OccupationFamily::ME, 0, 3, &mut rng(1)).is_err());
    }

    #[test]
    fn invalid_t_below_e() {
        assert!(initialize_balanced(OccupationFamily::ME, 4, 3, &mut rng(1)).is_err());
    }

    #[test]
    fn invalid_b_overflow() {
        assert!(initialize_balanced(OccupationFamily::B { layers: 2 }, 3, 7, &mut rng(1)).is_err());
    }

    #[test]
    fn deterministic_same_seed() {
        let a = initialize_balanced(OccupationFamily::ME, 6, 20, &mut rng(42)).unwrap();
        let b = initialize_balanced(OccupationFamily::ME, 6, 20, &mut rng(42)).unwrap();
        assert_eq!(a, b);
    }
}
