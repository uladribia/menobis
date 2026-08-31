//! Rust test-support port of the MENoBiS PA geographic generator.
//!
//! Original: `src/menobis/utilities/synthetic.py` (`generate_pa_geographic_network`,
//! `_preferential_support`, `_allocate_positive_integer_weights`).  Gate A
//! (fixed-(s,k) trace-from-exact-witness) requires a **proper** realistic
//! instance whose binary support has heterogeneous preferential-attachment
//! degrees and whose occupations follow the geographic score model — not the
//! older uniform random table.
//!
//! Three occupation patterns are supported (§8, §11):
//!
//! - [`OccupationPattern::Uniform`] — every support edge gets the same
//!   occupation `c` so `T/E = c` exactly (the plan's controlled `T/E`
//!   sensitivity cases);
//! - [`OccupationPattern::Balanced12`] — exactly half `1`s / half `2`s
//!   (`T/E = 1.5`); and
//! - [`OccupationPattern::PaGeographic`] — the realistic score allocation:
//!   extras spread multinomial-style with probabilities proportional to
//!   degree-attractiveness × distance-decay scores, `occ = 1 + extra`.
//!
//! Determinism: same `PaGeoConfig` + `OccupationPattern` + seed always
//! reproduces the identical support and witness (§34).  Coordinates and
//! occupations use a seed derived from `config.seed` so they never disturb
//! the support stream.  No hidden `thread_rng()`.

use std::collections::HashSet;

use menobis_core::OccNum;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

/// Generator configuration for one PA-geographic witness.
#[derive(Clone, Copy, Debug)]
pub struct PaGeoConfig {
    /// Number of nodes.
    pub node_count: usize,
    /// Target mean out-degree `d` (edge count ≈ `d · N`).
    pub average_degree: f64,
    /// Whether self-loops are allowed in the support.
    pub self_loops: bool,
    /// Seed driving the preferential-attachment RNG.
    pub seed: u64,
}

/// Occupation pattern applied on top of the PA support (§8, §11).
#[derive(Clone, Copy, Debug)]
pub enum OccupationPattern {
    /// Every support edge gets occupation `c` (so `T/E = c` exactly).
    Uniform(OccNum),
    /// The first `ceil(E/2)` support edges get occupation 2, the rest 1
    /// (so `T/E = 1.5` exactly, "roughly balanced").
    Balanced12,
    /// Realistic geographic-score allocation (port of
    /// `_allocate_positive_integer_weights`): extras are spread
    /// multinomial-style with probabilities proportional to
    /// degree-attractiveness × distance-decay scores, `occ = 1 + extra`,
    /// with total events `≈ events_per_edge · E`.
    PaGeographic { events_per_edge: f64 },
}

impl OccupationPattern {
    /// `T/E` realized by this pattern on any nonempty support.
    pub fn t_over_e(&self) -> f64 {
        match self {
            OccupationPattern::Uniform(c) => *c as f64,
            OccupationPattern::Balanced12 => 1.5,
            OccupationPattern::PaGeographic { events_per_edge } => *events_per_edge,
        }
    }
}

/// Binary support result with per-node degrees (needed for geo scores).
#[derive(Clone, Debug)]
pub struct PaGeoSupport {
    /// Occupied pairs in deterministic sorted order.
    pub edges: Vec<(u64, u64)>,
    /// Out-degree per node of the binary support.
    pub out_degree: Vec<u32>,
    /// In-degree per node of the binary support.
    pub in_degree: Vec<u32>,
}

/// One exact fixed-(s,k) witness and its derived constraints.
#[derive(Clone, Debug)]
pub struct PaGeoWitness {
    /// Node count.
    pub n: usize,
    /// Occupied pairs with positive occupations (the exact table).
    pub table: Vec<((u64, u64), OccNum)>,
    /// Out-strength sequence derived from the table.
    pub strength_out: Vec<OccNum>,
    /// In-strength sequence derived from the table.
    pub strength_in: Vec<OccNum>,
    /// Out-degree sequence derived from the table.
    pub degree_out: Vec<u32>,
    /// In-degree sequence derived from the table.
    pub degree_in: Vec<u32>,
}

impl PaGeoWitness {
    /// Number of occupied pairs `E`.
    pub fn e(&self) -> usize {
        self.table.len()
    }

    /// Total events `T = Σ t_ij`.
    pub fn total_events(&self) -> OccNum {
        self.table.iter().map(|&(_, o)| o).sum()
    }

    /// Fraction of occupied pairs with occupation 1.
    pub fn fraction_occupation_1(&self) -> f64 {
        let ones = self.table.iter().filter(|&&(_, o)| o == 1).count();
        ones as f64 / self.table.len().max(1) as f64
    }
}

/// Build the directed preferential-attachment support (port of
/// `_preferential_support`): a deterministic cycle seed then urn-style
/// preferential attachment from the accumulated out/in degree.
pub fn pa_geographic_support(config: &PaGeoConfig) -> PaGeoSupport {
    let n = config.node_count;
    let admissible = if config.self_loops {
        n * n
    } else {
        n * n.saturating_sub(1)
    };
    let edge_count = (config.average_degree * n as f64)
        .round()
        .min(admissible as f64) as usize;

    let mut rng = StdRng::seed_from_u64(config.seed);

    let mut edges: HashSet<(u64, u64)> = HashSet::with_capacity(edge_count);
    let mut out_degree = vec![0u32; n];
    let mut in_degree = vec![0u32; n];
    let mut source_urn: Vec<usize> = Vec::new();
    let mut target_urn: Vec<usize> = Vec::new();

    let add_edge = |edges: &mut HashSet<(u64, u64)>,
                    out_degree: &mut [u32],
                    in_degree: &mut [u32],
                    source_urn: &mut Vec<usize>,
                    target_urn: &mut Vec<usize>,
                    s: usize,
                    t: usize| {
        edges.insert((s as u64, t as u64));
        out_degree[s] += 1;
        in_degree[t] += 1;
        source_urn.push(s);
        target_urn.push(t);
    };

    // Deterministic cycle seed (every node starts with out-degree ≥ 1).
    for source in 0..n {
        if edges.len() >= edge_count {
            break;
        }
        let target = (source + 1) % n;
        if config.self_loops || source != target {
            add_edge(
                &mut edges,
                &mut out_degree,
                &mut in_degree,
                &mut source_urn,
                &mut target_urn,
                source,
                target,
            );
        }
    }

    // Urn-style preferential attachment.
    let max_attempts = 10_000usize.max(edge_count * 20);
    let mut attempts = 0usize;
    while edges.len() < edge_count && attempts < max_attempts {
        attempts += 1;
        let s = source_urn[rng.random_range(0..source_urn.len())];
        let t = target_urn[rng.random_range(0..target_urn.len())];
        if (!config.self_loops && s == t) || edges.contains(&(s as u64, t as u64)) {
            continue;
        }
        add_edge(
            &mut edges,
            &mut out_degree,
            &mut in_degree,
            &mut source_urn,
            &mut target_urn,
            s,
            t,
        );
    }

    // Fallback (only reachable when the urn phase stalls, e.g. very high
    // density): pick remaining pairs weighted by (out+1)(in+1) at source
    // and target, mirroring the Python complement loop.
    while edges.len() < edge_count {
        let mut remaining: Vec<(usize, usize)> = Vec::new();
        for s in 0..n {
            for t in 0..n {
                if (!config.self_loops && s == t) || edges.contains(&(s as u64, t as u64)) {
                    continue;
                }
                remaining.push((s, t));
            }
        }
        if remaining.is_empty() {
            break;
        }
        let weight = |s: usize, t: usize| {
            let out = out_degree[s] as f64 + 1.0;
            let inp = in_degree[t] as f64 + 1.0;
            out * inp
        };
        let total: f64 = remaining.iter().map(|&(s, t)| weight(s, t)).sum();
        let mut pick = rng.random::<f64>() * total;
        let mut chosen = remaining[0];
        for &(s, t) in &remaining {
            pick -= weight(s, t);
            chosen = (s, t);
            if pick <= 0.0 {
                break;
            }
        }
        add_edge(
            &mut edges,
            &mut out_degree,
            &mut in_degree,
            &mut source_urn,
            &mut target_urn,
            chosen.0,
            chosen.1,
        );
    }

    let mut ordered: Vec<(u64, u64)> = edges.iter().copied().collect();
    ordered.sort_unstable();
    PaGeoSupport {
        edges: ordered,
        out_degree,
        in_degree,
    }
}

/// Median of a positive slice (used as the distance scale, mirroring the
/// Python generator).
fn median_positive(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 1.0;
    }
    let mut v: Vec<f64> = values.iter().copied().filter(|&x| x > 0.0).collect();
    if v.is_empty() {
        return 1.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let mid = v.len() / 2;
    if v.len() % 2 == 0 {
        0.5 * (v[mid - 1] + v[mid])
    } else {
        v[mid]
    }
}

/// Build a full exact fixed-(s,k) witness: PA support + occupation
/// pattern, then derive the exact strength/degree constraints.
pub fn pa_geographic_witness(config: &PaGeoConfig, pattern: OccupationPattern) -> PaGeoWitness {
    let n = config.node_count;
    let sup = pa_geographic_support(config);
    let e = sup.edges.len();
    let mut table = Vec::with_capacity(e);
    match pattern {
        OccupationPattern::Uniform(c) => {
            for &(s, t) in &sup.edges {
                table.push(((s, t), c));
            }
        }
        OccupationPattern::Balanced12 => {
            // Support is in deterministic sorted order; the first
            // ceil(E/2) edges carry occupation 2, the rest 1.
            let half = e.div_ceil(2);
            for (i, &(s, t)) in sup.edges.iter().enumerate() {
                let occ = if i < half { 2 } else { 1 };
                table.push(((s, t), occ));
            }
        }
        OccupationPattern::PaGeographic { events_per_edge } => {
            // Coordinates on a derived stream (independent of support).
            let mut rng = StdRng::seed_from_u64(config.seed ^ 0x5A17_E5ED);
            let x: Vec<f64> = (0..n).map(|_| rng.random::<f64>()).collect();
            let y: Vec<f64> = (0..n).map(|_| rng.random::<f64>()).collect();
            let distances: Vec<f64> = sup
                .edges
                .iter()
                .map(|&(s, t)| (x[s as usize] - x[t as usize], y[s as usize] - y[t as usize]))
                .map(|(dx, dy)| dx.hypot(dy))
                .collect();
            let scale = median_positive(&distances);
            let decay = 2.0;
            let scores: Vec<f64> = sup
                .edges
                .iter()
                .zip(&distances)
                .map(|(&(s, t), &d)| {
                    let degree_part = (sup.out_degree[s as usize] as f64 + 1.0)
                        * (sup.in_degree[t as usize] as f64 + 1.0);
                    degree_part * (-decay * d / scale).exp()
                })
                .collect();
            // Cumulative probabilities + binary-search categorical draws
            // (O(E) build, O(log E) per extra unit) — a faithful
            // multinomial for `occ = 1 + extra`.
            let total: f64 = scores.iter().sum();
            let mut cum = Vec::with_capacity(e);
            let mut acc = 0.0;
            for s in &scores {
                acc += s / total;
                cum.push(acc);
            }
            let target_events = (events_per_edge * e as f64).round().max(e as f64) as u64;
            let extras = target_events - e as u64;
            let mut extra_count = vec![0u64; e];
            for _ in 0..extras {
                let u = rng.random::<f64>();
                let idx = cum.partition_point(|&c| c < u);
                extra_count[idx.min(e - 1)] += 1;
            }
            for (i, &(s, t)) in sup.edges.iter().enumerate() {
                table.push(((s, t), 1 + extra_count[i]));
            }
        }
    }

    let mut strength_out = vec![0u64; n];
    let mut strength_in = vec![0u64; n];
    let mut degree_out = vec![0u32; n];
    let mut degree_in = vec![0u32; n];
    for &((s, t), o) in &table {
        strength_out[s as usize] += o;
        strength_in[t as usize] += o;
        degree_out[s as usize] += 1;
        degree_in[t as usize] += 1;
    }
    PaGeoWitness {
        n,
        table,
        strength_out,
        strength_in,
        degree_out,
        degree_in,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_config_reproduces_same_support() {
        let cfg = PaGeoConfig {
            node_count: 200,
            average_degree: 8.0,
            self_loops: false,
            seed: 42,
        };
        let a = pa_geographic_support(&cfg);
        let b = pa_geographic_support(&cfg);
        assert_eq!(a.edges, b.edges);
        assert_eq!(a.out_degree, b.out_degree);
    }

    #[test]
    fn support_is_loopless_and_has_exact_degree_counts() {
        let cfg = PaGeoConfig {
            node_count: 200,
            average_degree: 8.0,
            self_loops: false,
            seed: 42,
        };
        let support = pa_geographic_support(&cfg);
        assert_eq!(support.edges.len(), 1600);
        let mut ko = vec![0u32; 200];
        let mut ki = vec![0u32; 200];
        for &(s, t) in &support.edges {
            assert_ne!(s, t, "no self-loops");
            ko[s as usize] += 1;
            ki[t as usize] += 1;
        }
        assert_eq!(ko.iter().sum::<u32>(), 1600);
        assert_eq!(ki.iter().sum::<u32>(), 1600);
        assert!(
            ko.iter().max().unwrap() > &12,
            "heterogeneous support expected"
        );
    }

    #[test]
    fn uniform_pattern_gives_exact_t_over_e_and_s_equals_c_times_k() {
        let cfg = PaGeoConfig {
            node_count: 200,
            average_degree: 8.0,
            self_loops: false,
            seed: 42,
        };
        for c in [1u64, 2, 3, 5, 10] {
            let w = pa_geographic_witness(&cfg, OccupationPattern::Uniform(c));
            assert_eq!(w.total_events(), c * w.e() as u64);
            let ones = w.table.iter().filter(|&&(_, o)| o == 1).count() as u64;
            assert_eq!(ones, if c == 1 { w.e() as u64 } else { 0 });
            for i in 0..w.n {
                assert_eq!(w.strength_out[i], c * w.degree_out[i] as u64);
                assert_eq!(w.strength_in[i], c * w.degree_in[i] as u64);
            }
        }
    }

    #[test]
    fn balanced12_gives_exact_1_5() {
        let cfg = PaGeoConfig {
            node_count: 200,
            average_degree: 8.0,
            self_loops: false,
            seed: 42,
        };
        let w = pa_geographic_witness(&cfg, OccupationPattern::Balanced12);
        let e = w.e();
        let twos = w.table.iter().filter(|&&(_, o)| o == 2).count();
        let ones = w.table.iter().filter(|&&(_, o)| o == 1).count();
        assert_eq!(twos, e.div_ceil(2));
        assert_eq!(ones, e - twos);
        assert!((w.total_events() as f64 / e as f64 - 1.5).abs() < 1e-9);
    }

    #[test]
    fn pa_geographic_pattern_is_realistic_and_exact() {
        let cfg = PaGeoConfig {
            node_count: 200,
            average_degree: 8.0,
            self_loops: false,
            seed: 42,
        };
        let w = pa_geographic_witness(
            &cfg,
            OccupationPattern::PaGeographic {
                events_per_edge: 8.0,
            },
        );
        assert_eq!(w.total_events(), (8.0 * w.e() as f64).round() as u64);
        // Realistic mixed occupations: some 1s, some larger values.
        let ones = w.table.iter().filter(|&&(_, o)| o == 1).count();
        assert!(ones > 0, "geo allocation must contain occupation-1 edges");
        assert!(
            w.table.iter().any(|&(_, o)| o >= 8),
            "geo allocation must have heavy edges"
        );
        // Exact constraints by construction (strengths/degrees derived).
        for i in 0..w.n {
            let so: u64 = w
                .table
                .iter()
                .filter(|&&((s, _), _)| s as usize == i)
                .map(|&(_, o)| o)
                .sum();
            assert_eq!(so, w.strength_out[i]);
        }
    }
}
