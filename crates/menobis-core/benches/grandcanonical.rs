//! Criterion benchmarks for the grand-canonical router.
//!
//! Measures end-to-end `sample_grandcanonical` throughput for every
//! supported constraint × family combination at N = 50 (≈2450 pairs).

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use rand::Rng;
use rand::SeedableRng;

use menobis_core::generation::grandcanonical::{sample_grandcanonical, GrandCanonicalCase};
use menobis_core::model::family::OccupationFamily;

const N: usize = 50;

fn random_f64s(len: usize, seed: u64) -> Vec<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..len)
        .map(|_| rng.random::<f64>().mul_add(0.8, 0.1))
        .collect()
}

fn make_xy(seed: u64) -> (Vec<f64>, Vec<f64>) {
    let x = random_f64s(N, seed);
    let y = random_f64s(N, seed.wrapping_add(1));
    (x, y)
}

/// Run one GC sample benchmark with the given parameter presets.
#[allow(clippy::too_many_arguments)]
fn bench_family(
    c: &mut Criterion,
    label: &str,
    family: OccupationFamily,
    case: GrandCanonicalCase,
    lam: Option<f64>,
    pi: Option<f64>,
    z: Option<Vec<f64>>,
    w: Option<Vec<f64>>,
    node_count: Option<usize>,
    q: Option<f64>,
    occ: Option<f64>,
) {
    let mut group = c.benchmark_group("gc");
    group.throughput(Throughput::Elements((N * N) as u64));
    group.bench_with_input(label, &(family, case), |b, &(fam, cs)| {
        let z_ref = z.as_deref();
        let w_ref = w.as_deref();
        b.iter_batched(
            || make_xy(42),
            |(x, y)| {
                let net = sample_grandcanonical(
                    cs, fam, &x, &y, lam, pi, z_ref, w_ref, None, None,
                    None, // gamma, coord_x, coord_y
                    node_count, q, occ, false, 42,
                )
                .unwrap();
                black_box(net.occ_nums.len())
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_strength(c: &mut Criterion) {
    for (fam, label) in [
        (OccupationFamily::ME, "strength/ME"),
        (OccupationFamily::B { layers: 5 }, "strength/B5"),
        (OccupationFamily::W { layers: 3 }, "strength/W3"),
    ] {
        bench_family(
            c,
            label,
            fam,
            GrandCanonicalCase::Strength,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
    }
}

fn bench_strength_edges(c: &mut Criterion) {
    for (fam, label) in [
        (OccupationFamily::ME, "strength_edges/ME"),
        (OccupationFamily::B { layers: 5 }, "strength_edges/B5"),
        (OccupationFamily::W { layers: 3 }, "strength_edges/W3"),
    ] {
        bench_family(
            c,
            label,
            fam,
            GrandCanonicalCase::StrengthEdges,
            Some(0.3),
            None,
            None,
            None,
            None,
            None,
            None,
        );
    }
}

fn bench_strength_degree(c: &mut Criterion) {
    let z = random_f64s(N, 99);
    let w = random_f64s(N, 101);
    for (fam, label) in [
        (OccupationFamily::ME, "strength_degree/ME"),
        (OccupationFamily::B { layers: 5 }, "strength_degree/B5"),
        (OccupationFamily::W { layers: 3 }, "strength_degree/W3"),
    ] {
        bench_family(
            c,
            label,
            fam,
            GrandCanonicalCase::StrengthDegree,
            None,
            None,
            Some(z.clone()),
            Some(w.clone()),
            None,
            None,
            None,
        );
    }
}

fn bench_degree_events(c: &mut Criterion) {
    for (fam, label) in [
        (OccupationFamily::ME, "degree_events/ME"),
        (OccupationFamily::B { layers: 5 }, "degree_events/B5"),
        (OccupationFamily::W { layers: 3 }, "degree_events/W3"),
    ] {
        bench_family(
            c,
            label,
            fam,
            GrandCanonicalCase::DegreeEvents,
            None,
            Some(0.5),
            None,
            None,
            None,
            None,
            None,
        );
    }
}

fn bench_edges_events(c: &mut Criterion) {
    for (fam, label) in [
        (OccupationFamily::ME, "edges_events/ME"),
        (OccupationFamily::B { layers: 5 }, "edges_events/B5"),
        (OccupationFamily::W { layers: 3 }, "edges_events/W3"),
    ] {
        bench_family(
            c,
            label,
            fam,
            GrandCanonicalCase::EdgesEvents,
            None,
            None,
            None,
            None,
            Some(N),
            Some(2.0),
            Some(10.0),
        );
    }
}

fn bench_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5))
        .sample_size(30)
}

criterion_group!(
    name = grandcanonical;
    config = bench_config();
    targets = bench_strength, bench_strength_edges, bench_strength_degree,
              bench_degree_events, bench_edges_events
);
criterion_main!(grandcanonical);
