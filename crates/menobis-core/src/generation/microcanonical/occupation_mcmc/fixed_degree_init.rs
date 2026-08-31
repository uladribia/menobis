//! Direct exact fixed-(s,k) initialization — **extras-first** constructor
//! (plan Part A–C, §6–§35).
//!
//! Architecture (combinatorial, no MCMC and no stationarity required):
//!
//! ```text
//! r = s_out − k_out, c = s_in − k_in               (checked residuals §8)
//!     -> slot-aware compressed extras transport    (§10–§21)
//!        row_slots = k_out, col_slots = k_in; every positive
//!        extras edge consumes one slot per endpoint (§11);
//!        pressure(mass, slots) = ceil(mass / slots) rows/columns,
//!        block x = min(row_mass, col_mass, edge_cap) per coordinate,
//!        coordinate never reused (§16); deterministic attempt 0,
//!        bounded randomized retries (§19–§20)
//!     -> extras support degrees ≤ k                  (§23)
//!     -> delta = k − extras support degrees          (§24)
//!     -> occupation-1 binary completion of the      (§25–§28)
//!        missing support slots on domain − extras
//!     -> t = 1 + y on extras, t = 1 on fillers       (§30)
//!     -> independent table validation +             (§31–§32)
//!        StrengthState post-validation (D = 0)
//! ```
//!
//! The extras determine the hard row/column co-joint structure; exact
//! degrees are completed afterwards (§0).  This replaces the failed
//! support-first constructor (exact-k support then residual allocation),
//! whose co-joint extras transport does not fit a generic k-draw — see
//! `docs/decisions/microcanonical-fixed-sk-direct-init.md` (historical)
//! and `docs/decisions/microcanonical-fixed-sk-extras-first-init.md`
//! (current).  The constructor needs no detailed balance (§2, §31).
//!
//! Complexity: extras transport `O(N)` candidate scans per extras edge,
//! memory `O(N + B)`; completion reuses the binary initializer
//! (`O(N·d)`).  Never an `N × N` matrix and never an explicit complete
//! admissible-pair vector.

use super::errors::FixedStrengthError;
use super::fixed_degrees::{degree_distance, ResidualDegreeTarget};
use super::problem::ResidualStrengthProblem;
use super::state::StrengthState;
use crate::generation::microcanonical::occupation_mcmc::domain::PairDomain;
use crate::model::family::OccupationFamily;
use crate::OccNum;
use rand::Rng;

/// Tuning knobs for the extras-first direct initializer (§17, §20).
/// Performance/safety limits only; never exposed through Python.
#[derive(Clone, Copy, Debug)]
pub struct ExactSkInitConfig {
    /// Maximum slot-aware extras transport attempts (§20).  Internal
    /// default 64.
    pub max_extras_attempts: usize,
    /// Maximum binary completion retries per extras table (§26–§27).
    /// Internal default 16.
    pub max_completion_attempts: usize,
    /// Randomized top-window size for column picking on attempts > 0
    /// (§20).  Internal default 8.
    pub randomized_top_window: usize,
}

impl Default for ExactSkInitConfig {
    fn default() -> Self {
        Self {
            max_extras_attempts: 64,
            max_completion_attempts: 16,
            randomized_top_window: 8,
        }
    }
}

/// Diagnostics collected by one [`initialize_exact_sk_extras_first`]
/// call (§34).  The obsolete support-first fields (support attempts,
/// greedy/flow fallbacks, incompatible supports) were removed when the
/// extras-first architecture became the sole active initializer
/// (plan Part I §60).
#[derive(Clone, Debug, Default)]
pub struct ExactSkInitDiagnostics {
    /// `Σ (s_out - k_out) == Σ (s_in - k_in)` (also `total - E`).
    pub residual_total: OccNum,
    /// Total slot-aware extras transport attempts consumed (extras-first).
    pub extras_attempts: usize,
    /// Extras attempts that stranded positive mass + extras tables
    /// discarded because filler completion failed (§21, §27).
    pub extras_failed_attempts: usize,
    /// Positive-`y` extras edges in the final extras support `B` (§33).
    pub extras_edges: usize,
    /// Occupation-1 filler edges completing the missing degree slots
    /// (`C`).  Every filler is occupation 1 (§33).
    pub filler_edges: usize,
    /// Binary completion attempts across all kept extras tables (§26).
    pub completion_attempts: usize,
    /// Binary completion failures across all kept extras tables (§26).
    pub completion_failed_attempts: usize,
    /// Number of occupation-1 edges in the final state — equals
    /// `filler_edges` because extras edges carry `y >= 1` ⇒ occupation ≥ 2
    /// (§33).
    pub occupation_one_edges: usize,
    /// `occupation_one_edges / occupied_count` (§33).  Not a validity
    /// gate: a mathematically valid target is never rejected solely
    /// because this fraction is small (trace-mobility diagnostic, Gate A).
    pub occupation_one_fraction: f64,
}

fn validate_exact_sk_state(
    state: &StrengthState,
    problem: &ResidualStrengthProblem,
    degree: &ResidualDegreeTarget,
) -> Result<(), FixedStrengthError> {
    if state.out_strengths != problem.strength_out {
        return Err(FixedStrengthError::InvalidResidual(format!(
            "direct init strengths mismatch: out {:?} != {:?}",
            state.out_strengths, problem.strength_out
        )));
    }
    if state.in_strengths != problem.strength_in {
        return Err(FixedStrengthError::InvalidResidual(format!(
            "direct init strengths mismatch: in {:?} != {:?}",
            state.in_strengths, problem.strength_in
        )));
    }
    let k_out: Vec<usize> = degree.out.iter().map(|&k| k as usize).collect();
    let k_in: Vec<usize> = degree.in_.iter().map(|&k| k as usize).collect();
    if state.row_occ_count != k_out || state.col_occ_count != k_in {
        return Err(FixedStrengthError::InvalidDegreeTarget(
            "direct init degree mismatch".into(),
        ));
    }
    if state.occupied_count() != degree.edge_count {
        return Err(FixedStrengthError::InvalidDegreeTarget(format!(
            "direct init occupied {} != residual E {}",
            state.occupied_count(),
            degree.edge_count
        )));
    }
    let d = degree_distance(&state.row_occ_count, &state.col_occ_count, degree);
    if d != 0 {
        return Err(FixedStrengthError::InvalidDegreeTarget(format!(
            "direct init is not on the degree fiber: D = {d}"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Extras-first constructor — plan Part A (§6–§22), Part B (§23–§29),
// Part C (§30–§35).  This is the sole active initializer; the legacy
// support-first path (exact-k support then residual allocation) was
// removed in Part I — see the decision records for why it failed (§2.3-
// §2.4, `microcanonical-fixed-sk-direct-init.md`).
// ---------------------------------------------------------------------------

/// Residual extras `r = s_out − k_out`, `c = s_in − k_in` (§8).
///
/// Checked subtraction plus the **B M=1 invariant** (user decision, §9):
/// a Bernoulli family has per-pair occupations in `{0, 1}`, so per-node
/// strength must equal per-node degree for any realizable target.  This
/// is a model fact independent of the sampling ensemble (microcanonical
/// or grand canonical) and is rejected here — before any constructor
/// logic activates.
///
/// Also requires the plan's preconditions (§8): `sum(r) == sum(c)` and
/// `sum(k_out) == sum(k_in)`.
fn residual_extras(
    problem: &ResidualStrengthProblem,
    degree_out: &[u32],
    degree_in: &[u32],
) -> Result<(Vec<OccNum>, Vec<OccNum>, OccNum), FixedStrengthError> {
    let n = problem.strength_out.len();
    if degree_out.len() != n || degree_in.len() != n {
        return Err(FixedStrengthError::InvalidResidual(format!(
            "degree vectors must have length {n}, got out={} in={}",
            degree_out.len(),
            degree_in.len()
        )));
    }
    if matches!(problem.family, OccupationFamily::B { layers: 1 }) {
        for i in 0..n {
            if problem.strength_out[i] != degree_out[i] as OccNum {
                return Err(FixedStrengthError::InvalidResidual(format!(
                    "node {i}: B M=1 (Bernoulli) forces strength_out == degree_out, \
                     got {} != {}",
                    problem.strength_out[i], degree_out[i]
                )));
            }
            if problem.strength_in[i] != degree_in[i] as OccNum {
                return Err(FixedStrengthError::InvalidResidual(format!(
                    "node {i}: B M=1 (Bernoulli) forces strength_in == degree_in, \
                     got {} != {}",
                    problem.strength_in[i], degree_in[i]
                )));
            }
        }
    }
    let mut r = Vec::with_capacity(n);
    let mut c = Vec::with_capacity(n);
    for i in 0..n {
        r.push(
            problem.strength_out[i]
                .checked_sub(degree_out[i] as OccNum)
                .ok_or_else(|| {
                    FixedStrengthError::InvalidResidual(format!(
                        "strength_out[{i}] = {} < degree_out[{i}] = {}",
                        problem.strength_out[i], degree_out[i]
                    ))
                })?,
        );
        c.push(
            problem.strength_in[i]
                .checked_sub(degree_in[i] as OccNum)
                .ok_or_else(|| {
                    FixedStrengthError::InvalidResidual(format!(
                        "strength_in[{i}] = {} < degree_in[{i}] = {}",
                        problem.strength_in[i], degree_in[i]
                    ))
                })?,
        );
    }
    let r_total: OccNum = r.iter().sum();
    let c_total: OccNum = c.iter().sum();
    if r_total != c_total {
        return Err(FixedStrengthError::InvalidResidual(format!(
            "residual totals unbalanced: sum(r) = {r_total} != sum(c) = {c_total}"
        )));
    }
    let k_out: u64 = degree_out.iter().map(|&k| k as u64).sum();
    let k_in: u64 = degree_in.iter().map(|&k| k as u64).sum();
    if k_out != k_in {
        return Err(FixedStrengthError::InvalidResidual(format!(
            "degree sums unbalanced: sum(k_out) = {k_out} != sum(k_in) = {k_in}"
        )));
    }
    debug_assert_eq!(
        r_total,
        problem.total - k_out,
        "sum(r) == total strength − E"
    );
    Ok((r, c, r_total))
}

/// Integer pressure of residual mass against remaining support slots
/// (§12): `ceil(mass / slots)` with widened arithmetic.  Callers
/// guarantee `slots > 0` — positive mass with zero slots is a stranded
/// row/column and fails the attempt before pressure is computed (§11).
fn constructed_pressure(mass: OccNum, slots: usize) -> u64 {
    debug_assert!(slots > 0);
    let m = mass as u128;
    let s = slots as u128;
    m.div_ceil(s) as u64
}

/// Row ordering (§13): higher pressure first, then larger mass, then
/// fewer slots, then node index (deterministic tie-break).
fn row_precedes(a: usize, b: usize, row_mass: &[OccNum], row_slots: &[usize]) -> bool {
    let (pa, ma, sa) = (
        constructed_pressure(row_mass[a], row_slots[a]),
        row_mass[a],
        row_slots[a],
    );
    let (pb, mb, sb) = (
        constructed_pressure(row_mass[b], row_slots[b]),
        row_mass[b],
        row_slots[b],
    );
    pa > pb
        || (pa == pb && ma > mb)
        || (pa == pb && ma == mb && sa < sb)
        || (pa == pb && ma == mb && sa == sb && a < b)
}

/// Column picking for a fixed row (§14, §20).
///
/// Candidates satisfy: positive column mass, positive column slots,
/// domain admissibility, and no extras-coordinate reuse (§16).  They are
/// ranked by pressure desc, mass desc, slots asc, index asc.  Attempt 0
/// takes the top candidate (deterministic, §19); later attempts pick
/// uniformly from the best `min(window, candidates)` with the caller RNG
/// (reproducible per seed, no hidden RNGs, §20).  `None` means no
/// candidate — the row is stranded and the attempt fails (§21).
#[allow(clippy::too_many_arguments)] // conceptual args mandated by plan §6/§14
fn pick_column(
    row: usize,
    col_mass: &[OccNum],
    col_slots: &[usize],
    domain: &PairDomain,
    extras_set: &std::collections::HashSet<(u64, u64)>,
    attempt: usize,
    window: usize,
    rng: &mut impl Rng,
) -> Option<usize> {
    let n = col_mass.len();
    let mut ranked: Vec<usize> = (0..n)
        .filter(|&j| {
            col_mass[j] > 0
                && col_slots[j] > 0
                && domain.is_admissible(row as u64, j as u64)
                && !extras_set.contains(&(row as u64, j as u64))
        })
        .collect();
    if ranked.is_empty() {
        return None;
    }
    ranked.sort_by(|&a, &b| {
        let (pa, ma, sa) = (
            constructed_pressure(col_mass[a], col_slots[a]),
            col_mass[a],
            col_slots[a],
        );
        let (pb, mb, sb) = (
            constructed_pressure(col_mass[b], col_slots[b]),
            col_mass[b],
            col_slots[b],
        );
        pb.cmp(&pa)
            .then_with(|| mb.cmp(&ma))
            .then_with(|| sa.cmp(&sb))
            .then_with(|| a.cmp(&b))
    });
    if attempt == 0 {
        Some(ranked[0])
    } else {
        let top = ranked.len().min(window.max(1));
        let idx = rng.random_range(0..top);
        Some(ranked[idx])
    }
}

/// Outcome of one slot-aware extras transport attempt (positive-`y`
/// edges only; each coordinate appears at most once, §16).
struct ExtrasTable {
    /// `((src, tgt), y)` with `y >= 1`, no coordinate reuse.
    edges: Vec<((u64, u64), OccNum)>,
    /// Per-row count of positive extras edges (extras support degree).
    out_degree: Vec<usize>,
    /// Per-column count of positive extras edges.
    in_degree: Vec<usize>,
    /// Coordinate set (duplicate protection / filler disjointness).
    set: std::collections::HashSet<(u64, u64)>,
}

/// Slot-aware compressed extras transport (§10–§21).
///
/// Maintains residual masses and remaining support slots (initialized to
/// `k_out` / `k_in`); every new positive extras edge consumes exactly one
/// slot at each endpoint (§11).  Rows are pressure-ordered
/// deterministically (§13); the column is picked per §14/§20.  A block
/// `x = min(row_mass, col_mass, edge_extra_cap)` is allocated per
/// coordinate and the coordinate is never reused (§16).  Returns `None`
/// when positive residual mass becomes stranded (an endpoint's slots run
/// out or the domain leaves no candidate) — the attempt is abandoned and
/// the caller restarts from the original residuals (§21).
///
/// The first version uses `O(N)` candidate scans for row/column
/// selection (acceptable for the N=1000 correctness spike, §18);
/// `O(N²)` memory is never allocated (§17, §35).
#[allow(clippy::too_many_arguments)] // conceptual args mandated by plan §6/§10–§20
fn construct_extras_slot_aware(
    problem: &ResidualStrengthProblem,
    degree_out: &[u32],
    degree_in: &[u32],
    r: &[OccNum],
    c: &[OccNum],
    edge_extra_cap: OccNum,
    attempt: usize,
    window: usize,
    rng: &mut impl Rng,
) -> Option<ExtrasTable> {
    let n = problem.strength_out.len();
    let mut row_mass = r.to_vec();
    let mut col_mass = c.to_vec();
    let mut row_slots: Vec<usize> = degree_out.iter().map(|&k| k as usize).collect();
    let mut col_slots: Vec<usize> = degree_in.iter().map(|&k| k as usize).collect();
    let mut edges: Vec<((u64, u64), OccNum)> = Vec::new();
    let mut out_degree = vec![0usize; n];
    let mut in_degree = vec![0usize; n];
    let mut set: std::collections::HashSet<(u64, u64)> = std::collections::HashSet::new();

    loop {
        // Row selection (§13): active rows only (mass > 0, slots > 0).
        let mut best: Option<usize> = None;
        for i in 0..n {
            if row_mass[i] == 0 {
                continue;
            }
            if row_slots[i] == 0 {
                // §11 hard slot invariant: mass cannot pass through slots.
                return None;
            }
            best = Some(match best {
                None => i,
                Some(b) if row_precedes(i, b, &row_mass, &row_slots) => i,
                Some(b) => b,
            });
        }
        let Some(i) = best else { break }; // every row residual satisfied

        // Column selection (§14).  No candidate => stranded for this row.
        let j = pick_column(
            i,
            &col_mass,
            &col_slots,
            &problem.domain,
            &set,
            attempt,
            window,
            rng,
        )?;

        let x = row_mass[i].min(col_mass[j]).min(edge_extra_cap);
        debug_assert!(x > 0, "block allocation must be positive");
        edges.push(((i as u64, j as u64), x));
        set.insert((i as u64, j as u64));
        row_mass[i] -= x;
        col_mass[j] -= x;
        row_slots[i] -= 1;
        col_slots[j] -= 1;
        out_degree[i] += 1;
        in_degree[j] += 1;
    }

    debug_assert!(row_mass.iter().all(|&m| m == 0));
    debug_assert!(col_mass.iter().all(|&m| m == 0));
    Some(ExtrasTable {
        edges,
        out_degree,
        in_degree,
        set,
    })
}

/// Missing degree slots `delta = k − extras_support_degrees` (§23–§24).
///
/// Checked subtraction; any extras support degree above the target `k`
/// is a constructor bug (§23).  Requires `sum(delta_out) == sum(delta_in)
/// == target_E − extras_edges.len()` (§24).
fn missing_degree_slots(
    degree_out: &[u32],
    degree_in: &[u32],
    extras_out: &[usize],
    extras_in: &[usize],
    target_e: usize,
    extras_edges: usize,
) -> Result<(Vec<u32>, Vec<u32>), FixedStrengthError> {
    let n = degree_out.len();
    let mut delta_out = Vec::with_capacity(n);
    let mut delta_in = Vec::with_capacity(n);
    for i in 0..n {
        if extras_out[i] > degree_out[i] as usize {
            return Err(FixedStrengthError::InvalidResidual(format!(
                "extras out-degree {} exceeds target k_out {} (row {i})",
                extras_out[i], degree_out[i]
            )));
        }
        if extras_in[i] > degree_in[i] as usize {
            return Err(FixedStrengthError::InvalidResidual(format!(
                "extras in-degree {} exceeds target k_in {} (column {i})",
                extras_in[i], degree_in[i]
            )));
        }
        delta_out.push(degree_out[i] - extras_out[i] as u32);
        delta_in.push(degree_in[i] - extras_in[i] as u32);
    }
    let sum_out: usize = delta_out.iter().map(|&d| d as usize).sum();
    let sum_in: usize = delta_in.iter().map(|&d| d as usize).sum();
    let expected = target_e - extras_edges;
    if sum_out != sum_in || sum_out != expected {
        return Err(FixedStrengthError::InvalidResidual(format!(
            "delta degree sums unbalanced: {sum_out} out != {sum_in} in; expected {expected} (= E − extras)"
        )));
    }
    Ok((delta_out, delta_in))
}

/// Binary exact-degree completion of the missing support slots
/// (§25–§28).
///
/// Reuses the domain-aware binary support initializer on
/// `(delta_out, delta_in)` with an admissibility closure that
/// additionally excludes the extras support — a filler coordinate must
/// never coincide with an extras coordinate (§25).  Retries up to
/// `max_attempts` times with the advancing caller RNG (§26–§27); returns
/// `(None, attempts, failures)` if every attempt fails so the caller can
/// discard the extras table and start a new extras transport (§27).
///
/// Does **not** invoke the full fixed-`(k,T)` MCMC to obtain one filler
/// support (§26).
#[allow(clippy::type_complexity)]
fn complete_support_to_exact_k(
    problem: &ResidualStrengthProblem,
    extras_set: &std::collections::HashSet<(u64, u64)>,
    delta_out: &[u32],
    delta_in: &[u32],
    rng: &mut impl Rng,
    max_attempts: usize,
) -> (Option<Vec<(u64, u64)>>, usize, usize) {
    let mut attempts = 0usize;
    let mut failures = 0usize;
    let self_loops = problem.domain.self_loops_allowed();
    for _ in 0..max_attempts {
        attempts += 1;
        let is_admissible =
            |s: u64, t: u64| problem.domain.is_admissible(s, t) && !extras_set.contains(&(s, t));
        match crate::generation::microcanonical::binary::initializer::
            greedy_directed_initialize_with_admissibility(
                delta_out,
                delta_in,
                self_loops,
                rng,
                is_admissible,
            ) {
            Ok(support) => return (Some(support.edges), attempts, failures),
            Err(_) => failures += 1,
        }
    }
    (None, attempts, failures)
}

/// Final occupation table (§30): `t = 1 + y` on extras coordinates,
/// `t = 1` on filler coordinates; extras and fillers are disjoint by
/// construction (§25), so the table has no duplicates.
fn build_state_from_extras_and_fillers(
    extras: &[((u64, u64), OccNum)],
    fillers: &[(u64, u64)],
) -> Vec<((u64, u64), OccNum)> {
    let mut table = Vec::with_capacity(extras.len() + fillers.len());
    for &((s, t), y) in extras {
        debug_assert!(y >= 1);
        table.push(((s, t), 1 + y));
    }
    for &(s, t) in fillers {
        table.push(((s, t), 1));
    }
    table
}

/// Independent pre-state validation of the candidate table (§31).
///
/// Requires positive occupations, domain admissibility (which encodes
/// the loop policy), B per-pair capacity, and coordinate uniqueness.
/// Returns the occupied count.
fn validate_constructed_table(
    problem: &ResidualStrengthProblem,
    table: &[((u64, u64), OccNum)],
) -> Result<usize, FixedStrengthError> {
    let mut seen = std::collections::HashSet::with_capacity(table.len());
    let cap = match problem.family {
        OccupationFamily::B { layers } => Some(layers as OccNum),
        _ => None,
    };
    for &((s, t), occ) in table {
        if occ == 0 {
            return Err(FixedStrengthError::InvalidResidual(format!(
                "zero occupation at ({s}, {t})"
            )));
        }
        if !problem.domain.is_admissible(s, t) {
            return Err(FixedStrengthError::InvalidResidual(format!(
                "inadmissible pair ({s}, {t}) in constructed table"
            )));
        }
        if let Some(m) = cap {
            if occ > m {
                return Err(FixedStrengthError::InvalidResidual(format!(
                    "B capacity: occupation {occ} > M={m} at ({s}, {t})"
                )));
            }
        }
        if !seen.insert((s, t)) {
            return Err(FixedStrengthError::InvalidResidual(format!(
                "duplicate coordinate ({s}, {t}) in constructed table"
            )));
        }
    }
    Ok(table.len())
}

/// Prototype extras-first exact-(s,k) constructor (plan §6–§22,
/// §30–§35).
///
/// ```text
/// r = s_out − k_out, c = s_in − k_in          (§8)
///     -> slot-aware compressed extras y        (§10–§21)
///     -> extras support degrees ≤ k            (§23)
///     -> delta = k − extras support degrees    (§24)
///     -> occupation-1 filler support on domain minus extras (§25–§28)
///     -> t = 1 + y on extras, t = 1 on fillers (§30)
///     -> independent table validation (§31)
///     -> StrengthState + post-validation (§32)
/// ```
///
/// The extras determine the hard row/column co-joint structure; exact
/// degrees are completed afterwards (§0).  This is the replacement for
/// the failed support-first constructor; the legacy `initialize_exact_sk`
/// remains active until the N=1000 gates pass (§7).
///
/// # Errors
///
/// - [`ExactSkExtrasFirstExhausted`](FixedStrengthError::ExactSkExtrasFirstExhausted)
///   when every extras/completion retry fails (§35) — retry exhaustion,
///   **not** mathematical infeasibility (§21, §27).
/// - [`InvalidResidual`](FixedStrengthError::InvalidResidual) for
///   invalid residual targets, including the B M=1 invariant (§8–§9).
pub fn initialize_exact_sk_extras_first(
    problem: &ResidualStrengthProblem,
    degree_out: &[u32],
    degree_in: &[u32],
    rng: &mut impl Rng,
    config: &ExactSkInitConfig,
) -> Result<(StrengthState, ExactSkInitDiagnostics), FixedStrengthError> {
    // Residual extras (incl. the B M=1 invariant) before anything else
    // activates (§8, §9).
    let (r, c, residual_total) = residual_extras(problem, degree_out, degree_in)?;
    let degree = ResidualDegreeTarget {
        out: degree_out.to_vec(),
        in_: degree_in.to_vec(),
        edge_count: degree_out.iter().map(|&k| k as usize).sum::<usize>(),
    };
    let n = problem.strength_out.len();

    // Per-edge extra capacity (§9): B keeps at most M−1 extras per pair
    // (M=1 was rejected by `residual_extras`); ME/W use the residual
    // total so the cap never binds before an endpoint is exhausted.
    let edge_cap: OccNum = match problem.family {
        OccupationFamily::B { layers } => (layers as OccNum).saturating_sub(1),
        _ => residual_total.max(1),
    };

    let mut diag = ExactSkInitDiagnostics {
        residual_total,
        ..ExactSkInitDiagnostics::default()
    };

    for extras_attempt in 0..config.max_extras_attempts {
        diag.extras_attempts = extras_attempt + 1;

        // ---- Stage 1: slot-aware compressed extras transport (§10–§21).
        let Some(extras) = construct_extras_slot_aware(
            problem,
            degree_out,
            degree_in,
            &r,
            &c,
            edge_cap,
            extras_attempt,
            config.randomized_top_window,
            rng,
        ) else {
            diag.extras_failed_attempts += 1;
            continue; // restart from the original residuals (§21)
        };

        // ---- Missing degree slots (§23–§24); violations are bugs.
        let (delta_out, delta_in) = missing_degree_slots(
            degree_out,
            degree_in,
            &extras.out_degree,
            &extras.in_degree,
            degree.edge_count,
            extras.edges.len(),
        )?;

        // ---- Stage 2: occupation-1 completion (§25–§28): retry the
        // ---- binary constructor; if every attempt fails, discard this
        // ---- extras table and start a new transport (§27).
        let (fillers, completion_attempts, completion_failures) = complete_support_to_exact_k(
            problem,
            &extras.set,
            &delta_out,
            &delta_in,
            rng,
            config.max_completion_attempts,
        );
        diag.completion_attempts += completion_attempts;
        diag.completion_failed_attempts += completion_failures;
        let Some(fillers) = fillers else {
            diag.extras_failed_attempts += 1;
            continue;
        };

        // ---- Build + validate + post-validate (§30–§32).
        let table = build_state_from_extras_and_fillers(&extras.edges, &fillers);
        validate_constructed_table(problem, &table)?;
        let state = StrengthState::new(n, table);
        validate_exact_sk_state(&state, problem, &degree)?;

        diag.extras_edges = extras.edges.len();
        diag.filler_edges = fillers.len();
        // Extras carry y ≥ 1 ⇒ occupation ≥ 2, so the occupation-1
        // edges are exactly the fillers (§33).
        diag.occupation_one_edges = fillers.len();
        diag.occupation_one_fraction = fillers.len() as f64 / state.occupied_count() as f64;

        return Ok((state, diag));
    }

    Err(FixedStrengthError::ExactSkExtrasFirstExhausted {
        extras_attempts: config.max_extras_attempts,
        extras_failures: diag.extras_failed_attempts,
        completion_failures: diag.completion_failed_attempts,
        residual_total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::microcanonical::occupation_mcmc::domain::PairDomain;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn residual_problem(
        family: OccupationFamily,
        so: Vec<OccNum>,
        si: Vec<OccNum>,
        sl: bool,
    ) -> ResidualStrengthProblem {
        let total: OccNum = so.iter().sum();
        ResidualStrengthProblem {
            family,
            strength_out: so.clone(),
            strength_in: si,
            total,
            domain: PairDomain::Complete {
                node_count: so.len(),
                self_loops: sl,
            },
        }
    }

    fn degree_target(ko: Vec<u32>, ki: Vec<u32>) -> ResidualDegreeTarget {
        let edge_count: usize = ko.iter().map(|&k| k as usize).sum();
        ResidualDegreeTarget {
            out: ko,
            in_: ki,
            edge_count,
        }
    }

    // -----------------------------------------------------------------
    // Part A/B: extras-first construction tests (§22, §29)
    // -----------------------------------------------------------------

    /// Run the extras-first initializer and return (state, diag).
    fn extras_first(
        problem: &ResidualStrengthProblem,
        degree: &ResidualDegreeTarget,
        seed: u64,
    ) -> (StrengthState, ExactSkInitDiagnostics) {
        let mut rng = StdRng::seed_from_u64(seed);
        initialize_exact_sk_extras_first(
            problem,
            &degree.out,
            &degree.in_,
            &mut rng,
            &ExactSkInitConfig::default(),
        )
        .unwrap_or_else(|e| panic!("seed {seed}: {e}"))
    }

    /// §22.1: zero residual (`s == k`) → no extras, all-ones state.
    #[test]
    fn extras_first_zero_extras_s_equals_k() {
        let problem = residual_problem(OccupationFamily::ME, vec![2, 2, 0], vec![1, 1, 2], true);
        let degree = degree_target(vec![2, 2, 0], vec![1, 1, 2]);
        let (state, diag) = extras_first(&problem, &degree, 5);
        assert_eq!(diag.residual_total, 0);
        assert_eq!(diag.extras_edges, 0);
        assert!(state.iter_occupied().all(|(_, o)| o == 1), "all-ones state");
        validate_exact_sk_state(&state, &problem, &degree).unwrap();
    }

    /// §22.2: one extras edge carries the whole residual through a single
    /// column; the rest of the support becomes occupation-1 fillers.
    #[test]
    fn extras_first_one_extras_edge() {
        // r = (2,0,0), c = (2,0,0), k_out = k_in = (1,1,1), E = 3.
        let problem = residual_problem(OccupationFamily::ME, vec![3, 1, 1], vec![3, 1, 1], true);
        let degree = degree_target(vec![1, 1, 1], vec![1, 1, 1]);
        let (state, diag) = extras_first(&problem, &degree, 5);
        assert_eq!(diag.extras_edges, 1, "exactly one extras edge");
        assert_eq!(diag.filler_edges, 2);
        // The extras edge (0,0) carries occupation 3; the two fillers are
        // occupation 1.
        assert_eq!(state.get(0, 0), 3);
        assert_eq!(state.out_strengths, vec![3, 1, 1]);
        assert_eq!(state.in_strengths, vec![3, 1, 1]);
        validate_exact_sk_state(&state, &problem, &degree).unwrap();
    }

    /// §22.3: the highest-pressure row is paired with the highest-pressure
    /// column — strength-heavy rows couple to strength-heavy columns.
    #[test]
    fn extras_first_high_pressure_row_pairs_high_pressure_column() {
        // r = (5,1), c = (1,5), k_out = k_in = (1,1), E = 2.
        let problem = residual_problem(OccupationFamily::ME, vec![6, 2], vec![2, 6], true);
        let degree = degree_target(vec![1, 1], vec![1, 1]);
        let (state, _diag) = extras_first(&problem, &degree, 1);
        // Deterministic attempt 0: row 0 (mass 5, pressure 5) must take
        // column 1 (mass 5, pressure 5), not column 0 (mass 1).
        assert_eq!(state.get(0, 1), 6, "row 0 -> column 1 with y=5");
        assert_eq!(state.get(1, 0), 2, "row 1 -> column 0 with y=1");
        assert_eq!(state.get(0, 0), 0);
        assert_eq!(state.get(1, 1), 0);
        assert_eq!(state.out_strengths, vec![6, 2]);
        assert_eq!(state.in_strengths, vec![2, 6]);
        validate_exact_sk_state(&state, &problem, &degree).unwrap();
    }

    /// §22.4/§22.5: extras support degrees never exceed the target `k`
    /// on either side — checked directly on the extras table.
    #[test]
    fn extras_first_support_caps_never_exceed_k() {
        // r = (3,2,1), c = (2,2,2), k_out = k_in = (2,1,1), E = 4.
        let problem = residual_problem(OccupationFamily::ME, vec![5, 3, 2], vec![4, 3, 3], true);
        let degree = degree_target(vec![2, 1, 1], vec![2, 1, 1]);
        let (state, diag) = extras_first(&problem, &degree, 7);
        assert!(diag.extras_edges > 0);
        let mut extras_out = [0usize; 3];
        let mut extras_in = [0usize; 3];
        let mut extras_edges = 0usize;
        for ((s, t), o) in state.iter_occupied() {
            if o > 1 {
                extras_out[s as usize] += 1;
                extras_in[t as usize] += 1;
                extras_edges += 1;
            }
        }
        assert_eq!(extras_edges, diag.extras_edges);
        for i in 0..3 {
            assert!(
                extras_out[i] <= degree.out[i] as usize,
                "row {i}: extras out-degree {} > k_out {}",
                extras_out[i],
                degree.out[i]
            );
            assert!(
                extras_in[i] <= degree.in_[i] as usize,
                "col {i}: extras in-degree {} > k_in {}",
                extras_in[i],
                degree.in_[i]
            );
        }
        validate_exact_sk_state(&state, &problem, &degree).unwrap();
    }

    /// §22.6: B extras never exceed `M − 1` (capacity binds on a block).
    #[test]
    fn extras_first_b_extra_at_most_m_minus_1() {
        // B M=3 (cap 2): r = (3,0), c = (1,2), k_out = (2,0), k_in = (1,1).
        // Row 0 must push mass 3 through its 2 slots; column 1 takes a
        // block capped at M−1 = 2, column 0 the remaining 1.
        let problem = residual_problem(
            OccupationFamily::B { layers: 3 },
            vec![5, 0],
            vec![2, 3],
            true,
        );
        let degree = degree_target(vec![2, 0], vec![1, 1]);
        let (state, diag) = extras_first(&problem, &degree, 4);
        assert_eq!(diag.extras_edges, 2);
        assert_eq!(diag.filler_edges, 0, "extras fill all k slots");
        for ((s, t), o) in state.iter_occupied() {
            assert!(o <= 3, "B capacity violated at ({s},{t}): occ={o}");
            assert!(o >= 2, "extras edges carry occ >= 2");
        }
        // y = occ − 1 must stay <= M−1 = 2.
        assert!(state.get(0, 1) - 1 <= 2 && state.get(0, 0) - 1 <= 2);
        validate_exact_sk_state(&state, &problem, &degree).unwrap();
    }

    /// §22.7: loopless domain — no self-loop anywhere in the state.
    ///
    /// The residual `r = c = (1,1,1,1)` on a directed 4-cycle of degree
    /// sequence `(1,1,1,1)` is feasible only off the diagonal: each
    /// single-slot row/column pair must be a perfect matching avoiding
    /// `(i,i)`.  (Naive small loopless fixtures with `k_in = (1,1,2)`
    /// turned out genuinely infeasible: the last row's only residual
    /// column is its own diagonal — the constructor correctly reports
    /// exhaustion.)
    #[test]
    fn extras_first_loopless_domain() {
        let problem = residual_problem(
            OccupationFamily::ME,
            vec![2, 2, 2, 2],
            vec![2, 2, 2, 2],
            false,
        );
        let degree = degree_target(vec![1, 1, 1, 1], vec![1, 1, 1, 1]);
        let (state, diag) = extras_first(&problem, &degree, 7);
        assert!(diag.extras_edges > 0);
        assert_eq!(diag.extras_edges, 4, "one extras edge per row/column");
        assert!(
            state.iter_occupied().all(|((s, t), _)| s != t),
            "no self-loops"
        );
        validate_exact_sk_state(&state, &problem, &degree).unwrap();
    }

    /// §22.8: CompleteMinus exclusion — a forbidden coordinate is never
    /// occupied, whether as extras or as filler.
    #[test]
    fn extras_first_complete_minus_exclusion() {
        let domain = PairDomain::CompleteMinus {
            node_count: 3,
            self_loops: true,
            excluded: std::collections::HashSet::from([(0u64, 1u64), (1u64, 0u64)]),
        };
        let problem =
            crate::generation::microcanonical::occupation_mcmc::problem::ResidualStrengthProblem {
                family: OccupationFamily::ME,
                strength_out: vec![5, 3, 2],
                strength_in: vec![4, 3, 3],
                total: 10,
                domain,
            };
        let degree = degree_target(vec![2, 1, 1], vec![2, 1, 1]);
        let (state, diag) = extras_first(&problem, &degree, 7);
        assert!(diag.extras_edges > 0);
        assert!(
            !state
                .iter_occupied()
                .any(|((s, t), _)| (s, t) == (0, 1) || (s, t) == (1, 0)),
            "excluded coordinates must stay empty"
        );
        validate_exact_sk_state(&state, &problem, &degree).unwrap();
    }

    /// §22.9: same seed reproduces the identical state (deterministic
    /// attempt 0 and reproducible retry RNG).
    #[test]
    fn extras_first_reproducibility() {
        let problem = residual_problem(OccupationFamily::ME, vec![5, 3, 2], vec![4, 3, 3], true);
        let degree = degree_target(vec![2, 1, 1], vec![2, 1, 1]);
        let sort_occ = |state: &StrengthState| -> Vec<((u64, u64), OccNum)> {
            let mut v: Vec<_> = state.iter_occupied().collect();
            v.sort_unstable();
            v
        };
        let a = sort_occ(&extras_first(&problem, &degree, 99).0);
        let b = sort_occ(&extras_first(&problem, &degree, 99).0);
        assert_eq!(a, b, "same seed must reproduce the same state");
    }

    /// §22.10: retry diversity — with `attempt > 0`, column picking from
    /// the randomized top window must be seed-dependent but reproducible;
    /// attempt 0 stays deterministic.
    #[test]
    fn extras_first_retry_diversity_mechanism() {
        let problem = residual_problem(OccupationFamily::ME, vec![2, 2, 2], vec![2, 2, 2], true);
        let n = problem.domain.node_count();
        let domain = &problem.domain;
        let col_mass = vec![4u64, 4, 4];
        let col_slots = vec![2usize, 2, 2];
        let set = std::collections::HashSet::new();
        let rng_pick = |seed: u64, attempt: usize| -> usize {
            let mut rng = StdRng::seed_from_u64(seed);
            pick_column(0, &col_mass, &col_slots, domain, &set, attempt, 8, &mut rng).unwrap()
        };
        // Attempt 0 is deterministic: index tie-break picks column 0.
        assert_eq!(rng_pick(1, 0), 0);
        assert_eq!(rng_pick(2, 0), 0);
        // Attempts > 0 randomize within the top window: different seeds
        // can pick different columns, same seed reproduces.
        let picks: Vec<usize> = (1..=20).map(|s| rng_pick(s, 3)).collect();
        let distinct: usize = picks.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(
            distinct > 1,
            "expected retry diversity across seeds, got {picks:?}"
        );
        assert_eq!(rng_pick(7, 3), rng_pick(7, 3), "same seed reproducible");
        assert!(picks.iter().all(|&j| j < n));
    }

    /// §22 (mandatory): extras row/column sums independently restore the
    /// residuals `r` and `c` — computed from the final state, not from
    /// the internal table.
    #[test]
    fn extras_first_extras_margins_restore_residuals() {
        // r = (3,2,1), c = (2,2,2), k_out = k_in = (2,1,1).
        let problem = residual_problem(OccupationFamily::ME, vec![5, 3, 2], vec![4, 3, 3], true);
        let degree = degree_target(vec![2, 1, 1], vec![2, 1, 1]);
        let (state, _diag) = extras_first(&problem, &degree, 7);
        let mut row_mass = [0u64; 3];
        let mut col_mass = [0u64; 3];
        for ((s, t), o) in state.iter_occupied() {
            // extras mass = y = occ − 1; fillers (occ 1) contribute 0.
            row_mass[s as usize] += o - 1;
            col_mass[t as usize] += o - 1;
        }
        assert_eq!(row_mass, [3, 2, 1], "rows restore r");
        assert_eq!(col_mass, [2, 2, 2], "columns restore c");
    }

    /// B M=1 (Bernoulli) with `s != k`: rejected early with a clear
    /// error before any constructor logic activates (ensemble-independent
    /// invariant — user decision / §9).
    #[test]
    fn extras_first_b_m1_nonzero_residual_rejected_early() {
        let problem = residual_problem(
            OccupationFamily::B { layers: 1 },
            vec![2, 0],
            vec![2, 0],
            true,
        );
        let degree = degree_target(vec![1, 1], vec![1, 1]);
        let mut rng = StdRng::seed_from_u64(1);
        match initialize_exact_sk_extras_first(
            &problem,
            &degree.out,
            &degree.in_,
            &mut rng,
            &ExactSkInitConfig::default(),
        ) {
            Err(FixedStrengthError::InvalidResidual(_)) => {}
            other => panic!("expected early InvalidResidual for B M=1 s != k, got {other:?}"),
        }
        // The same target must also be rejected by the shared target
        // validation (validate_degree_target) before any construction.
        use crate::generation::microcanonical::occupation_mcmc::fixed_degrees::validate_degree_target;
        let residual = problem.clone();
        assert!(
            validate_degree_target(&residual, &degree).is_err(),
            "validate_degree_target must reject B M=1 with s != k"
        );
    }

    /// B M=1 with `s == k` is valid: zero residual, all-ones support.
    #[test]
    fn extras_first_b_m1_zero_residual_succeeds() {
        let problem = residual_problem(
            OccupationFamily::B { layers: 1 },
            vec![2, 2, 0],
            vec![1, 1, 2],
            true,
        );
        let degree = degree_target(vec![2, 2, 0], vec![1, 1, 2]);
        let (state, diag) = extras_first(&problem, &degree, 5);
        assert_eq!(diag.residual_total, 0);
        assert_eq!(diag.extras_edges, 0);
        assert!(state.iter_occupied().all(|(_, o)| o == 1));
        validate_exact_sk_state(&state, &problem, &degree).unwrap();
    }
}
