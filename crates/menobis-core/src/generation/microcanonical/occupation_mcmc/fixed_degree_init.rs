//! Direct exact fixed-(s,k) initialization (recovery plan Part B, §13–§24).
//!
//! Architecture (no MCMC, no stationarity required — §2, §13):
//!
//! ```text
//! exact-k binary support constructor (domain-aware, §16)
//!     -> occupation 1 on every support edge
//!     -> residual strengths r = s_out - k_out, c = s_in - k_in
//!     -> fast greedy residual allocation (§21)
//!          | success? ----- yes ---------------+
//!          no                                  |
//!          v                                   v
//!     sparse integer max-flow fallback      exact residual (s,k) state
//!     (§22) on the same support              with D = 0 by construction
//!          |
//!          feasible? ----- yes --------------+
//!          no (support incompatible with s)  |
//!          v                                 |
//!     construct another exact-k support -----+
//! ```
//!
//! One incompatible support is never treated as global infeasibility
//! (§14): the retry loop may find another exact-k support on which the
//! residual strengths do fit.  The constructor is combinatorial and
//! needs no detailed balance (§2, §31).
//!
//! Complexity (§35): state `O(E)`, flow graph `O(N+E)`, fixed-pair
//! exclusion `O(F)` (via the residual `PairDomain`), never an `N × N`
//! matrix and never an explicit complete admissible-pair vector.

use super::errors::FixedStrengthError;
use super::fixed_degrees::{degree_distance, ResidualDegreeTarget};
use super::problem::ResidualStrengthProblem;
use super::state::StrengthState;
use crate::generation::microcanonical::occupation_mcmc::domain::PairDomain;
use crate::model::family::OccupationFamily;
use crate::OccNum;
use rand::Rng;

/// Tuning knobs for the direct initializer (§17).  Performance/safety
/// limits only; never exposed through Python.
#[derive(Clone, Copy, Debug)]
pub struct ExactSkInitConfig {
    /// Maximum number of exact-k support construction attempts before
    /// giving up (support-first path).  Internal default 32.
    pub max_support_attempts: usize,
    /// Maximum slot-aware extras transport attempts (extras-first path,
    /// §20).  Internal default 64.
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
            max_support_attempts: 32,
            max_extras_attempts: 64,
            max_completion_attempts: 16,
            randomized_top_window: 8,
        }
    }
}

/// Diagnostics collected by one [`initialize_exact_sk`] call (§26).
/// The `support_first` fields are used by the legacy support-first
/// path; the `extras_first` fields by [`initialize_exact_sk_extras_first`]
/// (§34).  Once the extras-first architecture is selected, the obsolete
/// support-first fields are removed (plan Part I §60, §34).
#[derive(Clone, Debug, Default)]
pub struct ExactSkInitDiagnostics {
    /// Total support construction attempts consumed (support-first).
    pub support_attempts: usize,
    /// Attempts where the fast greedy allocation succeeded (support-first).
    pub greedy_allocation_successes: usize,
    /// Attempts where the sparse max-flow fallback ran (support-first).
    pub flow_fallback_attempts: usize,
    /// Supports discarded because the residual strengths could not be
    /// allocated (support-first) — **not** global infeasibility (§14).
    pub incompatible_supports: usize,
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

/// Direct exact fixed-(s,k) constructor (§18).
///
/// Constructs an exact-k binary support, sets one occupation on every
/// support edge, allocates the residual strengths `s − k` (greedy first,
/// sparse max-flow fallback), and retries with another support when the
/// current one is strength-incompatible.  The returned state satisfies
/// the residual strengths, the residual degrees, exact residual `E`,
/// domain admissibility, and (for B) family capacity, with
/// `degree_distance == 0` by construction.
///
/// # Errors
///
/// - [`ExactSkInitializationExhausted`](FixedStrengthError::ExactSkInitializationExhausted)
///   when the retry budget is exhausted without a feasible support —
///   this is **not** a proof that the target is infeasible (§14, §33).
#[allow(unused_assignments)] // best_flow is consumed by the exhausted error (§33)
pub fn initialize_exact_sk(
    problem: &ResidualStrengthProblem,
    degree_out: &[u32],
    degree_in: &[u32],
    rng: &mut impl Rng,
    config: &ExactSkInitConfig,
) -> Result<(StrengthState, ExactSkInitDiagnostics), FixedStrengthError> {
    let degree = ResidualDegreeTarget {
        out: degree_out.to_vec(),
        in_: degree_in.to_vec(),
        edge_count: degree_out.iter().map(|&k| k as usize).sum::<usize>(),
    };
    let n = problem.domain.node_count();
    let e_res = degree.edge_count;

    // Residual strengths r = s_out - k_out, c = s_in - k_in (§20).
    let mut r: Vec<OccNum> = Vec::with_capacity(n);
    let mut c: Vec<OccNum> = Vec::with_capacity(n);
    for i in 0..n {
        r.push(
            problem.strength_out[i]
                .checked_sub(degree.out[i] as OccNum)
                .ok_or_else(|| {
                    FixedStrengthError::InvalidResidual(format!(
                        "strength_out[{i}] = {} < degree_out[{i}] = {}",
                        problem.strength_out[i], degree.out[i]
                    ))
                })?,
        );
        c.push(
            problem.strength_in[i]
                .checked_sub(degree.in_[i] as OccNum)
                .ok_or_else(|| {
                    FixedStrengthError::InvalidResidual(format!(
                        "strength_in[{i}] = {} < degree_in[{i}] = {}",
                        problem.strength_in[i], degree.in_[i]
                    ))
                })?,
        );
    }
    let residual_total: OccNum = r.iter().sum();
    debug_assert_eq!(residual_total, c.iter().sum::<OccNum>());
    debug_assert_eq!(
        residual_total,
        problem.total - e_res as OccNum,
        "sum(r) == total strength - E"
    );

    // Per-edge extra capacity: B holds M - 1 extras per pair; ME/W use
    // the residual total so the capacity never binds (§22).
    let edge_cap: OccNum = match problem.family {
        OccupationFamily::B { layers } => (layers as OccNum).saturating_sub(1),
        _ => residual_total.max(1),
    };

    let mut diag = ExactSkInitDiagnostics {
        residual_total,
        ..ExactSkInitDiagnostics::default()
    };
    let mut best_flow: Option<OccNum> = None;

    for attempt in 0..config.max_support_attempts {
        diag.support_attempts = attempt + 1;

        // ---- Step 1: construct an exact residual-k support (§19) ----
        let support = match construct_exact_k_support(problem, &degree, rng) {
            Ok(s) => s,
            Err(_) => continue, // another attempt
        };

        // ---- Steps 2-3: residual allocation, greedy then flow ----
        let extras = match allocate_residual_greedy(&support, &r, &c, edge_cap) {
            Some(extras) => {
                diag.greedy_allocation_successes += 1;
                extras
            }
            None => {
                diag.flow_fallback_attempts += 1;
                match allocate_residual_flow(&support, &r, &c, edge_cap) {
                    (Some(extras), flow) => {
                        best_flow = Some(best_flow.map_or(flow, |b| b.max(flow)));
                        extras
                    }
                    (None, partial) => {
                        // Record the best *partial* flow too (§33): the
                        // error reports the closest a support got to a
                        // full allocation.
                        best_flow = Some(best_flow.map_or(partial, |b| b.max(partial)));
                        diag.incompatible_supports += 1;
                        continue; // another exact-k support
                    }
                }
            }
        };

        // ---- Step 4: build the state and validate every invariant ----
        let table: Vec<((u64, u64), OccNum)> = support
            .iter()
            .zip(extras.iter())
            .map(|(&(s, t), &y)| ((s, t), 1 + y))
            .collect();
        let state = StrengthState::new(n, table);
        validate_exact_sk_state(&state, problem, &degree)?;
        return Ok((state, diag));
    }

    Err(FixedStrengthError::ExactSkInitializationExhausted {
        support_attempts: config.max_support_attempts,
        best_flow: best_flow.unwrap_or(0),
        residual_total,
    })
}

/// Construct one exact residual-k support via the domain-aware binary
/// support initializer (§19).
fn construct_exact_k_support(
    problem: &ResidualStrengthProblem,
    degree: &ResidualDegreeTarget,
    rng: &mut impl Rng,
) -> Result<Vec<(u64, u64)>, FixedStrengthError> {
    let support = crate::generation::microcanonical::binary::initializer::
        greedy_directed_initialize_with_admissibility(
        &degree.out,
        &degree.in_,
        problem.domain.self_loops_allowed(),
        rng,
        |src, tgt| problem.domain.is_admissible(src, tgt),
    )
    .map_err(|e| FixedStrengthError::InitializationFailed(e.to_string()))?;

    let out_deg = support.out_degree_sequence();
    let edges = support.edges;
    // Post-conditions (§19): exact counts, exact degrees, admissible
    // pairs, no duplicates, loop policy (constructor guarantees them).
    debug_assert_eq!(edges.len(), degree.edge_count);
    debug_assert_eq!(out_deg, degree.out);
    let mut seen = std::collections::HashSet::with_capacity(edges.len());
    for &(s, t) in &edges {
        debug_assert!(
            problem.domain.is_admissible(s, t),
            "support pair must be admissible"
        );
        debug_assert!(problem.domain.self_loops_allowed() || s != t, "loop policy");
        debug_assert!(seen.insert((s, t)), "duplicate support edge");
    }
    Ok(edges)
}

/// Exact residual (s,k) invariant validation (§24) — cheap O(E) checks
/// run once per successful construction.
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

/// Fast greedy residual allocation (§21).
///
/// Heuristic: repeatedly take a positive-residual row with the fewest
/// currently usable support neighbors, prefer a column with larger
/// positive `c_j`, allocate `x = min(r_i, c_j, edge remaining capacity)`.
/// No backtracking (§21): on failure the caller falls back to the sparse
/// max-flow.  Returns per-edge extras `y_ij` or `None` if stuck.
fn allocate_residual_greedy(
    support: &[(u64, u64)],
    r: &[OccNum],
    c: &[OccNum],
    cap: OccNum,
) -> Option<Vec<OccNum>> {
    let n = r.len();
    let e = support.len();
    let mut row_rem = r.to_vec();
    let mut col_rem = c.to_vec();
    let mut extras = vec![0u64; e];
    let mut rem_cap = vec![cap; e];

    // Row -> edge indices; column -> edge indices (O(E) memory).
    let mut edges_by_row: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut edges_by_col: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (idx, &(s, t)) in support.iter().enumerate() {
        edges_by_row[s as usize].push(idx);
        edges_by_col[t as usize].push(idx);
    }

    // Usable neighbor count per row: edges (i, j) with c_j > 0 and spare.
    let mut usable: Vec<usize> = edges_by_row
        .iter()
        .map(|es| {
            es.iter()
                .filter(|&&e| col_rem[support[e].1 as usize] > 0)
                .count()
        })
        .collect();

    loop {
        // Pick the positive-residual row with the fewest usable neighbors.
        let mut best = None;
        for i in 0..n {
            if row_rem[i] == 0 {
                continue;
            }
            if usable[i] == 0 {
                return None; // stuck: positive residual is unreachable
            }
            match best {
                None => best = Some(i),
                Some(b) if usable[i] < usable[b] => best = Some(i),
                _ => {}
            }
        }
        let Some(i) = best else { break };

        // Prefer the usable neighbor with the largest positive c_j.
        let mut best_e: Option<usize> = None;
        let mut best_c = 0u64;
        for &e in &edges_by_row[i] {
            let j = support[e].1 as usize;
            if col_rem[j] == 0 || rem_cap[e] == 0 {
                continue;
            }
            if best_e.is_none() || col_rem[j] > best_c {
                best_e = Some(e);
                best_c = col_rem[j];
            }
        }
        let e = best_e?;

        let j = support[e].1 as usize;
        let x = row_rem[i].min(col_rem[j]).min(rem_cap[e]);
        debug_assert!(x > 0);
        extras[e] += x;
        row_rem[i] -= x;
        col_rem[j] -= x;
        rem_cap[e] -= x;

        let col_zeroed = col_rem[j] == 0;
        let edge_saturated = rem_cap[e] == 0;
        if col_zeroed {
            // All edges into j become unusable (those that still have spare).
            for &f in &edges_by_col[j] {
                if rem_cap[f] > 0 {
                    usable[support[f].0 as usize] -= 1;
                }
            }
        }
        if edge_saturated && !col_zeroed {
            // The saturated edge itself stops being usable for its row.
            usable[i] -= 1;
        }
    }

    debug_assert!(row_rem.iter().all(|&x| x == 0));
    debug_assert!(col_rem.iter().all(|&x| x == 0));
    Some(extras)
}

/// Sparse integer max-flow fallback (§22), Dinic, `O(N + E)` memory.
///
/// Flow graph: `source -> row i` (cap `r_i`), `row i -> col j` for
/// support edges only (cap `edge_cap`), `col j -> sink` (cap `c_j`).
/// Returns `(Some(extras), flow)` when `flow == Σ r_i`, else
/// `(None, flow)` with `flow` = the best partial transport achieved — the
/// support is strength-incompatible and the caller retries another one.
fn allocate_residual_flow(
    support: &[(u64, u64)],
    r: &[OccNum],
    c: &[OccNum],
    edge_cap: OccNum,
) -> (Option<Vec<OccNum>>, OccNum) {
    let n = r.len();
    let total: OccNum = r.iter().sum();
    if total == 0 {
        return (Some(vec![0; support.len()]), 0);
    }
    // Node ids: 0 = source, 1..n+1 rows, n+1..2n+1 cols, 2n+1 = sink.
    let src = 0usize;
    let sink = 2 * n + 1;
    let mut dinic = Dinic::new(2 * n + 2);
    let row_node = |i: usize| 1 + i;
    let col_node = |j: usize| 1 + n + j;
    for (i, &ri) in r.iter().enumerate() {
        if ri > 0 {
            dinic.add_edge(src, row_node(i), ri);
        }
    }
    for (j, &cj) in c.iter().enumerate() {
        if cj > 0 {
            dinic.add_edge(col_node(j), sink, cj);
        }
    }
    // Row -> col for support edges only, deterministic order (§34).
    // `(row_node, forward_index)` per support edge; flow on the forward
    // edge is read later from its reverse edge's capacity.
    let mut row_edges: Vec<(usize, usize)> = Vec::with_capacity(support.len());
    for &(s, t) in support {
        if r[s as usize] > 0 && c[t as usize] > 0 {
            let from = row_node(s as usize);
            let idx = dinic.graph[from].len();
            dinic.add_edge(from, col_node(t as usize), edge_cap);
            row_edges.push((from, idx));
        } else {
            row_edges.push((usize::MAX, usize::MAX));
        }
    }

    let flow = dinic.max_flow(src, sink);
    if flow < total {
        return (None, flow);
    }
    let mut extras = vec![0u64; support.len()];
    for (e, &(from, idx)) in row_edges.iter().enumerate() {
        if from == usize::MAX {
            continue;
        }
        let to = dinic.graph[from][idx].to;
        let rev = dinic.graph[from][idx].rev;
        // Pushed flow on the forward edge == reverse edge's capacity.
        extras[e] = dinic.graph[to][rev].cap;
    }
    (Some(extras), flow)
}

// ---------------------------------------------------------------------------
// Dinic max-flow (bounded integer capacities) — private (§22)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct FlowEdge {
    to: usize,
    rev: usize,
    cap: OccNum,
}

struct Dinic {
    graph: Vec<Vec<FlowEdge>>,
    level: Vec<i32>,
    it: Vec<usize>,
}

impl Dinic {
    fn new(n: usize) -> Self {
        Self {
            graph: vec![Vec::new(); n],
            level: vec![0; n],
            it: vec![0; n],
        }
    }

    fn add_edge(&mut self, from: usize, to: usize, cap: OccNum) {
        let rev_f = self.graph[to].len();
        let rev_t = self.graph[from].len();
        self.graph[from].push(FlowEdge {
            to,
            rev: rev_f,
            cap,
        });
        self.graph[to].push(FlowEdge {
            to: from,
            rev: rev_t,
            cap: 0,
        });
    }

    fn bfs(&mut self, s: usize, t: usize) -> bool {
        self.level.iter_mut().for_each(|l| *l = -1);
        let mut q = std::collections::VecDeque::new();
        self.level[s] = 0;
        q.push_back(s);
        while let Some(v) = q.pop_front() {
            for e in &self.graph[v] {
                if e.cap > 0 && self.level[e.to] < 0 {
                    self.level[e.to] = self.level[v] + 1;
                    q.push_back(e.to);
                }
            }
        }
        self.level[t] >= 0
    }

    fn dfs(&mut self, v: usize, t: usize, f: OccNum) -> OccNum {
        if v == t {
            return f;
        }
        for i in self.it[v]..self.graph[v].len() {
            self.it[v] = i;
            let to = self.graph[v][i].to;
            let cap = self.graph[v][i].cap;
            if cap > 0 && self.level[v] < self.level[to] {
                let pushed = self.dfs(to, t, f.min(cap));
                if pushed > 0 {
                    self.graph[v][i].cap -= pushed;
                    let rev = self.graph[v][i].rev;
                    self.graph[to][rev].cap += pushed;
                    return pushed;
                }
            }
        }
        0
    }

    fn max_flow(&mut self, s: usize, t: usize) -> OccNum {
        let mut flow = 0;
        const INF: OccNum = OccNum::MAX;
        while self.bfs(s, t) {
            self.it.iter_mut().for_each(|x| *x = 0);
            loop {
                let pushed = self.dfs(s, t, INF);
                if pushed == 0 {
                    break;
                }
                flow += pushed;
            }
        }
        flow
    }
}

// ---------------------------------------------------------------------------
// Extras-first prototype — plan Part A (§6–§22), Part B (§23–§29),
// Part C (§30–§35).  Kept separate from the legacy support-first
// `initialize_exact_sk` until the N=1000 gates pass (§7); the old path
// is not deleted while the new constructor is being validated.
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

    #[test]
    fn flow_trivial_one_edge_matches_residual() {
        // §23.1: one row, one column, residual 5, one support edge.
        let support = vec![(0u64, 1u64)];
        let (extras, flow) = allocate_residual_flow(&support, &[5, 0], &[0, 5], 1000);
        assert_eq!(flow, 5);
        assert_eq!(extras.unwrap(), vec![5]);
    }

    #[test]
    fn flow_solves_plan_decomposition_graph() {
        // §23.2: A->X, A->Y, B->X with r=(1,1), c=(1,1): flow must route
        // A->Y=1 and B->X=1 (even if a naive greedy could get stuck).
        let support = vec![(0u64, 0u64), (0, 1), (1, 0)];
        let (extras, flow) = allocate_residual_flow(&support, &[1, 1], &[1, 1], 1000);
        assert_eq!(flow, 2);
        // edges: (0,0)=A->X, (0,1)=A->Y, (1,0)=B->X
        assert_eq!(extras.unwrap(), vec![0, 1, 1], "A->Y=1, B->X=1");
    }

    #[test]
    fn flow_incompatible_support_returns_none() {
        // Row B has positive residual but no support edge: max flow < R.
        let support = vec![(0u64, 0u64)]; // only A->X
        let out = allocate_residual_flow(&support, &[1, 1], &[1, 1], 1000);
        assert!(out.0.is_none(), "flow must be < total residual");
    }

    #[test]
    fn flow_respects_b_capacity() {
        // §23.4: a single edge with M-1 = 1 cannot carry residual 2.
        let support = vec![(0u64, 1u64)];
        assert!(allocate_residual_flow(&support, &[2, 0], &[0, 2], 1)
            .0
            .is_none());
        let (extras, flow) = allocate_residual_flow(&support, &[2, 0], &[0, 2], 2);
        assert_eq!(flow, 2);
        assert_eq!(extras.unwrap(), vec![2]);
    }

    #[test]
    fn flow_extraction_reproduces_residuals() {
        // §23.5: extracted per-edge flows reproduce row/column residuals.
        let support = vec![(0u64, 0u64), (0, 1), (1, 0), (1, 1)];
        let r = vec![3, 4];
        let c = vec![2, 5];
        let (extras, flow) = allocate_residual_flow(&support, &r, &c, 1000);
        assert_eq!(flow, 7);
        let extras = extras.unwrap();
        let mut co = vec![0u64; 2];
        let mut ci = vec![0u64; 2];
        for (e, &(s, t)) in support.iter().enumerate() {
            co[s as usize] += extras[e];
            ci[t as usize] += extras[e];
        }
        assert_eq!(co, r);
        assert_eq!(ci, c);
    }

    #[test]
    fn greedy_trap_flow_fallback_succeeds() {
        // Greedy trap (allocation level): r=(2,2), c=(2,1,1) on support
        // A->X, A->Y, B->X, B->Z.  The greedy fills X from A entirely
        // (both rows have 2 usable neighbors; index tie goes to A), then
        // B is stranded (X saturated, Z too small).  Max-flow rescues it;
        // the full initializer must return the exact (s,k) state (§37.3)
        // whichever exact-k support the binary constructor builds.
        let so = vec![4, 4, 0]; // s = k + r, k=(2,2,0)
        let si = vec![3, 2, 3]; // s = k + c, k=(1,1,2)
        let problem = residual_problem(OccupationFamily::ME, so, si, true);
        let degree = degree_target(vec![2, 2, 0], vec![1, 1, 2]);
        for seed in [9u64, 10, 11, 12] {
            let mut rng = StdRng::seed_from_u64(seed);
            let (state, diag) = initialize_exact_sk(
                &problem,
                &degree.out,
                &degree.in_,
                &mut rng,
                &ExactSkInitConfig::default(),
            )
            .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
            validate_exact_sk_state(&state, &problem, &degree).unwrap();
            assert!(
                diag.greedy_allocation_successes + diag.flow_fallback_attempts >= 1,
                "seed {seed}: no allocation succeeded"
            );
            state.debug_validate();
        }
    }

    #[test]
    fn greedy_trap_allocation_level() {
        // Pure-allocation check of the same trap (§37.3): greedy stuck
        // (no backtracking), flow succeeds.
        let support = vec![(0u64, 0u64), (0, 1), (1, 0), (1, 2)];
        let r = vec![2, 2, 0];
        let c = vec![2, 1, 1];
        assert!(
            allocate_residual_greedy(&support, &r, &c, 1000).is_none(),
            "greedy must be stuck on this system"
        );
        let (extras, flow) = allocate_residual_flow(&support, &r, &c, 1000);
        assert_eq!(flow, 4);
        let extras = extras.unwrap();
        // Extraction must reproduce the residuals exactly (§23.5).
        let mut co = vec![0u64; 3];
        let mut ci = vec![0u64; 3];
        for (e, &(s, t)) in support.iter().enumerate() {
            co[s as usize] += extras[e];
            ci[t as usize] += extras[e];
        }
        assert_eq!(co, r);
        assert_eq!(ci, c);
    }

    #[test]
    fn greedy_solves_plan_decomposition_graph() {
        let support = vec![(0u64, 0u64), (0, 1), (1, 0)];
        let extras = allocate_residual_greedy(&support, &[1, 1], &[1, 1], 1000).unwrap();
        assert_eq!(extras, vec![0, 1, 1], "A->Y=1, B->X=1");
    }

    #[test]
    fn all_ones_state_residuals_zero() {
        // §37.1: s = k, residual total zero -> all-ones state.
        let so = vec![2u64, 2, 0];
        let si = vec![1u64, 1, 2];
        let problem = residual_problem(OccupationFamily::ME, so, si, true);
        let degree = degree_target(vec![2, 2, 0], vec![1, 1, 2]);
        let mut rng = StdRng::seed_from_u64(5);
        let (state, diag) = initialize_exact_sk(
            &problem,
            &degree.out,
            &degree.in_,
            &mut rng,
            &ExactSkInitConfig::default(),
        )
        .unwrap();
        assert_eq!(diag.residual_total, 0);
        assert!(state.iter_occupied().all(|(_, o)| o == 1));
        validate_exact_sk_state(&state, &problem, &degree).unwrap();
    }

    #[test]
    fn exhaustion_after_max_attempts() {
        // Strengths demand residual mass but k=(0,0,0) admits no support
        // edge, so every attempt's allocation fails: exhausted after the
        // retry budget (never labelled globally infeasible, §14/§33).
        let so = vec![1u64, 1, 0];
        let si = vec![1u64, 1, 0];
        let problem = residual_problem(OccupationFamily::ME, so, si, false);
        let degree = degree_target(vec![0, 0, 0], vec![0, 0, 0]);
        let mut rng = StdRng::seed_from_u64(2);
        match initialize_exact_sk(
            &problem,
            &degree.out,
            &degree.in_,
            &mut rng,
            &ExactSkInitConfig {
                max_support_attempts: 4,
                ..ExactSkInitConfig::default()
            },
        ) {
            Err(FixedStrengthError::ExactSkInitializationExhausted {
                support_attempts,
                best_flow,
                residual_total,
            }) => {
                assert_eq!(support_attempts, 4);
                assert_eq!(residual_total, 2);
                assert_eq!(best_flow, 0);
            }
            other => panic!("expected ExactSkInitializationExhausted, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Phase 6: tiny ME/B/W + fixed-pair initializer cases (§25, §37)
    // -----------------------------------------------------------------

    fn full_to_residual(
        family: OccupationFamily,
        so: Vec<OccNum>,
        si: Vec<OccNum>,
        sl: bool,
        fixed: Vec<(u64, u64, OccNum)>,
    ) -> (
        ResidualStrengthProblem,
        crate::generation::microcanonical::occupation_mcmc::fixed_degrees::ResidualDegreeTarget,
    ) {
        let n = so.len();
        let problem =
            crate::generation::microcanonical::occupation_mcmc::problem::FixedStrengthProblem::new(
                family,
                so,
                si,
                PairDomain::Complete {
                    node_count: n,
                    self_loops: sl,
                },
                fixed.clone(),
            )
            .unwrap();
        let residual = problem.into_residual().unwrap();
        use crate::generation::microcanonical::occupation_mcmc::fixed_degrees::residualize_degree_target;
        (
            residual,
            residualize_degree_target(&[2u32, 2, 2], &[2u32, 2, 2], &fixed)
                .map_err(|e| panic!("{e}"))
                .unwrap(),
        )
    }

    #[test]
    fn full_init_b_capacity_enforced() {
        // B M=3: occupations must stay ≤ 3 (extras ≤ M-1 = 2).
        // N=2 self-loops allowed, s=[3,3]/[3,3], k=(2,2)/(2,2).
        let problem = residual_problem(
            OccupationFamily::B { layers: 3 },
            vec![3, 3],
            vec![3, 3],
            true,
        );
        let degree = degree_target(vec![2, 2], vec![2, 2]);
        let mut rng = StdRng::seed_from_u64(4);
        let (state, _diag) = initialize_exact_sk(
            &problem,
            &degree.out,
            &degree.in_,
            &mut rng,
            &ExactSkInitConfig::default(),
        )
        .unwrap();
        for (_, o) in state.iter_occupied() {
            assert!(o <= 3, "B capacity violated: occ={o}");
        }
        validate_exact_sk_state(&state, &problem, &degree).unwrap();
    }

    #[test]
    fn full_init_w_family_exact() {
        // W M=2, N=3: no capacity cap, exact strengths/degrees.
        // s = k + r with r=(2,1,0), c=(1,2,0), feasible on a support
        // that routes row 1 into column Y (capacity 2).
        let so = vec![4u64, 2, 1];
        let si = vec![2u64, 4, 1];
        let problem = residual_problem(OccupationFamily::W { layers: 2 }, so, si, true);
        let degree = degree_target(vec![2, 1, 1], vec![1, 2, 1]);
        for seed in 1..=10u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let (state, _diag) = initialize_exact_sk(
                &problem,
                &degree.out,
                &degree.in_,
                &mut rng,
                &ExactSkInitConfig::default(),
            )
            .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
            validate_exact_sk_state(&state, &problem, &degree).unwrap();
        }
    }

    #[test]
    fn one_support_incompatible_another_compatible() {
        // §14/§37: the same degree sequence k_out=(1,2,0), k_in=(1,1,1)
        // admits two supports; one is strength-incompatible with
        // r=(0,3,0), c=(2,1,0), the other compatible — an incompatible
        // support is NOT global infeasibility (§14).
        let support_bad = vec![(0u64, 0u64), (1, 1), (1, 2)]; // A->X, B->Y, B->Z
        let support_good = vec![(0u64, 2u64), (1, 0), (1, 1)]; // A->Z, B->X, B->Y
        let r = vec![0u64, 3, 0];
        let c = vec![2u64, 1, 0];
        assert!(
            allocate_residual_flow(&support_bad, &r, &c, 1000)
                .0
                .is_none(),
            "B has no column with enough residual on the bad support"
        );
        let (extras, flow) = allocate_residual_flow(&support_good, &r, &c, 1000);
        assert_eq!(flow, 3, "B->X=2, B->Y=1");
        assert_eq!(extras.unwrap(), vec![0, 2, 1]);
    }

    #[test]
    fn positive_fixed_pair_via_residualization() {
        // §25/§37.6: fixed (0,1,1) on s=[3,3,3], k=(2,2,2)/(2,2,2) ->
        // residual s=[2,3,3]/[3,2,3], residual k=(1,2,2)/(2,1,2); the
        // residual initializer must never reoccupy (0,1).
        let (residual, degree) = full_to_residual(
            OccupationFamily::ME,
            vec![3, 3, 3],
            vec![3, 3, 3],
            true,
            vec![(0, 1, 1)],
        );
        let mut rng = StdRng::seed_from_u64(8);
        let (state, _diag) = initialize_exact_sk(
            &residual,
            &degree.out,
            &degree.in_,
            &mut rng,
            &ExactSkInitConfig::default(),
        )
        .unwrap();
        validate_exact_sk_state(&state, &residual, &degree).unwrap();
        assert!(
            !state.iter_occupied().any(|((s, t), _)| s == 0 && t == 1),
            "fixed positive pair must be excluded from the residual support"
        );
        assert_eq!(state.occupied_count(), 5, "residual E = 6 - 1 fixed pair");
    }

    #[test]
    fn zero_fixed_pair_keeps_coordinate_forbidden() {
        // §25/§37.7: fixed (0,1,0) subtracts nothing from strengths or
        // degrees but keeps the coordinate forbidden.
        let (residual, degree) = full_to_residual(
            OccupationFamily::ME,
            vec![3, 3, 3],
            vec![3, 3, 3],
            true,
            vec![(0, 1, 0)],
        );
        let mut rng = StdRng::seed_from_u64(13);
        let (state, _diag) = initialize_exact_sk(
            &residual,
            &degree.out,
            &degree.in_,
            &mut rng,
            &ExactSkInitConfig::default(),
        )
        .unwrap();
        validate_exact_sk_state(&state, &residual, &degree).unwrap();
        assert_eq!(
            state.occupied_count(),
            6,
            "zero fixed pair keeps residual E"
        );
        assert_eq!(state.out_strengths, vec![3, 3, 3]);
        assert!(
            !state.iter_occupied().any(|((s, t), _)| s == 0 && t == 1),
            "zero fixed pair coordinate must stay forbidden"
        );
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
