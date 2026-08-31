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
use crate::model::family::OccupationFamily;
use crate::OccNum;
use rand::Rng;

/// Tuning knobs for the direct initializer (§17).  Performance/safety
/// limits only; never exposed through Python.
#[derive(Clone, Copy, Debug)]
pub struct ExactSkInitConfig {
    /// Maximum number of exact-k support construction attempts before
    /// giving up.  Internal default 32.
    pub max_support_attempts: usize,
}

impl Default for ExactSkInitConfig {
    fn default() -> Self {
        Self {
            max_support_attempts: 32,
        }
    }
}

/// Diagnostics collected by one [`initialize_exact_sk`] call (§26).
#[derive(Clone, Debug, Default)]
pub struct ExactSkInitDiagnostics {
    /// Total support construction attempts consumed.
    pub support_attempts: usize,
    /// Attempts where the fast greedy allocation succeeded.
    pub greedy_allocation_successes: usize,
    /// Attempts where the sparse max-flow fallback ran.
    pub flow_fallback_attempts: usize,
    /// Supports discarded because the residual strengths could not be
    /// allocated (greedy and flow both failed) — **not** global
    /// infeasibility (§14).
    pub incompatible_supports: usize,
    /// `Σ (s_out - k_out) == Σ (s_in - k_in)` (also `total - E`).
    pub residual_total: OccNum,
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
}
