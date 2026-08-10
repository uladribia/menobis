//! Legacy Dinic max-flow for fixed-strength feasibility checking.
//!
//! Strength feasibility is a capacitated transportation problem: find a
//! non-negative integer matrix with given row sums (out-strengths),
//! column sums (in-strengths), and per-cell upper bounds (family
//! capacity + domain restrictions).
//!
//! This module provides a Dinic max-flow implementation for exact
//! feasibility testing on restricted (non-complete) domains.  It is
//! retained as a **test oracle** for validating faster heuristics.
//!
//! ⚠ **Never import this in production code.** Always prefer the
//! compressed aggregated matching or other scalable algorithms.

use menobis_core::generation::microcanonical::occupation_mcmc::FlowTable;
use menobis_core::model::family::OccupationFamily;
use menobis_core::OccNum;

// ──────────────────────────────────────────────
// Dinic max-flow (bounded integer capacities)
// ──────────────────────────────────────────────

#[derive(Clone)]
struct Edge {
    to: usize,
    rev: usize,
    cap: OccNum,
}

/// Dinic max-flow on a directed graph with integer capacities.
struct Dinic {
    #[allow(dead_code)]
    n: usize,
    graph: Vec<Vec<Edge>>,
    level: Vec<i32>,
    it: Vec<usize>,
}

impl Dinic {
    fn new(n: usize) -> Self {
        Self {
            n,
            graph: vec![vec![]; n],
            level: vec![0; n],
            it: vec![0; n],
        }
    }

    fn add_edge(&mut self, from: usize, to: usize, cap: OccNum) {
        let rev_f = self.graph[to].len();
        let rev_t = self.graph[from].len();
        self.graph[from].push(Edge {
            to,
            rev: rev_f,
            cap,
        });
        self.graph[to].push(Edge {
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
            let e = self.graph[v][i].to; // cannot borrow mutably twice
            let cap = self.graph[v][i].cap;
            if cap > 0 && self.level[v] < self.level[e] {
                let pushed = self.dfs(e, t, f.min(cap));
                if pushed > 0 {
                    self.graph[v][i].cap -= pushed;
                    let rev = self.graph[v][i].rev;
                    self.graph[e][rev].cap += pushed;
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

// ──────────────────────────────────────────────
// Public feasibility API
// ──────────────────────────────────────────────

/// Check whether a residual fixed-strength problem is feasible and, if
/// so, construct an initial occupation table using max flow.
///
/// Builds a flow network:
///
/// ```text
/// source → out-nodes (cap = s_i^out)
/// out-nodes → in-nodes (cap = min(capacity, s_i^out, s_j^in)) for admissible pairs
/// in-nodes → sink (cap = s_j^in)
/// ```
pub fn feasibility_max_flow(
    strength_out: &[OccNum],
    strength_in: &[OccNum],
    _family: OccupationFamily,
    admissible_pairs: &[(u64, u64)],
    max_capacity: OccNum,
) -> Result<Option<FlowTable>, String> {
    let n = strength_out.len();
    let total: OccNum = strength_out.iter().sum();

    if total == 0 {
        return Ok(None);
    }

    // Node indices:
    // 0 = source
    // 1..n+1 = out-nodes
    // n+1..2n+1 = in-nodes
    // 2n+1 = sink
    let src = 0usize;
    let sink = 2 * n + 1;
    let node_count = 2 * n + 2;

    let mut dinic = Dinic::new(node_count);

    // Source → out-nodes
    for (i, &s) in strength_out.iter().enumerate() {
        if s > 0 {
            dinic.add_edge(src, 1 + i, s);
        }
    }

    // In-nodes → sink
    for (j, &s) in strength_in.iter().enumerate() {
        if s > 0 {
            dinic.add_edge(1 + n + j, sink, s);
        }
    }

    // Out-nodes → in-nodes for admissible pairs
    let out_node = |i: usize| 1 + i;
    let in_node = |j: usize| 1 + n + j;

    for &(s, t) in admissible_pairs {
        let i = s as usize;
        let j = t as usize;
        if i >= n || j >= n {
            continue;
        }
        let cap = max_capacity.min(strength_out[i]).min(strength_in[j]);
        if cap > 0 {
            dinic.add_edge(out_node(i), in_node(j), cap);
        }
    }

    // Run max flow.
    let flow = dinic.max_flow(src, sink);

    if flow != total {
        return Err(format!(
            "max flow {flow} < required total {total}: infeasible transportation problem"
        ));
    }

    // Extract the occupation table from reverse edges.
    let mut table = Vec::new();
    for &(s, t) in admissible_pairs {
        let i = s as usize;
        let j = t as usize;
        if i >= n || j >= n {
            continue;
        }
        // Find the edge out_i → in_j and read flow from the reverse edge.
        let u = out_node(i);
        for e in &dinic.graph[in_node(j)] {
            if e.to == u {
                // e is the reverse edge (in_j → out_i), whose cap holds
                // the flow on the forward edge.
                let occ = e.cap;
                if occ > 0 {
                    table.push(((s, t), occ));
                }
                break;
            }
        }
    }

    Ok(Some(table))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_2x2_feasible() {
        let out = vec![5u64, 5];
        let inp = vec![5u64, 5];
        let pairs = vec![(0, 0), (0, 1), (1, 0), (1, 1)];
        let result = feasibility_max_flow(&out, &inp, OccupationFamily::ME, &pairs, OccNum::MAX);
        assert!(result.is_ok());
        let table = result.unwrap().unwrap();
        let total: OccNum = table.iter().map(|(_, o)| o).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn infeasible_exceeds_row_capacity() {
        // Row 0 has total 10, but only 5 capacity per pair, and only 1 pair
        let out = vec![10u64, 0];
        let inp = vec![5u64, 5];
        let pairs = vec![(0, 0)]; // only (0,0) admissible
        let result = feasibility_max_flow(&out, &inp, OccupationFamily::ME, &pairs, OccNum::MAX);
        assert!(result.is_err());
    }

    #[test]
    fn b_capacity_respected() {
        // B(2): each cell max 2. Square 3x3 with one active row.
        let out = vec![5u64, 0, 0];
        let inp = vec![2u64, 2, 1]; // total = 5
        let pairs = vec![(0, 0), (0, 1), (0, 2)];
        let result = feasibility_max_flow(&out, &inp, OccupationFamily::B { layers: 2 }, &pairs, 2);
        assert!(result.is_ok());
        let table = result.unwrap().unwrap();
        for &(_, occ) in &table {
            assert!(occ <= 2);
        }
    }

    #[test]
    fn zero_total() {
        let out = vec![0u64; 3];
        let inp = vec![0u64; 3];
        let pairs = vec![];
        let result = feasibility_max_flow(&out, &inp, OccupationFamily::ME, &pairs, OccNum::MAX);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn single_pair() {
        let out = vec![7u64];
        let inp = vec![7u64];
        let pairs = vec![(0, 0)];
        let result = feasibility_max_flow(&out, &inp, OccupationFamily::ME, &pairs, OccNum::MAX);
        assert!(result.is_ok());
        let table = result.unwrap().unwrap();
        assert_eq!(table, vec![((0, 0), 7)]);
    }
}
