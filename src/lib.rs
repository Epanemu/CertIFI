//! Core branch-and-bound search for the "subgroup with maximum violation"
//! problem, implementing the algorithm and bounds proven in
//! `subgroup_pruning_proof.tex` (canonical/order-restricted generation of
//! conjunctions of attribute-value literals, with an anti-monotone
//! optimistic bound used to prune whole subtrees).
//!
//! This module intentionally implements exactly the two bound formulas from
//! the proof:
//!   * MSD:          U(S) = max( P_D1(X_S), P_D2(X_S) )
//!   * SPSF / FPSF:  U(S) = P(X_S) * max(p_base, 1 - p_base)
//! via the `Metric::Msd` / `Metric::Linear` variants below (the corollary in
//! the proof notes that a single algorithm, parameterized by the (U, V)
//! pair, suffices for all three measures -- FPSF reuses the SPSF-style
//! linear bound because the Python layer restricts the population to
//! `y == 0` *before* the search ever sees the data).
//!
//! The heavy lifting (row filtering, sum accumulation, recursion) is done
//! on flat Rust slices with the GIL released, so it scales to the sample
//! counts typical of tabular fairness-auditing datasets.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::time::{Duration, Instant};

/// A single literal: "column `col` must equal `val`" (val in {0, 1}).
#[derive(Clone, Copy, Debug)]
struct Lit {
    col: usize,
    val: u8,
}

/// Flattened, order-restricted per-attribute literal domains. `offsets` has
/// length `n_items + 1`; the literals for attribute `i` (in the fixed
/// global attribute order used for canonical generation) are
/// `lits[offsets[i]..offsets[i+1]]`.
struct Items {
    offsets: Vec<usize>,
    lits: Vec<Lit>,
}

impl Items {
    fn n_items(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
    #[inline]
    fn lits_of(&self, idx: usize) -> &[Lit] {
        &self.lits[self.offsets[idx]..self.offsets[idx + 1]]
    }
}

/// The optimistic bound U(S) and exact value V(S) for one metric, expressed
/// in terms of the two accumulator sums (ca, cb) of a subgroup S:
///   * Msd:    ca = weight of X_S in D1, cb = weight of X_S in D2.
///   * Linear: ca = weight of X_S with h(x)=1, cb = weight of X_S with
///             h(x)=0 (restricted to y=0 beforehand, for FPSF).
#[derive(Clone, Copy)]
enum Metric {
    Msd {
        den_a: f64,
        den_b: f64,
    },
    Linear {
        den: f64,
        p_base: f64,
        delta_bar: f64,
    },
}

impl Metric {
    #[inline]
    fn bound(&self, ca: f64, cb: f64) -> f64 {
        match *self {
            Metric::Msd { den_a, den_b } => {
                let a = if den_a > 0.0 { ca / den_a } else { 0.0 };
                let b = if den_b > 0.0 { cb / den_b } else { 0.0 };
                a.max(b)
            }
            Metric::Linear { den, delta_bar, .. } => (ca + cb) / den * delta_bar,
        }
    }

    #[inline]
    fn value(&self, ca: f64, cb: f64) -> f64 {
        match *self {
            Metric::Msd { den_a, den_b } => {
                let a = if den_a > 0.0 { ca / den_a } else { 0.0 };
                let b = if den_b > 0.0 { cb / den_b } else { 0.0 };
                (a - b).abs()
            }
            Metric::Linear { den, p_base, .. } => {
                (ca * (1.0 - p_base) - cb * p_base).abs() / den
            }
        }
    }
}

/// Why the search stopped. `Optimal` is the only outcome carrying the
/// proof's exactness guarantee; the others are early-exit safety valves
/// (or a deliberate restriction of the search space, for `MaxDepth`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum StopReason {
    Optimal,
    MaxNodes,
    TimeLimit,
    MaxDepth,
}

impl StopReason {
    fn as_str(&self) -> &'static str {
        match self {
            StopReason::Optimal => "optimal",
            StopReason::MaxNodes => "max_nodes",
            StopReason::TimeLimit => "time_limit",
            StopReason::MaxDepth => "max_depth",
        }
    }
}

struct SearchState {
    tau: f64,
    best: Vec<Lit>,
    nodes: u64,
    /// Optional safety valves: if set, the search stops (returning the best
    /// subgroup found so far, NOT necessarily the certified global optimum)
    /// once this many nodes have been visited, or the deadline passes.
    /// `None` for both preserves the algorithm's exactness guarantee
    /// exactly as proven.
    max_nodes: Option<u64>,
    deadline: Option<Instant>,
    /// Maximum number of literals (rule size) a subgroup may have. This
    /// restricts the search space itself (Algorithm 1 is otherwise
    /// unchanged) rather than cutting an in-progress search short.
    max_depth: Option<usize>,
    /// Minimum required coverage (sum of `wc` over the subgroup's rows).
    /// Sound by the same coverage-monotonicity lemma the whole algorithm
    /// rests on (X_S' subset-eq X_S for S preceq S', so any measure of
    /// X_S is non-increasing as literals are added): once a subgroup's
    /// coverage drops below this floor, every descendant's coverage is
    /// also below it, so the whole subtree can be discarded outright.
    /// Like `max_depth`, this restricts the search space/problem
    /// definition rather than cutting an in-progress search short.
    min_coverage: Option<f64>,
    /// Set once `max_nodes` or the deadline is exceeded; short-circuits
    /// the remainder of the search (each `dfs` frame returns immediately).
    aborted_nodes: bool,
    aborted_time: bool,
}

impl SearchState {
    #[inline]
    fn aborted(&self) -> bool {
        self.aborted_nodes || self.aborted_time
    }

    /// The reason to report once the search has finished (whether that's
    /// natural completion or an early exit). `depth_restrictive` should be
    /// true iff `max_depth` was set to something less than the number of
    /// attributes, i.e. could plausibly have excluded the true optimum.
    fn stop_reason(&self, depth_restrictive: bool) -> StopReason {
        if self.aborted_time {
            StopReason::TimeLimit
        } else if self.aborted_nodes {
            StopReason::MaxNodes
        } else if depth_restrictive {
            StopReason::MaxDepth
        } else {
            StopReason::Optimal
        }
    }
}

/// Filter `rows` down to those where column `lit.col` equals `lit.val`,
/// returning the surviving row indices plus the accumulated (ca, cb, cc)
/// sums (`cc` is the coverage accumulator used for `min_coverage`,
/// independent of whichever metric-specific weighting `wa`/`wb` carry).
#[inline]
fn filter_and_sum(
    x: &[u8],
    n_cols: usize,
    wa: &[f64],
    wb: &[f64],
    wc: &[f64],
    rows: &[u32],
    lit: Lit,
) -> (Vec<u32>, f64, f64, f64) {
    let mut new_rows = Vec::with_capacity(rows.len());
    let mut ca = 0.0f64;
    let mut cb = 0.0f64;
    let mut cc = 0.0f64;
    for &r in rows {
        if x[r as usize * n_cols + lit.col] == lit.val {
            new_rows.push(r);
            ca += wa[r as usize];
            cb += wb[r as usize];
            cc += wc[r as usize];
        }
    }
    (new_rows, ca, cb, cc)
}

/// Depth-first branch and bound over the canonical generation order:
/// attributes are visited in a fixed order (index `idx`, `idx+1`, ...), and
/// each subgroup either fixes one literal of the current attribute (moving
/// deeper immediately, to raise `tau` fast) or skips it entirely and moves
/// on. Every subgroup is therefore generated by exactly one parent chain,
/// so pruning at a node suppresses every one of its supersets for free.
#[allow(clippy::too_many_arguments)]
fn dfs(
    idx: usize,
    rows: &[u32],
    x: &[u8],
    n_cols: usize,
    wa: &[f64],
    wb: &[f64],
    wc: &[f64],
    items: &Items,
    metric: &Metric,
    path: &mut Vec<Lit>,
    state: &mut SearchState,
) {
    if idx >= items.n_items() || state.aborted() {
        return;
    }
    // A subgroup may only grow up to `max_depth` literals; beyond that,
    // remaining attributes may only be skipped (never fixed). This
    // restricts the search space itself, so it is checked independently
    // of the node/time abort flags.
    let depth_ok = state.max_depth.map_or(true, |d| path.len() < d);
    if depth_ok {
        for &lit in items.lits_of(idx) {
            if state.aborted() {
                return;
            }
            let (new_rows, ca, cb, cc) = filter_and_sum(x, n_cols, wa, wb, wc, rows, lit);
            state.nodes += 1;
            if let Some(limit) = state.max_nodes {
                if state.nodes >= limit {
                    state.aborted_nodes = true;
                }
            }
            // Checking the wall clock on every node would add needless
            // syscall overhead on fast-pruning branches; every 1024 nodes
            // bounds the worst-case overshoot past the deadline tightly
            // enough in practice.
            if state.nodes % 1024 == 0 {
                if let Some(deadline) = state.deadline {
                    if Instant::now() >= deadline {
                        state.aborted_time = true;
                    }
                }
            }
            if let Some(min_cov) = state.min_coverage {
                if cc < min_cov {
                    // Coverage-monotonicity lemma: every descendant has
                    // coverage <= cc < min_cov, so none can be feasible.
                    continue;
                }
            }
            let b = metric.bound(ca, cb);
            if b <= state.tau {
                // Bound-transfer lemma: every descendant of this node has
                // V <= U(S) <= tau, so the whole subtree is pruned.
                continue;
            }
            let v = metric.value(ca, cb);
            path.push(lit);
            if v > state.tau {
                state.tau = v;
                state.best = path.clone();
            }
            dfs(
                idx + 1, &new_rows, x, n_cols, wa, wb, wc, items, metric, path, state,
            );
            path.pop();
        }
    }
    // Skip attribute `idx` entirely (no literal on it) and move on. This is
    // always allowed, even at the depth cap, since it adds no literal.
    dfs(idx + 1, rows, x, n_cols, wa, wb, wc, items, metric, path, state);
}

/// Reorder attributes (and the literals within each attribute) by
/// descending single-step "promise" (their bound against the full
/// population). Pure speed heuristic -- the proof shows correctness does
/// not depend on traversal/attribute order, only how quickly `tau` rises.
fn reorder_by_promise(
    items_in: Vec<Vec<Lit>>,
    x: &[u8],
    n_cols: usize,
    wa: &[f64],
    wb: &[f64],
    wc: &[f64],
    n: usize,
    metric: &Metric,
) -> Vec<Vec<Lit>> {
    let all_rows: Vec<u32> = (0..n as u32).collect();
    let mut scored_items: Vec<(f64, Vec<Lit>)> = items_in
        .into_iter()
        .map(|lits| {
            let mut scored: Vec<(f64, Lit)> = lits
                .into_iter()
                .map(|lit| {
                    let (_, ca, cb, _cc) = filter_and_sum(x, n_cols, wa, wb, wc, &all_rows, lit);
                    (metric.bound(ca, cb), lit)
                })
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let promise = scored.first().map(|(s, _)| *s).unwrap_or(0.0);
            (promise, scored.into_iter().map(|(_, l)| l).collect())
        })
        .collect();
    scored_items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    scored_items.into_iter().map(|(_, l)| l).collect()
}

#[pymodule]
mod _core {
    use super::*;

    /// Run the branch-and-bound search.
    ///
    /// Parameters mirror the flattened representation built by the Python
    /// layer:
    ///   * `x`: (n, n_cols) uint8 binary matrix.
    ///   * `item_offsets`: (n_items + 1,) int64, CSR-style offsets into
    ///     `lit_cols`/`lit_vals` delimiting each attribute's literal domain.
    ///   * `lit_cols`, `lit_vals`: flattened (column, required value) pairs.
    ///   * `wa`, `wb`: (n,) float64 per-sample accumulator weights.
    ///   * `wc`: (n,) float64 per-sample coverage weights, used only for
    ///     `min_coverage` (independent of the metric-specific `wa`/`wb`).
    ///   * `metric_code`: 0 = MSD (needs den_a, den_b), 1 = linear/SPSF-style
    ///     (needs den, p_base, delta_bar; also used for FPSF).
    ///
    /// `min_coverage`, if given, restricts the search to subgroups whose
    /// total `wc`-coverage is at least this much -- sound by the same
    /// coverage-monotonicity lemma the whole algorithm rests on, since
    /// coverage can only shrink as literals are added. Like `max_depth`,
    /// this restricts the search space/problem definition itself, so a
    /// search that completes under a binding `min_coverage` is still
    /// reported as "optimal" (it is the true optimum *of that restricted
    /// problem*), unlike `max_nodes`/`time_limit`.
    ///
    /// `max_nodes` / `time_limit_secs`, if given, cap the search as safety
    /// valves against the algorithm's exponential worst case; the search
    /// then returns the best subgroup found so far rather than the
    /// certified optimum. `max_depth`, if given, restricts the search
    /// space itself to subgroups with at most that many literals. Leave
    /// all three None to keep the full exactness guarantee.
    ///
    /// Returns `(violation, literals, nodes_visited, reason)` where
    /// `literals` is a list of `(column, value)` pairs describing the
    /// winning subgroup (in the order they were fixed), `nodes_visited` is
    /// a diagnostic counter (subgroups for which the bound was evaluated),
    /// and `reason` is one of "optimal", "max_nodes", "time_limit", or
    /// "max_depth" (only "optimal" carries the full certified-optimum
    /// guarantee).
    ///
    /// `gamma`, if given, seeds the running best-found value `tau` at
    /// `gamma` instead of 0 -- exactly Algorithm 1, just started from a
    /// higher initial threshold. The bound-transfer lemma's pruning rule
    /// ("prune if U(S) <= tau") is correct for *any* starting tau, since it
    /// only ever discards a subgroup S' once V(S') <= tau_at_prune_time <=
    /// tau_final (tau is non-decreasing), so nothing that could have beaten
    /// the final tau is ever discarded. Two outcomes result: if some
    /// subgroup exceeds gamma, the search still finds the true global max
    /// exactly (identical result to gamma=None/0, just fewer nodes
    /// visited); if none does, the search still runs to completion (when
    /// `reason == "optimal"`) having proven no subgroup exceeds gamma, and
    /// returns literals=[] with violation=gamma (the threshold itself,
    /// reflected back as a certified upper bound -- NOT a computed exact
    /// value, since exact values below gamma are never computed).
    #[pyfunction]
    #[pyo3(signature = (x, item_offsets, lit_cols, lit_vals, wa, wb, wc, metric_code, den_a, den_b, den, p_base, delta_bar, max_nodes=None, time_limit_secs=None, max_depth=None, gamma=None, min_coverage=None))]
    #[allow(clippy::too_many_arguments)]
    fn search(
        py: Python<'_>,
        x: PyReadonlyArray2<u8>,
        item_offsets: PyReadonlyArray1<i64>,
        lit_cols: PyReadonlyArray1<i64>,
        lit_vals: PyReadonlyArray1<u8>,
        wa: PyReadonlyArray1<f64>,
        wb: PyReadonlyArray1<f64>,
        wc: PyReadonlyArray1<f64>,
        metric_code: i64,
        den_a: f64,
        den_b: f64,
        den: f64,
        p_base: f64,
        delta_bar: f64,
        max_nodes: Option<u64>,
        time_limit_secs: Option<f64>,
        max_depth: Option<usize>,
        gamma: Option<f64>,
        min_coverage: Option<f64>,
    ) -> PyResult<(f64, Vec<(i64, u8)>, u64, String)> {
        let x_arr = x.as_array();
        let shape = x_arr.shape();
        let (n, n_cols) = (shape[0], shape[1]);
        // Flatten into a contiguous row-major buffer for fast raw indexing
        // in the hot loop (ndarray's `.iter()` always walks in logical
        // row-major order regardless of the underlying strides).
        let x_flat: Vec<u8> = x_arr.iter().cloned().collect();

        let wa_s = wa
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("wa must be contiguous: {e}")))?;
        let wb_s = wb
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("wb must be contiguous: {e}")))?;
        let wc_s = wc
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("wc must be contiguous: {e}")))?;
        if wa_s.len() != n || wb_s.len() != n || wc_s.len() != n {
            return Err(PyValueError::new_err(
                "wa/wb/wc length must match the number of rows of x",
            ));
        }

        let offsets_s = item_offsets
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("item_offsets must be contiguous: {e}")))?;
        let cols_s = lit_cols
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("lit_cols must be contiguous: {e}")))?;
        let vals_s = lit_vals
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("lit_vals must be contiguous: {e}")))?;
        if cols_s.len() != vals_s.len() {
            return Err(PyValueError::new_err(
                "lit_cols and lit_vals must have the same length",
            ));
        }

        let n_items = offsets_s.len().saturating_sub(1);
        let mut items_raw: Vec<Vec<Lit>> = Vec::with_capacity(n_items);
        for i in 0..n_items {
            let start = offsets_s[i];
            let end = offsets_s[i + 1];
            if start < 0 || end < start || end as usize > cols_s.len() {
                return Err(PyValueError::new_err("invalid item_offsets"));
            }
            let mut lits = Vec::with_capacity((end - start) as usize);
            for j in start as usize..end as usize {
                let col = cols_s[j];
                if col < 0 || col as usize >= n_cols {
                    return Err(PyValueError::new_err("literal column index out of range"));
                }
                lits.push(Lit {
                    col: col as usize,
                    val: vals_s[j],
                });
            }
            items_raw.push(lits);
        }

        let metric = match metric_code {
            0 => Metric::Msd { den_a, den_b },
            1 => Metric::Linear {
                den,
                p_base,
                delta_bar,
            },
            _ => return Err(PyValueError::new_err("unknown metric_code (expected 0 or 1)")),
        };

        if let Some(t) = time_limit_secs {
            if !(t > 0.0) {
                return Err(PyValueError::new_err("time_limit_secs must be positive"));
            }
        }
        if let Some(g) = gamma {
            if !g.is_finite() {
                return Err(PyValueError::new_err("gamma must be a finite number"));
            }
        }
        if let Some(mc) = min_coverage {
            if !mc.is_finite() || mc < 0.0 {
                return Err(PyValueError::new_err(
                    "min_coverage must be a finite, non-negative number",
                ));
            }
        }
        let depth_restrictive = max_depth.map_or(false, |d| d < n_items);
        let deadline = time_limit_secs.map(|t| Instant::now() + Duration::from_secs_f64(t));
        let tau0 = gamma.unwrap_or(0.0);

        let (tau, best, nodes, reason) = py.detach(move || {
            let items_sorted =
                reorder_by_promise(items_raw, &x_flat, n_cols, wa_s, wb_s, wc_s, n, &metric);
            let mut offsets = Vec::with_capacity(items_sorted.len() + 1);
            offsets.push(0usize);
            for it in &items_sorted {
                offsets.push(offsets.last().unwrap() + it.len());
            }
            let items = Items {
                offsets,
                lits: items_sorted.into_iter().flatten().collect(),
            };
            let all_rows: Vec<u32> = (0..n as u32).collect();
            let mut state = SearchState {
                tau: tau0,
                best: Vec::new(),
                nodes: 0,
                max_nodes,
                deadline,
                max_depth,
                min_coverage,
                aborted_nodes: false,
                aborted_time: false,
            };
            let mut path = Vec::new();
            dfs(
                0, &all_rows, &x_flat, n_cols, wa_s, wb_s, wc_s, &items, &metric, &mut path,
                &mut state,
            );
            let reason = state.stop_reason(depth_restrictive);
            (state.tau, state.best, state.nodes, reason)
        });

        let best_out: Vec<(i64, u8)> = best.iter().map(|l| (l.col as i64, l.val)).collect();
        Ok((tau, best_out, nodes, reason.as_str().to_string()))
    }
}
