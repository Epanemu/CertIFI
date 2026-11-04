"""Validation, one-hot/group handling, and the shared search entry point.

All three public metrics (``msd``, ``spsf``, ``fpsf`` in :mod:`certifi_learn.exact._api`)
funnel through :func:`_run_core` here, which is the one place that talks to
the compiled Rust extension. Keeping this single implementation shared is
what the "one algorithm, three measures" corollary of the proof is really
about: the three public functions are thin, metric-specific wrappers that
compute the right accumulator weights and normalization constants, then
hand off to the same search.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .. import _core


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def _validate_binary_matrix(X, name: str = "X") -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {X.shape}")
    if X.size and not np.all((X == 0) | (X == 1)):
        raise ValueError(f"{name} must be binary (all entries in {{0, 1}})")
    return np.ascontiguousarray(X, dtype=np.uint8)


def _validate_binary_vector(v, n: int, name: str) -> np.ndarray:
    v = np.asarray(v).reshape(-1)
    if v.shape[0] != n:
        raise ValueError(
            f"{name} must have length {n} (the number of rows of X), got {v.shape[0]}"
        )
    if v.size and not np.all((v == 0) | (v == 1)):
        raise ValueError(f"{name} must be binary (all entries in {{0, 1}})")
    return np.ascontiguousarray(v, dtype=np.uint8)


def _validate_probability_vector(v, n: int, name: str) -> np.ndarray:
    """Like `_validate_binary_vector`, but accepts any value in [0, 1], not
    just exactly 0 or 1 -- for a "soft"/probabilistic prediction or
    membership score (e.g. a classifier's predicted probability, rather
    than its thresholded hard label). A hard 0/1 input is accepted
    unchanged (it's just the boundary of the allowed range), so this is a
    strict superset of the binary case: nothing that validated before
    stops validating now.
    """
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if v.shape[0] != n:
        raise ValueError(
            f"{name} must have length {n} (the number of rows of X), got {v.shape[0]}"
        )
    if v.size and not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must contain only finite values")
    if v.size and (np.any(v < 0.0) or np.any(v > 1.0)):
        raise ValueError(
            f"{name} must contain only values in [0, 1] (0/1 for a hard "
            "label, or a probability/soft score in between)"
        )
    return np.ascontiguousarray(v)


def _validate_weight(w, n: int, name: str = "sample_weight") -> np.ndarray:
    if w is None:
        return np.ones(n, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if w.shape[0] != n:
        raise ValueError(f"{name} must have length {n}, got {w.shape[0]}")
    if np.any(w < 0):
        raise ValueError(f"{name} must be non-negative")
    return np.ascontiguousarray(w)


def _normalize_group(g) -> list:
    if isinstance(g, (int, np.integer)):
        return [int(g)]
    return [int(c) for c in g]


def _validate_groups(n_features: int, groups) -> list:
    """Normalize `groups` to a list of list[int], and validate its
    structure (column indices in range, no column claimed by more than one
    group) up front -- before any statistical checks run against X -- so
    that a bad `groups` argument always fails with a clear, specific
    message rather than a downstream indexing error.
    """
    normalized = []
    if not groups:
        return normalized
    covered = set()
    for g in groups:
        cols = _normalize_group(g)
        for c in cols:
            if c < 0 or c >= n_features:
                raise ValueError(
                    f"group column index {c} out of range [0, {n_features})"
                )
            if c in covered:
                raise ValueError(f"column {c} appears in more than one group")
            covered.add(c)
        normalized.append(cols)
    return normalized


def _check_one_hot_groups(X: np.ndarray, groups: list) -> None:
    """Sanity-check that declared one-hot groups actually look one-hot
    (at most one indicator column set per row); catches the common mistake
    of grouping columns that were not actually one-hot encoded.

    `groups` must already be normalized/validated via `_validate_groups`.
    """
    for cols in groups:
        if len(cols) < 2:
            continue
        row_sums = X[:, cols].sum(axis=1)
        if np.any(row_sums > 1):
            raise ValueError(
                f"columns {cols} were declared as a one-hot group via `groups`, "
                "but at least one row has more than one of them set to 1"
            )


def _build_items(n_features: int, groups: list, allow_negations: bool) -> list:
    """Build the list of attributes (items), each a list of (col, val)
    literals, from the (already validated/normalized) `groups` declaration.

    * Columns not mentioned in `groups` are treated as independent plain
      binary attributes, whose domain is always {(c, 0), (c, 1)} -- a plain
      binary attribute's two values are equally "positive", so
      `allow_negations` does not apply to it.
    * Columns in a one-hot group [c1..ck] default to the categorical,
      positive-only domain [(c1, 1), ..., (ck, 1)] (choose at most one
      category). With `allow_negations=True`, the negated literals
      [(c1, 0), ..., (ck, 0)] ("not this category") are added too.
    """
    items = []
    covered = set()
    for cols in groups:
        covered.update(cols)
        lits = [(c, 1) for c in cols]
        if allow_negations:
            lits += [(c, 0) for c in cols]
        items.append(lits)
    for c in range(n_features):
        if c not in covered:
            items.append([(c, 0), (c, 1)])
    return items


def _items_to_arrays(items: Sequence[Sequence[tuple]]):
    offsets = [0]
    cols: list[int] = []
    vals: list[int] = []
    for lits in items:
        for c, v in lits:
            cols.append(c)
            vals.append(v)
        offsets.append(len(cols))
    return (
        np.asarray(offsets, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(vals, dtype=np.uint8),
    )


# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------


class Subgroup:
    """The winning subgroup found by the search: a conjunction of
    attribute-value literals, together with its violation value.
    """

    #: reasons a search can stop; only "optimal" carries the full
    #: certified-global-optimum guarantee proven for the algorithm.
    REASONS = ("optimal", "max_nodes", "time_limit", "max_depth")

    def __init__(
        self,
        violation: float,
        literals: list,
        feature_names=None,
        reason: str = "optimal",
        gamma=None,
    ):
        self.violation = float(violation)
        #: list of (column_index, value) pairs, value in {0, 1}
        self.literals = list(literals)
        #: one of Subgroup.REASONS; "optimal" unless a search limit bound.
        self.reason = reason
        #: True iff reason == "optimal": `violation` is then only the best
        #: value found (or the best within the depth cap), not necessarily
        #: the certified global optimum.
        self.exact = reason == "optimal"
        #: the `gamma` threshold passed in, if any (None otherwise). When
        #: `literals` is empty and `gamma` was set, `violation == gamma`:
        #: this is the threshold reflected back as a certified upper bound
        #: ("nothing exceeds gamma"), NOT a computed exact maximum -- exact
        #: values below gamma are never evaluated, by design.
        self.gamma = None if gamma is None else float(gamma)
        self._feature_names = feature_names

    def describe(self, feature_names=None) -> str:
        names = feature_names if feature_names is not None else self._feature_names
        if not self.literals:
            if self.gamma is not None:
                cond = f"<no subgroup with violation > {self.gamma:g} found>"
            else:
                cond = "<empty subgroup: no violation found>"
        else:
            parts = []
            for col, val in self.literals:
                if names is not None:
                    parts.append(f"{names[col]}={val}")
                else:
                    parts.append(f"x[{col}]={val}")
            cond = " AND ".join(parts)
        if not self.exact:
            explanation = {
                "max_nodes": "search truncated by max_nodes: not a certified optimum",
                "time_limit": "search truncated by time_limit: not a certified optimum",
                "max_depth": "search restricted by max_depth: optimal only among "
                "subgroups within that size",
            }[self.reason]
            cond += f" [{explanation}]"
        return cond

    def mask(self, X) -> np.ndarray:
        """Boolean membership mask of this subgroup over `X`."""
        X = np.asarray(X)
        m = np.ones(X.shape[0], dtype=bool)
        for col, val in self.literals:
            m &= X[:, col] == val
        return m

    def support(self, X, sample_weight=None) -> float:
        """(Weighted) number of rows of `X` satisfying this subgroup."""
        m = self.mask(X)
        if sample_weight is None:
            return float(m.sum())
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        return float(sample_weight[m].sum())

    def __repr__(self) -> str:
        return f"Subgroup(violation={self.violation:.6g}, condition='{self.describe()}')"


# --------------------------------------------------------------------------
# Shared search entry point
# --------------------------------------------------------------------------


def _run_core(
    X: np.ndarray,
    wa: np.ndarray,
    wb: np.ndarray,
    wc: np.ndarray,
    items: list,
    metric_code: int,
    feature_names=None,
    den_a: float = 0.0,
    den_b: float = 0.0,
    den: float = 0.0,
    p_base: float = 0.0,
    delta_bar: float = 0.0,
    max_nodes=None,
    time_limit=None,
    max_depth=None,
    gamma=None,
    min_coverage=None,
) -> Subgroup:
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive")
    if max_depth is not None and max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if gamma is not None and not np.isfinite(gamma):
        raise ValueError("gamma must be a finite number")
    if min_coverage is not None and (not np.isfinite(min_coverage) or min_coverage < 0):
        raise ValueError("min_coverage must be a finite, non-negative number")

    item_offsets, lit_cols, lit_vals = _items_to_arrays(items)
    X = np.ascontiguousarray(X, dtype=np.uint8)
    wa = np.ascontiguousarray(wa, dtype=np.float64)
    wb = np.ascontiguousarray(wb, dtype=np.float64)
    wc = np.ascontiguousarray(wc, dtype=np.float64)

    violation, lits, _nodes, reason = _core.search(
        X,
        item_offsets,
        lit_cols,
        lit_vals,
        wa,
        wb,
        wc,
        int(metric_code),
        float(den_a),
        float(den_b),
        float(den),
        float(p_base),
        float(delta_bar),
        None if max_nodes is None else int(max_nodes),
        None if time_limit is None else float(time_limit),
        None if max_depth is None else int(max_depth),
        None if gamma is None else float(gamma),
        None if min_coverage is None else float(min_coverage),
    )
    literals = [(int(c), int(v)) for c, v in lits]
    return Subgroup(
        violation=float(violation),
        literals=literals,
        feature_names=feature_names,
        reason=reason,
        gamma=gamma,
    )
