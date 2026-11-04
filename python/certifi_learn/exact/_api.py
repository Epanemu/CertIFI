"""Public API: msd, spsf, fpsf.

Each function validates its inputs, computes the metric-specific
accumulator weights and normalization constants implied by the definitions
in the proof, and delegates the actual search to
:func:`certifi_learn.exact._common._run_core` -- a single shared implementation, per the
proof's "one algorithm, three measures" corollary. Keeping three thin,
explicitly-typed public wrappers (rather than one function with a
``metric=`` switch) avoids exposing arguments that only make sense for one
metric, e.g. ``y`` is only meaningful for FPSF.
"""

from __future__ import annotations

import numpy as np

from ._common import (
    Subgroup,
    _build_items,
    _check_one_hot_groups,
    _run_core,
    _validate_binary_matrix,
    _validate_binary_vector,
    _validate_groups,
    _validate_probability_vector,
    _validate_weight,
)

__all__ = ["msd", "spsf", "fpsf", "Subgroup"]


def msd(
    X,
    yhat,
    sample_weight=None,
    groups=None,
    allow_negations: bool = False,
    feature_names=None,
    max_nodes=None,
    time_limit=None,
    max_depth=None,
    gamma=None,
    min_coverage=None,
) -> Subgroup:
    """Maximum Subgroup Discrepancy between two groups.

    MSD(D1, D2; A) = sup over subgroups S of |P_D1(X_S) - P_D2(X_S)|.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Binary feature matrix (values in {0, 1}) covering both groups.
    yhat : array-like of shape (n_samples,)
        Splits the rows of X into the two groups being compared: D1 is
        weighted by `yhat`, D2 by `1 - yhat`. For a hard 0/1 `yhat` this is
        the usual disjoint split (D1 = rows where yhat == 1, D2 = rows
        where yhat == 0). Values strictly between 0 and 1 are also
        accepted, giving each row soft/fractional membership in both
        groups at once (e.g. a continuous group-membership score) --
        every formula and the bound-transfer pruning argument both still
        hold exactly (see `Notes on probabilistic yhat` below); this is a
        strict superset of the hard-label case, not a separate mode.
    sample_weight : array-like of shape (n_samples,), optional
        Non-negative weight per row (e.g. when a row represents multiple
        real samples). Defaults to all ones.
    groups : list, optional
        Declares one-hot-encoded categorical feature groups so the search
        does not combine incompatible literals (e.g. "color=red AND
        color=green"). Each element is either a single column index
        (an ordinary column, equivalent to leaving it out of `groups`) or a
        list/tuple of column indices that together one-hot-encode a single
        categorical attribute. Columns not mentioned are treated as
        independent plain binary attributes.
    allow_negations : bool, default False
        If True, one-hot groups may also be matched negatively (e.g.
        "color != red"), in addition to the default positive-only
        ("color == red") literals. Has no effect on plain (ungrouped)
        binary columns, whose two-valued domain already covers both senses.
    feature_names : sequence of str, optional
        Human-readable names for the columns of X, used when describing the
        winning subgroup.
    max_nodes : int, optional
        Safety valve against the algorithm's exponential worst case: if
        given, the search stops after visiting this many nodes and returns
        the best subgroup found so far. Leave as None (default) to keep
        the algorithm's exactness guarantee.
    time_limit : float, optional
        Safety valve, in seconds: if given, the search stops once this
        much wall-clock time has elapsed and returns the best subgroup
        found so far. Checked periodically (every ~1024 nodes), so the
        actual overshoot is small but non-zero. Leave as None (default) to
        keep the algorithm's exactness guarantee.
    max_depth : int, optional
        Restricts the search to subgroups with at most this many literals
        (rule size). Unlike `max_nodes`/`time_limit`, this changes the
        search space itself rather than cutting an in-progress search
        short: the search still runs to completion, but only over
        depth-bounded subgroups, so the result is optimal *among those*,
        not necessarily the unrestricted global optimum. Leave as None
        (default) to search subgroups of any size.
    gamma : float, optional
        Threshold of interest: if given, seeds the search's running best
        with `gamma` instead of 0, so any branch whose bound does not
        exceed `gamma` is pruned immediately -- typically much faster than
        an unconstrained search when you only care whether *some* subgroup
        beats a threshold, not the exact maximum below it. Two outcomes:
        if some subgroup's violation exceeds `gamma`, the search still
        finds the true global optimum exactly (identical result to
        `gamma=None`, just faster); if none does, the search still runs to
        completion (when `Subgroup.exact` is True) and returns
        `Subgroup.literals = []` with `Subgroup.violation = gamma` -- a
        certified upper bound ("no subgroup exceeds gamma"), NOT a
        computed exact value, since values below `gamma` are never
        evaluated. Leave as None (default) for an unconstrained search.
    min_coverage : float, optional
        Minimum required (weighted) number of rows the subgroup must cover
        -- counted over *all* rows of `X` regardless of `yhat` (i.e. the
        combined D1+D2 support), matching a typical "the subgroup must be
        big enough to matter" minimum-support constraint. Sound by the
        same coverage-monotonicity fact the whole algorithm relies on
        (coverage can only shrink as literals are added), so this
        restricts the search space rather than approximating: the search
        still runs to completion and finds the true optimum *among
        subgroups meeting the floor*. Leave as None (default, equivalent
        to `min_coverage=0`) for no minimum.

    `Subgroup.reason` reports which (if any) of `max_nodes`/`time_limit`/
    `max_depth` applied: one of "optimal", "max_nodes", "time_limit",
    "max_depth" (`Subgroup.exact` is a convenience alias for
    `reason == "optimal"`). If more than one limit is set and could apply,
    whichever one actually curtailed the search is reported (an exhausted
    node budget or deadline takes precedence over a depth cap that merely
    bounds what is being *certified*). `gamma` and `min_coverage` do not
    have their own reason -- like `groups`, they define which problem is
    being solved (what counts as an improvement, and which subgroups are
    even feasible), not why the search stopped; a search bounded only by
    `gamma`/`min_coverage` that runs to completion is still "optimal".

    Notes on probabilistic yhat
    ----------------------------
    The bound-transfer lemma the whole search relies on only ever needs
    two facts about the accumulator weights: they are non-negative, and
    they can only shrink (coverage-monotonicity) as literals restrict the
    row set further. Both hold for `wa = w * yhat` and
    `wb = w * (1 - yhat)` regardless of whether `yhat` is a hard 0/1 label
    or a continuous value in [0, 1] -- so the search remains exactly
    correct (same proof, same pruning rule, no approximation) with a
    probabilistic `yhat`; it exactly reduces to the disjoint-groups
    formula when `yhat` happens to be hard 0/1.

    Returns
    -------
    Subgroup
        The winning subgroup, with `.violation` set to the MSD value.
    """
    X = _validate_binary_matrix(X)
    n, n_features = X.shape
    yhat = _validate_probability_vector(yhat, n, "yhat")
    w = _validate_weight(sample_weight, n)
    groups = _validate_groups(n_features, groups)
    _check_one_hot_groups(X, groups)

    wa = w * yhat  # (soft) weight in D1
    wb = w * (1.0 - yhat)  # (soft) weight in D2
    den_a = float(wa.sum())
    den_b = float(wb.sum())
    if den_a <= 0.0 or den_b <= 0.0:
        raise ValueError(
            "both D1 (weighted by yhat) and D2 (weighted by 1 - yhat) "
            "must have positive total weight for MSD"
        )

    items = _build_items(n_features, groups, allow_negations)
    return _run_core(
        X, wa, wb, w, items, metric_code=0, den_a=den_a, den_b=den_b,
        feature_names=feature_names, max_nodes=max_nodes,
        time_limit=time_limit, max_depth=max_depth, gamma=gamma,
        min_coverage=min_coverage,
    )


def spsf(
    X,
    yhat,
    sample_weight=None,
    groups=None,
    allow_negations: bool = False,
    feature_names=None,
    max_nodes=None,
    time_limit=None,
    max_depth=None,
    gamma=None,
    min_coverage=None,
) -> Subgroup:
    """Statistical Parity Subgroup Fairness violation of a classifier.

    SPSF(h, S) = P(X_S) * |p_base - P(h(x) = 1 | X_S)|,
    p_base = P(h(x) = 1).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Binary feature matrix (values in {0, 1}).
    yhat : array-like of shape (n_samples,)
        Classifier output h(x) for each row -- either a hard 0/1 label, or
        a probability/soft score in [0, 1] (e.g. a calibrated classifier's
        predicted probability of the positive class), in which case
        `p_base` and the per-subgroup rate become weighted averages of
        `yhat` rather than counts of a hard label. Both are handled by the
        exact same formula and are exactly correct for the same reason
        given in :func:`msd`'s "Notes on probabilistic yhat".
    sample_weight, groups, allow_negations, feature_names, max_nodes, time_limit, max_depth, gamma, min_coverage
        See :func:`msd`. `min_coverage` counts over all rows of `X` (there
        is no D1/D2 split here, so this is simply the subgroup's total
        weighted row count).

    Returns
    -------
    Subgroup
        The winning subgroup, with `.violation` set to the SPSF value.
    """
    X = _validate_binary_matrix(X)
    n, n_features = X.shape
    yhat = _validate_probability_vector(yhat, n, "yhat")
    w = _validate_weight(sample_weight, n)
    groups = _validate_groups(n_features, groups)
    _check_one_hot_groups(X, groups)

    den = float(w.sum())
    if den <= 0.0:
        raise ValueError("total sample weight must be positive")
    p_base = float((w * yhat).sum() / den)
    delta_bar = max(p_base, 1.0 - p_base)

    wa = w * yhat
    wb = w * (1 - yhat)

    items = _build_items(n_features, groups, allow_negations)
    return _run_core(
        X,
        wa,
        wb,
        w,
        items,
        metric_code=1,
        den=den,
        p_base=p_base,
        delta_bar=delta_bar,
        feature_names=feature_names,
        max_nodes=max_nodes,
        time_limit=time_limit,
        max_depth=max_depth,
        gamma=gamma,
        min_coverage=min_coverage,
    )


def fpsf(
    X,
    yhat,
    y,
    sample_weight=None,
    groups=None,
    allow_negations: bool = False,
    feature_names=None,
    max_nodes=None,
    time_limit=None,
    max_depth=None,
    gamma=None,
    min_coverage=None,
) -> Subgroup:
    """False Positive Subgroup Fairness violation of a classifier.

    FPSF(h, S) = P(X_S, y=0) * |p_base,0 - P(h(x) = 1 | X_S, y=0)|,
    p_base,0 = P(h(x) = 1 | y=0).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Binary feature matrix (values in {0, 1}).
    yhat : array-like of shape (n_samples,)
        Classifier output h(x) for each row -- either a hard 0/1 label, or
        a probability/soft score in [0, 1]; see :func:`spsf`'s note on
        probabilistic `yhat` (the same reasoning applies here, restricted
        to the y == 0 population).
    y : array-like of shape (n_samples,)
        True binary label for each row (always a hard 0/1 label, unlike
        `yhat` -- this defines *which population* is being audited, not a
        weighted rate within it); the search population is restricted
        to rows with y == 0 (the false-positive population), while the
        denominator in FPSF's P(X_S, y=0) uses the *full* population.
    sample_weight, groups, allow_negations, feature_names, max_nodes, time_limit, max_depth, gamma
        See :func:`msd`.
    min_coverage : float, optional
        Unlike `msd`/`spsf`, this counts only over the y == 0 (audited)
        population -- the population the search actually runs over --
        since y == 1 rows are excluded from the search entirely. See
        :func:`msd` for the general behavior.

    Returns
    -------
    Subgroup
        The winning subgroup, with `.violation` set to the FPSF value.
    """
    X = _validate_binary_matrix(X)
    n, n_features = X.shape
    yhat = _validate_probability_vector(yhat, n, "yhat")
    y = _validate_binary_vector(y, n, "y")
    w = _validate_weight(sample_weight, n)
    groups = _validate_groups(n_features, groups)
    _check_one_hot_groups(X, groups)

    den = float(w.sum())  # full population weight, BEFORE restricting to y=0
    if den <= 0.0:
        raise ValueError("total sample weight must be positive")

    neg_mask = y == 0
    w0 = w[neg_mask]
    if float(w0.sum()) <= 0.0:
        raise ValueError("no negative-label (y == 0) samples available for FPSF")

    yhat0 = yhat[neg_mask]
    p_base0 = float((w0 * yhat0).sum() / w0.sum())
    delta_bar = max(p_base0, 1.0 - p_base0)

    X0 = X[neg_mask]
    wa = w0 * yhat0
    wb = w0 * (1 - yhat0)

    items = _build_items(n_features, groups, allow_negations)
    return _run_core(
        X0,
        wa,
        wb,
        w0,
        items,
        metric_code=1,
        den=den,
        p_base=p_base0,
        delta_bar=delta_bar,
        feature_names=feature_names,
        max_nodes=max_nodes,
        time_limit=time_limit,
        max_depth=max_depth,
        gamma=gamma,
        min_coverage=min_coverage,
    )
