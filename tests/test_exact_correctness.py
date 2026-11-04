"""Brute-force cross-checks of the branch-and-bound search.

For small random synthetic datasets, we enumerate every valid subgroup
(every conjunction of literals respecting the `groups`/`allow_negations`
constraints, with at most one literal per attribute) directly in Python and
compare the true maximum violation against what the exact search finds. This
empirically validates that the pruning implemented in the Rust core never
discards the optimal subgroup, i.e. that the algorithm's optimality
guarantee is actually realized by the implementation.
"""

import itertools

import numpy as np
import pytest

from certifi_learn.exact import fpsf, msd, spsf


def _attribute_domains(n_features, groups, allow_negations):
    """Mirror certifi_learn.exact._common._build_items, but return domains as plain
    (col, val) tuples per attribute, independently of the package under
    test (this file must not import certifi_learn.exact._common, to keep the brute
    force honestly independent of the implementation it is checking)."""
    items = []
    covered = set()
    if groups:
        for g in groups:
            cols = [g] if isinstance(g, int) else list(g)
            covered.update(cols)
            lits = [(c, 1) for c in cols]
            if allow_negations:
                lits += [(c, 0) for c in cols]
            items.append(lits)
    for c in range(n_features):
        if c not in covered:
            items.append([(c, 0), (c, 1)])
    return items


def _all_subgroups(n_features, groups, allow_negations, max_depth=None):
    """Every non-empty conjunction using at most one literal per attribute,
    and (if given) at most `max_depth` literals total."""
    items = _attribute_domains(n_features, groups, allow_negations)
    # Each attribute contributes: "unused" (None) or one of its literals.
    choices = [[None] + lits for lits in items]
    for combo in itertools.product(*choices):
        lits = [c for c in combo if c is not None]
        if lits and (max_depth is None or len(lits) <= max_depth):
            yield lits


def _mask_for(X, lits):
    m = np.ones(X.shape[0], dtype=bool)
    for col, val in lits:
        m &= X[:, col] == val
    return m


def _brute_msd(X, yhat, sample_weight, groups, allow_negations, max_depth=None, min_coverage=None):
    w = sample_weight
    d1 = yhat == 1
    d2 = yhat == 0
    den1 = w[d1].sum()
    den2 = w[d2].sum()
    best = 0.0
    for lits in _all_subgroups(X.shape[1], groups, allow_negations, max_depth=max_depth):
        m = _mask_for(X, lits)
        if min_coverage is not None and w[m].sum() < min_coverage:
            continue
        p1 = w[m & d1].sum() / den1
        p2 = w[m & d2].sum() / den2
        best = max(best, abs(p1 - p2))
    return best


def _brute_spsf(X, yhat, sample_weight, groups, allow_negations, max_depth=None, min_coverage=None):
    w = sample_weight
    den = w.sum()
    p_base = (w * yhat).sum() / den
    best = 0.0
    for lits in _all_subgroups(X.shape[1], groups, allow_negations, max_depth=max_depth):
        m = _mask_for(X, lits)
        cov = w[m].sum()
        if cov <= 0:
            continue
        if min_coverage is not None and cov < min_coverage:
            continue
        p_hat = (w[m] * yhat[m]).sum() / cov
        v = (cov / den) * abs(p_base - p_hat)
        best = max(best, v)
    return best


def _brute_fpsf(X, yhat, y, sample_weight, groups, allow_negations, max_depth=None, min_coverage=None):
    w = sample_weight
    den = w.sum()
    neg = y == 0
    w0, yhat0, X0 = w[neg], yhat[neg], X[neg]
    p_base0 = (w0 * yhat0).sum() / w0.sum()
    best = 0.0
    for lits in _all_subgroups(X.shape[1], groups, allow_negations, max_depth=max_depth):
        m = _mask_for(X0, lits)
        cov = w0[m].sum()
        if cov <= 0:
            continue
        if min_coverage is not None and cov < min_coverage:
            continue
        p_hat = (w0[m] * yhat0[m]).sum() / cov
        v = (cov / den) * abs(p_base0 - p_hat)
        best = max(best, v)
    return best


def _make_data(rng, n=300, d=7, one_hot_width=3):
    """d plain binary columns, plus one one-hot group of width
    `one_hot_width` appended at the end (columns d..d+one_hot_width)."""
    X_bin = rng.integers(0, 2, size=(n, d))
    cat = rng.integers(0, one_hot_width, size=n)
    X_oh = np.eye(one_hot_width, dtype=int)[cat]
    X = np.concatenate([X_bin, X_oh], axis=1)
    groups = [list(range(d, d + one_hot_width))]
    yhat = rng.integers(0, 2, size=n)
    y = rng.integers(0, 2, size=n)
    return X, yhat, y, groups


SEEDS = [0, 1, 2, 3, 4]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("use_groups", [False, True])
@pytest.mark.parametrize("allow_negations", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_msd_matches_bruteforce(seed, use_groups, allow_negations, weighted):
    rng = np.random.default_rng(seed)
    X, yhat, y, groups = _make_data(rng)
    if not use_groups:
        groups = None
    w = rng.uniform(0.5, 2.0, size=X.shape[0]) if weighted else np.ones(X.shape[0])

    got = msd(
        X, yhat, sample_weight=w if weighted else None,
        groups=groups, allow_negations=allow_negations,
    )
    want = _brute_msd(X, yhat, w, groups, allow_negations)
    assert got.violation == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("use_groups", [False, True])
@pytest.mark.parametrize("allow_negations", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_spsf_matches_bruteforce(seed, use_groups, allow_negations, weighted):
    rng = np.random.default_rng(seed)
    X, yhat, y, groups = _make_data(rng)
    if not use_groups:
        groups = None
    w = rng.uniform(0.5, 2.0, size=X.shape[0]) if weighted else np.ones(X.shape[0])

    got = spsf(
        X, yhat, sample_weight=w if weighted else None,
        groups=groups, allow_negations=allow_negations,
    )
    want = _brute_spsf(X, yhat, w, groups, allow_negations)
    assert got.violation == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("use_groups", [False, True])
@pytest.mark.parametrize("allow_negations", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_fpsf_matches_bruteforce(seed, use_groups, allow_negations, weighted):
    rng = np.random.default_rng(seed)
    X, yhat, y, groups = _make_data(rng)
    if not use_groups:
        groups = None
    w = rng.uniform(0.5, 2.0, size=X.shape[0]) if weighted else np.ones(X.shape[0])

    got = fpsf(
        X, yhat, y, sample_weight=w if weighted else None,
        groups=groups, allow_negations=allow_negations,
    )
    want = _brute_fpsf(X, yhat, y, w, groups, allow_negations)
    assert got.violation == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("use_groups", [False, True])
@pytest.mark.parametrize("max_depth", [0, 1, 2])
def test_msd_matches_bruteforce_with_max_depth(seed, use_groups, max_depth):
    rng = np.random.default_rng(seed)
    X, yhat, y, groups = _make_data(rng)
    if not use_groups:
        groups = None
    w = np.ones(X.shape[0])

    got = msd(X, yhat, groups=groups, max_depth=max_depth)
    want = _brute_msd(X, yhat, w, groups, allow_negations=False, max_depth=max_depth)
    assert got.violation == pytest.approx(want, abs=1e-9)
    assert len(got.literals) <= max_depth


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("max_depth", [0, 1, 2])
def test_spsf_matches_bruteforce_with_max_depth(seed, max_depth):
    rng = np.random.default_rng(seed)
    X, yhat, y, groups = _make_data(rng)
    groups = None
    w = np.ones(X.shape[0])

    got = spsf(X, yhat, max_depth=max_depth)
    want = _brute_spsf(X, yhat, w, groups, allow_negations=False, max_depth=max_depth)
    assert got.violation == pytest.approx(want, abs=1e-9)
    assert len(got.literals) <= max_depth


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("use_groups", [False, True])
@pytest.mark.parametrize("min_coverage", [0, 50, 150, 1000])
def test_msd_matches_bruteforce_with_min_coverage(seed, use_groups, min_coverage):
    rng = np.random.default_rng(seed)
    X, yhat, y, groups = _make_data(rng)
    if not use_groups:
        groups = None
    w = np.ones(X.shape[0])

    got = msd(X, yhat, groups=groups, min_coverage=min_coverage)
    want = _brute_msd(X, yhat, w, groups, allow_negations=False, min_coverage=min_coverage)
    assert got.violation == pytest.approx(want, abs=1e-9)
    if got.literals:
        # coverage is counted over ALL rows (both D1 and D2), regardless
        # of yhat -- matches the reference MIO tool's n_min semantics.
        assert got.support(X) >= min_coverage


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("min_coverage", [0, 50, 150, 1000])
def test_spsf_matches_bruteforce_with_min_coverage(seed, min_coverage):
    rng = np.random.default_rng(seed)
    X, yhat, y, groups = _make_data(rng)
    groups = None
    w = np.ones(X.shape[0])

    got = spsf(X, yhat, min_coverage=min_coverage)
    want = _brute_spsf(X, yhat, w, groups, allow_negations=False, min_coverage=min_coverage)
    assert got.violation == pytest.approx(want, abs=1e-9)
    if got.literals:
        assert got.support(X) >= min_coverage


def test_min_coverage_impossible_finds_nothing():
    rng = np.random.default_rng(0)
    X, yhat, y, groups = _make_data(rng, n=100)
    result = msd(X, yhat, min_coverage=1000)  # larger than the whole dataset
    assert result.literals == []
    assert result.reason == "optimal"
    assert result.exact is True


def test_min_coverage_zero_matches_unconstrained():
    rng = np.random.default_rng(1)
    X, yhat, y, groups = _make_data(rng)
    default = msd(X, yhat)
    explicit = msd(X, yhat, min_coverage=0)
    assert default.violation == pytest.approx(explicit.violation)
    assert default.literals == explicit.literals


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("use_groups", [False, True])
def test_msd_gamma_below_true_max_matches_unconstrained(seed, use_groups):
    rng = np.random.default_rng(seed)
    X, yhat, y, groups = _make_data(rng)
    if not use_groups:
        groups = None
    w = np.ones(X.shape[0])

    full = msd(X, yhat, groups=groups)
    true_max = _brute_msd(X, yhat, w, groups, allow_negations=False)
    assert full.violation == pytest.approx(true_max, abs=1e-9)

    # A gamma strictly below the true max must not change the answer at
    # all (same violation, same winning subgroup) -- only how much of the
    # tree gets pruned along the way.
    capped = msd(X, yhat, groups=groups, gamma=true_max - 0.01)
    assert capped.violation == pytest.approx(full.violation, abs=1e-9)
    assert capped.literals == full.literals
    assert capped.reason == "optimal"


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("use_groups", [False, True])
def test_msd_gamma_at_or_above_true_max_finds_nothing(seed, use_groups):
    rng = np.random.default_rng(seed)
    X, yhat, y, groups = _make_data(rng)
    if not use_groups:
        groups = None
    w = np.ones(X.shape[0])

    true_max = _brute_msd(X, yhat, w, groups, allow_negations=False)

    # gamma exactly at the true max: the "beyond" semantics are strict, so
    # nothing should be reported as found (ties don't count as exceeding).
    at_max = msd(X, yhat, groups=groups, gamma=true_max)
    assert at_max.literals == []
    assert at_max.violation == pytest.approx(true_max, abs=1e-9)
    assert at_max.exact is True

    above_max = msd(X, yhat, groups=groups, gamma=true_max + 0.05)
    assert above_max.literals == []
    assert above_max.violation == pytest.approx(true_max + 0.05, abs=1e-9)
    assert above_max.exact is True


@pytest.mark.parametrize("seed", SEEDS)
def test_spsf_gamma_matches_bruteforce_boundary(seed):
    rng = np.random.default_rng(seed)
    X, yhat, y, groups = _make_data(rng)
    groups = None
    w = np.ones(X.shape[0])

    true_max = _brute_spsf(X, yhat, w, groups, allow_negations=False)

    below = spsf(X, yhat, gamma=true_max - 0.01)
    assert below.violation == pytest.approx(true_max, abs=1e-9)
    assert below.literals != []

    above = spsf(X, yhat, gamma=true_max + 0.01)
    assert above.literals == []
    assert above.violation == pytest.approx(true_max + 0.01, abs=1e-9)


def test_subgroup_mask_and_support_consistency():
    rng = np.random.default_rng(42)
    X, yhat, y, groups = _make_data(rng)
    result = msd(X, yhat, groups=groups)
    mask = result.mask(X)
    assert mask.sum() == result.support(X)
    for col, val in result.literals:
        assert np.all(X[mask, col] == val)
