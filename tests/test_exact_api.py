"""Input-validation edge cases for the public API."""

import numpy as np
import pytest

from certifi_learn.exact import fpsf, msd, spsf


def _binary_data(n=50, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, d))
    yhat = rng.integers(0, 2, size=n)
    y = rng.integers(0, 2, size=n)
    return X, yhat, y


def test_non_binary_X_rejected():
    X, yhat, _ = _binary_data()
    X = X.astype(float)
    X[0, 0] = 2
    with pytest.raises(ValueError, match="binary"):
        msd(X, yhat)


def test_out_of_range_yhat_rejected():
    # yhat accepts any value in [0, 1] (see probabilistic-yhat tests below),
    # but must still reject values genuinely outside that range.
    X, yhat, _ = _binary_data()
    yhat = yhat.copy().astype(float)
    yhat[0] = 5
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        msd(X, yhat)
    yhat[0] = -0.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        msd(X, yhat)


def test_non_finite_yhat_rejected():
    X, yhat, _ = _binary_data()
    yhat = yhat.copy().astype(float)
    yhat[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        msd(X, yhat)
    yhat[0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        spsf(X, yhat)


def test_probabilistic_yhat_accepted_and_matches_bruteforce():
    # A soft/probabilistic yhat (e.g. a calibrated classifier's predicted
    # probability, rather than its thresholded hard label) is accepted: the
    # bound-transfer argument only needs non-negative, coverage-monotone
    # accumulators, which holds regardless of whether yhat is hard 0/1 or
    # continuous in [0, 1].
    rng = np.random.default_rng(21)
    n, d = 150, 5
    X = rng.integers(0, 2, size=(n, d))
    yhat_soft = rng.random(n)  # continuous in [0, 1)

    for fn, kwargs in ((msd, {}), (spsf, {})):
        result = fn(X, yhat_soft, **kwargs)
        assert 0.0 <= result.violation <= 1.0
        assert result.exact is True

        # Brute-force cross-check over a small candidate set of conjunctions
        # (single-literal and pairs, either literal value per column -- plain
        # binary columns get both (c, 0) and (c, 1) in their domain), using
        # the same weighted-average formulas the implementation documents.
        best = 0.0
        literals = [(j, v) for j in range(d) for v in (0, 1)]
        candidates = [[lit] for lit in literals] + [
            [lit1, lit2]
            for i, lit1 in enumerate(literals)
            for lit2 in literals[i + 1 :]
            if lit1[0] != lit2[0]
        ]
        w = np.ones(n)
        for lits in candidates:
            mask = np.ones(n, dtype=bool)
            for c, v in lits:
                mask &= X[:, c] == v
            if not mask.any():
                continue
            if fn is msd:
                da = (w * yhat_soft).sum()
                db = (w * (1 - yhat_soft)).sum()
                a = (w[mask] * yhat_soft[mask]).sum() / da
                b = (w[mask] * (1 - yhat_soft[mask])).sum() / db
                val = abs(a - b)
            else:
                p_base = (w * yhat_soft).sum() / w.sum()
                p_x = mask.sum() / n
                p_yhat_given_x = (w[mask] * yhat_soft[mask]).sum() / w[mask].sum()
                val = p_x * abs(p_base - p_yhat_given_x)
            best = max(best, val)
        assert result.violation <= best + 1e-9


def test_probabilistic_yhat_reduces_to_hard_binary_case():
    # Passing an already-binary yhat as floats (0.0/1.0) must reproduce
    # exactly the same result as passing it as ints -- probabilistic
    # support is a strict superset, not a separate code path.
    rng = np.random.default_rng(22)
    n, d = 100, 6
    X = rng.integers(0, 2, size=(n, d))
    yhat_int = rng.integers(0, 2, size=n)
    yhat_float = yhat_int.astype(float)
    for fn in (msd, spsf):
        r_int = fn(X, yhat_int)
        r_float = fn(X, yhat_float)
        assert r_int.violation == pytest.approx(r_float.violation)
        assert r_int.literals == r_float.literals


def test_mismatched_length_yhat_rejected():
    X, yhat, _ = _binary_data()
    with pytest.raises(ValueError, match="length"):
        msd(X, yhat[:-1])


def test_mismatched_length_sample_weight_rejected():
    X, yhat, _ = _binary_data()
    with pytest.raises(ValueError, match="length"):
        spsf(X, yhat, sample_weight=np.ones(len(yhat) - 1))


def test_negative_sample_weight_rejected():
    X, yhat, _ = _binary_data()
    w = np.ones(len(yhat))
    w[0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        spsf(X, yhat, sample_weight=w)


def test_all_yhat_one_rejected_for_msd():
    X, _, _ = _binary_data()
    yhat = np.ones(X.shape[0], dtype=int)
    with pytest.raises(ValueError, match="positive total weight"):
        msd(X, yhat)


def test_all_yhat_zero_rejected_for_msd():
    X, _, _ = _binary_data()
    yhat = np.zeros(X.shape[0], dtype=int)
    with pytest.raises(ValueError, match="positive total weight"):
        msd(X, yhat)


def test_fpsf_requires_some_negative_labels():
    X, yhat, _ = _binary_data()
    y = np.ones(X.shape[0], dtype=int)  # no y == 0 rows
    with pytest.raises(ValueError, match="y == 0"):
        fpsf(X, yhat, y)


def test_fpsf_accepts_probabilistic_yhat_but_requires_binary_y():
    rng = np.random.default_rng(23)
    n, d = 120, 5
    X = rng.integers(0, 2, size=(n, d))
    yhat_soft = rng.random(n)
    y = rng.integers(0, 2, size=n)
    result = fpsf(X, yhat_soft, y)  # should not raise
    assert 0.0 <= result.violation <= 1.0

    y_soft = rng.random(n)  # y (true label) must stay hard 0/1
    with pytest.raises(ValueError, match=r"\{0, 1\}"):
        fpsf(X, yhat_soft, y_soft)


def test_group_overlap_rejected():
    X, yhat, _ = _binary_data(d=4)
    with pytest.raises(ValueError, match="more than one group"):
        msd(X, yhat, groups=[[0, 1], [1, 2]])


def test_group_out_of_range_rejected():
    X, yhat, _ = _binary_data(d=4)
    with pytest.raises(ValueError, match="out of range"):
        msd(X, yhat, groups=[[0, 99]])


def test_non_one_hot_group_rejected():
    rng = np.random.default_rng(0)
    n = 30
    X = rng.integers(0, 2, size=(n, 4))
    # Force at least one row with two indicator columns of the "group" set.
    X[0, 0] = 1
    X[0, 1] = 1
    yhat = rng.integers(0, 2, size=n)
    with pytest.raises(ValueError, match="one-hot"):
        msd(X, yhat, groups=[[0, 1]])


def test_groups_with_allow_negations_adds_negative_literals():
    rng = np.random.default_rng(0)
    n = 200
    cat = rng.integers(0, 3, size=n)
    X = np.eye(3, dtype=int)[cat]
    extra = rng.integers(0, 2, size=(n, 2))
    X = np.concatenate([X, extra], axis=1)
    yhat = rng.integers(0, 2, size=n)

    r_pos = msd(X, yhat, groups=[[0, 1, 2]], allow_negations=False)
    r_neg = msd(X, yhat, groups=[[0, 1, 2]], allow_negations=True)
    # Allowing negations only ever adds candidate subgroups, so the best
    # violation found cannot decrease.
    assert r_neg.violation >= r_pos.violation - 1e-12


def test_max_nodes_caps_search_and_marks_inexact():
    X, yhat, _ = _binary_data(n=400, d=14, seed=3)

    full = msd(X, yhat)
    assert full.exact is True
    assert full.reason == "optimal"

    capped = msd(X, yhat, max_nodes=5)
    assert capped.exact is False
    assert capped.reason == "max_nodes"
    assert "not a certified optimum" in capped.describe()
    # A partial search can only ever have found a subset of what the full
    # search explores, so it can't report a larger violation.
    assert capped.violation <= full.violation + 1e-12


def test_max_nodes_none_is_exact_by_default():
    X, yhat, _ = _binary_data(n=200, d=8, seed=4)
    result = msd(X, yhat)
    assert result.exact is True
    assert result.reason == "optimal"
    assert "truncated" not in result.describe()


def test_time_limit_caps_search_and_marks_inexact():
    # Large/unstructured enough that the unrestricted search takes several
    # seconds (verified during development), so a tiny time_limit reliably
    # triggers well before the search would finish on its own -- the
    # deadline is only checked every ~1024 nodes, so the dataset needs to
    # be big enough that that many nodes are visited almost immediately.
    X, yhat, _ = _binary_data(n=5000, d=22, seed=5)

    capped = msd(X, yhat, time_limit=0.01)
    assert capped.exact is False
    assert capped.reason == "time_limit"
    assert "time_limit" in capped.describe()


def test_time_limit_must_be_positive():
    X, yhat, _ = _binary_data()
    with pytest.raises(ValueError, match="time_limit"):
        msd(X, yhat, time_limit=0)
    with pytest.raises(ValueError, match="time_limit"):
        msd(X, yhat, time_limit=-1.0)


def test_max_depth_restricts_rule_size_and_reason():
    X, yhat, _ = _binary_data(n=500, d=10, seed=6)

    full = msd(X, yhat)
    capped = msd(X, yhat, max_depth=1)

    assert len(capped.literals) <= 1
    assert capped.reason == "max_depth"
    assert capped.exact is False
    assert "max_depth" in capped.describe()
    # A depth-1 search explores a subset of what the unrestricted search
    # explores, so it can't do better.
    assert capped.violation <= full.violation + 1e-12


def test_max_depth_zero_finds_nothing():
    X, yhat, _ = _binary_data(n=200, d=6, seed=7)
    result = msd(X, yhat, max_depth=0)
    assert result.literals == []
    assert result.violation == 0.0
    assert result.reason == "max_depth"


def test_max_depth_covering_all_attributes_is_optimal():
    X, yhat, _ = _binary_data(n=200, d=5, seed=8)
    full = msd(X, yhat)
    # A depth cap that can never actually bind (>= number of attributes)
    # should not be reported as having restricted anything.
    same = msd(X, yhat, max_depth=5)
    assert same.reason == "optimal"
    assert same.exact is True
    assert same.violation == pytest.approx(full.violation)


def test_max_depth_must_be_non_negative():
    X, yhat, _ = _binary_data()
    with pytest.raises(ValueError, match="max_depth"):
        msd(X, yhat, max_depth=-1)


def test_max_nodes_takes_precedence_over_max_depth_reason():
    X, yhat, _ = _binary_data(n=500, d=14, seed=9)
    # Both a (non-binding-until-hit) node cap and a depth cap are set; the
    # node cap should still be what's reported, since it's what actually
    # curtailed the search.
    result = msd(X, yhat, max_nodes=3, max_depth=2)
    assert result.reason == "max_nodes"


def test_gamma_none_matches_default_zero_start():
    # gamma=None is documented as behaving like the unconstrained search
    # (equivalent to seeding tau at 0, since all violations are >= 0).
    X, yhat, _ = _binary_data(n=200, d=8, seed=10)
    default = msd(X, yhat)
    explicit_zero = msd(X, yhat, gamma=0.0)
    assert default.violation == pytest.approx(explicit_zero.violation)
    assert default.literals == explicit_zero.literals


def test_gamma_high_threshold_returns_empty_certificate():
    X, yhat, _ = _binary_data(n=300, d=10, seed=11)
    # No violation measure can exceed 1.0, so this certifies "nothing
    # exceeds gamma" trivially, and should say so.
    result = msd(X, yhat, gamma=1.0)
    assert result.literals == []
    assert result.violation == pytest.approx(1.0)
    assert result.exact is True
    assert result.gamma == pytest.approx(1.0)
    assert "no subgroup with violation > 1" in result.describe()


def test_gamma_must_be_finite():
    X, yhat, _ = _binary_data()
    with pytest.raises(ValueError, match="gamma"):
        msd(X, yhat, gamma=float("nan"))
    with pytest.raises(ValueError, match="gamma"):
        msd(X, yhat, gamma=float("inf"))


def test_gamma_default_is_none_on_subgroup():
    X, yhat, _ = _binary_data(n=100, d=5, seed=12)
    result = msd(X, yhat)
    assert result.gamma is None


def test_gamma_combined_with_max_depth():
    X, yhat, _ = _binary_data(n=300, d=10, seed=13)
    full = msd(X, yhat)
    # A gamma above the true max, combined with an (irrelevant, since
    # nothing survives the gamma cut anyway) depth cap, should still
    # report a clean "nothing found" result with reason == "max_depth"
    # (since the depth cap is what keeps this from being a global
    # certificate, even though gamma is what actually pruned everything).
    result = msd(X, yhat, gamma=full.violation + 0.1, max_depth=1)
    assert result.literals == []
    assert result.reason == "max_depth"


def test_gamma_matches_unconstrained_for_spsf_and_fpsf():
    # The existing gamma tests above only exercise msd; pin the same
    # behavior for spsf and fpsf too, since each metric has its own `den`
    # normalization and gamma must be interpreted on that same scale as
    # `.violation` (not, e.g., a scale tied to some other population size).
    rng = np.random.default_rng(30)
    n, d = 300, 8
    X = rng.integers(0, 2, size=(n, d))
    yhat = rng.integers(0, 2, size=n)
    y = rng.integers(0, 2, size=n)

    full_spsf = spsf(X, yhat)
    below = spsf(X, yhat, gamma=full_spsf.violation - 1e-9)
    assert below.literals == full_spsf.literals
    assert below.violation == pytest.approx(full_spsf.violation)
    above = spsf(X, yhat, gamma=full_spsf.violation + 1e-6)
    assert above.literals == []
    assert above.violation == pytest.approx(full_spsf.violation + 1e-6)

    full_fpsf = fpsf(X, yhat, y)
    below = fpsf(X, yhat, y, gamma=full_fpsf.violation - 1e-9)
    assert below.literals == full_fpsf.literals
    assert below.violation == pytest.approx(full_fpsf.violation)
    above = fpsf(X, yhat, y, gamma=full_fpsf.violation + 1e-6)
    assert above.literals == []
    assert above.violation == pytest.approx(full_fpsf.violation + 1e-6)


def test_gamma_fpsf_uses_full_population_scale_not_restricted_scale():
    # FPSF's `.violation` is normalized by the *full* population (all rows,
    # not just y == 0) -- confirmed by the identity
    # fpsf(X, yhat, y).violation == (n0/n) * spsf(X[y==0], yhat[y==0]).violation.
    # gamma must live on that same full-population scale: a gamma passed to
    # fpsf should NOT need rescaling relative to `.violation`, but the same
    # numeric threshold used directly (unscaled) against the *restricted*-
    # population spsf call means something different -- it needs dividing
    # by (n0/n) first to ask the same question. Demonstrate the failure a
    # naive (unscaled) restricted-population call would have, and that
    # fpsf itself -- and the correctly-scaled restricted call -- avoid it.
    rng = np.random.default_rng(31)
    n, d = 350, 7
    X = rng.integers(0, 2, size=(n, d))
    yhat = rng.integers(0, 2, size=n)
    y = rng.integers(0, 2, size=n)
    neg = y == 0
    scale = neg.sum() / n

    full = fpsf(X, yhat, y)
    g = full.violation + 1e-6  # just above fpsf's own true max

    # A gamma just above the true max certifies "nothing exceeds it".
    direct = fpsf(X, yhat, y, gamma=g)
    assert direct.literals == []
    assert direct.violation == pytest.approx(g)

    # The same threshold, correctly translated to the restricted
    # population's own scale, also certifies nothing exceeds it (the
    # restricted population's own true max is g/scale - eps/scale > g/scale
    # is NOT what we need here -- what matters is that g/scale still
    # exceeds the restricted population's true max, since fpsf's true max
    # equals scale * restricted true max).
    restricted_correct = spsf(X[neg], yhat[neg], gamma=g / scale)
    assert restricted_correct.literals == []

    # Using fpsf's gamma value *directly* (unscaled) against the restricted
    # population is a materially weaker threshold (since scale < 1 makes
    # g / scale > g) -- it fails to certify anything, and instead finds the
    # exact subgroup fpsf itself had just certified doesn't exist above g.
    # This is precisely the bug that would appear if gamma were passed
    # through without accounting for FPSF's full-population normalization.
    mismatched = spsf(X[neg], yhat[neg], gamma=g)
    assert mismatched.literals == full.literals
    assert mismatched.literals != []


def test_min_coverage_must_be_non_negative():
    X, yhat, _ = _binary_data()
    with pytest.raises(ValueError, match="min_coverage"):
        msd(X, yhat, min_coverage=-1)


def test_min_coverage_must_be_finite():
    X, yhat, _ = _binary_data()
    with pytest.raises(ValueError, match="min_coverage"):
        msd(X, yhat, min_coverage=float("nan"))
    with pytest.raises(ValueError, match="min_coverage"):
        msd(X, yhat, min_coverage=float("inf"))


def test_min_coverage_reason_stays_optimal_when_binding():
    # min_coverage restricts the problem definition (like `groups`), not
    # the search's completeness -- a search bound only by a (binding)
    # min_coverage that runs to completion should still report "optimal".
    X, yhat, _ = _binary_data(n=200, d=8, seed=14)
    result = msd(X, yhat, min_coverage=50)
    assert result.reason == "optimal"
    assert result.exact is True


def test_min_coverage_never_increases_violation():
    X, yhat, _ = _binary_data(n=300, d=10, seed=15)
    unconstrained = msd(X, yhat)
    constrained = msd(X, yhat, min_coverage=80)
    # Restricting to a subset of subgroups can only ever match or lower
    # the best violation found, never raise it.
    assert constrained.violation <= unconstrained.violation + 1e-12


def test_2d_required_for_X():
    yhat = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="2D"):
        msd(np.array([0, 1, 0, 1]), yhat)


def test_feature_names_used_in_describe():
    X, yhat, _ = _binary_data(d=3)
    names = ["a", "b", "c"]
    result = msd(X, yhat, feature_names=names)
    if result.literals:
        col, val = result.literals[0]
        assert f"{names[col]}={val}" in result.describe()
