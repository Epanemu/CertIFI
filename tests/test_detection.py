import numpy as np
import pytest

from certifi_learn import find_subgroup, measures


def _random_data(seed, n=300, d=6):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d)) < 0.4
    y_true = rng.random(n) < 0.5
    # predictions biased against the subgroup x0 AND x1
    y_pred = rng.random(n) < np.where(X[:, 0] & X[:, 1], 0.2, 0.6)
    return X, y_true, y_pred


def _has_highs():
    try:
        import pyomo.environ as pyo

        return pyo.SolverFactory("appsi_highs").available(exception_flag=False)
    except ImportError:
        return False


@pytest.mark.parametrize("measure", ["msd", "spsf", "fpsf"])
def test_exact_value_matches_measure(measure):
    X, y_true, y_pred = _random_data(0)
    y = y_true if measure == "msd" else y_pred
    sg = find_subgroup(X, y, measure, y_true=y_true)
    assert sg.kind == "conjunction" and sg.backend == "exact" and sg.optimal
    assert np.array_equal(sg.mask, sg.contains(X))
    if measure == "msd":
        expected = measures.msd(y, sg.mask)
    elif measure == "spsf":
        expected = measures.spsf(y, sg.mask)
    else:
        expected = measures.fpsf(y_true, y, sg.mask)
    assert sg.value == pytest.approx(expected)


def test_gamma_threshold():
    X, y_true, y_pred = _random_data(1)
    best = find_subgroup(X, y_pred, "spsf")
    assert find_subgroup(X, y_pred, "spsf", gamma=best.value + 1e-6) is None
    assert find_subgroup(X, y_pred, "spsf", gamma=best.value / 2) is not None


def test_invalid_combinations():
    X, y_true, y_pred = _random_data(2)
    with pytest.raises(ValueError):
        find_subgroup(X, y_pred, "spsf", subgroups="linear", backend="exact")
    with pytest.raises(ValueError):
        find_subgroup(X, y_pred, "fpsf")  # y_true missing


@pytest.mark.skipif(not _has_highs(), reason="pyomo with HiGHS not available")
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("measure", ["msd", "spsf", "fpsf"])
def test_exact_and_mio_agree(seed, measure):
    X, y_true, y_pred = _random_data(seed)
    y = y_true if measure == "msd" else y_pred
    exact = find_subgroup(X, y, measure, y_true=y_true, backend="exact")
    mio = find_subgroup(X, y, measure, y_true=y_true, backend="mio", solver="highs", time_limit=60)
    assert mio.optimal
    assert exact.value == pytest.approx(mio.value, abs=1e-6)


@pytest.mark.skipif(not _has_highs(), reason="pyomo with HiGHS not available")
def test_mio_linear_subgroups():
    X, y_true, y_pred = _random_data(3, n=200, d=4)
    sg = find_subgroup(X.astype(float), y_pred, "spsf", subgroups="linear", solver="highs", time_limit=30)
    assert sg.kind == "linear" and sg.backend == "mio"
    # linear subgroups include all conjunctions, so they are at least as biased
    conj = find_subgroup(X, y_pred, "spsf")
    assert sg.value >= conj.value - 1e-4
