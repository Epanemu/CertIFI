import numpy as np
import pandas as pd
import pytest

from certifi_learn import FairDNFClassifier, FairLinearClassifier, find_subgroup, measures


def _has_gurobi():
    try:
        import gurobipy as gp

        with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gurobi(), reason="Gurobi (with a license) not available")


def _data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    prot = rng.random((n, 3)) < 0.5
    X = np.hstack([rng.normal(size=(n, 3)), prot.astype(float)])
    # labels depend on a protected attribute -> an unconstrained model is unfair
    y = (X[:, 0] + 1.5 * prot[:, 0] + 0.3 * rng.normal(size=n)) > 0.7
    return X, y, prot


@pytest.mark.parametrize("backend", ["exact", "mio"])
def test_linear_respects_gamma_on_train(backend):
    X, y, prot = _data()
    gamma = 0.01
    clf = FairLinearClassifier(gamma=gamma, measure="fpsf", time_limit=60, detection_backend=backend)
    clf.fit(X, y, prot)
    y_hat = clf.predict(X)
    assert y_hat.shape == y.shape and y_hat.dtype == bool
    worst = find_subgroup(prot, y_hat, "fpsf", y_true=y)
    assert worst.value <= gamma + 1e-3
    assert clf.n_cuts_ == len(clf.cut_subgroups_) == len(clf.describe_cut_subgroups())


def test_linear_unconstrained_and_dataframe():
    X, y, prot = _data(1)
    df = pd.DataFrame(X[:, :3], columns=["a", "b", "c"])
    df_prot = pd.DataFrame({"sex": np.where(prot[:, 0], "F", "M"), "group": np.where(prot[:, 1], "x", "y")})
    clf = FairLinearClassifier(measure=None, time_limit=30).fit(df, y, df_prot)
    assert clf.n_cuts_ == 0
    assert 0.5 < clf.score(df, y) <= 1.0


def test_dnf():
    X, y, prot = _data(2, n=200)
    Xb = X[:, :3] > 0
    clf = FairDNFClassifier(n_terms=2, gamma=0.02, measure="fpsf", time_limit=60).fit(Xb, y, prot)
    y_hat = clf.predict(Xb)
    assert measures.fpsf(y, y_hat, find_subgroup(prot, y_hat, "fpsf", y_true=y).mask) <= 0.02 + 1e-3
    assert isinstance(clf.describe(), str)


def test_linear_with_negated_subgroups():
    X, y, prot = _data(3)
    clf = FairLinearClassifier(gamma=0.01, measure="fpsf", subgroup_negations=True, time_limit=60).fit(X, y, prot)
    worst = find_subgroup(prot, clf.predict(X), "fpsf", y_true=y, negations=True)
    assert worst.value <= 0.01 + 1e-3
    assert len(clf.protected_feature_names_) == 2 * prot.shape[1]
