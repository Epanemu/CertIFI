"""Interpretable classifiers trained under intersectional fairness constraints.

The classifiers are trained by Mixed-Integer Optimization (MIO, solved by Gurobi).
The (exponentially many) fairness constraints, one per subgroup, are added lazily:
whenever the solver finds a new candidate classifier, the most biased subgroup is
searched for and, if its bias exceeds ``gamma``, a constraint for it is added.

Both classifiers follow the scikit-learn interface, with the protected attributes
passed separately: ``fit(X, y, X_protected)``, ``predict(X)``.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .encoding import TabularEncoder

__all__ = ["FairLinearClassifier", "FairDNFClassifier"]

# public measure names -> internal fairness types
_MEASURES = {
    None: "",
    "fpsf": "FPSF",
    "fpsf_by_sd": "MSD0",
    "spsf": "SPSF",
    "msd": "MSD",
}
_SUBGROUPS = {"conjunction": "rule", "linear": "linear"}


class _FairClassifier(ClassifierMixin, BaseEstimator):
    def _check_params(self):
        if self.measure not in _MEASURES:
            raise ValueError(f"measure must be one of {list(_MEASURES)}")
        if self.subgroups not in _SUBGROUPS:
            raise ValueError(f"subgroups must be one of {list(_SUBGROUPS)}")
        if self.detection_backend not in ("exact", "mio"):
            raise ValueError("detection_backend must be 'exact' or 'mio'")
        if self.subgroup_negations and self.subgroups != "conjunction":
            raise ValueError("subgroup_negations is supported only for conjunction subgroups")

    def _encode_protected(self, X_protected, y):
        self.protected_encoder_ = TabularEncoder(X_protected, y, self.categorical)
        if self.subgroups == "conjunction":
            self.protected_feature_names_ = self.protected_encoder_.binary_names(self.subgroup_negations)
            return self.protected_encoder_.binary(X_protected, self.subgroup_negations)
        self.protected_feature_names_ = self.protected_encoder_.numeric_names()
        return self.protected_encoder_.numeric(X_protected)

    def _store_solver_info(self, mio):
        self.n_cuts_ = mio.n_cuts
        self.cut_subgroups_ = list(mio.cut_subgroups)
        self.termination_condition_ = str(mio.mio_result.solver.termination_condition)

    def describe_cut_subgroups(self) -> list[str]:
        """Readable descriptions of the subgroups that were added as fairness constraints."""
        out = []
        for desc in self.cut_subgroups_:
            if self.subgroups == "conjunction":
                out.append(" AND ".join(self.protected_feature_names_[j] for j in desc) or "(all samples)")
            else:
                coef, thr = desc[:-1], desc[-1]
                terms = [f"{c:+.3g}*{n}" for c, n in zip(coef, self.protected_feature_names_) if c != 0]
                out.append(" ".join(terms) + f" >= {thr:.3g}")
        return out


class FairLinearClassifier(_FairClassifier):
    """Linear classifier ``X @ coef_ >= threshold_`` trained to optimality (balanced
    0-1 loss) under an intersectional fairness constraint.

    Args:
        gamma: fairness threshold; every subgroup must have bias at most gamma.
        measure: bias measure constrained: "fpsf" (False Positive Subgroup Fairness),
            "fpsf_by_sd" (FPSF constraints, violating subgroups found by maximizing
            Subgroup Discrepancy on negative samples), "spsf" (Statistical Parity
            Subgroup Fairness), "msd" (Maximum Subgroup Discrepancy), or None
            (no fairness constraint).
        subgroups: "conjunction" (conjunctions of protected attribute values) or
            "linear" (linear threshold functions of the protected attributes).
        subgroup_negations: allow negated literals in conjunction subgroups
            (e.g. ``RAC1P != White``), i.e., constrain a larger set of subgroups.
        sparsity: weight of the penalty on the proportion of non-zero coefficients.
        time_limit: time limit of the solver in seconds.
        detection_backend: search for violating subgroups: "exact" (default) or
            "mio". Linear subgroups always use "mio".
        warmstart: warm start the solver with a logistic regression model.
        categorical: column names treated as categorical when DataFrames are passed
            (default: all non-numeric columns).
        verbose: print the solver output.
    """

    def __init__(
        self,
        gamma: float = 0.01,
        measure: str | None = "fpsf",
        subgroups: str = "conjunction",
        subgroup_negations: bool = False,
        sparsity: float = 0.0,
        time_limit: float = 600,
        detection_backend: str = "exact",
        warmstart: bool = True,
        categorical: list | None = None,
        verbose: bool = False,
    ):
        self.gamma = gamma
        self.measure = measure
        self.subgroups = subgroups
        self.subgroup_negations = subgroup_negations
        self.sparsity = sparsity
        self.time_limit = time_limit
        self.detection_backend = detection_backend
        self.warmstart = warmstart
        self.categorical = categorical
        self.verbose = verbose

    def fit(self, X, y, X_protected):
        """Train the classifier.

        Args:
            X: features used by the classifier (array or DataFrame).
            y: binary labels.
            X_protected: protected attributes (array or DataFrame); binary if
                ``subgroups="conjunction"`` and an array is passed.
        """
        from ._linear import LinearFairClassifier

        self._check_params()
        y = np.asarray(y).astype(bool)
        self.encoder_ = TabularEncoder(X, y, self.categorical)
        X_enc = self.encoder_.numeric(X)
        X_prot = self._encode_protected(X_protected, y)

        mio = LinearFairClassifier(
            gamma=self.gamma,
            fairness_type=_MEASURES[self.measure],
            subgroup_type=_SUBGROUPS[self.subgroups],
            sparsity_term=self.sparsity,
            detection_backend=self.detection_backend,
        )
        coef, threshold = mio.find_classifier(
            X_enc, X_prot, y, time_limit=self.time_limit, epsilon=1e-3,
            warmstart=self.warmstart, verbose=self.verbose,
        )
        self.coef_ = np.asarray(coef, dtype=float)
        self.threshold_ = float(threshold)
        self.feature_names_ = self.encoder_.numeric_names()
        self.classes_ = np.array([False, True])
        self._store_solver_info(mio)
        return self

    def decision_function(self, X) -> np.ndarray:
        return self.encoder_.numeric(X) @ self.coef_ - self.threshold_

    def predict(self, X) -> np.ndarray:
        return self.decision_function(X) >= 0


class FairDNFClassifier(_FairClassifier):
    """DNF (rule set) classifier: predicts positive if any of ``n_terms`` conjunctions
    of (binarized) features holds. Trained to optimality (balanced 0-1 loss) under an
    intersectional fairness constraint.

    Args:
        n_terms: number of terms (rules) of the DNF.
        gamma, measure, subgroups, subgroup_negations, time_limit,
            detection_backend, categorical, verbose: see :class:`FairLinearClassifier`.

    Features passed as an array must be binary; their negations are added
    automatically, so the rules can use both ``x`` and ``NOT x``.
    """

    def __init__(
        self,
        n_terms: int = 3,
        gamma: float = 0.01,
        measure: str | None = "fpsf",
        subgroups: str = "conjunction",
        subgroup_negations: bool = False,
        time_limit: float = 600,
        detection_backend: str = "exact",
        categorical: list | None = None,
        verbose: bool = False,
    ):
        self.n_terms = n_terms
        self.gamma = gamma
        self.measure = measure
        self.subgroups = subgroups
        self.subgroup_negations = subgroup_negations
        self.time_limit = time_limit
        self.detection_backend = detection_backend
        self.categorical = categorical
        self.verbose = verbose

    def fit(self, X, y, X_protected):
        """Train the classifier (see :meth:`FairLinearClassifier.fit`)."""
        from ._dnf import DNFFairClassifier

        self._check_params()
        y = np.asarray(y).astype(bool)
        self.encoder_ = TabularEncoder(X, y, self.categorical)
        X_bin = self.encoder_.binary(X, negations=True)
        X_prot = self._encode_protected(X_protected, y)

        mio = DNFFairClassifier(
            gamma=self.gamma,
            fairness_type=_MEASURES[self.measure],
            subgroup_type=_SUBGROUPS[self.subgroups],
            detection_backend=self.detection_backend,
        )
        self.terms_ = mio.find_dnf(X_bin, X_prot, y, n_terms=self.n_terms, time_limit=self.time_limit, verbose=self.verbose)
        self.feature_names_ = self.encoder_.binary_names(negations=True)
        self.classes_ = np.array([False, True])
        self._store_solver_info(mio)
        return self

    def predict(self, X) -> np.ndarray:
        X_bin = self.encoder_.binary(X, negations=True)
        y_hat = np.zeros(X_bin.shape[0], dtype=bool)
        for term in self.terms_:
            y_hat |= np.all(X_bin[:, term], axis=1)
        return y_hat

    def describe(self) -> str:
        """The learned DNF as readable text."""
        rules = [" AND ".join(self.feature_names_[j] for j in term) or "TRUE" for term in self.terms_]
        return "\n OR ".join(f"({r})" for r in rules)
