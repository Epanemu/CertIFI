"""Detection of the most biased subgroup.

Subgroups are described over (binarized) protected attributes either as
conjunctions of attributes (e.g. ``sex=female AND race=white``) or as linear
threshold functions ``X @ coef >= threshold``.

Two backends are available:

* ``"exact"``: exact branch-and-bound search implemented in Rust. Fast, needs no
  solver, supports conjunction subgroups only.
* ``"mio"``: Mixed-Integer Optimization formulation solved by Gurobi or HiGHS
  (requires ``pyomo`` and the solver). Supports conjunction and linear subgroups.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from . import measures

__all__ = ["find_subgroup", "DetectedSubgroup"]

MEASURES = ("msd", "spsf", "fpsf")
SUBGROUP_TYPES = ("conjunction", "linear")


@dataclass
class DetectedSubgroup:
    """A detected subgroup and its violation on the data it was detected on."""

    kind: str  # "conjunction" or "linear"
    measure: str
    value: float  # violation of `measure` on the input data
    mask: np.ndarray = field(repr=False)  # membership of the input samples
    optimal: bool  # True if proven to be the most biased subgroup
    backend: str
    literals: list[int] | None = None  # conjunction: indices of columns that must be 1
    negated_literals: list[int] | None = None  # conjunction: indices of columns that must be 0
    coef: np.ndarray | None = None  # linear: X @ coef >= threshold
    threshold: float | None = None

    def contains(self, X) -> np.ndarray:
        """Boolean membership of the subgroup for (new) data X."""
        X = np.asarray(X)
        if self.kind == "conjunction":
            Xb = X.astype(bool)
            mask = np.ones(X.shape[0], dtype=bool)
            for j in self.literals:
                mask &= Xb[:, j]
            for j in self.negated_literals or []:
                mask &= ~Xb[:, j]
            return mask
        return X @ self.coef >= self.threshold

    def describe(self, feature_names=None) -> str:
        """Human-readable description of the subgroup."""
        if self.kind == "conjunction":
            def name(j):
                return str(feature_names[j]) if feature_names is not None else f"x{j}"

            parts = [name(j) for j in self.literals] + [f"NOT {name(j)}" for j in self.negated_literals or []]
            return " AND ".join(parts) if parts else "(all samples)"
        terms = []
        for j, c in enumerate(self.coef):
            if c != 0:
                name = str(feature_names[j]) if feature_names is not None else f"x{j}"
                terms.append(f"{c:+.3g}*{name}")
        return " ".join(terms) + f" >= {self.threshold:.3g}"


def find_subgroup(
    X,
    y,
    measure: str = "msd",
    *,
    y_true=None,
    subgroups: str = "conjunction",
    backend: str = "auto",
    time_limit: float = 300,
    negations: bool = False,
    n_min: int = 0,
    gamma: float | None = None,
    solver: str = "gurobi",
    warmstart: bool = True,
    verbose: bool = False,
) -> DetectedSubgroup | None:
    """Find the subgroup with the highest bias.

    Args:
        X: protected attributes, shape (n, d). Binary for conjunction subgroups;
            any numeric encoding for linear subgroups.
        y: for ``measure="msd"``, the binary vector splitting the data into the two
            compared distributions (e.g., true labels, to find bias in data, or
            predictions); for ``"spsf"`` and ``"fpsf"``, the classifier predictions.
        measure: "msd" (Subgroup Discrepancy; MSD for conjunctions), "spsf"
            (Statistical Parity Subgroup Fairness) or "fpsf" (False Positive
            Subgroup Fairness, requires ``y_true``).
        y_true: true labels, required for ``measure="fpsf"``.
        subgroups: "conjunction" or "linear".
        backend: "exact", "mio", or "auto" (exact for conjunctions, MIO for linear).
        time_limit: time limit of the search in seconds.
        negations: also allow negated literals in conjunctions (e.g. ``NOT race = White``).
        n_min: minimal number of samples in the subgroup.
        gamma: if given, only a subgroup with violation above gamma is returned,
            otherwise None (useful for checking gamma-fairness).
        solver: "gurobi" or "highs" (MIO backend only).
        warmstart: warm start linear subgroups with logistic regression (MIO only).
        verbose: print solver output / log progress.

    Returns:
        The detected subgroup, or None if ``gamma`` is given and no subgroup exceeds it.
    """
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES}")
    if subgroups not in SUBGROUP_TYPES:
        raise ValueError(f"subgroups must be one of {SUBGROUP_TYPES}")
    if backend == "auto":
        backend = "exact" if subgroups == "conjunction" else "mio"
    if backend not in ("exact", "mio"):
        raise ValueError("backend must be one of 'auto', 'exact', 'mio'")
    if backend == "exact" and subgroups == "linear":
        raise ValueError("Linear subgroups are supported only by the 'mio' backend")
    if measure == "fpsf" and y_true is None:
        raise ValueError("measure='fpsf' requires y_true")
    if negations and subgroups != "conjunction":
        raise ValueError("negations are supported only for conjunction subgroups")

    X = np.asarray(X)
    y = np.asarray(y).astype(bool)
    if y_true is not None:
        y_true = np.asarray(y_true).astype(bool)
    group_type = "rule" if subgroups == "conjunction" else "linear"
    X_search = X.astype(bool) if subgroups == "conjunction" else X.astype(float)
    d = X.shape[1]
    if negations:
        # negated literals as extra columns: index d + j means "column j is 0"
        X_search = np.hstack([X_search, ~X_search])

    common = dict(verbose=verbose, n_min=n_min, time_limit=time_limit, group_type=group_type, return_opt_flag=True)

    if backend == "exact":
        from ._exact_backend import ExactDetector

        desc, optimal = ExactDetector().find_subgroup(
            X_search, y, bias_type=measure, gamma=gamma, y_true=y_true, **common
        )
    else:
        from ._mio import MIODetector

        detector = MIODetector()
        mio_args = dict(solver=solver, warmstart=warmstart, **common)
        if measure == "fpsf":
            # FPSF is SPSF on the negative samples, scaled by P(y=0)
            neg = ~y_true
            desc, optimal = detector.find_subgroup(
                X_search[neg], y[neg], bias_type="spsf",
                gamma=None if gamma is None else gamma / neg.mean(), **mio_args
            )
        else:
            desc, optimal = detector.find_subgroup(X_search, y, bias_type=measure, gamma=gamma, **mio_args)

    if desc is None:
        if not optimal:
            warnings.warn(
                "The search stopped (time limit) before finding a subgroup with violation above gamma; "
                "this does not certify gamma-fairness."
            )
        return None

    if subgroups == "conjunction":
        result = DetectedSubgroup(
            "conjunction", measure, 0.0, None, bool(optimal), backend,
            literals=[int(j) for j in desc if j < d], negated_literals=[int(j) - d for j in desc if j >= d],
        )
    else:
        result = DetectedSubgroup(
            "linear", measure, 0.0, None, bool(optimal), backend,
            coef=np.asarray(desc[:-1], dtype=float), threshold=float(desc[-1]),
        )
    result.mask = result.contains(X_search[:, :d])
    if measure == "msd":
        result.value = float(measures.msd(y, result.mask))
    elif measure == "spsf":
        result.value = float(measures.spsf(y, result.mask))
    else:
        result.value = float(measures.fpsf(y_true, y, result.mask))
    return result
