"""Exact (Rust) subgroup search with the same interface as :class:`MIODetector`.

Only conjunction ("rule") subgroups are supported. Differences to the MIO detector:

* ``gamma`` is a strict bound: a subgroup is returned only if its violation is
  strictly greater than ``gamma`` (``None`` otherwise).
* ``solver``, ``warmstart`` and ``warmstart_model`` are accepted and ignored.
* ``bias_type="fpsf"`` is supported directly (requires ``y_true``).
* Rows are deduplicated before the search, duplicate counts are passed as
  sample weights (this does not change the result, only speeds up the search).
"""

from __future__ import annotations

import logging

import numpy as np

from . import exact

logger = logging.getLogger(__name__)

__all__ = ["ExactDetector"]


def _dedupe(X, *labels):
    """Collapse duplicate ``(X, *labels)`` rows into unique rows plus their counts.

    Returns ``(X_dedup, label_0_dedup, ..., counts)``.
    """
    n, d = X.shape
    cols = [np.asarray(X, dtype=np.int64)]
    for lbl in labels:
        cols.append(np.asarray(lbl, dtype=np.int64).reshape(n, 1))
    stacked = np.concatenate(cols, axis=1)
    uniq, counts = np.unique(stacked, axis=0, return_counts=True)
    X_dedup = uniq[:, :d]
    label_dedups = tuple(uniq[:, d + i] for i in range(len(labels)))
    return (X_dedup,) + label_dedups + (counts.astype(float),)


class ExactDetector:
    """Find the conjunction subgroup with maximum violation via the exact Rust search."""

    def __init__(self) -> None:
        self.last_result = None

    @staticmethod
    def _extract_gap(result: exact.Subgroup) -> tuple[float, float, float]:
        """(gap, bound, value): zero gap if the search finished, unknown (inf) otherwise."""
        if result.exact:
            return 0.0, result.violation, result.violation
        return float("inf"), float("inf"), result.violation

    def find_subgroup(
        self,
        X,
        y,
        verbose: bool = False,
        n_min: int = 0,
        time_limit: int = 300,
        bias_type: str = "msd",
        group_type: str = "rule",
        probabilistic: bool = False,
        return_opt_flag: bool = False,
        return_gap: bool = False,
        warmstart_model: dict | None = None,
        warmstart: bool = True,
        gamma: float | None = None,
        solver: str = "gurobi",
        y_true=None,
    ):
        """Find the conjunction maximizing the violation.

        Args:
            X: binary protected attributes, shape (n, d).
            y: for "msd", the two groups to compare (e.g., labels or predictions);
                for "spsf" and "fpsf", the classifier output. Hard 0/1, or soft
                scores in [0, 1] if ``probabilistic=True``.
            bias_type: "msd", "spsf" or "fpsf".
            y_true: true labels, required for "fpsf".
            gamma: return only a subgroup with violation strictly above gamma.

        Returns:
            list[int] of column indices of the conjunction (``X[:, j] == 1``), or
            None if no subgroup exceeds ``gamma``. With ``return_opt_flag``, also
            whether the search finished (proven optimal); with ``return_gap``,
            also (gap, bound, value).
        """
        if group_type != "rule":
            raise NotImplementedError(
                "The exact search supports only conjunction ('rule') subgroups; "
                "use the MIO backend for linear subgroups."
            )
        if bias_type not in ("msd", "spsf", "fpsf"):
            raise ValueError("Incorrect bias type, choose one of ['msd', 'spsf', 'fpsf']")

        X = np.asarray(X)
        y = np.asarray(y, dtype=np.float64) if probabilistic else np.asarray(y)
        if y.shape != (X.shape[0],):
            raise ValueError("y must be a vector with one value per row of X")
        n = y.shape[0]

        if not probabilistic:
            y_bool = y.astype(bool)
            # with a single class in y, there is nothing to compare (and msd weights are undefined)
            size1 = int(np.sum(y_bool))
            if bias_type in ("msd", "spsf") and (size1 == 0 or size1 == n):
                if return_opt_flag:
                    return [], True
                return []

        d = X.shape[1]
        groups = [[j] for j in range(d)]  # positive literals only, as in the MIO formulation
        n_min_active = float(n_min) if n_min > 0 else None
        kwargs = dict(
            groups=groups,
            allow_negations=False,
            time_limit=time_limit,
            gamma=gamma,
            min_coverage=n_min_active,
        )

        if bias_type in ("msd", "spsf"):
            metric_fn = exact.msd if bias_type == "msd" else exact.spsf
            if probabilistic:
                result = metric_fn(X, y, **kwargs)
            else:
                X_dedup, y_dedup, counts = _dedupe(X, y_bool.astype(np.uint8))
                result = metric_fn(X_dedup, y_dedup, sample_weight=counts, **kwargs)
        else:  # fpsf
            if y_true is None:
                raise ValueError("bias_type='fpsf' requires y_true (the true labels)")
            y_true = np.asarray(y_true)
            if y_true.shape != (X.shape[0],):
                raise ValueError("y_true must be a vector with one value per row of X")
            if probabilistic:
                result = exact.fpsf(X, y, y_true, **kwargs)
            else:
                X_dedup, yhat_dedup, ytrue_dedup, counts = _dedupe(
                    X, y_bool.astype(np.uint8), y_true.astype(np.uint8)
                )
                result = exact.fpsf(X_dedup, yhat_dedup, ytrue_dedup, sample_weight=counts, **kwargs)

        self.last_result = result
        rule = [col for col, _val in result.literals]

        if verbose:
            logger.info(
                "exact search finished: bias_type=%s rule=%s violation=%.6g reason=%s",
                bias_type, rule, result.violation, result.reason,
            )

        if gamma is not None and not rule:
            # nothing above gamma; proven if the search finished, otherwise just not found in time
            if return_opt_flag:
                return None, result.exact
            return None

        if return_opt_flag:
            return rule, result.exact
        if return_gap:
            return (rule,) + self._extract_gap(result)
        return rule

    def find_subgroup_with_ingroup(
        self,
        X,
        y,
        verbose: bool = False,
        n_min: int = 0,
        time_limit: int = 300,
        bias_type: str = "msd",
        group_type: str = "rule",
        probabilistic: bool = False,
        return_opt_flag: bool = False,
        warmstart_model: dict | None = None,
        gamma: float | None = None,
        solver: str = "gurobi",
        y_true=None,
    ):
        """Like :meth:`find_subgroup`, but also returns the boolean membership mask."""
        group_desc, opt = self.find_subgroup(
            X,
            y,
            verbose=verbose,
            n_min=n_min,
            time_limit=time_limit,
            bias_type=bias_type,
            group_type=group_type,
            probabilistic=probabilistic,
            return_opt_flag=True,
            gamma=gamma,
            y_true=y_true,
        )
        if group_desc is None:
            ingroup = None
        else:
            X_bin = np.asarray(X).astype(bool)
            ingroup = np.ones(X_bin.shape[0], dtype=bool)
            for feat_i in group_desc:
                ingroup &= X_bin[:, feat_i]

        if return_opt_flag:
            return group_desc, ingroup, opt
        return group_desc, ingroup
