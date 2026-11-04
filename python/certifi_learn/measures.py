"""Measures of intersectional bias for a given subgroup.

All functions take boolean vectors (0/1 values are converted) and return the
(absolute) violation; with ``get_direction=True`` they also return whether the
subgroup is over- (True) or under-represented in the positive class.
"""

import numpy as np

__all__ = ["msd", "spsf", "fpsf"]


def _as_bool(v) -> np.ndarray:
    v = np.asarray(v)
    return v if v.dtype == bool else v == 1


def msd(y, ingroup, get_direction: bool = False):
    """Subgroup Discrepancy between the samples with y=1 and y=0:
    |P(S | y=1) - P(S | y=0)|. Maximized over conjunction subgroups, this is MSD.
    """
    y, ingroup = _as_bool(y), _as_bool(ingroup)
    if y.shape != ingroup.shape:
        raise ValueError("ingroup and y have different shape")
    p1, p0 = np.mean(ingroup[y]), np.mean(ingroup[~y])
    if get_direction:
        return np.abs(p1 - p0), p1 > p0
    return np.abs(p1 - p0)


def spsf(y_pred, ingroup, get_direction: bool = False):
    """Statistical Parity Subgroup Fairness: P(S) * |P(h=1) - P(h=1 | S)|."""
    y_pred, ingroup = _as_bool(y_pred), _as_bool(ingroup)
    p_group = np.mean(ingroup)
    p_pos = np.mean(y_pred)
    p_joint = np.mean(ingroup & y_pred)
    if get_direction:
        return np.abs(p_group * p_pos - p_joint), p_group * p_pos > p_joint
    return np.abs(p_group * p_pos - p_joint)


def fpsf(y_true, y_pred, ingroup, get_direction: bool = False):
    """False Positive Subgroup Fairness: P(S, y=0) * |P(h=1 | y=0) - P(h=1 | S, y=0)|.

    ``y_pred`` may also contain probabilities of the positive class.
    """
    y_true, ingroup = _as_bool(y_true), _as_bool(ingroup)
    y_pred = np.asarray(y_pred)
    if not (y_true.shape == y_pred.shape == ingroup.shape):
        raise ValueError("y_true, y_pred and ingroup have different shapes")
    p_group = np.mean(ingroup & ~y_true)
    p_pos = np.mean(y_pred[~y_true])
    p_joint = np.sum(y_pred[ingroup & ~y_true]) / y_true.shape[0]
    if get_direction:
        return np.abs(p_group * p_pos - p_joint), p_group * p_pos > p_joint
    return np.abs(p_group * p_pos - p_joint)
