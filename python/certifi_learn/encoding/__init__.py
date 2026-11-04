"""Encoding of tabular data: one-hot encoding / scaling (DataHandler) and
binarization of features into boolean literals (Binarizer)."""

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .binarizer import Bin, Binarizer
from .data_handler import DataHandler

__all__ = ["TabularEncoder", "DataHandler", "Binarizer", "Bin"]


class TabularEncoder:
    """Encodes tabular data for the models.

    For a DataFrame, categorical columns (``categorical``, by default all non-numeric
    columns) are one-hot encoded and numeric ones scaled (:meth:`numeric`), or all
    features are binarized into boolean literals, e.g. ``SEX == Female`` or
    ``AGEP in [18, 25)`` (:meth:`binary`). Arrays are passed through unchanged
    (``binary`` can add negated columns).
    """

    def __init__(self, X, y=None, categorical=None):
        self.is_frame = isinstance(X, pd.DataFrame)
        if not self.is_frame:
            self.n_features = np.asarray(X).shape[1]
            return
        if categorical is None:
            categorical = [c for c in X.columns if not is_numeric_dtype(X[c])]
        categ_map = {c: [] for c in X.columns if c in categorical}
        y = None if y is None else pd.Series(np.asarray(y).astype(bool))
        self.dhandler = DataHandler.from_data(X, y, categ_map=categ_map)
        self.binarizer = Binarizer(self.dhandler)

    def numeric(self, X) -> np.ndarray:
        if self.is_frame:
            return self.dhandler.encode(X)
        return np.asarray(X, dtype=float)

    def binary(self, X, negations: bool) -> np.ndarray:
        if self.is_frame:
            # without full negations, still include both values of binary features (e.g. sex)
            return self.binarizer.encode(X, include_negations=negations, include_binary_negations=True)
        X = np.asarray(X).astype(bool)
        return np.hstack([X, ~X]) if negations else X

    def numeric_names(self) -> list[str]:
        if self.is_frame:
            return list(self.dhandler.get_feature_names(encoded=True))
        return [f"x{j}" for j in range(self.n_features)]

    def binary_names(self, negations: bool) -> list[str]:
        if self.is_frame:
            return list(self.binarizer.feature_names(include_negations=negations, include_binary_negations=True))
        names = [f"x{j}" for j in range(self.n_features)]
        return names + [f"NOT x{j}" for j in range(self.n_features)] if negations else names
