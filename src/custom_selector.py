from typing import List, Union

import pandas as pd

from sklearn.feature_selection import f_classif
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, corr_threshold: float = 0.9):
        """Column selector"""
        self.corr_threshold = corr_threshold
        self.selected = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fits the transformer to the input."""
        corrs = X.corr().abs().stack().reset_index()
        corrs.columns = ['feature_1', 'feature_2', 'abs_corr']
        corrs = corrs[(corrs.feature_1 != corrs.feature_2) & (corrs["abs_corr"] >= self.corr_threshold)]
        corrs = corrs.sort_values("abs_corr", ascending=False)

        table = pd.DataFrame([dict(zip(X.columns, 
                                       f_classif(X, y)[0]))],
                             index=["F-Value"]).T

        table.sort_values("F-Value", ascending=False, inplace=True)

        self.selected = []
        for feature in table.index:
            if feature not in corrs["feature_1"].values or \
                all([x not in self.selected for x in corrs[corrs["feature_1"] == feature]["feature_2"].values]):
                self.selected.append(feature)
        return self

    def transform(self, X: pd.DataFrame):
        """
        Transforms a dataframe by selecting the specified columns.
        """
        assert isinstance(X, pd.DataFrame)
        Xcopy = X.copy()
        try:
            X_out = Xcopy[self.selected]
            if isinstance(X_out, pd.Series):
                X_out = X_out.to_frame()
            return X_out
        except KeyError:
            cols_error = list(set(self.selected) - set(Xcopy.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)