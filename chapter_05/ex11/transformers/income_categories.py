import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class IncomeCategoriesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformer for creating income buckets"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        income_cat = pd.cut(
            X["MedInc"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5]
        )
        return np.array(income_cat).reshape(-1, 1)  # transform 1D array to 2D array

    def get_feature_names_out(self, feature_names=None):
        return ["income_category"]
