import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureCombinationsEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for combined features"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X["AveRooms"] / X["AveOccup"]
        bedrooms_per_room = X["AveBedrms"] / X["AveRooms"]
        population_per_household = X["Population"] / X["AveOccup"]
        return np.c_[rooms_per_household, bedrooms_per_room, population_per_household]

    def get_feature_names_out(self, feature_names=None):
        return ["rooms_per_household", "bedrooms_per_room", "population_per_household"]
