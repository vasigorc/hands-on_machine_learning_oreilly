import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class EnhancedGeographicFeatures(BaseEstimator, TransformerMixin):
    """Creates distance-based geographic features"""

    # Major California city coordinates
    SF_COORDS = (37.77, -122.42)  # San Francisco
    SJ_COORDS = (37.34, -121.89)  # San Jose
    LA_COORDS = (34.05, -118.24)  # Los Angeles
    SD_COORDS = (32.772, -117.16)  # San Diego

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        lat = X["Latitude"]
        long = X["Longitude"]

        # Calculate distances to major cities
        sf_distance = np.sqrt(
            (lat - self.SF_COORDS[0]) ** 2 + (long - self.SF_COORDS[1]) ** 2
        )
        sj_distance = np.sqrt(
            (lat - self.SJ_COORDS[0]) ** 2 + (long - self.SJ_COORDS[1]) ** 2
        )
        la_distance = np.sqrt(
            (lat - self.LA_COORDS[0]) ** 2 + (long - self.LA_COORDS[1]) ** 2
        )
        sd_distance = np.sqrt(
            (lat - self.SD_COORDS[0]) ** 2 + (long - self.SD_COORDS[1]) ** 2
        )

        # Calculate minimum distance to any major city
        min_distance = np.minimum(
            np.minimum(np.minimum(sf_distance, sj_distance), la_distance), sd_distance
        )

        # Calculate coastal proximmity
        coastal_distance = (lat - 36) - (long + 122) / 2

        return np.c_[
            sf_distance,
            sj_distance,
            la_distance,
            sd_distance,
            min_distance,
            coastal_distance,
        ]

    def get_feature_names_out(self, feature_names=None):
        return [
            "sf_distance",
            "sj_distance",
            "la_distance",
            "sd_distance",
            "min_city_distance",
            "coastal_distance",
        ]
