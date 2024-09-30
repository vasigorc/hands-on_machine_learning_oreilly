import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import (
    ColumnTransformer,
    make_column_selector,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def column_ratio(X):
    return X[:, [0]] / X[:, [1]]


def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out


def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler(),
    )


log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler(),
)


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(
            self.n_clusters, n_init=10, random_state=self.random_state  # type: ignore[arg-type]
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)

default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")
)


def get_preprocessor():
    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)
    default_num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    )
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")
    )

    return ColumnTransformer(
        [
            ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
            ("people_per_house", ratio_pipeline(), ["population", "households"]),
            (
                "log",
                log_pipeline,
                [
                    "total_bedrooms",
                    "total_rooms",
                    "population",
                    "households",
                    "median_income",
                ],
            ),
            ("geo", cluster_simil, ["latitude", "longitude"]),
            ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        ],
        remainder=default_num_pipeline, # type: ignore[arg-type]
    )
