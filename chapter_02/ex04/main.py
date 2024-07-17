import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from scipy.stats import loguniform, expon
from common import TransformerFromRegressor
from common import get_train_test_data, get_preprocessor
from svr import rnd_search, svr_kernel_h, svr_c_h, svr_gamma_h

"""
Try creating a custom transformer that trains a _k_ - nearest neighbors regressor 
(`sklearn.neighbors.KNeighborsRegressor`) in its `fit()` method, and outputs the model's predictions 
in its `transform()` method. Then add this feature to the preprocessing pipeline, using latitude and 
longitude as the inputs to this transformer. This will add a feature in the model that corresponds 
to the housing median price of the nearest districts.
"""

knn_regressor = KNeighborsRegressor(n_neighbors=3, weights="distance")
knn_transformer = TransformerFromRegressor(knn_regressor)

# now we have to tweak our default preprocessing to include this transformer
preprocessing = get_preprocessor()
transformers = [
    (name, clone(transformer), columns)
    for name, transformer, columns in preprocessing.transformers
]
geo_index = [name for name, _, _ in transformers].index("geo")
transformers[geo_index] = ("geo", knn_transformer, ["latitude", "longitude"])
prepreocessing_with_geo = ColumnTransformer(transformers)

housing, housing_labels, strat_test_set = get_train_test_data()

new_param_distribs = {
    "svr__kernel": ["linear", "rbf"],
    "svr__C": loguniform(20, 200_000),
    "svr__gamma": expon(scale=1.0),
}

rnd_search.param_distributions = new_param_distribs

# train the model
rnd_search.fit(housing.iloc[:5000], housing_labels.iloc[:5000])

# build the new pipeline
geo_pipeline = Pipeline(
    [
        ("preprocessing", prepreocessing_with_geo),
        (
            "svr",
            SVR(
                C=rnd_search.best_params_[svr_c_h],
                gamma=rnd_search.best_params_[svr_gamma_h],
                kernel=rnd_search.best_params_[svr_kernel_h],
            ),
        ),
    ]
)

housing, housing_labels, _ = get_train_test_data()

geo_pipeline_rmses = -cross_val_score(
    geo_pipeline,
    housing.iloc[:5000],
    housing_labels.iloc[:5000],
    scoring="neg_root_mean_squared_error",
    cv=3,
)

print("RMSE Statistics: \n")
rmse_stats = pd.Series(geo_pipeline_rmses).describe()
print(rmse_stats)
