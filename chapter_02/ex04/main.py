import pandas as pd
from common import get_train_test_data
from geo import prepreocessing_with_geo
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from svr import (
    randomized_hyperparameter_distributions,
    rnd_search,
    svr_c_h,
    svr_gamma_h,
    svr_kernel_h,
)

housing, housing_labels, strat_test_set = get_train_test_data()

rnd_search.param_distributions = randomized_hyperparameter_distributions

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
