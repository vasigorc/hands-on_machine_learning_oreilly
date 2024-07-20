from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from scipy.stats import loguniform, expon

from common import get_preprocessor

preprocessing = get_preprocessor()

# hyperparameter fields
svr_kernel_h = "svr__kernel"
svr_c_h = "svr__C"
svr_gamma_h = "svr__gamma"

hyperparameter_values = [
    {
        svr_kernel_h: ["linear"],
        svr_c_h: [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0],
    },
    {
        svr_kernel_h: ["rbf"],
        svr_c_h: [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0],
        svr_gamma_h: [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    },
]


# from Exercise 2
"""
Note that we used `expon()` distribution for `gamma`, with a scale of 1, so `RandomSearch` mostly
searched for values roughly of that scale.

We used `loguniform()` distribution for `C`, meaning we did not have a clue what the optimal scale
of `C` was before running the random search.
"""
randomized_hyperparameter_distributions = {
    svr_kernel_h: ["linear", "rbf"],
    svr_c_h: loguniform(20, 200_000),
    svr_gamma_h: expon(scale=1.0),
}

svr_pipeline = make_pipeline(preprocessing, SVR())
rnd_search = RandomizedSearchCV(
    svr_pipeline,
    param_distributions=hyperparameter_values,
    n_iter=10,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42,
)
