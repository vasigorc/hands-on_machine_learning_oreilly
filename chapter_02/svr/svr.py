from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR

from common import get_preprocessor

preprocessing = get_preprocessor()

# hyperparameter fields
svr_kernel_h = "svr__kernel"
svr_c_h = "svr__C"
svr_gamma_h = "svr__gamma"

param_grid = [
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

svr_pipeline = make_pipeline(preprocessing, SVR())
rnd_search = RandomizedSearchCV(
    svr_pipeline,
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42,
)
