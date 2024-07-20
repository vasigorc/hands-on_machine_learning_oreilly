from common import get_train_test_data
from geo import prepreocessing_with_geo
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from svr import randomized_hyperparameter_distributions, svr_kernel_h

"""
Automatically explore some preparation options using `GridSearchCV`.

NB In the book they actually use `RandomizedSearchCV` flavor of `GridSearchCV`
"""
housing, housing_labels, _ = get_train_test_data()

# Then,we can re--use part of it in the geo-enhanced search
geo_pipeline = Pipeline(
    [
        ("preprocessing", prepreocessing_with_geo),
        (
            "svr",
            SVR(),
        ),
    ]
)

geo_param_distributions = {
    k: v
    for k, v in randomized_hyperparameter_distributions.items()
    if k != svr_kernel_h
}

geo_param_distributions.update(
    {
        "preprocessing__geo__estimator__n_neighbors": range(1, 30),
        "preprocessing__geo__estimator__weights": ["distance", "uniform"],
    }
)

geo_rnd_search = RandomizedSearchCV(
    geo_pipeline,
    param_distributions=geo_param_distributions,
    n_iter=50,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42,
)

geo_rnd_search.fit(housing.iloc[:5000], housing_labels.iloc[:5000])

# Print the best parameters found
print("Best parameters:", geo_rnd_search.best_params_)

# Print the best score achieved
geo_rnd_search_rmses = -geo_rnd_search.best_score_
print("Best score:", geo_rnd_search_rmses)

"""
This should roughly print:
Best parameters: {'preprocessing__geo__estimator__n_neighbors': 20, 'preprocessing__geo__estimator__weights': 'distance', 'svr__C': 55456.48365602121, 'svr__gamma': 0.006976409181650647}
Best score: 108386.28622878295
"""
