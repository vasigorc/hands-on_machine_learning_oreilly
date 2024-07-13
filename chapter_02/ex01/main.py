from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
import pandas as pd

from common import get_train_test_data, get_preprocessor

"""
Exercise 01: Try  a support vector machine regressor (`sklearn.svm.SVR`) with various hyperparameters, such as 
`kernel = "linear"` (with various values for the C hyperparameter) or `kernel="rbf"` (with various 
values for the C and `gamma` hyperparameters). Note that support vector machines don't scale well 
to large datasets, so you should probably train your model on just the first 5000 instances of the 
training set and use only 3-fold cross-validation, or else it will take hours. Don't worry about 
what the hyperparameters mean for now; we'll discuss them in Chapter 5. How does the best SVR 
predictor perform?

&&

Exercise 02: Try replacing the `GridSearchCV` with a `RandomizedSearchCV`


NB: these are the meaniing of the hyperparameters

1. `C`:

- represents the regularization (constraining the model to make it simpler and reduce the risk of overfitting) parameter
- it controls the trade-off between achieving a low error on the training data and minimizing the norm of the weights (i.e. model complexity)
- a small value for `C` makes the decision surface smooth, while a larger value of `C` aims to fit the training data as well as possible

2. `kernel`

- Common `kernel` types to be used in the algorithm:
    - `linear`
    - `poly`: Polynomial kernel
    - `rbf`: [default] Radial basis function (RBF) kernel (also called Gaussian kernel)
    - `sigmoid`
- this choice affects the ability of the model to fit non-linear relationships in the data

3. `gamma`

- this parameter is specific to `rbf`, `poly` and `sigmoid` kernels
- it defines how far the influence of a single training example reaches, with low values meaning 'far' and high values meaning 'close'
- the higher the `gamma` value, the closer other examples must be to be affected
- for `rbf` `kernel`, it acts as a parameter for the Gaussian function. A small `gamma` value means a Gaussian with a large variance 
and thus a smoother decision boundary, while a large `gamma` value leads to a Gaussian with a small variance, potentially leading to
overfitting
"""

# TODO Read-up about Gaussian function and its' parameters, and its variance (Math)
# TODO Read-up about norm weights

housing, housing_labels, strat_test_set = get_train_test_data()
preprocessing = get_preprocessor()

param_grid = [
    {
        "svr__kernel": ["linear"],
        "svr__C": [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0],
    },
    {
        "svr__kernel": ["rbf"],
        "svr__C": [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0],
        "svr__gamma": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
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
rnd_search.fit(housing.iloc[:5000], housing_labels.iloc[:5000])

# Add print statements to see the results
print("Best parameters:", rnd_search.best_params_)
print("Best RMSE:", -rnd_search.best_score_)

# Create a DataFrame of the results and print the top 5
rscv_res = pd.DataFrame(rnd_search.cv_results_)
rscv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
print("\nTop 5 models:")
print(
    rscv_res[["params", "mean_test_score", "std_test_score"]]
    .head()
    .to_string(index=False)
)

# Print the best model's performance on the test set
best_model = rnd_search.best_estimator_
test_rmse = -best_model.score(
    strat_test_set.drop("median_house_value", axis=1),
    strat_test_set["median_house_value"],
)
print(f"\nTest set RMSE: {test_rmse:.2f}")

"""
The above should roughly print the following:
Best parameters: {'svr__kernel': 'linear', 'svr__C': 5000.0}
Best RMSE: 73266.16126417006

Top 5 models:
                                                      params  mean_test_score  std_test_score
                 {'svr__kernel': 'linear', 'svr__C': 5000.0}    -73266.161264     2235.633346
{'svr__kernel': 'rbf', 'svr__gamma': 0.01, 'svr__C': 5000.0}    -79820.226074      907.830628
{'svr__kernel': 'rbf', 'svr__gamma': 0.01, 'svr__C': 1000.0}    -99030.456305     1123.584387
                   {'svr__kernel': 'linear', 'svr__C': 10.0}   -105522.031636      946.674236
 {'svr__kernel': 'rbf', 'svr__gamma': 0.05, 'svr__C': 100.0}   -111730.164331     1080.843694

Test set RMSE: -0.62

This is roughly twice as worse as `RandomForrestRegressor`
"""
