import pandas as pd
from common import get_preprocessor, get_train_test_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
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

"""
Exercise 03: Try adding a `SelectFromModel` transformer in the preparation pipeline to select only the most
important attributes
"""

housing, housing_labels, strat_test_set = get_train_test_data()
preprocessing = get_preprocessor()

rnd_search.param_distributions = randomized_hyperparameter_distributions

# train the model
rnd_search.fit(housing.iloc[:5000], housing_labels.iloc[:5000])

min_feature_importance = 0.005
selector = SelectFromModel(
    RandomForestRegressor(random_state=42), threshold=min_feature_importance
)

svr = SVR(
    C=rnd_search.best_params_[svr_c_h],
    gamma=rnd_search.best_params_[svr_gamma_h],
    kernel=rnd_search.best_params_[svr_kernel_h],
)

svr_pipeline_with_select = Pipeline(
    [("preprocessing", preprocessing), ("selector", selector), ("svr", svr)]
)


"""
Scikit-Learn's cross-validation features expect a utility function (greater is better)
rather than a cost function (lower is better), so the scoring function is actually the
opposite of the RMSE. It's a negative value, so you need to switch the sign of the output
to get the RMSE scores.

cross-val-score splits the training set into 3 non-overlapping subsets
"""
selector_rmses = -cross_val_score(
    svr_pipeline_with_select,
    housing.iloc[:5000],
    housing_labels.iloc[:5000],
    scoring="neg_root_mean_squared_error",
    cv=3,
)

print("RMSE Statistics: \n")
rmse_stats = pd.Series(selector_rmses).describe()
print(rmse_stats)

print("Feature importance scores")
# first we need to train the selector
svr_pipeline_with_select.fit(housing.iloc[:5000], housing_labels.iloc[:5000])

# we need to match number of features in housing and selector
# get feature names after preprocessing
preprocessed_features = preprocessing.fit_transform(housing.iloc[:5000])
feature_names = preprocessing.get_feature_names_out()

selected_features = [
    feature
    for feature, selected in zip(feature_names, selector.get_support())
    if selected
]
feature_importance = selector.estimator_.feature_importances_[selector.get_support()]
for feature, importance in zip(selected_features, feature_importance):
    print(f"{feature}: {importance}")

print(f"\nNumber of original features: {len(housing.columns)}")
print(f"Number of features after preprocessing: {len(feature_names)}")
print(f"Number of selected features: {len(selected_features)}")

"""
This module, when run, should roughly print out something as follows:

python3 -m ex03.main
RMSE Statistics:

count        3.000000
mean     57106.007292
std        890.463390
min      56425.910779
25%      56602.057050
50%      56778.203321
75%      57446.055549
max      58113.907777
dtype: float64
Feature importance scores
bedrooms__ratio: 0.019954181314926697
rooms_per_house__ratio: 0.024248804696219116
people_per_house__ratio: 0.11045230012752531
log__total_bedrooms: 0.008998734714923746
log__total_rooms: 0.010655919710952502
log__population: 0.009324877584051101
log__households: 0.009130603305982343
log__median_income: 0.4727032906348246
geo__Cluster 0 similarity: 0.021495102131050405
geo__Cluster 1 similarity: 0.007694400334301747
geo__Cluster 2 similarity: 0.027954378190200403
geo__Cluster 3 similarity: 0.023498042204069366
geo__Cluster 4 similarity: 0.011602464255126857
geo__Cluster 5 similarity: 0.008702242991321918
geo__Cluster 6 similarity: 0.006823493026189014
geo__Cluster 7 similarity: 0.014433382392541263
geo__Cluster 8 similarity: 0.02012215199054772
geo__Cluster 9 similarity: 0.01313440552826559
cat__ocean_proximity_INLAND: 0.1412630474957612
remainder__housing_median_age: 0.03327087325969262

Number of original features: 9
Number of features after preprocessing: 24
Number of selected features: 20

The current model's RMSE is higher than the baseline (baseline mean was 47,002.93), which indicates that it's performing worse.

This means that, on average, the current model's predictions are off by about 10,103 more units (presumably dollars) compared to the baseline model.
To calculate the percentage difference:
(57,106.01 - 47,002.93) / 47,002.93 * 100 ≈ 21.5%
So, the current model is performing approximately 21.5% worse than the baseline in terms of RMSE.
"""
