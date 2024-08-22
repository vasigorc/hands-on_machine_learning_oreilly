from scipy.stats import randint as sp_randint
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

"""
`fetch_openml` returns the inputs as a Panda DataFrame and the labels as a Pandas Series. Since MNIST
dataset contains images, and DataFrames aren't ideal for that, so it's preferable to set as_frame=False
to get the data as NumPy arrays instead 
"""
mnist = fetch_openml("mnist_784", as_frame=False)

# tuple containing the input data and the targets
X, y = mnist.data, mnist.target

"""
MNIST dataset returned by `fetch_openml` is already split into a training set (first 60k images) and
a test set (last 10k images). Training set is already shuffled
"""
X_train, X_test, y_train, y_test = X[:60_000], X[60_000:], y[:60_000], y[60_000:]

y_train_5 = y_train == "5"  # True for all 5s

"""
Exercise 1. Try to build a classifier for the MNIST dataset that achieves over 97% accuracy on the 
test set. Hint: the `KNeighborsClassifier` works quite well for this task; you just need to find 
good hyperparameter values (try a grid search on the weights and n_neighbors hyperparameters)
"""
# create a KNeighborsClassifier object
knn_clf = KNeighborsClassifier()
# define the hyperparameter grid
param_dist = {
    "weights": ["uniform", "distance"],
    "n_neighbors": sp_randint(6, 16),
}  # range from 5 to 15
"""
Other available `scoring` parameters could be:
- f1_score: Combines precision and recall into a single metric.
- precision: Measures the proportion of positive predictions that are actually positive.
- recall: Measures the proportion of actual positive cases that are correctly predicted as positive.
- roc_auc_score: Measures the area under the receiver operating characteristic curve (ROC curve).
"""
random_search = RandomizedSearchCV(
    knn_clf, param_dist, n_iter=10, cv=3, scoring="accuracy"
)

# fit randomized_search to the data
random_search.fit(X_train, y_train)

# get the best parameters
# this should equal `{'n_neighbors': 7, 'weights': 'distance'}` and will be used by `random_search` for prediction
best_params = random_search.best_params_

best_model = random_search.best_estimator_

# predict on the training data with the best model
y_pred = best_model.predict(X_test)

# calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"KNeighborsClassifier's regularized accuracy is {accuracy * 100:.2f}%")
