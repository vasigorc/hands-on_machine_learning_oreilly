from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

from common import *

"""
Exercise 1. Try to build a classifier for the MNIST dataset that achieves over 97% accuracy on the 
test set. Hint: the `KNeighborsClassifier` works quite well for this task; you just need to find 
good hyperparameter values (try a grid search on the weights and n_neighbors hyperparameters)
"""


def create_knn_random_search():
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
    return RandomizedSearchCV(knn_clf, param_dist, n_iter=10, cv=3, scoring="accuracy")


def fit_and_get_best_model(X_train, y_train):
    random_search = create_knn_random_search()
    random_search.fit(X_train, y_train)
    return random_search
