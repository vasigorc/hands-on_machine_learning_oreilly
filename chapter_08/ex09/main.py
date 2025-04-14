"""
Exercise 9.

Load the MNIST dataset (introduced in Chapter 3) and split it into a training set and a
test set (take the first 60,000 instances for training, and the remaining 10,000 for testing). Train
a random forest classifier on the dataset and time how long it takes, then evaluate the resulting model
on the test set. Next, use PCA to reduce the dataset's dimensionality, with an explained variance ratio
of 95%. Train a new random forest classifier on the reduced dataset and see how long it takes. Was training
much faster? Next, evaluate the classifier on the test set. How does it compare to the previous classifier?
Try again with an `SGDClassifier`. How much does PCA help now?
"""

import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from functools import wraps
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
import numpy as np


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, *kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time

    return wrapper


@timeit
def evaluate_single_classifier(X_train, y_train, X_test, y_test, clf):
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    return clf, accuracy


def classifiers_factory():
    return [
        ("random_forest", RandomForestClassifier(random_state=42)),
        ("sgd", SGDClassifier(random_state=42)),
    ]


def main():
    # Step 1. Load and split the dataset
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data
    y = mnist.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10_000, random_state=42
    )

    # Step 2. Train and test classifiers without PCA
    base_classifiers = classifiers_factory()
    for name, clf in base_classifiers:
        (_, accuracy), timing = evaluate_single_classifier(
            X_train, y_train, X_test, y_test, clf
        )
        print(
            f"{name} took {timing:.4f} seconds to execute. It had the accuracy of {accuracy:.4f}"
        )

    # Step 3. Use PCA to reduce dimensions by keeping variance at 95%
    # find the optimal number of dimensions so as to keep variance at 95%
    pca = PCA()
    pca.fit(X_train)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1
    print(
        f"optimal number of retained dimensions for the MNIST dataset (to keep variance at 95%) is {d}"
    )
    # obtain the reduced training set
    pca = PCA(n_components=d)
    X_train_reduced = pca.fit_transform(X_train)

    # and the reduced test set
    # we need to use transform() instead of fit_transform(), this
    # 1. ensures we use the same principal components in training and testing
    # 2. makes sure that the model is evaluated on data transformed in the same way as it was trained
    # 2. prevents data leakage
    X_test_reduced = pca.transform(X_test)

    # Step 4. Train and test classifiers with PCA
    classifiers_with_pca = classifiers_factory()
    for name, clf in classifiers_with_pca:
        (_, accuracy), timing = evaluate_single_classifier(
            X_train_reduced, y_train, X_test_reduced, y_test, clf
        )
        print(
            f"{name} with PCA took {timing:.4f} seconds to execute. It had the accuracy of {accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
