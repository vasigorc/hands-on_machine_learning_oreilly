"""
Exercise 8:

Load the MNIST dataset (introduced in Chapter 3), and split it into a training set, a validation set,
and a test set (e.g. use 50_000 instances for training, 10_000 each for validation and testing). Then
train the various classifiers, such as a random forest classifier, an extra-treees classifier, and
a SVM classifier. Next, try to combine them into an ensemble that outperforms each individual classifier
on the validation set, using _soft_ or _hard_ voting. Once you have found one, try it on the test
set. How much better does it perform compared to the individual classifiers?
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)


def evaluate_single_classifier(X_train, y_train, X_validation, y_validation, clf):
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_validation, y_validation)
    return clf, accuracy


def main():
    # Step 1. Split the data
    mnist: Bunch = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X: np.ndarray = mnist.data
    y: np.ndarray = mnist.target

    # First split to get the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=10_000, random_state=42
    )
    # split the remaining data to get training and validation sets
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_temp, y_temp, test_size=10_000, random_state=42
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_validation.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Step 2. Evaluate individual classifiers
    rf_clf, rf_acc = evaluate_single_classifier(
        X_train,
        y_train,
        X_validation,
        y_validation,
        RandomForestClassifier(random_state=42),
    )
    print(
        f"RandomForestClassifier's accuracy on the validation set is {rf_acc * 100:.2f}%"
    )

    et_clf, et_acc = evaluate_single_classsifier(
        X_train,
        y_train,
        X_validation,
        y_validation,
        ExtraTreesClassifier(random_state=42),
    )
    print(
        f"ExtraTreesClassifier's accuracy on the validation set is {et_acc * 100:.2f}%"
    )

    svm_clf, svm_acc = evaluate_single_classsifier(
        X_train,
        y_train,
        X_validation,
        y_validation,
        SVC(probability=True, random_state=42),
    )
    print(f"SVC's accuracy on the validation set is {svm_acc * 100:.2f}%")

    # Step 3. Combine individual classifiers into an Ensemble
    # VotingClassifier with hard voting
    voting_clf_hard = VotingClassifier(
        estimators=[("rf", rf_clf), ("et", et_clf), ("svc", svm_clf)], voting="hard"
    )
    vt_clf_hard, vt_hard_acc = evaluate_single_classsifier(
        X_train, y_train, X_validation, y_validation, voting_clf_hard
    )
    print(
        f"VotingClassifier's accuracy on the validation set with 'hard' voting is {vt_hard_acc * 100:.2f}%"
    )

    # VotingClassifier with soft voting
    voting_clf_soft = VotingClassifier(
        estimators=[("rf", rf_clf), ("et", et_clf), ("svc", svm_clf)], voting="soft"
    )
    vt_clf_soft, vt_soft_acc = evaluate_single_classsifier(
        X_train, y_train, X_validation, y_validation, voting_clf_soft
    )
    print(
        f"VotingClassifier's accuracy on the validation set with 'soft' voting is {vt_soft_acc * 100:.2f}%"
    )

    # Step 4. Try the optimall ensemble on the test set
    voting_clf_soft_test_accuracy = voting_clf_soft.score(X_test, y_test)
    print(
        f"VotingClassifier's accuracy on the test set with 'soft' voting is {voting_clf_soft_test_accuracy * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
"""
The program prints:

    Training set size: 50000
    Validation set size: 10000
    Test set size: 10000

    RandomForestClassifier's accuracy on the validation set is 96.92%
    ExtraTreesClassifier's accuracy on the validation set is 97.15%
    SVC's accuracy on the validation set is 97.88%
    VotingClassifier's accuracy on the validation set with 'hard' voting is 97.44%
    VotingClassifier's accuracy on the validation set with 'soft' voting is 97.91%
    VotingClassifier's accuracy on the test set with 'soft' voting is 97.67%
"""
