"""
Exercise 9.

Run the individual classifiers from the previous exercise to make predictions on the validaiton set,
and create a new training set with the resulting predictions: each training instance is a vector containing
the set of predictions from all your classifiers for an image, and the target is the image's class.
Train a classifier on this new training set. Congratulations - you have just trained a blender, and
together with the classifiers it forms a stacking ensemble! Now evaluate the ensemble on the test set.
For each image in the test set, make predictions with all your classifiers, then feed the predictions
to the blender to get the ensemble's predictions. How does it compare to the voting classifier you trained
earlier? Now try again using a `StackingClassifier` instead. Do you get better performance? If so, why?
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
from chapter_07.common.model_utils import evaluate_single_classifier


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
    # Step 2. Run the individual classifiers from the previous exercise to make predictions on the validation set.
    individual_classifiers = [
        ("RandomForestClassifier", RandomForestClassifier(random_state=42)),
        ("ExtraTreesClassifier", ExtraTreesClassifier(random_state=42)),
        ("SVC", SVC(probability=True, random_state=42)),
    ]
    pass


if __name__ == "__main__":
    main()
