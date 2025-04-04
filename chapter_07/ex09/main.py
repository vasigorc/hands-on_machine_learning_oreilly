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

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
)
from chapter_07.common.model_utils import (
    evaluate_single_classifier,
    train_and_evaluate_classifiers,
)
from chapter_07.common.data_utils import load_mnist_dataset, split_mnist_dataset


def main():
    # Step 1. Split the data
    X, y = load_mnist_dataset()

    X_train, X_validation, X_test, y_train, y_validation, y_test = split_mnist_dataset(
        X, y
    )

    # Step 2. Evaluate individual classifiers
    classifiers = [
        ("random_forest", RandomForestClassifier(random_state=42)),
        ("extra_trees", ExtraTreesClassifier(random_state=42)),
        ("svc", SVC(probability=True, random_state=42)),
    ]

    trained_models = train_and_evaluate_classifiers(
        classifiers, X_train, y_train, X_validation, y_validation
    )

    # Step 3. Get predictions from each classifier on the validation set
    # Blender matrix needs a feature matrix where each row is a training example (instance)
    # and a column in the index of each individual classifier
    validation_meta_features = np.column_stack(
        [clf.predict_proba(X_validation) for _, clf in trained_models]
    )

    # Step 4. Train a LogisticRegression classification algorithm on the new training set &
    # Step 5. Evaluate the ensemble on the test set. after training the blender.
    # Picking SVC since it performed best on the individual classifiers' level
    blender = LogisticRegression(random_state=42)

    # we also need meta features for the test set in order to assess correctly
    test_meta_features = np.column_stack(
        [clf.predict_proba(X_test) for _, clf in trained_models]
    )
    _, blender_acc = evaluate_single_classifier(
        validation_meta_features, y_validation, test_meta_features, y_test, blender
    )
    print(
        f"Blender's (LogisitRegression model) accuracy on the test set is {blender_acc * 100:.4f}%"
    )

    # Step 6. Try stacking using Scikit-Learn's `StackingClassifier`
    # We should use "fresh" estimators and an untrained final estimator,
    # because `StackingClassifier` handles the training process internally
    # and in a specific way
    stacking_clf = StackingClassifier(
        estimators=[
            ("random_forest", RandomForestClassifier(random_state=42)),
            ("extra_trees", ExtraTreesClassifier(random_state=42)),
            ("svc", SVC(probability=True, random_state=42)),
        ],
        cv=5,
    )
    stacking_clf.fit(X_train, y_train)
    stacking_clf_acc = stacking_clf.score(X_test, y_test)
    print(
        f"StackingClassifier's accuracy on the test set is {stacking_clf_acc * 100:.4f}%"
    )


if __name__ == "__main__":
    main()
    """
    The program prints:

        Training set size: 50000
        Validation set size: 10000
        Test set size: 10000

        random_forest's accuracy on the validation set is 0.9692
        extra_trees's accuracy on the validation set is 0.9715
        svc's accuracy on the validation set is 0.9788
        Blender's (LogisitRegression model) accuracy on the test set is 97.7600%
        StackingClassifier's accuracy on the test set is 97.7500%
"""
