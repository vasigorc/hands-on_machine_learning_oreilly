"""
Exercise 8:

Load the MNIST dataset (introduced in Chapter 3), and split it into a training set, a validation set,
and a test set (e.g. use 50_000 instances for training, 10_000 each for validation and testing). Then
train the various classifiers, such as a random forest classifier, an extra-treees classifier, and
a SVM classifier. Next, try to combine them into an ensemble that outperforms each individual classifier
on the validation set, using _soft_ or _hard_ voting. Once you have found one, try it on the test
set. How much better does it perform compared to the individual classifiers?
"""

from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from chapter_07.common.model_utils import evaluate_single_classifier, train_and_evaluate_classifiers
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
        ("svc", SVC(probability=True, random_state=42))
    ]

    trained_models = train_and_evaluate_classifiers(
        classifiers, X_train, y_train, X_validation, y_validation
    )

    # Step 3. Combine individual classifiers into an Ensemble
    # VotingClassifier with hard voting
    voting_clf_hard = VotingClassifier(estimators=trained_models, voting="hard")

    vt_clf_hard, vt_hard_acc = evaluate_single_classifier(
        X_train, y_train, X_validation, y_validation, voting_clf_hard
    )
    print(
        f"VotingClassifier's accuracy on the validation set with 'hard' voting is {vt_hard_acc * 100:.2f}%"
    )

    # VotingClassifier with soft voting
    voting_clf_soft = VotingClassifier(estimators=trained_models, voting="soft")
    vt_clf_soft, vt_soft_acc = evaluate_single_classifier(
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
