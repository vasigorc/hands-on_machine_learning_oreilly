"""
Exercise 8. Grow a forrest by following these steps:

a. Continuing the previous exercise, generate 1000 subsets of the training set, each containing 100
instances selected randomly. Hint: you can use Scikit-Learn's `ShufffleSplit` class for this

b. Train one decision tree on each subset, using the best hyperparameters' values found in the previous
exercise. Evaluate these 1000 decision trees on the test set. Since they were trained on smaller sets,
these decision trees will likely perform worse than the first decision tree, achieving only about 80%
accuracy.

c. Now comes the magic. For each test set instance, generate the predictions of the 1000 decision trees
and keep only the most frequent prediction (you can use SciPy's `mode` function for this). This approach
gives you _majority-vote-predictions_ over the test set.

d. Evaluate thesee predictions on the test set: you should obtain a slightly higher accuracy than your
first model (between 0.5 and 1.5 % improvement). Congratulations, you have trained a random forrest
classifier.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from chapter_06.common.data_utils import generate_moons_dataset, split_train_test
from chapter_06.common.model_utils import (
    evaluate_model,
    find_best_hyperparameters,
    train_model_with_best_params,
)


def main():
    # creates 10_000 samples
    X, y = generate_moons_dataset()

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Steap  a. Generate training subsets, each containing 100 instances
    shuffle_split = ShuffleSplit(n_splits=1000, train_size=100, random_state=42)

    # Get the indices of samples selected for the training subset
    # The result will be a 2D array containing arrays of 100 indices pointing to rows within X_train
    subsets = [train_index for train_index, _ in shuffle_split.split(X_train)]

    grid_search = find_best_hyperparameters(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"Best hyperparameters: {best_params}")

    # Step b. Train a decision tree on each subset, evaluate them on the test set
    subset_decision_trees = [
        train_model_with_best_params(
            X_train[subset_indices], y_train[subset_indices], best_params
        )
        for subset_indices in subsets
    ]
    # Collect accuracies for each tree
    accuracies = [
        evaluate_model(tree, X_test, y_test) for tree in subset_decision_trees
    ]

    print(f"Mean accuracy of individual trees: {np.mean(accuracies):.4f}")
    print(f"Standard deviation: {np.std(accuracies):.4f}")
    print(
        f"Min accuracy: {np.min(accuracies):.4f}, Max accuracy: {np.max(accuracies):.4f}"
    )

    # Step c. Generate predictions from all trees and take majority vote
    raw_predictions = np.array([tree.predict(X_test) for tree in subset_decision_trees])
    """
    all_predictions is a 2D NumPy array with:
    - 1000 rows (one for each tree)
    - Each row having the length of X_test (actually 2000 in our case)
    So the shape is (1000, 2000):
    all_predictions = [
            [tree1's prediction for sample1, tree1's prediction for sample2, ...],
            [tree2's prediction for sample1, tree2's prediction for sample2, ...],
            ...
            [tree1000's prediction for sample1, tree1000's prediction for sample2, ...]
        ]
    We need to transpose this to get shape (2000, 1000), so that for each test
    sample, we have predictions from all trees grouped by samples:
    all_predictions = [
            [tree1's prediction for sample1, tree2's prediction for sample1, ..., tree1000's prediction for sample1],
            [tree1's prediction for sample2, tree2's prediction for sample2, ..., tree1000's prediction for sample2],
            ...
        ]
    """
    grouped_predictions = raw_predictions.T
    # identify the most common prediction by sample using `mode` function
    # we need axis=1, to get the most common predictions across all 1000 trees
    # if we left this at default (axis=0), we would be answering which prediction each tree makes most
    # often overall
    ensemble_predictions = stats.mode(grouped_predictions, axis=1)[0]

    # Step d. Evaluate grouped_predictions on thee test set
    ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)

    best_single_tree = train_model_with_best_params(X_train, y_train, best_params)
    single_tree_accuracy = evaluate_model(best_single_tree, X_test, y_test)

    print(f"Single tree accuracy: {single_tree_accuracy:.4f}")
    print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
    print(f"Improvement: {(ensemble_accuracy - single_tree_accuracy) * 100:.2f}%")


if __name__ == "__main__":
    main()
