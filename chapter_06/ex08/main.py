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

from sklearn.model_selection import ShuffleSplit
from chapter_06.common.data_utils import generate_moons_dataset, split_train_test
from chapter_06.common.model_utils import (
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
    subseet_decision_trees = [
        train_model_with_best_params(
            X_train[subset_indices], y_train[subset_indices], best_params
        )
        for subset_indices in subsets
    ]
    pass


if __name__ == "__main__":
    main()
