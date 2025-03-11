"""
Exercise 7. Train and fine-tune a decision tree for the moons dataset by following these steps:

a. Use `make_moons(n_samples=10000, noise=0.4)` to generate a moons dataset.
b. Use `test_train_split()` to split the dataset into a training set and a test set.
c. Use grid search with cross-validation (with the help of the `GridSearchCV` class) to find good hyperparameter
values for a `DecisionTreeClassifier`. Hint: try various values for `max_leaf_nodes`.
d. Train it on the full training set using these hyperparameters, and measure your model's performance
on the test set. You should get roughly 85% to 87% accuracy.
"""

from chapter_06.common.data_utils import generate_moons_dataset, split_train_test
from chapter_06.common.model_utils import (
    evaluate_model,
    find_best_hyperparameters,
    train_model_with_best_params,
)


def main():
    # Step a: Generate the moons dataset
    X, y = generate_moons_dataset()

    # Step b: Split into training and test sets
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Step c: Find best hyperparameters
    grid_search = find_best_hyperparameters(X_train, y_train)

    print(f"Best hyperparameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Test score with grid search model: {grid_search.score(X_test, y_test):.4f}")

    # Step d: Train on full training set with best parameters
    best_model = train_model_with_best_params(
        X_train, y_train, grid_search.best_params_
    )

    # Evaluate on test set
    test_accuracy = evaluate_model(best_model, X_test, y_test)
    """
    Supposed to print:
    Final model's test accuracy: 0.8700
    """
    print(f"Final model's test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
