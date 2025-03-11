from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

"""
Exercise 7. Train and fine-tune a decision tree for the moons dataset by following these steps:
  a. Use `make_moons(n_samples=10000, noise=0.4)` to generate a moons dataset.
  b. Use `test_train_split()` to split the dataset into a training set and a test set.
  c. Use grid search with cross-validation (with the help of the `GridSearchCV` class)
    to find good hyperparameter values for a `DecisionTreeClassifier`. Hint: try various
    values for `max_leaf_nodes`.
  d. Train it on the full training set using these hyperparameters, and measure
    your model's performance on the test set. You should get roughly 85% to 87% accuracy.
"""


def find_best_hyperparameters(X_train, y_train, random_state=42):
    """
    Find the best hyperparameters for a DecisionTreeClassifier using GridSearchCV.

    Args:
        X_train: training features
        y_train: training labels
        random_state: random seed for reproducibility

    Returns:
        grid_search: Fitted GridSearchCV object with best parameters
    """
    tree_clf = DecisionTreeClassifier(random_state=random_state)
    param_grid = [
        {"max_depth": [None, 3, 6, 9, 12, 14]},  # None means just unbounded
        {"max_leaf_nodes": [10, 20, 30, 40, 50, 80, 100]},
    ]
    grid_search = GridSearchCV(
        tree_clf, param_grid, cv=3, scoring="accuracy", return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def train_model_with_best_params(X_train, y_train, best_params, random_state=42):
    """
    Train a model with the best parameters found.

    Args:
        X_train: training features
        y_train: training labels
        best_params: Dictionary of best parameters
        random_state: random seed for reproducibility

    Returns:
        Trained DecisionTreeClassifier
    """
    best_tree = DecisionTreeClassifier(random_state=random_state, **best_params)
    best_tree.fit(X_train, y_train)
    return best_tree


def main():
    # Step a. Generate the moons dataset
    X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

    # Step b. Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step c. Find best hyperparameters
    grid_search = find_best_hyperparameters(X_train, y_train)

    print(f"Best hyperparameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Test score with grid search model: {grid_search.score(X_test, y_test):.4f}")

    # Step d. Train on a full training set with best parameters
    best_model = train_model_with_best_params(
        X_train, y_train, grid_search.best_params_
    )

    # Evaluate efficiency on the test set
    test_accuracy = best_model.score(X_test, y_test)
    print(f"Final model's test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
